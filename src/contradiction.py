from pathlib import Path
from copy import deepcopy
import random

from data import load_kb
from evidence import stable_seed
from corruptions import (
    load_jsonl,
    save_jsonl,
    build_corruption_indexes,
    choose_target_step,
    choose_replacement_entity,
)


# ============================================================
# CONFIG
# ============================================================

CLEAN_DATA_PATH = Path(
    "data/processed/metaqa_pilot_clean.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "metaqa_pilot_contradiction.jsonl"
)

FAILURE_PATH = Path(
    "data/processed/"
    "metaqa_pilot_contradiction_failures.jsonl"
)

DEFAULT_SEED = 42


# ============================================================
# 1. CHOOSE CONTEXT SLOT
# ============================================================

def choose_context_index(
    item,
    seed=DEFAULT_SEED,
):
    """
    Contradictory triple-i əlavə edərkən evidence budget
    artmasın deyə bir context slot seçirik.

    Support triple heç vaxt silinmir.

    Selection deterministic-dir.
    """

    context_indices = []

    for index, entry in enumerate(
        item["evidence"]
    ):

        if entry["role"] == "context":
            context_indices.append(
                index
            )

    if not context_indices:
        return None

    rng = random.Random(
        stable_seed(
            item["qid"]
            + ":contradiction_context_slot",
            seed,
        )
    )

    return rng.choice(
        context_indices
    )


# ============================================================
# 2. APPLY CONTRADICTION
# ============================================================

def apply_contradiction(
    clean_item,
    indexes,
    position="answer-adjacent",
    seed=DEFAULT_SEED,
):
    """
    True support triple saxlanılır.

    Əlavə olaraq eyni relation üçün type-compatible
    false conflicting triple yaradılır.

    Evidence size sabit qalması üçün bir context triple
    həmin contradiction ilə əvəz olunur.
    """

    # --------------------------------------------------------
    # A. Choose answer-adjacent support step
    # --------------------------------------------------------

    target = choose_target_step(
        item=clean_item,
        position=position,
        seed=seed,
    )

    if target is None:
        return (
            None,
            "no_target_support_step",
        )

    target_answer = target[
        "target_answer"
    ]

    step = target[
        "step"
    ]

    # --------------------------------------------------------
    # B. Generate type-compatible false entity
    # --------------------------------------------------------

    replacement = choose_replacement_entity(
        item=clean_item,
        step=step,
        indexes=indexes,
        seed=seed,
    )

    if replacement is None:
        return (
            None,
            "no_false_entity_candidate",
        )

    original_triple = tuple(
        step["triple"]
    )

    contradictory_triple = tuple(
        replacement[
            "corrupted_triple"
        ]
    )

    # --------------------------------------------------------
    # C. Choose context slot
    # --------------------------------------------------------

    context_index = choose_context_index(
        clean_item,
        seed=seed,
    )

    if context_index is None:
        return (
            None,
            "no_context_slot",
        )

    removed_context_triple = tuple(
        clean_item[
            "evidence"
        ][context_index][
            "triple"
        ]
    )

    # --------------------------------------------------------
    # D. Copy clean item
    # --------------------------------------------------------

    contradiction_item = deepcopy(
        clean_item
    )

    contradiction_item[
        "clean_evidence"
    ] = deepcopy(
        clean_item["evidence"]
    )

    contradiction_item[
        "clean_support_triples"
    ] = deepcopy(
        clean_item[
            "support_triples"
        ]
    )

    # --------------------------------------------------------
    # E. Replace ONLY a context slot
    # --------------------------------------------------------

    contradiction_item[
        "evidence"
    ][context_index] = {
        "triple":
            list(
                contradictory_triple
            ),

        "role":
            "contradiction",
    }

    # --------------------------------------------------------
    # F. False answer
    # --------------------------------------------------------

    injected_false_answer = None

    if (
        position == "answer-adjacent"
        and step["to_entity"]
        == target_answer
    ):
        injected_false_answer = (
            replacement[
                "replacement_entity"
            ]
        )

    # --------------------------------------------------------
    # G. Manifest
    # --------------------------------------------------------

    manifest = {

        "corruption_type":
            "contradictory_insertion",

        "corruption_severity":
            1,

        "corrupted_position":
            position,

        "target_gold_answer":
            target_answer,

        "hop_index":
            step["hop_index"],

        "relation":
            step["relation"],

        "reverse":
            step["reverse"],

        "replaced_side":
            replacement[
                "replaced_side"
            ],

        "true_triple":
            list(
                original_triple
            ),

        "contradictory_triple":
            list(
                contradictory_triple
            ),

        "replacement_entity":
            replacement[
                "replacement_entity"
            ],

        "injected_false_answer":
            injected_false_answer,

        "removed_context_index":
            context_index,

        "removed_context_triple":
            list(
                removed_context_triple
            ),

        "evidence_budget_preserved":
            True,

        "seed":
            seed,
    }

    contradiction_item[
        "corruption_type"
    ] = "contradictory_insertion"

    contradiction_item[
        "corrupted_position"
    ] = position

    contradiction_item[
        "injected_false_answer"
    ] = injected_false_answer

    contradiction_item[
        "corruption_manifest"
    ] = manifest

    return (
        contradiction_item,
        "ok",
    )


# ============================================================
# 3. VALIDATE CONTRADICTION
# ============================================================

def validate_contradiction(
    clean_item,
    contradiction_item,
    indexes,
):
    """
    Main invariants:

    1. Evidence count unchanged.
    2. True support triple remains present.
    3. Exactly one context slot changed.
    4. New triple has same relation.
    5. New triple is absent from true KG.
    6. New triple conflicts with true support triple.
    """

    clean_evidence = clean_item[
        "evidence"
    ]

    bad_evidence = contradiction_item[
        "evidence"
    ]

    # --------------------------------------------------------
    # Same evidence budget
    # --------------------------------------------------------

    if len(clean_evidence) != len(
        bad_evidence
    ):
        return (
            False,
            "evidence_size_changed",
        )

    manifest = contradiction_item[
        "corruption_manifest"
    ]

    true_triple = tuple(
        manifest[
            "true_triple"
        ]
    )

    contradiction = tuple(
        manifest[
            "contradictory_triple"
        ]
    )

    # --------------------------------------------------------
    # True fact must remain
    # --------------------------------------------------------

    true_support_present = False

    for entry in bad_evidence:

        if (
            entry["role"] == "support"
            and tuple(
                entry["triple"]
            ) == true_triple
        ):
            true_support_present = True
            break

    if not true_support_present:

        return (
            False,
            "true_support_removed",
        )

    # --------------------------------------------------------
    # Exactly one evidence slot changes
    # --------------------------------------------------------

    changed_indices = []

    for index, (
        clean_entry,
        bad_entry,
    ) in enumerate(
        zip(
            clean_evidence,
            bad_evidence,
        )
    ):

        if clean_entry != bad_entry:
            changed_indices.append(
                index
            )

    if len(changed_indices) != 1:

        return (
            False,
            f"changed_slots="
            f"{len(changed_indices)}",
        )

    changed_index = changed_indices[0]

    if (
        changed_index
        != manifest[
            "removed_context_index"
        ]
    ):
        return (
            False,
            "wrong_changed_slot",
        )

    # Original slot must have been context
    if (
        clean_evidence[
            changed_index
        ]["role"]
        != "context"
    ):
        return (
            False,
            "support_slot_replaced",
        )

    # New role must be contradiction
    if (
        bad_evidence[
            changed_index
        ]["role"]
        != "contradiction"
    ):
        return (
            False,
            "wrong_inserted_role",
        )

    # --------------------------------------------------------
    # Same relation
    # --------------------------------------------------------

    if (
        true_triple[1]
        != contradiction[1]
    ):
        return (
            False,
            "relation_changed",
        )

    # --------------------------------------------------------
    # False triple cannot exist in KG
    # --------------------------------------------------------

    if contradiction in indexes[
        "kg_triples"
    ]:
        return (
            False,
            "contradiction_is_true_fact",
        )

    # --------------------------------------------------------
    # Actual difference required
    # --------------------------------------------------------

    if true_triple == contradiction:

        return (
            False,
            "contradiction_equals_true",
        )

    return (
        True,
        "ok",
    )


# ============================================================
# 4. BUILD DATASET
# ============================================================

def build_contradiction_dataset(
    clean_items,
    indexes,
    position="answer-adjacent",
    seed=DEFAULT_SEED,
):
    """
    Bütün clean pilot üçün contradiction condition yaradır.
    """

    successful = []
    failures = []

    for clean_item in clean_items:

        bad_item, reason = (
            apply_contradiction(
                clean_item=clean_item,
                indexes=indexes,
                position=position,
                seed=seed,
            )
        )

        if bad_item is None:

            failures.append(
                {
                    "qid":
                        clean_item[
                            "qid"
                        ],

                    "hop":
                        clean_item[
                            "hop"
                        ],

                    "reason":
                        reason,
                }
            )

            continue

        valid, validation_reason = (
            validate_contradiction(
                clean_item=
                    clean_item,

                contradiction_item=
                    bad_item,

                indexes=
                    indexes,
            )
        )

        if not valid:

            failures.append(
                {
                    "qid":
                        clean_item[
                            "qid"
                        ],

                    "hop":
                        clean_item[
                            "hop"
                        ],

                    "reason":
                        validation_reason,
                }
            )

            continue

        successful.append(
            bad_item
        )

    return (
        successful,
        failures,
    )


# ============================================================
# 5. SAVE / PRINT EXAMPLE
# ============================================================

def print_example(
    clean_item,
    bad_item,
):
    """
    Human-readable clean vs contradiction example.
    """

    m = bad_item[
        "corruption_manifest"
    ]

    print()
    print("=" * 70)
    print("CONTRADICTION EXAMPLE")
    print("=" * 70)

    print(
        "QID:",
        clean_item["qid"],
    )

    print(
        "Hop:",
        clean_item["hop"],
    )

    print(
        "Question:",
        clean_item["question"],
    )

    print(
        "Gold:",
        clean_item["gold_answers"],
    )

    print(
        "Target gold:",
        m[
            "target_gold_answer"
        ],
    )

    print(
        "Position:",
        m[
            "corrupted_position"
        ],
    )

    print()

    print("TRUE:")
    print(
        " | ".join(
            m["true_triple"]
        )
    )

    print()

    print("FALSE CONTRADICTION:")
    print(
        " | ".join(
            m[
                "contradictory_triple"
            ]
        )
    )

    print()

    print(
        "Injected false answer:",
        m[
            "injected_false_answer"
        ],
    )

    print()

    print(
        "REMOVED CONTEXT:"
    )

    print(
        " | ".join(
            m[
                "removed_context_triple"
            ]
        )
    )

    print()

    print(
        "Evidence size:",
        len(
            clean_item["evidence"]
        ),
        "->",
        len(
            bad_item["evidence"]
        ),
    )

    print()

    print(
        "CONTRADICTION CONDITION EVIDENCE:"
    )

    print()

    for entry in bad_item[
        "evidence"
    ]:

        triple = entry[
            "triple"
        ]

        print(
            f"{entry['role'].upper():13} | "
            f"{triple[0]} | "
            f"{triple[1]} | "
            f"{triple[2]}"
        )


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Clean evidence
    # --------------------------------------------------------

    print(
        "Loading frozen clean pilot..."
    )

    clean_items = load_jsonl(
        CLEAN_DATA_PATH
    )

    print(
        f"Clean items: "
        f"{len(clean_items)}"
    )

    # --------------------------------------------------------
    # KG
    # --------------------------------------------------------

    print(
        "Loading KG..."
    )

    triples = load_kb()

    print(
        f"KG triples: "
        f"{len(triples)}"
    )

    indexes = build_corruption_indexes(
        triples
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print()
    print(
        "Generating contradictions..."
    )
    print()

    bad_items, failures = (
        build_contradiction_dataset(
            clean_items=
                clean_items,

            indexes=
                indexes,

            position=
                "answer-adjacent",

            seed=
                DEFAULT_SEED,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_jsonl(
        bad_items,
        OUTPUT_PATH,
    )

    save_jsonl(
        failures,
        FAILURE_PATH,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "Successful:",
        f"{len(bad_items)}/"
        f"{len(clean_items)}"
    )

    print(
        "Failures:",
        len(failures),
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print(
        "Failure log:",
        FAILURE_PATH,
    )

    hop_counts = {
        1: 0,
        2: 0,
        3: 0,
    }

    for item in bad_items:

        hop_counts[
            item["hop"]
        ] += 1

    print(
        "Hop counts:",
        hop_counts,
    )

    reason_counts = {}

    for failure in failures:

        reason = failure[
            "reason"
        ]

        reason_counts[
            reason
        ] = (
            reason_counts.get(
                reason,
                0,
            )
            + 1
        )

    print(
        "Failure reasons:",
        reason_counts,
    )

    # --------------------------------------------------------
    # Example
    # --------------------------------------------------------

    if bad_items:

        clean_lookup = {
            item["qid"]: item
            for item in clean_items
        }

        example_bad = bad_items[0]

        example_clean = clean_lookup[
            example_bad["qid"]
        ]

        print_example(
            example_clean,
            example_bad,
        )