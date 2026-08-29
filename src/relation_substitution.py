from pathlib import Path
from copy import deepcopy
import json
import random

from data import load_kb
from evidence import stable_seed
from corruptions import (
    load_jsonl,
    save_jsonl,
    build_corruption_indexes,
)


# ============================================================
# CONFIG
# ============================================================

CLEAN_DATA_PATH = Path(
    "data/processed/metaqa_pilot_clean.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "metaqa_pilot_relation_substitution.jsonl"
)

FAILURE_PATH = Path(
    "data/processed/"
    "metaqa_pilot_relation_substitution_failures.jsonl"
)

DEFAULT_SEED = 42


# ============================================================
# 1. RELATION COMPATIBILITY
# ============================================================

RELATION_FAMILIES = {

    "person_relation": {
        "directed_by",
        "written_by",
        "starred_actors",
    },

}


def get_relation_family(relation):
    """
    Relation hansı schema-compatible family-yə daxildir?
    """

    for family_name, relations in RELATION_FAMILIES.items():

        if relation in relations:

            return (
                family_name,
                relations,
            )

    return (
        None,
        None,
    )


# ============================================================
# 2. COLLECT ELIGIBLE SUPPORT STEPS
# ============================================================

def collect_eligible_steps(
    item,
    position="answer-adjacent",
):
    """
    Relation substitution üçün uyğun support steps tapır.

    Şərtlər:
    - tələb olunan position
    - relation compatible family-də olmalıdır
    """

    candidates = []

    for answer, paths in item[
        "support_paths"
    ].items():

        if not paths:
            continue

        path = paths[0]

        for step in path:

            if step["position"] != position:
                continue

            family_name, family_relations = (
                get_relation_family(
                    step["relation"]
                )
            )

            if family_name is None:
                continue

            candidates.append(
                {
                    "target_answer": answer,
                    "step": step,
                    "family_name": family_name,
                }
            )

    return candidates


# ============================================================
# 3. CHOOSE TARGET STEP
# ============================================================

def choose_relation_target(
    item,
    position="answer-adjacent",
    seed=DEFAULT_SEED,
):
    """
    Bir eligible support step deterministic seçilir.
    """

    candidates = collect_eligible_steps(
        item=item,
        position=position,
    )

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["target_answer"],
            x["step"]["hop_index"],
            x["step"]["relation"],
            x["step"]["from_entity"],
            x["step"]["to_entity"],
        ),
    )

    rng = random.Random(
        stable_seed(
            item["qid"]
            + ":relation_target:"
            + position,
            seed,
        )
    )

    return rng.choice(
        candidates
    )


# ============================================================
# 4. CHOOSE REPLACEMENT RELATION
# ============================================================

def choose_replacement_relation(
    item,
    step,
    indexes,
    seed=DEFAULT_SEED,
):
    """
    Original:

        (head, relation, tail)

    Corrupted:

        (head, new_relation, tail)

    Head və tail dəyişmir.

    New relation:
    - same compatibility family
    - original relation-dan fərqli
    - resulting triple real KG-də olmamalıdır
    """

    original_triple = tuple(
        step["triple"]
    )

    head, relation, tail = (
        original_triple
    )

    family_name, family_relations = (
        get_relation_family(
            relation
        )
    )

    if family_name is None:
        return None

    candidate_relations = []

    for new_relation in family_relations:

        if new_relation == relation:
            continue

        corrupted_triple = (
            head,
            new_relation,
            tail,
        )

        # New triple artıq KG-də həqiqi fact olmasın
        if corrupted_triple in indexes[
            "kg_triples"
        ]:
            continue

        candidate_relations.append(
            (
                new_relation,
                corrupted_triple,
            )
        )

    if not candidate_relations:
        return None

    candidate_relations = sorted(
        candidate_relations,
        key=lambda x: x[0],
    )

    rng = random.Random(
        stable_seed(
            item["qid"]
            + ":relation_replacement:"
            + relation
            + ":"
            + str(
                step["hop_index"]
            ),
            seed,
        )
    )

    new_relation, corrupted_triple = (
        rng.choice(
            candidate_relations
        )
    )

    return {
        "original_relation": relation,
        "replacement_relation": new_relation,
        "corrupted_triple": corrupted_triple,
        "family_name": family_name,
    }


# ============================================================
# 5. REPLACE EVIDENCE TRIPLE
# ============================================================

def replace_evidence_triple(
    evidence,
    original_triple,
    corrupted_triple,
):
    """
    Evidence order/size dəyişmir.

    Yalnız ONE support triple dəyişir.
    """

    output = deepcopy(
        evidence
    )

    count = 0

    for entry in output:

        triple = tuple(
            entry["triple"]
        )

        if (
            entry["role"] == "support"
            and triple == original_triple
        ):

            entry["triple"] = list(
                corrupted_triple
            )

            count += 1

    if count != 1:

        raise ValueError(
            f"Expected 1 evidence replacement, got {count}"
        )

    return output


# ============================================================
# 6. REPLACE SUPPORT METADATA
# ============================================================

def replace_support_triple_list(
    support_triples,
    original_triple,
    corrupted_triple,
):
    """
    support_triples metadata-nı corrupted version-a uyğunlaşdırır.
    """

    output = []

    count = 0

    for triple in support_triples:

        triple_tuple = tuple(
            triple
        )

        if triple_tuple == original_triple:

            output.append(
                list(
                    corrupted_triple
                )
            )

            count += 1

        else:

            output.append(
                triple
            )

    if count != 1:

        raise ValueError(
            f"Expected 1 support replacement, got {count}"
        )

    return output


# ============================================================
# 7. APPLY RELATION SUBSTITUTION
# ============================================================

def apply_relation_substitution(
    clean_item,
    indexes,
    position="answer-adjacent",
    seed=DEFAULT_SEED,
):
    """
    Clean evidence E:

        (h, r, t)

    becomes:

        (h, r', t)
    """

    target = choose_relation_target(
        item=clean_item,
        position=position,
        seed=seed,
    )

    if target is None:
        return None, "no_eligible_support_relation"

    step = target[
        "step"
    ]

    target_answer = target[
        "target_answer"
    ]

    replacement = (
        choose_replacement_relation(
            item=clean_item,
            step=step,
            indexes=indexes,
            seed=seed,
        )
    )

    if replacement is None:
        return None, "no_valid_relation_replacement"

    original_triple = tuple(
        step["triple"]
    )

    corrupted_triple = tuple(
        replacement[
            "corrupted_triple"
        ]
    )

    corrupted_item = deepcopy(
        clean_item
    )

    # --------------------------------------------------------
    # Preserve clean data
    # --------------------------------------------------------

    corrupted_item[
        "clean_evidence"
    ] = deepcopy(
        clean_item["evidence"]
    )

    corrupted_item[
        "clean_support_triples"
    ] = deepcopy(
        clean_item["support_triples"]
    )

    # --------------------------------------------------------
    # Modify evidence
    # --------------------------------------------------------

    corrupted_item[
        "evidence"
    ] = replace_evidence_triple(
        evidence=clean_item[
            "evidence"
        ],
        original_triple=original_triple,
        corrupted_triple=corrupted_triple,
    )

    # --------------------------------------------------------
    # Modify support metadata
    # --------------------------------------------------------

    corrupted_item[
        "support_triples"
    ] = replace_support_triple_list(
        support_triples=
            clean_item[
                "support_triples"
            ],
        original_triple=
            original_triple,
        corrupted_triple=
            corrupted_triple,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "corruption_type":
            "relation_substitution",

        "corruption_severity":
            1,

        "corrupted_position":
            position,

        "target_gold_answer":
            target_answer,

        "hop_index":
            step["hop_index"],

        "reverse":
            step["reverse"],

        "relation_family":
            replacement[
                "family_name"
            ],

        "original_relation":
            replacement[
                "original_relation"
            ],

        "replacement_relation":
            replacement[
                "replacement_relation"
            ],

        "original_triple":
            list(
                original_triple
            ),

        "corrupted_triple":
            list(
                corrupted_triple
            ),

        # Relation corruption changes predicate,
        # not answer entity.
        "injected_false_answer":
            None,

        "seed":
            seed,
    }

    corrupted_item[
        "corruption_type"
    ] = "relation_substitution"

    corrupted_item[
        "corrupted_position"
    ] = position

    corrupted_item[
        "injected_false_answer"
    ] = None

    corrupted_item[
        "corruption_manifest"
    ] = manifest

    return (
        corrupted_item,
        "ok",
    )


# ============================================================
# 8. VALIDATION
# ============================================================

def validate_relation_substitution(
    clean_item,
    corrupted_item,
    indexes,
):
    """
    Relation substitution sanity checks.

    Expected:

    - evidence size unchanged
    - exactly one evidence triple changed
    - head unchanged
    - tail unchanged
    - relation changed
    - original/replacement same compatible family
    - corrupted triple absent from true KG
    """

    clean_evidence = clean_item[
        "evidence"
    ]

    bad_evidence = corrupted_item[
        "evidence"
    ]

    if len(clean_evidence) != len(
        bad_evidence
    ):

        return (
            False,
            "evidence_size_changed",
        )

    changed = []

    for index, (
        clean_entry,
        bad_entry,
    ) in enumerate(
        zip(
            clean_evidence,
            bad_evidence,
        )
    ):

        if (
            clean_entry["role"]
            != bad_entry["role"]
        ):

            return (
                False,
                "role_changed",
            )

        if (
            clean_entry["triple"]
            != bad_entry["triple"]
        ):

            changed.append(
                index
            )

    if len(changed) != 1:

        return (
            False,
            f"changed_count={len(changed)}",
        )

    manifest = corrupted_item[
        "corruption_manifest"
    ]

    original = tuple(
        manifest[
            "original_triple"
        ]
    )

    corrupted = tuple(
        manifest[
            "corrupted_triple"
        ]
    )

    # Head unchanged
    if original[0] != corrupted[0]:

        return (
            False,
            "head_changed",
        )

    # Tail unchanged
    if original[2] != corrupted[2]:

        return (
            False,
            "tail_changed",
        )

    # Relation MUST change
    if original[1] == corrupted[1]:

        return (
            False,
            "relation_not_changed",
        )

    # False fact
    if corrupted in indexes[
        "kg_triples"
    ]:

        return (
            False,
            "corrupted_triple_is_true",
        )

    original_family, _ = (
        get_relation_family(
            original[1]
        )
    )

    corrupted_family, _ = (
        get_relation_family(
            corrupted[1]
        )
    )

    if (
        original_family is None
        or original_family
        != corrupted_family
    ):

        return (
            False,
            "relation_family_mismatch",
        )

    return (
        True,
        "ok",
    )


# ============================================================
# 9. BUILD DATASET
# ============================================================

def build_relation_dataset(
    clean_items,
    indexes,
    position="answer-adjacent",
    seed=DEFAULT_SEED,
):
    """
    Relation substitution yalnız feasible items-də yaradılır.
    """

    successful = []
    failures = []

    for clean_item in clean_items:

        corrupted_item, reason = (
            apply_relation_substitution(
                clean_item=clean_item,
                indexes=indexes,
                position=position,
                seed=seed,
            )
        )

        if corrupted_item is None:

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
            validate_relation_substitution(
                clean_item=
                    clean_item,

                corrupted_item=
                    corrupted_item,

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
            corrupted_item
        )

    return (
        successful,
        failures,
    )


# ============================================================
# 10. PRINT EXAMPLE
# ============================================================

def print_example(
    clean_item,
    corrupted_item,
):

    manifest = corrupted_item[
        "corruption_manifest"
    ]

    print()
    print("=" * 70)
    print("RELATION SUBSTITUTION EXAMPLE")
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
        "Position:",
        manifest[
            "corrupted_position"
        ],
    )

    print(
        "Family:",
        manifest[
            "relation_family"
        ],
    )

    print()

    print("ORIGINAL:")

    print(
        " | ".join(
            manifest[
                "original_triple"
            ]
        )
    )

    print()

    print("CORRUPTED:")

    print(
        " | ".join(
            manifest[
                "corrupted_triple"
            ]
        )
    )

    print()

    print(
        "Relation:",
        manifest[
            "original_relation"
        ],
        "->",
        manifest[
            "replacement_relation"
        ],
    )


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":

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

    print(
        "Loading KG..."
    )

    triples = load_kb()

    print(
        f"KG triples: "
        f"{len(triples)}"
    )

    print(
        "Building corruption indexes..."
    )

    indexes = build_corruption_indexes(
        triples
    )

    print()
    print(
        "Generating relation substitutions..."
    )
    print()

    corrupted_items, failures = (
        build_relation_dataset(
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

    save_jsonl(
        corrupted_items,
        OUTPUT_PATH,
    )

    save_jsonl(
        failures,
        FAILURE_PATH,
    )

    print(
        "Successful:",
        f"{len(corrupted_items)}/"
        f"{len(clean_items)}"
    )

    print(
        "Infeasible/failed:",
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

    # --------------------------------------------------------
    # Hop distribution
    # --------------------------------------------------------

    hop_counts = {
        1: 0,
        2: 0,
        3: 0,
    }

    for item in corrupted_items:

        hop_counts[
            item["hop"]
        ] += 1

    print(
        "Hop counts:",
        hop_counts,
    )

    # --------------------------------------------------------
    # Failure reasons
    # --------------------------------------------------------

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

    if corrupted_items:

        clean_lookup = {
            item["qid"]: item
            for item in clean_items
        }

        corrupted_example = (
            corrupted_items[0]
        )

        clean_example = (
            clean_lookup[
                corrupted_example[
                    "qid"
                ]
            ]
        )

        print_example(
            clean_example,
            corrupted_example,
        )