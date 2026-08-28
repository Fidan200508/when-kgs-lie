from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import json
import random

from data import load_kb
from evidence import stable_seed


# ============================================================
# CONFIG
# ============================================================

CLEAN_DATA_PATH = Path(
    "data/processed/metaqa_pilot_clean.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "metaqa_pilot_entity_substitution.jsonl"
)

DEFAULT_SEED = 42


# ============================================================
# 1. LOAD / SAVE JSONL
# ============================================================

def load_jsonl(path):
    """
    JSONL dataset-i oxuyur.
    """

    items = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            items.append(
                json.loads(line)
            )

    return items


def save_jsonl(items, path):
    """
    JSONL dataset-i save edir.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for item in items:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# 2. BUILD KG CORRUPTION INDEXES
# ============================================================

def build_corruption_indexes(triples):
    """
    Corruption üçün KG-dən 3 əsas index qururuq.

    kg_triples:
        real KG triples set-i

    heads_by_relation:
        hər relation üçün mümkün head entities

    tails_by_relation:
        hər relation üçün mümkün tail entities
    """

    kg_triples = set()

    heads_by_relation = defaultdict(
        set
    )

    tails_by_relation = defaultdict(
        set
    )

    for triple in triples:

        head = triple["head"]
        relation = triple["relation"]
        tail = triple["tail"]

        kg_triples.add(
            (
                head,
                relation,
                tail,
            )
        )

        heads_by_relation[
            relation
        ].add(
            head
        )

        tails_by_relation[
            relation
        ].add(
            tail
        )

    return {
        "kg_triples":
            kg_triples,

        "heads_by_relation":
            heads_by_relation,

        "tails_by_relation":
            tails_by_relation,
    }


# ============================================================
# 3. GET ALL SUPPORT-PATH ENTITIES
# ============================================================

def get_support_entities(item):
    """
    Hazırkı clean support paths-də istifadə olunan
    bütün entities-i toplayır.

    Replacement zamanı bunlardan qaçmağa çalışırıq ki
    corruption təsadüfən başqa gold/support entity-yə
    çevrilməsin.
    """

    entities = set()

    entities.add(
        item["topic_entity"]
    )

    entities.update(
        item["gold_answers"]
    )

    for answer, paths in item[
        "support_paths"
    ].items():

        for path in paths:

            for step in path:

                entities.add(
                    step[
                        "from_entity"
                    ]
                )

                entities.add(
                    step[
                        "to_entity"
                    ]
                )

    return entities


# ============================================================
# 4. CHOOSE TARGET SUPPORT STEP
# ============================================================

def choose_target_step(
    item,
    position="answer-adjacent",
    seed=DEFAULT_SEED,
):
    """
    Hansı support edge-i corrupt edəcəyimizi seçir.

    Hazırda primary pilot üçün:

        position = answer-adjacent

    istifadə edirik.

    Multi-answer question-dadırsa,
    gold answers-dan biri deterministic seçilir.
    """

    candidates = []

    for answer, paths in item[
        "support_paths"
    ].items():

        if not paths:
            continue

        # Canonical selected path
        path = paths[0]

        for step in path:

            if step[
                "position"
            ] != position:
                continue

            candidates.append(
                {
                    "target_answer":
                        answer,

                    "step":
                        step,
                }
            )

    if not candidates:
        return None

    # deterministic ordering
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
            + f":target:{position}",
            seed,
        )
    )

    return rng.choice(
        candidates
    )


# ============================================================
# 5. BUILD TYPE-COMPATIBLE REPLACEMENT
# ============================================================

def choose_replacement_entity(
    item,
    step,
    indexes,
    seed=DEFAULT_SEED,
):
    """
    Traversal destination entity-ni type-compatible
    false entity ilə əvəz edir.

    Forward traversal:

        head --r--> tail

    destination = tail
    => relation-in başqa tail entity-si seçilir.

    Reverse traversal:

        tail --r^-1--> head

    destination = head
    => relation-in başqa head entity-si seçilir.
    """

    original_triple = tuple(
        step["triple"]
    )

    head, relation, tail = (
        original_triple
    )

    reverse = step[
        "reverse"
    ]

    original_destination = step[
        "to_entity"
    ]

    # -------------------------------------
    # Which side must be replaced?
    # -------------------------------------

    if reverse:

        # Traversal:
        #
        # tail -> head
        #
        # destination = original HEAD

        candidate_pool = indexes[
            "heads_by_relation"
        ][relation]

        replaced_side = "head"

    else:

        # Traversal:
        #
        # head -> tail
        #
        # destination = original TAIL

        candidate_pool = indexes[
            "tails_by_relation"
        ][relation]

        replaced_side = "tail"

    # -------------------------------------
    # Entities we should not reuse
    # -------------------------------------

    forbidden_entities = get_support_entities(
        item
    )

    forbidden_entities.add(
        original_destination
    )

    candidates = []

    for candidate in candidate_pool:

        if candidate in forbidden_entities:
            continue

        # ---------------------------------
        # Construct false triple
        # ---------------------------------

        if replaced_side == "head":

            corrupted_triple = (
                candidate,
                relation,
                tail,
            )

        else:

            corrupted_triple = (
                head,
                relation,
                candidate,
            )

        # ---------------------------------
        # CRITICAL:
        # corrupted triple must NOT already
        # exist as a true KG fact.
        # ---------------------------------

        if corrupted_triple in indexes[
            "kg_triples"
        ]:
            continue

        candidates.append(
            (
                candidate,
                corrupted_triple,
            )
        )

    if not candidates:
        return None

    # deterministic order
    candidates = sorted(
        candidates,
        key=lambda x: (
            x[0],
            x[1],
        ),
    )

    rng = random.Random(
        stable_seed(
            item["qid"]
            + ":entity_replacement:"
            + relation
            + ":"
            + str(
                step["hop_index"]
            ),
            seed,
        )
    )

    replacement_entity, corrupted_triple = (
        rng.choice(candidates)
    )

    return {
        "replacement_entity":
            replacement_entity,

        "corrupted_triple":
            corrupted_triple,

        "replaced_side":
            replaced_side,
    }


# ============================================================
# 6. REPLACE TRIPLE INSIDE EVIDENCE
# ============================================================

def replace_evidence_triple(
    evidence,
    original_triple,
    corrupted_triple,
):
    """
    Evidence order-u dəyişmədən yalnız target support
    triple-i replace edir.

    Bu çox vacibdir:

        clean order == corrupted order
    """

    new_evidence = deepcopy(
        evidence
    )

    replacement_count = 0

    for evidence_item in new_evidence:

        current_triple = tuple(
            evidence_item[
                "triple"
            ]
        )

        if (
            evidence_item["role"]
            == "support"
            and current_triple
            == original_triple
        ):

            evidence_item[
                "triple"
            ] = list(
                corrupted_triple
            )

            replacement_count += 1

    if replacement_count != 1:

        raise ValueError(
            "Expected exactly one support "
            f"triple replacement, got "
            f"{replacement_count}. "
            f"Original={original_triple}"
        )

    return new_evidence


# ============================================================
# 7. UPDATE SUPPORT TRIPLES
# ============================================================

def replace_support_triple_list(
    support_triples,
    original_triple,
    corrupted_triple,
):
    """
    support_triples metadata-nı da corrupted
    version-a uyğunlaşdırır.
    """

    output = []

    replacement_count = 0

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

            replacement_count += 1

        else:

            output.append(
                triple
            )

    if replacement_count != 1:

        raise ValueError(
            "support_triples replacement "
            f"count={replacement_count}"
        )

    return output


# ============================================================
# 8. APPLY ENTITY SUBSTITUTION
# ============================================================

def apply_entity_substitution(
    clean_item,
    indexes,
    position="answer-adjacent",
    seed=DEFAULT_SEED,
):
    """
    Clean evidence E-dən corrupted evidence E' yaradır.

    Yalnız ONE support step dəyişir.
    """

    target = choose_target_step(
        item=clean_item,
        position=position,
        seed=seed,
    )

    if target is None:
        return None

    target_answer = target[
        "target_answer"
    ]

    step = target[
        "step"
    ]

    replacement = (
        choose_replacement_entity(
            item=clean_item,
            step=step,
            indexes=indexes,
            seed=seed,
        )
    )

    if replacement is None:
        return None

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

    # -------------------------------------
    # Preserve clean metadata
    # -------------------------------------

    corrupted_item[
        "clean_evidence"
    ] = deepcopy(
        clean_item[
            "evidence"
        ]
    )

    corrupted_item[
        "clean_support_triples"
    ] = deepcopy(
        clean_item[
            "support_triples"
        ]
    )

    # -------------------------------------
    # Replace evidence triple
    # -------------------------------------

    corrupted_item[
        "evidence"
    ] = replace_evidence_triple(
        evidence=clean_item[
            "evidence"
        ],
        original_triple=
            original_triple,
        corrupted_triple=
            corrupted_triple,
    )

    # -------------------------------------
    # Replace support metadata
    # -------------------------------------

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

    # -------------------------------------
    # Corruption metadata
    # -------------------------------------

    injected_false_answer = None

    # Answer-adjacent corruption-da traversal
    # destination target answer-dır.
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

    corruption_manifest = {

        "corruption_type":
            "entity_substitution",

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

        "original_destination_entity":
            step["to_entity"],

        "replacement_entity":
            replacement[
                "replacement_entity"
            ],

        "original_triple":
            list(
                original_triple
            ),

        "corrupted_triple":
            list(
                corrupted_triple
            ),

        "injected_false_answer":
            injected_false_answer,

        "seed":
            seed,
    }

    corrupted_item[
        "corruption_type"
    ] = "entity_substitution"

    corrupted_item[
        "corrupted_position"
    ] = position

    corrupted_item[
        "injected_false_answer"
    ] = injected_false_answer

    corrupted_item[
        "corruption_manifest"
    ] = corruption_manifest

    return corrupted_item


# ============================================================
# 9. VALIDATE ENTITY CORRUPTION
# ============================================================

def validate_entity_substitution(
    clean_item,
    corrupted_item,
    indexes,
):
    """
    Automatic sanity checks.

    True olması lazım olan conditions:

    1. Evidence count dəyişməyib.
    2. Yalnız bir evidence triple dəyişib.
    3. Role/order dəyişməyib.
    4. Relation dəyişməyib.
    5. Corrupted triple KG-də true fact deyil.
    """

    if corrupted_item is None:
        return False, "corruption_none"

    clean_evidence = clean_item[
        "evidence"
    ]

    bad_evidence = corrupted_item[
        "evidence"
    ]

    if len(clean_evidence) != len(
        bad_evidence
    ):
        return False, "evidence_size_changed"

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

        # Role/order invariant
        if (
            clean_entry["role"]
            != bad_entry["role"]
        ):
            return False, "role_changed"

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
            f"changed_triples={len(changed)}"
        )

    manifest = corrupted_item[
        "corruption_manifest"
    ]

    original_triple = tuple(
        manifest[
            "original_triple"
        ]
    )

    corrupted_triple = tuple(
        manifest[
            "corrupted_triple"
        ]
    )

    # Relation must stay same
    if (
        original_triple[1]
        != corrupted_triple[1]
    ):
        return False, "relation_changed"

    # New triple MUST be false
    if corrupted_triple in indexes[
        "kg_triples"
    ]:
        return False, "corrupted_triple_is_true"

    # Replacement cannot be original destination
    if (
        manifest[
            "replacement_entity"
        ]
        == manifest[
            "original_destination_entity"
        ]
    ):
        return False, "replacement_not_changed"

    return True, "ok"


# ============================================================
# 10. BUILD CORRUPTED PILOT
# ============================================================

def build_entity_substitution_dataset(
    clean_items,
    indexes,
    position="answer-adjacent",
    seed=DEFAULT_SEED,
):
    """
    Bütün clean pilot items üçün paired
    entity-substitution condition yaradır.
    """

    corrupted_items = []

    failures = []

    for clean_item in clean_items:

        corrupted_item = (
            apply_entity_substitution(
                clean_item=clean_item,
                indexes=indexes,
                position=position,
                seed=seed,
            )
        )

        if corrupted_item is None:

            failures.append(
                (
                    clean_item["qid"],
                    "could_not_generate",
                )
            )

            continue

        valid, reason = (
            validate_entity_substitution(
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
                (
                    clean_item["qid"],
                    reason,
                )
            )

            continue

        corrupted_items.append(
            corrupted_item
        )

    return (
        corrupted_items,
        failures,
    )


# ============================================================
# 11. PRINT EXAMPLE
# ============================================================

def print_example(
    clean_item,
    corrupted_item,
):
    """
    Human-readable clean vs corrupted comparison.
    """

    manifest = corrupted_item[
        "corruption_manifest"
    ]

    print()
    print("=" * 70)
    print("ENTITY SUBSTITUTION EXAMPLE")
    print("=" * 70)

    print()
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
        "Gold answers:",
        clean_item[
            "gold_answers"
        ],
    )

    print(
        "Target gold answer:",
        manifest[
            "target_gold_answer"
        ],
    )

    print(
        "Position:",
        manifest[
            "corrupted_position"
        ],
    )

    print(
        "Reverse:",
        manifest[
            "reverse"
        ],
    )

    print()

    print(
        "ORIGINAL:"
    )

    print(
        " | ".join(
            manifest[
                "original_triple"
            ]
        )
    )

    print()

    print(
        "CORRUPTED:"
    )

    print(
        " | ".join(
            manifest[
                "corrupted_triple"
            ]
        )
    )

    print()

    print(
        "Replacement entity:",
        manifest[
            "replacement_entity"
        ],
    )

    print(
        "Injected false answer:",
        manifest[
            "injected_false_answer"
        ],
    )

    print()

    print(
        "CLEAN EVIDENCE -> CORRUPTED EVIDENCE"
    )

    print()

    clean_evidence = clean_item[
        "evidence"
    ]

    bad_evidence = corrupted_item[
        "evidence"
    ]

    for clean_entry, bad_entry in zip(
        clean_evidence,
        bad_evidence,
    ):

        clean_triple = clean_entry[
            "triple"
        ]

        bad_triple = bad_entry[
            "triple"
        ]

        if clean_triple == bad_triple:

            print(
                " SAME | "
                + " | ".join(
                    clean_triple
                )
            )

        else:

            print(
                "CLEAN | "
                + " | ".join(
                    clean_triple
                )
            )

            print(
                "  BAD | "
                + " | ".join(
                    bad_triple
                )
            )


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    # -------------------------------------
    # Load clean frozen evidence
    # -------------------------------------

    print(
        "Loading clean pilot..."
    )

    clean_items = load_jsonl(
        CLEAN_DATA_PATH
    )

    print(
        f"Clean items: "
        f"{len(clean_items)}"
    )

    # -------------------------------------
    # Load KG
    # -------------------------------------

    print(
        "Loading KG..."
    )

    triples = load_kb()

    print(
        f"KG triples: "
        f"{len(triples)}"
    )

    # -------------------------------------
    # Build corruption indexes
    # -------------------------------------

    print(
        "Building corruption indexes..."
    )

    indexes = build_corruption_indexes(
        triples
    )

    print(
        "Relations indexed:",
        len(
            indexes[
                "heads_by_relation"
            ]
        ),
    )

    # -------------------------------------
    # Generate
    # -------------------------------------

    print()
    print(
        "Generating entity substitutions..."
    )
    print()

    corrupted_items, failures = (
        build_entity_substitution_dataset(
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

    # -------------------------------------
    # Save
    # -------------------------------------

    save_jsonl(
        corrupted_items,
        OUTPUT_PATH,
    )

    # -------------------------------------
    # Summary
    # -------------------------------------

    print(
        f"Successful corruptions: "
        f"{len(corrupted_items)}/"
        f"{len(clean_items)}"
    )

    print(
        f"Failures: "
        f"{len(failures)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    if failures:

        print()
        print(
            "First failures:"
        )

        for failure in failures[:10]:

            print(
                failure
            )

    # -------------------------------------
    # Hop counts
    # -------------------------------------

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

    # -------------------------------------
    # Show first example
    # -------------------------------------

    if corrupted_items:

        first_bad = (
            corrupted_items[0]
        )

        clean_lookup = {
            item["qid"]: item
            for item in clean_items
        }

        first_clean = clean_lookup[
            first_bad["qid"]
        ]

        print_example(
            first_clean,
            first_bad,
        )