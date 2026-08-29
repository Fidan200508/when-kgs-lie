from pathlib import Path
from copy import deepcopy
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
    "data/processed/metaqa_pilot_rerouting.jsonl"
)

FAILURE_PATH = Path(
    "data/processed/metaqa_pilot_rerouting_failures.jsonl"
)

DEFAULT_SEED = 42


# ============================================================
# 1. GET STRUCTURAL SLOT POOL
# ============================================================

def get_slot_pool(
    step,
    intermediate_entity,
    indexes,
):
    """
    Intermediate entity original KG triple-da hansı tərəfdədirsə,
    həmin relation üçün eyni tərəfdə görünən entity-ləri qaytarır.

    Example:

        Movie | directed_by | Person

    Movie -> head slot
    Person -> tail slot

    Bu, replacement node-un structural/type compatibility-sini
    qorumaq üçündür.
    """

    head, relation, tail = tuple(
        step["triple"]
    )

    if intermediate_entity == head:

        return (
            set(
                indexes[
                    "heads_by_relation"
                ][relation]
            ),
            "head",
        )

    if intermediate_entity == tail:

        return (
            set(
                indexes[
                    "tails_by_relation"
                ][relation]
            ),
            "tail",
        )

    return (
        set(),
        None,
    )


# ============================================================
# 2. REPLACE ONE ENTITY INSIDE A TRIPLE
# ============================================================

def replace_entity_in_triple(
    triple,
    old_entity,
    new_entity,
):
    """
    Triple daxilində old_entity-ni new_entity ilə əvəz edir.

    Relation dəyişmir.
    """

    head, relation, tail = tuple(
        triple
    )

    if head == old_entity:
        head = new_entity

    if tail == old_entity:
        tail = new_entity

    return (
        head,
        relation,
        tail,
    )


# ============================================================
# 3. COLLECT VALID INTERMEDIATE TARGETS
# ============================================================

def collect_rerouting_targets(item):
    """
    Rerouting üçün valid intermediate nodes tapır.

    1-hop:
        A -> B
        intermediate yoxdur.

    2-hop:
        A -> B -> C
             ^
             B intermediate

    3-hop:
        A -> B -> C -> D
             ^    ^
             B və C intermediate

    IMPORTANT:

    MetaQA exact-hop WALK olduğu üçün bəzən belə path mümkündür:

        A -> B -> A

    və iki traversal step eyni underlying KG triple-dan gələ bilər.

    Belə degenerate case-ləri rerouting experiment-dən çıxarırıq,
    çünki iki DISTINCT adjacent support fact dəyişmək istəyirik.
    """

    candidates = []

    if item["hop"] < 2:
        return candidates

    for answer, paths in item[
        "support_paths"
    ].items():

        if not paths:
            continue

        # Canonical support path
        path = paths[0]

        if len(path) < 2:
            continue

        for i in range(
            len(path) - 1
        ):

            left_step = path[i]
            right_step = path[i + 1]

            intermediate_entity = (
                left_step[
                    "to_entity"
                ]
            )

            # ------------------------------------------------
            # Path connectivity:
            #
            # A -> B
            #      B -> C
            # ------------------------------------------------

            if (
                right_step[
                    "from_entity"
                ]
                != intermediate_entity
            ):
                continue

            # ------------------------------------------------
            # Intermediate həqiqətən intermediate olsun.
            #
            # Cyclic walks səbəbilə topic/gold node bəzən
            # arada yenidən görünə bilər.
            # Bunları rerouting target kimi istifadə etmirik.
            # ------------------------------------------------

            if (
                intermediate_entity
                == item["topic_entity"]
            ):
                continue

            if (
                intermediate_entity
                in item["gold_answers"]
            ):
                continue

            left_triple = tuple(
                left_step[
                    "triple"
                ]
            )

            right_triple = tuple(
                right_step[
                    "triple"
                ]
            )

            # ------------------------------------------------
            # CRITICAL FIX:
            #
            # Adjacent traversal steps eyni underlying
            # KG triple-dırsa reject.
            #
            # Bu əvvəlki:
            #
            # ValueError: right replacement count=0
            #
            # problemini yaradan case idi.
            # ------------------------------------------------

            if left_triple == right_triple:
                continue

            candidates.append(
                {
                    "target_gold_answer":
                        answer,

                    "intermediate_index":
                        i + 1,

                    "intermediate_entity":
                        intermediate_entity,

                    "left_step":
                        left_step,

                    "right_step":
                        right_step,
                }
            )

    return candidates


# ============================================================
# 4. DETERMINISTIC TARGET ORDER
# ============================================================

def get_ordered_rerouting_targets(
    item,
    seed=DEFAULT_SEED,
):
    """
    Bütün valid intermediate targets-i deterministic şəkildə
    sıralayıb shuffle edir.

    Niyə bir target yox, hamısı?

    3-hop-da:

        A -> B -> C -> D

    həm B, həm də C reroute edilə bilər.

    Əgər B üçün compatible replacement yoxdur,
    C-ni də sınamaq istəyirik.
    """

    candidates = collect_rerouting_targets(
        item
    )

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: (
            x[
                "target_gold_answer"
            ],
            x[
                "intermediate_index"
            ],
            x[
                "intermediate_entity"
            ],
        ),
    )

    rng = random.Random(
        stable_seed(
            item["qid"]
            + ":rerouting_targets",
            seed,
        )
    )

    rng.shuffle(
        candidates
    )

    return candidates


# ============================================================
# 5. CHOOSE TYPE-COMPATIBLE REPLACEMENT NODE
# ============================================================

def choose_replacement_node(
    item,
    target,
    indexes,
    seed=DEFAULT_SEED,
):
    """
    Original chain:

        A -> B -> C

    Rerouted chain:

        A -> B' -> C

    B' hər iki adjacent relation üçün B ilə eyni structural
    slot-da görülmüş entity olmalıdır.

    Buna görə:

        candidate_pool =
            left_relation_pool
            INTERSECTION
            right_relation_pool

    istifadə edirik.
    """

    intermediate = target[
        "intermediate_entity"
    ]

    left_step = target[
        "left_step"
    ]

    right_step = target[
        "right_step"
    ]

    # --------------------------------------------------------
    # Get compatible entities for the LEFT relation
    # --------------------------------------------------------

    left_pool, left_slot = (
        get_slot_pool(
            left_step,
            intermediate,
            indexes,
        )
    )

    # --------------------------------------------------------
    # Get compatible entities for the RIGHT relation
    # --------------------------------------------------------

    right_pool, right_slot = (
        get_slot_pool(
            right_step,
            intermediate,
            indexes,
        )
    )

    if (
        not left_pool
        or not right_pool
    ):
        return None

    # Must satisfy both relation roles.
    candidate_pool = (
        left_pool
        & right_pool
    )

    # --------------------------------------------------------
    # Entities we do not allow as replacement
    # --------------------------------------------------------

    forbidden = {
        intermediate,
        item["topic_entity"],
        *item["gold_answers"],
    }

    original_left = tuple(
        left_step["triple"]
    )

    original_right = tuple(
        right_step["triple"]
    )

    candidates = []

    for candidate in candidate_pool:

        if candidate in forbidden:
            continue

        # ----------------------------------------------------
        # Replace B -> B'
        # ----------------------------------------------------

        new_left = (
            replace_entity_in_triple(
                original_left,
                intermediate,
                candidate,
            )
        )

        new_right = (
            replace_entity_in_triple(
                original_right,
                intermediate,
                candidate,
            )
        )

        # Something must actually change.
        if new_left == original_left:
            continue

        if new_right == original_right:
            continue

        # Rerouted edges should remain distinct.
        if new_left == new_right:
            continue

        # ----------------------------------------------------
        # Both injected edges must be absent from the
        # original KG.
        #
        # Therefore this is a controlled synthetic reroute,
        # not another true path from the KG.
        # ----------------------------------------------------

        if new_left in indexes[
            "kg_triples"
        ]:
            continue

        if new_right in indexes[
            "kg_triples"
        ]:
            continue

        candidates.append(
            {
                "replacement_node":
                    candidate,

                "left_slot":
                    left_slot,

                "right_slot":
                    right_slot,

                "new_left_triple":
                    new_left,

                "new_right_triple":
                    new_right,
            }
        )

    if not candidates:
        return None

    # Deterministic candidate order
    candidates = sorted(
        candidates,
        key=lambda x:
            x[
                "replacement_node"
            ],
    )

    rng = random.Random(
        stable_seed(
            item["qid"]
            + ":rerouting_replacement:"
            + intermediate,
            seed,
        )
    )

    return rng.choice(
        candidates
    )


# ============================================================
# 6. REPLACE TWO SUPPORT TRIPLES IN EVIDENCE
# ============================================================

def replace_two_support_triples(
    evidence,
    original_left,
    original_right,
    new_left,
    new_right,
):
    """
    Evidence size/order dəyişmir.

    Intermediate node-a toxunan iki DISTINCT support triple
    dəyişdirilir.
    """

    if original_left == original_right:
        raise ValueError(
            "Rerouting requires two distinct original triples."
        )

    output = deepcopy(
        evidence
    )

    left_count = 0
    right_count = 0

    for entry in output:

        if entry["role"] != "support":
            continue

        triple = tuple(
            entry["triple"]
        )

        if triple == original_left:

            entry["triple"] = list(
                new_left
            )

            left_count += 1

        elif triple == original_right:

            entry["triple"] = list(
                new_right
            )

            right_count += 1

    if left_count != 1:

        raise ValueError(
            f"left replacement count="
            f"{left_count}"
        )

    if right_count != 1:

        raise ValueError(
            f"right replacement count="
            f"{right_count}"
        )

    return output


# ============================================================
# 7. UPDATE SUPPORT_TRIPLES METADATA
# ============================================================

def replace_support_list(
    support_triples,
    original_left,
    original_right,
    new_left,
    new_right,
):
    """
    support_triples metadata-da eyni iki replacement edilir.
    """

    if original_left == original_right:
        return None

    output = []

    left_count = 0
    right_count = 0

    for triple in support_triples:

        triple_tuple = tuple(
            triple
        )

        if triple_tuple == original_left:

            output.append(
                list(
                    new_left
                )
            )

            left_count += 1

        elif triple_tuple == original_right:

            output.append(
                list(
                    new_right
                )
            )

            right_count += 1

        else:

            output.append(
                triple
            )

    if left_count != 1:
        return None

    if right_count != 1:
        return None

    return output


# ============================================================
# 8. APPLY REROUTING
# ============================================================

def apply_rerouting(
    clean_item,
    indexes,
    seed=DEFAULT_SEED,
):
    """
    Controlled intermediate-node rerouting.

    Original:

        A -> B -> C

    Corrupted:

        A -> B' -> C

    Preserved:
        - adjacent relation labels
        - evidence size
        - evidence order
        - support role

    Changed:
        - intermediate node
        - two adjacent support triples
    """

    # --------------------------------------------------------
    # A. Get ALL possible intermediate targets
    # --------------------------------------------------------

    targets = (
        get_ordered_rerouting_targets(
            clean_item,
            seed=seed,
        )
    )

    if not targets:

        # 1-hop:
        if clean_item["hop"] < 2:

            return (
                None,
                "no_intermediate_node",
            )

        # Multi-hop but only degenerate/cyclic invalid targets.
        return (
            None,
            "no_valid_distinct_edge_target",
        )

    # --------------------------------------------------------
    # B. Try every target until a replacement is found
    # --------------------------------------------------------

    selected_target = None
    selected_replacement = None

    for target in targets:

        replacement = (
            choose_replacement_node(
                item=clean_item,
                target=target,
                indexes=indexes,
                seed=seed,
            )
        )

        if replacement is None:
            continue

        selected_target = target
        selected_replacement = (
            replacement
        )

        break

    if (
        selected_target is None
        or selected_replacement is None
    ):

        return (
            None,
            "no_type_compatible_replacement",
        )

    target = selected_target
    replacement = selected_replacement

    # --------------------------------------------------------
    # C. Extract original and corrupted triples
    # --------------------------------------------------------

    left_step = target[
        "left_step"
    ]

    right_step = target[
        "right_step"
    ]

    original_left = tuple(
        left_step[
            "triple"
        ]
    )

    original_right = tuple(
        right_step[
            "triple"
        ]
    )

    new_left = tuple(
        replacement[
            "new_left_triple"
        ]
    )

    new_right = tuple(
        replacement[
            "new_right_triple"
        ]
    )

    # Defensive check
    if original_left == original_right:

        return (
            None,
            "same_underlying_edge",
        )

    # --------------------------------------------------------
    # D. Preserve clean item
    # --------------------------------------------------------

    bad_item = deepcopy(
        clean_item
    )

    bad_item[
        "clean_evidence"
    ] = deepcopy(
        clean_item[
            "evidence"
        ]
    )

    bad_item[
        "clean_support_triples"
    ] = deepcopy(
        clean_item[
            "support_triples"
        ]
    )

    bad_item[
        "clean_support_paths"
    ] = deepcopy(
        clean_item[
            "support_paths"
        ]
    )

    # --------------------------------------------------------
    # E. Modify evidence
    # --------------------------------------------------------

    try:

        bad_item[
            "evidence"
        ] = (
            replace_two_support_triples(
                evidence=
                    clean_item[
                        "evidence"
                    ],

                original_left=
                    original_left,

                original_right=
                    original_right,

                new_left=
                    new_left,

                new_right=
                    new_right,
            )
        )

    except ValueError as exc:

        return (
            None,
            "evidence_replacement_failed:"
            + str(exc),
        )

    # --------------------------------------------------------
    # F. Modify support_triples metadata
    # --------------------------------------------------------

    updated_support = (
        replace_support_list(
            support_triples=
                clean_item[
                    "support_triples"
                ],

            original_left=
                original_left,

            original_right=
                original_right,

            new_left=
                new_left,

            new_right=
                new_right,
        )
    )

    if updated_support is None:

        return (
            None,
            "support_metadata_replacement_failed",
        )

    bad_item[
        "support_triples"
    ] = updated_support

    # --------------------------------------------------------
    # G. Corruption position
    # --------------------------------------------------------

    intermediate_index = (
        target[
            "intermediate_index"
        ]
    )

    if clean_item["hop"] == 2:

        position = (
            "intermediate"
        )

    elif intermediate_index == 1:

        position = (
            "early-intermediate"
        )

    else:

        position = (
            "late-intermediate"
        )

    # --------------------------------------------------------
    # H. Manifest
    # --------------------------------------------------------

    manifest = {

        "corruption_type":
            "intermediate_node_rerouting",

        "corruption_severity":
            1,

        "target_gold_answer":
            target[
                "target_gold_answer"
            ],

        "corrupted_position":
            position,

        "intermediate_index":
            intermediate_index,

        "original_intermediate_node":
            target[
                "intermediate_entity"
            ],

        "replacement_node":
            replacement[
                "replacement_node"
            ],

        "original_left_triple":
            list(
                original_left
            ),

        "original_right_triple":
            list(
                original_right
            ),

        "corrupted_left_triple":
            list(
                new_left
            ),

        "corrupted_right_triple":
            list(
                new_right
            ),

        "left_relation":
            left_step[
                "relation"
            ],

        "right_relation":
            right_step[
                "relation"
            ],

        "left_reverse":
            left_step.get(
                "reverse",
                False,
            ),

        "right_reverse":
            right_step.get(
                "reverse",
                False,
            ),

        "left_slot":
            replacement[
                "left_slot"
            ],

        "right_slot":
            replacement[
                "right_slot"
            ],

        "changed_support_triples":
            2,

        "injected_false_answer":
            None,

        "seed":
            seed,
    }

    bad_item[
        "corruption_type"
    ] = (
        "intermediate_node_rerouting"
    )

    bad_item[
        "corrupted_position"
    ] = position

    bad_item[
        "injected_false_answer"
    ] = None

    bad_item[
        "corruption_manifest"
    ] = manifest

    return (
        bad_item,
        "ok",
    )


# ============================================================
# 9. VALIDATION
# ============================================================

def validate_rerouting(
    clean_item,
    bad_item,
    indexes,
):
    """
    Rerouting invariants:

    1. Evidence size unchanged.
    2. Evidence roles/order unchanged.
    3. Exactly 2 evidence triples changed.
    4. Changed entries are support entries.
    5. Relations unchanged.
    6. Both corrupted triples absent from original KG.
    7. Intermediate node changed.
    8. Two corrupted triples remain distinct.
    """

    clean_evidence = (
        clean_item[
            "evidence"
        ]
    )

    bad_evidence = (
        bad_item[
            "evidence"
        ]
    )

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

        # Role/order must remain unchanged
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

            changed_indices.append(
                index
            )

            # Only support evidence should change
            if (
                clean_entry["role"]
                != "support"
            ):

                return (
                    False,
                    "non_support_changed",
                )

    # Exactly two adjacent support facts should change
    if len(changed_indices) != 2:

        return (
            False,
            f"changed_count="
            f"{len(changed_indices)}",
        )

    manifest = bad_item[
        "corruption_manifest"
    ]

    original_left = tuple(
        manifest[
            "original_left_triple"
        ]
    )

    original_right = tuple(
        manifest[
            "original_right_triple"
        ]
    )

    corrupted_left = tuple(
        manifest[
            "corrupted_left_triple"
        ]
    )

    corrupted_right = tuple(
        manifest[
            "corrupted_right_triple"
        ]
    )

    # --------------------------------------------------------
    # Distinct original edges
    # --------------------------------------------------------

    if original_left == original_right:

        return (
            False,
            "original_edges_not_distinct",
        )

    # --------------------------------------------------------
    # Distinct corrupted edges
    # --------------------------------------------------------

    if corrupted_left == corrupted_right:

        return (
            False,
            "corrupted_edges_not_distinct",
        )

    # --------------------------------------------------------
    # Relations preserved
    # --------------------------------------------------------

    if (
        original_left[1]
        != corrupted_left[1]
    ):

        return (
            False,
            "left_relation_changed",
        )

    if (
        original_right[1]
        != corrupted_right[1]
    ):

        return (
            False,
            "right_relation_changed",
        )

    # --------------------------------------------------------
    # Corrupted edges must be false relative to clean KG
    # --------------------------------------------------------

    if corrupted_left in indexes[
        "kg_triples"
    ]:

        return (
            False,
            "left_corruption_is_true",
        )

    if corrupted_right in indexes[
        "kg_triples"
    ]:

        return (
            False,
            "right_corruption_is_true",
        )

    # --------------------------------------------------------
    # Node actually changed
    # --------------------------------------------------------

    if (
        manifest[
            "original_intermediate_node"
        ]
        == manifest[
            "replacement_node"
        ]
    ):

        return (
            False,
            "node_not_changed",
        )

    return (
        True,
        "ok",
    )


# ============================================================
# 10. BUILD DATASET
# ============================================================

def build_rerouting_dataset(
    clean_items,
    indexes,
    seed=DEFAULT_SEED,
):
    """
    Builds feasible rerouting subset.

    Expected:
        1-hop -> infeasible
        2-hop -> potentially feasible
        3-hop -> potentially feasible
    """

    successful = []
    failures = []

    for clean_item in clean_items:

        bad_item, reason = (
            apply_rerouting(
                clean_item=
                    clean_item,

                indexes=
                    indexes,

                seed=
                    seed,
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
            validate_rerouting(
                clean_item=
                    clean_item,

                bad_item=
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
# 11. PRINT EXAMPLE
# ============================================================

def print_example(
    clean_item,
    bad_item,
):
    """
    Human-readable rerouting example.
    """

    manifest = bad_item[
        "corruption_manifest"
    ]

    print()
    print("=" * 72)
    print(
        "INTERMEDIATE NODE REROUTING EXAMPLE"
    )
    print("=" * 72)

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
        manifest[
            "target_gold_answer"
        ],
    )

    print()

    print(
        "Original intermediate:",
        manifest[
            "original_intermediate_node"
        ],
    )

    print(
        "Replacement node:",
        manifest[
            "replacement_node"
        ],
    )

    print(
        "Position:",
        manifest[
            "corrupted_position"
        ],
    )

    print()

    print(
        "ORIGINAL LEFT:"
    )

    print(
        " | ".join(
            manifest[
                "original_left_triple"
            ]
        )
    )

    print(
        "CORRUPTED LEFT:"
    )

    print(
        " | ".join(
            manifest[
                "corrupted_left_triple"
            ]
        )
    )

    print()

    print(
        "ORIGINAL RIGHT:"
    )

    print(
        " | ".join(
            manifest[
                "original_right_triple"
            ]
        )
    )

    print(
        "CORRUPTED RIGHT:"
    )

    print(
        " | ".join(
            manifest[
                "corrupted_right_triple"
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


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load frozen clean pilot
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
    # Load KG
    # --------------------------------------------------------

    print(
        "Loading KG..."
    )

    triples = load_kb()

    print(
        f"KG triples: "
        f"{len(triples)}"
    )

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    print(
        "Building corruption indexes..."
    )

    indexes = (
        build_corruption_indexes(
            triples
        )
    )

    # --------------------------------------------------------
    # Generate rerouting condition
    # --------------------------------------------------------

    print()
    print(
        "Generating intermediate-node rerouting..."
    )
    print()

    bad_items, failures = (
        build_rerouting_dataset(
            clean_items=
                clean_items,

            indexes=
                indexes,

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

    # --------------------------------------------------------
    # Successful hop counts
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Failure reason counts
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
    # Failure counts by hop
    # --------------------------------------------------------

    failure_hop_counts = {
        1: 0,
        2: 0,
        3: 0,
    }

    for failure in failures:

        hop = failure[
            "hop"
        ]

        if hop in failure_hop_counts:

            failure_hop_counts[
                hop
            ] += 1

    print(
        "Failure hop counts:",
        failure_hop_counts,
    )

    # --------------------------------------------------------
    # Print one successful example
    # --------------------------------------------------------

    if bad_items:

        clean_lookup = {
            item["qid"]: item
            for item in clean_items
        }

        bad_example = bad_items[0]

        clean_example = (
            clean_lookup[
                bad_example[
                    "qid"
                ]
            ]
        )

        print_example(
            clean_example,
            bad_example,
        )