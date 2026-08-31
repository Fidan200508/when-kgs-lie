from pathlib import Path
from collections import Counter, defaultdict
import random

from data import load_kb, load_questions
from evidence import (
    CONTEXT_SIZE,
    DEFAULT_SEED,
    TARGET_PER_HOP,
    build_adjacency,
    build_clean_evidence,
    normalize_kb_triples,
    save_jsonl,
    stable_seed,
    validate_clean_item,
)


# ============================================================
# PATHS
# ============================================================

OUTPUT_PATH = Path(
    "data/processed/metaqa_pilot_clean.jsonl"
)

REJECTION_PATH = Path(
    "results/raw/clean_semantic_rejections.jsonl"
)

PACKET_AUDIT_PATH = Path(
    "results/raw/clean_packet_answer_set_audit.jsonl"
)


# ============================================================
# 1. BUILD ADJACENCY FROM THE ACTUAL EVIDENCE PACKET
# ============================================================

def build_packet_adjacency(item):
    """
    Builds a bidirectional adjacency graph using ONLY the triples
    that the LLM will actually see.

    This is important because we want to know whether the full
    clean evidence packet itself permits additional non-gold
    answers.

    Example:

        Movie | directed_by | Person

    allows traversal:

        Movie -> Person
        Person -> Movie

    while keeping the same relation label.
    """

    adjacency = defaultdict(list)

    for entry in item["evidence"]:

        head, relation, tail = (
            entry["triple"]
        )

        adjacency[head].append(
            (
                tail,
                relation,
            )
        )

        adjacency[tail].append(
            (
                head,
                relation,
            )
        )

    return adjacency


# ============================================================
# 2. FIND ALL ANSWERS REACHABLE FROM THE FULL PACKET
# ============================================================

def infer_answer_set_from_packet(item):
    """
    Finds every answer that can be reached from the topic entity
    using:

        - exactly the required number of hops
        - exactly the question-derived relation profile
        - the required final target relation

    Crucially, this operates over ALL evidence shown to the model,
    not only the originally selected support paths.

    Node repetition is allowed, matching the MetaQA support-path
    construction used in evidence.py.
    """

    hop = item["hop"]

    topic = item[
        "topic_entity"
    ]

    target_relation = item[
        "target_relation"
    ]

    required_profile = Counter(
        item[
            "required_relation_counts"
        ]
    )

    # The semantic relation profile must explain every hop.
    if (
        sum(
            required_profile.values()
        )
        != hop
    ):
        return set()

    adjacency = (
        build_packet_adjacency(
            item
        )
    )

    relation_counter = Counter()

    answers = set()

    # ========================================================
    # DFS
    # ========================================================

    def dfs(
        current_entity,
        depth,
    ):

        # ----------------------------------------------------
        # End of exact-hop walk
        # ----------------------------------------------------

        if depth == hop:

            if (
                relation_counter
                == required_profile
            ):

                answers.add(
                    current_entity
                )

            return

        # ----------------------------------------------------
        # Check whether remaining required relations can still
        # fit into remaining hops.
        # ----------------------------------------------------

        remaining_steps = (
            hop
            - depth
        )

        required_remaining = sum(
            max(
                0,
                required_profile[
                    relation
                ]
                - relation_counter[
                    relation
                ],
            )
            for relation
            in required_profile
        )

        if (
            required_remaining
            > remaining_steps
        ):
            return

        # ----------------------------------------------------
        # Explore evidence graph
        # ----------------------------------------------------

        for (
            next_entity,
            relation,
        ) in adjacency.get(
            current_entity,
            [],
        ):

            # Relation must belong to the semantic profile.
            if (
                relation
                not in required_profile
            ):
                continue

            # Do not use a relation more often than allowed.
            if (
                relation_counter[
                    relation
                ]
                >= required_profile[
                    relation
                ]
            ):
                continue

            # ------------------------------------------------
            # FINAL EDGE HARD CONSTRAINT
            # ------------------------------------------------

            if (
                depth
                == hop - 1
                and relation
                != target_relation
            ):
                continue

            # ------------------------------------------------
            # Traverse
            # ------------------------------------------------

            relation_counter[
                relation
            ] += 1

            dfs(
                next_entity,
                depth + 1,
            )

            relation_counter[
                relation
            ] -= 1

            if (
                relation_counter[
                    relation
                ]
                == 0
            ):
                del relation_counter[
                    relation
                ]

    # ========================================================
    # START
    # ========================================================

    dfs(
        topic,
        0,
    )

    return answers


# ============================================================
# 3. FULL PACKET ANSWER-SET VALIDATION
# ============================================================

def validate_packet_answer_set(item):
    """
    A clean item is valid only if:

        answers reachable from ALL evidence
        ==
        benchmark gold answer set

    This catches hidden ambiguity introduced by:

        - context triples
        - support-path recombination
        - repeated-node paths
        - cycles
        - multiple support paths interacting
    """

    reachable = (
        infer_answer_set_from_packet(
            item
        )
    )

    gold = set(
        item[
            "gold_answers"
        ]
    )

    # ========================================================
    # MISMATCH
    # ========================================================

    if reachable != gold:

        return (
            False,
            {
                "reason":
                    "packet_answer_set_mismatch",

                "reachable_answers":
                    sorted(
                        reachable
                    ),

                "gold_answers":
                    sorted(
                        gold
                    ),

                "extra_answers":
                    sorted(
                        reachable
                        - gold
                    ),

                "missing_answers":
                    sorted(
                        gold
                        - reachable
                    ),
            },
        )

    # ========================================================
    # VALID
    # ========================================================

    return (
        True,
        {
            "reason":
                "ok",

            "reachable_answers":
                sorted(
                    reachable
                ),

            "gold_answers":
                sorted(
                    gold
                ),

            "extra_answers":
                [],

            "missing_answers":
                [],
        },
    )


# ============================================================
# 4. BUILD FINAL PILOT
# ============================================================

def build_final_pilot(
    triples,
    adjacency,
):
    """
    Builds the final 300-item pilot:

        100 x 1-hop
        100 x 2-hop
        100 x 3-hop

    Candidate item must pass BOTH:

        1. original semantic validation
        2. full evidence packet answer-set validation
    """

    pilot = []

    rejections = []

    packet_audit = []

    # ========================================================
    # EACH HOP
    # ========================================================

    for hop in (
        1,
        2,
        3,
    ):

        questions = list(
            load_questions(
                hop
            )
        )

        # Same deterministic sampling logic.
        rng = random.Random(
            stable_seed(
                f"pilot-hop-{hop}",
                DEFAULT_SEED,
            )
        )

        rng.shuffle(
            questions
        )

        accepted = []

        rejection_counts = (
            Counter()
        )

        # ====================================================
        # SCAN QUESTIONS UNTIL 100 VALID ITEMS
        # ====================================================

        for question_item in questions:

            (
                clean_item,
                reason,
            ) = build_clean_evidence(
                question_item=
                    question_item,

                triples=
                    triples,

                adjacency=
                    adjacency,

                context_size=
                    CONTEXT_SIZE,

                seed=
                    DEFAULT_SEED,
            )

            # ------------------------------------------------
            # Original construction rejection
            # ------------------------------------------------

            if clean_item is None:

                rejection_counts[
                    reason
                ] += 1

                rejections.append(
                    {
                        "qid":
                            question_item[
                                "qid"
                            ],

                        "hop":
                            hop,

                        "question":
                            question_item[
                                "question"
                            ],

                        "reason":
                            reason,
                    }
                )

                continue

            # ------------------------------------------------
            # Original semantic validator
            # ------------------------------------------------

            (
                valid,
                base_reason,
            ) = validate_clean_item(
                clean_item
            )

            if not valid:

                reason = (
                    "validation:"
                    + base_reason
                )

                rejection_counts[
                    reason
                ] += 1

                rejections.append(
                    {
                        "qid":
                            question_item[
                                "qid"
                            ],

                        "hop":
                            hop,

                        "question":
                            question_item[
                                "question"
                            ],

                        "reason":
                            reason,
                    }
                )

                continue

            # ------------------------------------------------
            # NEW:
            # FULL EVIDENCE PACKET VALIDATION
            # ------------------------------------------------

            (
                packet_ok,
                packet_info,
            ) = validate_packet_answer_set(
                clean_item
            )

            packet_audit.append(
                {
                    "qid":
                        clean_item[
                            "qid"
                        ],

                    "hop":
                        hop,

                    "question":
                        clean_item[
                            "question"
                        ],

                    **packet_info,
                }
            )

            # ------------------------------------------------
            # Packet ambiguity => reject
            # ------------------------------------------------

            if not packet_ok:

                reason = (
                    "packet_answer_set_mismatch"
                )

                rejection_counts[
                    reason
                ] += 1

                rejections.append(
                    {
                        "qid":
                            clean_item[
                                "qid"
                            ],

                        "hop":
                            hop,

                        "question":
                            clean_item[
                                "question"
                            ],

                        "reason":
                            reason,

                        "extra_answers":
                            packet_info[
                                "extra_answers"
                            ],

                        "missing_answers":
                            packet_info[
                                "missing_answers"
                            ],
                    }
                )

                continue

            # ------------------------------------------------
            # ACCEPT
            # ------------------------------------------------

            accepted.append(
                clean_item
            )

            if (
                len(accepted)
                >= TARGET_PER_HOP
            ):
                break

        # ====================================================
        # SAVE HOP
        # ====================================================

        pilot.extend(
            accepted
        )

        print(
            f"{hop}-hop final clean items: "
            f"{len(accepted)}/"
            f"{TARGET_PER_HOP}"
        )

        print(
            "  Rejections:",
            dict(
                rejection_counts
            ),
        )

    return (
        pilot,
        rejections,
        packet_audit,
    )


# ============================================================
# 5. FINAL VALIDATION
# ============================================================

def final_validation(pilot):
    """
    Runs all validation again AFTER construction.
    """

    hop_counts = Counter(
        item[
            "hop"
        ]
        for item in pilot
    )

    base_failures = []

    packet_failures = []

    # ========================================================
    # CHECK EVERY FINAL ITEM
    # ========================================================

    for item in pilot:

        # ----------------------------------------------------
        # Original validator
        # ----------------------------------------------------

        (
            valid,
            reason,
        ) = validate_clean_item(
            item
        )

        if not valid:

            base_failures.append(
                {
                    "qid":
                        item[
                            "qid"
                        ],

                    "reason":
                        reason,
                }
            )

        # ----------------------------------------------------
        # Full evidence validator
        # ----------------------------------------------------

        (
            packet_ok,
            packet_info,
        ) = validate_packet_answer_set(
            item
        )

        if not packet_ok:

            packet_failures.append(
                {
                    "qid":
                        item[
                            "qid"
                        ],

                    **packet_info,
                }
            )

    # ========================================================
    # EXPECTED SIZE
    # ========================================================

    size_ok = (
        len(pilot)
        == 300
        and hop_counts
        == Counter(
            {
                1: 100,
                2: 100,
                3: 100,
            }
        )
    )

    return {
        "hop_counts":
            dict(
                sorted(
                    hop_counts.items()
                )
            ),

        "base_failures":
            base_failures,

        "packet_failures":
            packet_failures,

        "size_ok":
            size_ok,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Loading KG..."
    )

    raw_triples = (
        load_kb()
    )

    triples = (
        normalize_kb_triples(
            raw_triples
        )
    )

    print(
        "KG triples:",
        len(
            triples
        ),
    )

    print(
        "Building adjacency..."
    )

    adjacency = (
        build_adjacency(
            triples
        )
    )

    print(
        "Entities:",
        len(
            adjacency
        ),
    )

    print()
    print(
        "Building final ambiguity-free clean pilot..."
    )
    print()

    (
        pilot,
        rejections,
        packet_audit,
    ) = build_final_pilot(
        triples,
        adjacency,
    )

    # ========================================================
    # SAVE FINAL CLEAN
    # ========================================================

    save_jsonl(
        pilot,
        OUTPUT_PATH,
    )

    save_jsonl(
        rejections,
        REJECTION_PATH,
    )

    save_jsonl(
        packet_audit,
        PACKET_AUDIT_PATH,
    )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    result = (
        final_validation(
            pilot
        )
    )

    print()

    print(
        "Total pilot items:",
        len(
            pilot
        ),
    )

    print(
        "Hop counts:",
        result[
            "hop_counts"
        ],
    )

    print(
        "Base semantic failures:",
        len(
            result[
                "base_failures"
            ]
        ),
    )

    print(
        "Packet answer-set failures:",
        len(
            result[
                "packet_failures"
            ]
        ),
    )

    print()

    print(
        "Saved clean pilot:",
        OUTPUT_PATH,
    )

    print(
        "Saved rejection log:",
        REJECTION_PATH,
    )

    print(
        "Saved packet audit:",
        PACKET_AUDIT_PATH,
    )

    print()

    # ========================================================
    # PASS / FAIL
    # ========================================================

    print(
        "Pilot size check:",
        (
            "PASS"
            if result[
                "size_ok"
            ]
            else "FAIL"
        ),
    )

    print(
        "Base semantic check:",
        (
            "PASS"
            if not result[
                "base_failures"
            ]
            else "FAIL"
        ),
    )

    print(
        "Full evidence answer-set check:",
        (
            "PASS"
            if not result[
                "packet_failures"
            ]
            else "FAIL"
        ),
    )

    # ========================================================
    # HARD FAILURE
    # ========================================================

    if (
        not result[
            "size_ok"
        ]
        or result[
            "base_failures"
        ]
        or result[
            "packet_failures"
        ]
    ):

        raise RuntimeError(
            "Final clean pilot validation failed."
        )