from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import json
import re

from evidence import stable_seed
from corruptions import (
    load_jsonl,
    save_jsonl,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SEED = 42


DATASETS = {
    "clean": (
        Path(
            "data/processed/"
            "metaqa_pilot_clean.jsonl"
        ),
        Path(
            "data/processed/"
            "metaqa_pilot_clean_anonymized.jsonl"
        ),
    ),

    "entity_substitution": (
        Path(
            "data/processed/"
            "metaqa_pilot_entity_substitution.jsonl"
        ),
        Path(
            "data/processed/"
            "metaqa_pilot_entity_substitution_anonymized.jsonl"
        ),
    ),

    "relation_substitution": (
        Path(
            "data/processed/"
            "metaqa_pilot_relation_substitution.jsonl"
        ),
        Path(
            "data/processed/"
            "metaqa_pilot_relation_substitution_anonymized.jsonl"
        ),
    ),

    "contradiction": (
        Path(
            "data/processed/"
            "metaqa_pilot_contradiction.jsonl"
        ),
        Path(
            "data/processed/"
            "metaqa_pilot_contradiction_anonymized.jsonl"
        ),
    ),

    "rerouting": (
        Path(
            "data/processed/"
            "metaqa_pilot_rerouting.jsonl"
        ),
        Path(
            "data/processed/"
            "metaqa_pilot_rerouting_anonymized.jsonl"
        ),
    ),
}


MAP_OUTPUT_PATH = Path(
    "data/processed/"
    "metaqa_pilot_anonymization_maps.jsonl"
)


# ============================================================
# 1. ENTITY FIELDS
# ============================================================

ENTITY_SCALAR_FIELDS = {
    "topic_entity",
    "from_entity",
    "to_entity",
    "target_gold_answer",
    "replacement_entity",
    "original_destination_entity",
    "replacement_node",
    "original_intermediate_node",
    "injected_false_answer",
}


ENTITY_LIST_FIELDS = {
    "gold_answers",
}


TRIPLE_FIELDS = {
    "triple",
    "original_triple",
    "corrupted_triple",
    "true_triple",
    "contradictory_triple",
    "removed_context_triple",
    "original_left_triple",
    "original_right_triple",
    "corrupted_left_triple",
    "corrupted_right_triple",
}


TRIPLE_LIST_FIELDS = {
    "support_triples",
    "clean_support_triples",
    "context_triples",
}


# ============================================================
# 2. COLLECT ENTITIES FROM TRIPLE
# ============================================================

def collect_from_triple(
    triple,
    entities,
):
    """
    KG triple:

        [head, relation, tail]

    relation anonymized edilmir.

    Yalnız:
        head
        tail

    entity mapping-ə daxil olur.
    """

    if not isinstance(
        triple,
        (list, tuple),
    ):
        return

    if len(triple) != 3:
        return

    head, _, tail = triple

    if isinstance(head, str):
        entities.add(head)

    if isinstance(tail, str):
        entities.add(tail)


# ============================================================
# 3. RECURSIVELY COLLECT ENTITIES
# ============================================================

def collect_entities_recursive(
    obj,
    entities,
    parent_key=None,
):
    """
    Item-in bütün relevant metadata-sından
    entity labels toplanır.

    Relation labels toplanmır.
    """

    if isinstance(obj, dict):

        for key, value in obj.items():

            # -----------------------------------------------
            # Entity scalar
            # -----------------------------------------------

            if key in ENTITY_SCALAR_FIELDS:

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value
                ):
                    entities.add(
                        value
                    )

                continue

            # -----------------------------------------------
            # Entity list
            # -----------------------------------------------

            if key in ENTITY_LIST_FIELDS:

                if isinstance(
                    value,
                    list,
                ):

                    for entity in value:

                        if isinstance(
                            entity,
                            str,
                        ):
                            entities.add(
                                entity
                            )

                continue

            # -----------------------------------------------
            # One triple
            # -----------------------------------------------

            if key in TRIPLE_FIELDS:

                collect_from_triple(
                    value,
                    entities,
                )

                continue

            # -----------------------------------------------
            # List of triples
            # -----------------------------------------------

            if key in TRIPLE_LIST_FIELDS:

                if isinstance(
                    value,
                    list,
                ):

                    for triple in value:

                        collect_from_triple(
                            triple,
                            entities,
                        )

                continue

            # -----------------------------------------------
            # Recursive fallback
            # -----------------------------------------------

            collect_entities_recursive(
                value,
                entities,
                parent_key=key,
            )

    elif isinstance(obj, list):

        for value in obj:

            collect_entities_recursive(
                value,
                entities,
                parent_key=
                    parent_key,
            )


# ============================================================
# 4. COLLECT ENTITIES FOR ONE ITEM
# ============================================================

def collect_entities_from_item(item):
    entities = set()

    collect_entities_recursive(
        item,
        entities,
    )

    return entities


# ============================================================
# 5. CREATE PER-QID ANONYMIZATION MAP
# ============================================================

def build_qid_mapping(
    qid,
    entities,
    seed=DEFAULT_SEED,
):
    """
    Mapping PER QUESTION yaradılır.

    Eyni qid-in clean/corrupted versions-larında
    eyni real entity -> eyni anonymous ID.

    Amma fərqli qid-lərdə mapping müstəqildir.

    Beləliklə cross-question identity də leak etmir.
    """

    entities = sorted(
        entities,
        key=lambda entity:
            (
                stable_seed(
                    (
                        f"{qid}:"
                        f"anonymous:"
                        f"{entity}"
                    ),
                    seed,
                ),
                entity,
            ),
    )

    width = max(
        4,
        len(
            str(
                len(entities)
            )
        ),
    )

    mapping = {}

    for index, entity in enumerate(
        entities,
        start=1,
    ):

        mapping[
            entity
        ] = (
            "Entity_"
            + str(index).zfill(
                width
            )
        )

    return mapping


# ============================================================
# 6. REPLACE TOPIC IN QUESTION
# ============================================================

def replace_topic_in_question(
    question,
    natural_topic,
    anonymous_topic,
):
    """
    MetaQA question-da explicit entity topic-dir.

    Example:

        who directed Captain America

    becomes:

        who directed Entity_0007

    Relation words və question semantics dəyişmir.
    """

    if not question:
        return question

    if not natural_topic:
        return question

    # Avoid replacing entity names as part of another word.
    pattern = (
        r"(?<!\w)"
        + re.escape(
            natural_topic
        )
        + r"(?!\w)"
    )

    result = re.sub(
        pattern,
        anonymous_topic,
        question,
        flags=re.IGNORECASE,
    )

    return result


# ============================================================
# 7. ANONYMIZE ONE TRIPLE
# ============================================================

def anonymize_triple(
    triple,
    mapping,
):
    """
    [head, relation, tail]

    becomes:

    [Entity_x, relation, Entity_y]
    """

    if not isinstance(
        triple,
        (list, tuple),
    ):
        return triple

    if len(triple) != 3:
        return triple

    head, relation, tail = triple

    new_head = mapping.get(
        head,
        head,
    )

    new_tail = mapping.get(
        tail,
        tail,
    )

    return [
        new_head,
        relation,
        new_tail,
    ]


# ============================================================
# 8. RECURSIVE ANONYMIZATION
# ============================================================

def anonymize_recursive(
    obj,
    mapping,
    parent_key=None,
):
    """
    Entity metadata-nı anonymize edir.

    qid, relation labels, role labels və s.
    dəyişmir.
    """

    if isinstance(obj, dict):

        output = {}

        for key, value in obj.items():

            # -----------------------------------------------
            # Question fields separately handled later
            # -----------------------------------------------

            if key in {
                "question",
                "question_raw",
            }:

                output[
                    key
                ] = value

                continue

            # -----------------------------------------------
            # Scalar entity
            # -----------------------------------------------

            if key in ENTITY_SCALAR_FIELDS:

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value in mapping
                ):

                    output[
                        key
                    ] = mapping[
                        value
                    ]

                else:

                    output[
                        key
                    ] = value

                continue

            # -----------------------------------------------
            # Entity list
            # -----------------------------------------------

            if key in ENTITY_LIST_FIELDS:

                if isinstance(
                    value,
                    list,
                ):

                    output[
                        key
                    ] = [
                        mapping.get(
                            entity,
                            entity,
                        )
                        for entity
                        in value
                    ]

                else:

                    output[
                        key
                    ] = value

                continue

            # -----------------------------------------------
            # Single triple
            # -----------------------------------------------

            if key in TRIPLE_FIELDS:

                output[
                    key
                ] = anonymize_triple(
                    value,
                    mapping,
                )

                continue

            # -----------------------------------------------
            # List of triples
            # -----------------------------------------------

            if key in TRIPLE_LIST_FIELDS:

                if isinstance(
                    value,
                    list,
                ):

                    output[
                        key
                    ] = [
                        anonymize_triple(
                            triple,
                            mapping,
                        )
                        for triple
                        in value
                    ]

                else:

                    output[
                        key
                    ] = value

                continue

            # -----------------------------------------------
            # Recurse
            # -----------------------------------------------

            output[
                key
            ] = anonymize_recursive(
                value,
                mapping,
                parent_key=key,
            )

        return output

    if isinstance(obj, list):

        return [
            anonymize_recursive(
                value,
                mapping,
                parent_key=
                    parent_key,
            )
            for value
            in obj
        ]

    # Exact entity-valued strings inside nested structures
    if (
        isinstance(
            obj,
            str,
        )
        and obj in mapping
    ):

        return mapping[
            obj
        ]

    return obj


# ============================================================
# 9. ANONYMIZE ONE ITEM
# ============================================================

def anonymize_item(
    item,
    mapping,
):
    output = (
        anonymize_recursive(
            deepcopy(
                item
            ),
            mapping,
        )
    )

    natural_topic = (
        item[
            "topic_entity"
        ]
    )

    anonymous_topic = mapping[
        natural_topic
    ]

    # Clean question
    output[
        "question"
    ] = replace_topic_in_question(
        question=
            item[
                "question"
            ],

        natural_topic=
            natural_topic,

        anonymous_topic=
            anonymous_topic,
    )

    # Raw question
    output[
        "question_raw"
    ] = replace_topic_in_question(
        question=
            item.get(
                "question_raw",
                item[
                    "question"
                ],
            ),

        natural_topic=
            natural_topic,

        anonymous_topic=
            anonymous_topic,
    )

    output[
        "label_mode"
    ] = "anonymized"

    output[
        "anonymization_map_id"
    ] = item[
        "qid"
    ]

    output[
        "anonymization_seed"
    ] = DEFAULT_SEED

    return output


# ============================================================
# 10. LOAD ALL CONDITIONS
# ============================================================

def load_all_conditions():
    loaded = {}

    for (
        condition,
        (
            input_path,
            output_path,
        ),
    ) in DATASETS.items():

        print(
            f"Loading {condition}: "
            f"{input_path}"
        )

        items = load_jsonl(
            input_path
        )

        print(
            f"  Items: "
            f"{len(items)}"
        )

        loaded[
            condition
        ] = {
            "input_path":
                input_path,

            "output_path":
                output_path,

            "items":
                items,
        }

    return loaded


# ============================================================
# 11. BUILD UNION ENTITY SET PER QID
# ============================================================

def build_union_entities(
    datasets,
):
    """
    Critical design:

    Eyni QID üçün mapping clean + ALL corruption
    conditions union-dan qurulur.

    Buna görə:

        Rod Holcomb

    clean-də və contradiction-da eyni Entity_ID alacaq.

    Corruption ilə gələn yeni entity də həmin map-ə daxildir.
    """

    entities_by_qid = defaultdict(
        set
    )

    for condition_data in (
        datasets.values()
    ):

        for item in (
            condition_data[
                "items"
            ]
        ):

            qid = item[
                "qid"
            ]

            entities = (
                collect_entities_from_item(
                    item
                )
            )

            entities_by_qid[
                qid
            ].update(
                entities
            )

    return entities_by_qid


# ============================================================
# 12. BUILD MAPS
# ============================================================

def build_all_maps(
    entities_by_qid,
):
    maps = {}

    for qid, entities in (
        entities_by_qid.items()
    ):

        maps[
            qid
        ] = build_qid_mapping(
            qid=
                qid,

            entities=
                entities,

            seed=
                DEFAULT_SEED,
        )

    return maps


# ============================================================
# 13. VALIDATE NATURAL VS ANONYMIZED
# ============================================================

def validate_pair(
    natural,
    anonymized,
    mapping,
):
    """
    Checks that anonymization changes labels only.

    Must preserve:
    - qid
    - hop
    - evidence count
    - evidence role
    - relation labels
    - evidence order
    """

    if (
        natural[
            "qid"
        ]
        != anonymized[
            "qid"
        ]
    ):

        return (
            False,
            "qid_changed",
        )

    if (
        natural[
            "hop"
        ]
        != anonymized[
            "hop"
        ]
    ):

        return (
            False,
            "hop_changed",
        )

    natural_evidence = (
        natural[
            "evidence"
        ]
    )

    anonymous_evidence = (
        anonymized[
            "evidence"
        ]
    )

    if len(
        natural_evidence
    ) != len(
        anonymous_evidence
    ):

        return (
            False,
            "evidence_size_changed",
        )

    for (
        natural_entry,
        anonymous_entry,
    ) in zip(
        natural_evidence,
        anonymous_evidence,
    ):

        # Role unchanged
        if (
            natural_entry[
                "role"
            ]
            != anonymous_entry[
                "role"
            ]
        ):

            return (
                False,
                "evidence_role_changed",
            )

        natural_triple = (
            natural_entry[
                "triple"
            ]
        )

        anonymous_triple = (
            anonymous_entry[
                "triple"
            ]
        )

        # Relation unchanged
        if (
            natural_triple[1]
            != anonymous_triple[1]
        ):

            return (
                False,
                "relation_changed",
            )

        # Head correctly mapped
        if (
            anonymous_triple[0]
            != mapping.get(
                natural_triple[0],
                natural_triple[0],
            )
        ):

            return (
                False,
                "head_mapping_error",
            )

        # Tail correctly mapped
        if (
            anonymous_triple[2]
            != mapping.get(
                natural_triple[2],
                natural_triple[2],
            )
        ):

            return (
                False,
                "tail_mapping_error",
            )

    # Topic entity must be anonymized
    expected_topic = mapping[
        natural[
            "topic_entity"
        ]
    ]

    if (
        anonymized[
            "topic_entity"
        ]
        != expected_topic
    ):

        return (
            False,
            "topic_mapping_error",
        )

    # Gold answers must be mapped
    expected_gold = [
        mapping[
            answer
        ]
        for answer in (
            natural[
                "gold_answers"
            ]
        )
    ]

    if (
        anonymized[
            "gold_answers"
        ]
        != expected_gold
    ):

        return (
            False,
            "gold_mapping_error",
        )

    # Natural topic should no longer appear in question,
    # unless topic == anonymous label (should never happen).
    natural_topic = (
        natural[
            "topic_entity"
        ]
    )

    if (
        natural_topic
        != expected_topic
    ):

        pattern = (
            r"(?<!\w)"
            + re.escape(
                natural_topic
            )
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            anonymized[
                "question"
            ],
            flags=re.IGNORECASE,
        ):

            return (
                False,
                "natural_topic_leak",
            )

    return (
        True,
        "ok",
    )


# ============================================================
# 14. SAVE MAP FILE
# ============================================================

def save_maps(maps):
    rows = []

    for qid in sorted(
        maps.keys()
    ):

        mapping = maps[
            qid
        ]

        rows.append(
            {
                "qid":
                    qid,

                "seed":
                    DEFAULT_SEED,

                "num_entities":
                    len(
                        mapping
                    ),

                "mapping":
                    mapping,
            }
        )

    save_jsonl(
        rows,
        MAP_OUTPUT_PATH,
    )


# ============================================================
# 15. PRINT EXAMPLE
# ============================================================

def print_example(
    natural,
    anonymized,
):
    print()
    print("=" * 78)
    print("NATURAL VS ANONYMIZED EXAMPLE")
    print("=" * 78)

    print(
        "QID:",
        natural[
            "qid"
        ],
    )

    print(
        "Hop:",
        natural[
            "hop"
        ],
    )

    print()

    print(
        "NATURAL QUESTION:"
    )

    print(
        natural[
            "question"
        ]
    )

    print()

    print(
        "ANONYMIZED QUESTION:"
    )

    print(
        anonymized[
            "question"
        ]
    )

    print()

    print(
        "NATURAL GOLD:",
        natural[
            "gold_answers"
        ],
    )

    print(
        "ANONYMIZED GOLD:",
        anonymized[
            "gold_answers"
        ],
    )

    print()

    print(
        "NATURAL EVIDENCE"
    )

    for entry in (
        natural[
            "evidence"
        ]
    ):

        triple = (
            entry[
                "triple"
            ]
        )

        print(
            f"{entry['role'].upper():13} | "
            f"{triple[0]} | "
            f"{triple[1]} | "
            f"{triple[2]}"
        )

    print()

    print(
        "ANONYMIZED EVIDENCE"
    )

    for entry in (
        anonymized[
            "evidence"
        ]
    ):

        triple = (
            entry[
                "triple"
            ]
        )

        print(
            f"{entry['role'].upper():13} | "
            f"{triple[0]} | "
            f"{triple[1]} | "
            f"{triple[2]}"
        )


# ============================================================
# 16. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    datasets = (
        load_all_conditions()
    )

    # --------------------------------------------------------
    # Union entity sets
    # --------------------------------------------------------

    print()
    print(
        "Building per-QID entity unions..."
    )

    entities_by_qid = (
        build_union_entities(
            datasets
        )
    )

    print(
        "QIDs:",
        len(
            entities_by_qid
        ),
    )

    # --------------------------------------------------------
    # Maps
    # --------------------------------------------------------

    print(
        "Building deterministic anonymization maps..."
    )

    maps = build_all_maps(
        entities_by_qid
    )

    save_maps(
        maps
    )

    print(
        "Saved maps:",
        MAP_OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Anonymize every condition
    # --------------------------------------------------------

    all_failures = []

    output_by_condition = {}

    print()

    for (
        condition,
        condition_data,
    ) in datasets.items():

        natural_items = (
            condition_data[
                "items"
            ]
        )

        anonymized_items = []

        condition_failures = []

        for natural_item in natural_items:

            qid = natural_item[
                "qid"
            ]

            mapping = maps[
                qid
            ]

            anonymous_item = (
                anonymize_item(
                    item=
                        natural_item,

                    mapping=
                        mapping,
                )
            )

            valid, reason = (
                validate_pair(
                    natural=
                        natural_item,

                    anonymized=
                        anonymous_item,

                    mapping=
                        mapping,
                )
            )

            if not valid:

                condition_failures.append(
                    {
                        "qid":
                            qid,

                        "condition":
                            condition,

                        "reason":
                            reason,
                    }
                )

                all_failures.append(
                    {
                        "qid":
                            qid,

                        "condition":
                            condition,

                        "reason":
                            reason,
                    }
                )

                continue

            anonymized_items.append(
                anonymous_item
            )

        output_path = (
            condition_data[
                "output_path"
            ]
        )

        save_jsonl(
            anonymized_items,
            output_path,
        )

        output_by_condition[
            condition
        ] = anonymized_items

        print(
            f"{condition}: "
            f"{len(anonymized_items)}/"
            f"{len(natural_items)} "
            f"validated"
        )

        print(
            "  Saved:",
            output_path,
        )

        if condition_failures:

            print(
                "  Failures:",
                len(
                    condition_failures
                ),
            )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    print()

    print(
        "Total anonymization validation failures:",
        len(
            all_failures
        ),
    )

    # --------------------------------------------------------
    # Example
    # --------------------------------------------------------

    natural_clean = (
        datasets[
            "clean"
        ][
            "items"
        ]
    )

    anonymous_clean = (
        output_by_condition[
            "clean"
        ]
    )

    if (
        natural_clean
        and anonymous_clean
    ):

        print_example(
            natural=
                natural_clean[0],

            anonymized=
                anonymous_clean[0],
        )