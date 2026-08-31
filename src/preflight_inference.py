from collections import Counter

from finalize_clean import (
    validate_packet_answer_set,
)

from inference import (
    DATASETS,
    generate_answer,
    load_jsonl,
    load_model,
)

from prompts import (
    build_user_prompt,
)


# ============================================================
# FINAL EXPECTED DATASET SIZES
# ============================================================

EXPECTED_SIZES = {

    (
        "clean",
        "natural",
    ):
        300,

    (
        "clean",
        "anonymized",
    ):
        300,

    (
        "entity_substitution",
        "natural",
    ):
        300,

    (
        "entity_substitution",
        "anonymized",
    ):
        300,

    (
        "relation_substitution",
        "natural",
    ):
        156,

    (
        "relation_substitution",
        "anonymized",
    ):
        156,

    (
        "contradiction",
        "natural",
    ):
        300,

    (
        "contradiction",
        "anonymized",
    ):
        300,

    (
        "rerouting",
        "natural",
    ):
        197,

    (
        "rerouting",
        "anonymized",
    ):
        197,
}


# ============================================================
# FORBIDDEN PROMPT METADATA
# ============================================================

FORBIDDEN_PROMPT_TERMS = [

    "corruption_manifest",

    "gold_answers",

    "injected_false_answer",

    "SUPPORT |",

    "CONTEXT |",

    "CONTRADICTION |",
]


# ============================================================
# MAIN
# ============================================================

def main():

    loaded = {}

    static_ok = True

    # ========================================================
    # 1. DATASET SIZES + DUPLICATE QIDS
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "1. DATASET SIZE + UNIQUE QID CHECK"
    )

    print(
        "=" * 80
    )

    for spec in DATASETS:

        key = (
            spec[
                "condition"
            ],
            spec[
                "label_mode"
            ],
        )

        items = (
            load_jsonl(
                spec[
                    "input"
                ]
            )
        )

        loaded[
            key
        ] = items

        expected = (
            EXPECTED_SIZES[
                key
            ]
        )

        qids = [

            item[
                "qid"
            ]

            for item
            in items
        ]

        size_ok = (
            len(
                items
            )
            == expected
        )

        unique_ok = (
            len(
                qids
            )
            == len(
                set(
                    qids
                )
            )
        )

        ok = (
            size_ok
            and unique_ok
        )

        static_ok = (
            static_ok
            and ok
        )

        print(
            f"{key[0]:22} "
            f"{key[1]:11} "
            f"N={len(items):3} "
            f"expected={expected:3} "
            f"unique={unique_ok} "
            f"=> "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # ========================================================
    # 2. NATURAL / ANONYMIZED QID PAIRING
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "2. NATURAL / ANONYMIZED PAIRING"
    )

    print(
        "=" * 80
    )

    conditions = [

        "clean",

        "entity_substitution",

        "relation_substitution",

        "contradiction",

        "rerouting",
    ]

    for condition in conditions:

        natural_qids = {

            item[
                "qid"
            ]

            for item
            in loaded[
                (
                    condition,
                    "natural",
                )
            ]
        }

        anonymous_qids = {

            item[
                "qid"
            ]

            for item
            in loaded[
                (
                    condition,
                    "anonymized",
                )
            ]
        }

        ok = (
            natural_qids
            == anonymous_qids
        )

        static_ok = (
            static_ok
            and ok
        )

        print(
            f"{condition:22} "
            f"paired_qids={ok}"
        )

    # ========================================================
    # 3. CORRUPTED QIDS MUST BE SUBSET OF CLEAN
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "3. CORRUPTED QIDS ARE CLEAN SUBSETS"
    )

    print(
        "=" * 80
    )

    clean_qids = {

        item[
            "qid"
        ]

        for item
        in loaded[
            (
                "clean",
                "natural",
            )
        ]
    }

    for condition in conditions[
        1:
    ]:

        qids = {

            item[
                "qid"
            ]

            for item
            in loaded[
                (
                    condition,
                    "natural",
                )
            ]
        }

        ok = (
            qids
            <= clean_qids
        )

        static_ok = (
            static_ok
            and ok
        )

        print(
            f"{condition:22} "
            f"subset_of_clean={ok}"
        )

    # ========================================================
    # 4. HOP DISTRIBUTIONS
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "4. HOP DISTRIBUTIONS"
    )

    print(
        "=" * 80
    )

    for condition in conditions:

        counts = Counter(

            item[
                "hop"
            ]

            for item
            in loaded[
                (
                    condition,
                    "natural",
                )
            ]
        )

        print(
            f"{condition:22} "
            f"{dict(sorted(counts.items()))}"
        )

    # ========================================================
    # 5. FULL CLEAN PACKET ANSWER SET
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "5. FULL CLEAN PACKET ANSWER-SET CHECK"
    )

    print(
        "=" * 80
    )

    packet_failures = []

    for item in loaded[
        (
            "clean",
            "natural",
        )
    ]:

        (
            ok,
            info,
        ) = (
            validate_packet_answer_set(
                item
            )
        )

        if not ok:

            packet_failures.append(
                {
                    "qid":
                        item[
                            "qid"
                        ],

                    **info,
                }
            )

    print(
        "Packet failures:",
        len(
            packet_failures
        ),
    )

    if packet_failures:

        static_ok = False

        print(
            "First failures:",
            packet_failures[
                :5
            ],
        )

    else:

        print(
            "PASS - every full clean evidence packet "
            "yields exactly the gold answer set"
        )

    # ========================================================
    # 6. PROMPT LEAKAGE
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "6. PROMPT LEAKAGE CHECK"
    )

    print(
        "=" * 80
    )

    leaks = []

    for (
        condition,
        label_mode,
    ), items in loaded.items():

        for item in items:

            prompt = (
                build_user_prompt(
                    item
                )
            )

            for forbidden in (
                FORBIDDEN_PROMPT_TERMS
            ):

                if forbidden in prompt:

                    leaks.append(
                        (
                            item[
                                "qid"
                            ],

                            condition,

                            label_mode,

                            forbidden,
                        )
                    )

    print(
        "Leak count:",
        len(
            leaks
        ),
    )

    if leaks:

        static_ok = False

        print(
            "First leaks:",
            leaks[
                :10
            ],
        )

    else:

        print(
            "PASS - no internal metadata exposed"
        )

    # ========================================================
    # 7. ANONYMIZED NODE CHECK
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "7. ANONYMIZED EVIDENCE CHECK"
    )

    print(
        "=" * 80
    )

    anonymous_failures = []

    for condition in conditions:

        for item in loaded[
            (
                condition,
                "anonymized",
            )
        ]:

            for entry in item[
                "evidence"
            ]:

                head, _, tail = (
                    entry[
                        "triple"
                    ]
                )

                if not head.startswith(
                    "Entity_"
                ):

                    anonymous_failures.append(
                        (
                            item[
                                "qid"
                            ],

                            condition,

                            "head",

                            head,
                        )
                    )

                if not tail.startswith(
                    "Entity_"
                ):

                    anonymous_failures.append(
                        (
                            item[
                                "qid"
                            ],

                            condition,

                            "tail",

                            tail,
                        )
                    )

    print(
        "Anonymous label failures:",
        len(
            anonymous_failures
        ),
    )

    if anonymous_failures:

        static_ok = False

        print(
            "First failures:",
            anonymous_failures[
                :10
            ],
        )

    else:

        print(
            "PASS - all evidence nodes anonymized"
        )

    # ========================================================
    # 8. NATURAL VS ANON STRUCTURE
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "8. NATURAL / ANONYMIZED STRUCTURE CHECK"
    )

    print(
        "=" * 80
    )

    structure_failures = []

    for condition in conditions:

        natural_by_qid = {

            item[
                "qid"
            ]:
                item

            for item
            in loaded[
                (
                    condition,
                    "natural",
                )
            ]
        }

        anonymous_by_qid = {

            item[
                "qid"
            ]:
                item

            for item
            in loaded[
                (
                    condition,
                    "anonymized",
                )
            ]
        }

        for (
            qid,
            natural,
        ) in natural_by_qid.items():

            anonymous = (
                anonymous_by_qid[
                    qid
                ]
            )

            # ------------------------------------------------
            # Evidence count
            # ------------------------------------------------

            if (
                len(
                    natural[
                        "evidence"
                    ]
                )
                != len(
                    anonymous[
                        "evidence"
                    ]
                )
            ):

                structure_failures.append(
                    (
                        qid,
                        condition,
                        "evidence_size",
                    )
                )

                continue

            # ------------------------------------------------
            # Evidence order / role / relation
            # ------------------------------------------------

            for (
                natural_entry,
                anonymous_entry,
            ) in zip(

                natural[
                    "evidence"
                ],

                anonymous[
                    "evidence"
                ],
            ):

                if (
                    natural_entry[
                        "role"
                    ]
                    != anonymous_entry[
                        "role"
                    ]
                ):

                    structure_failures.append(
                        (
                            qid,
                            condition,
                            "role_order",
                        )
                    )

                    break

                if (
                    natural_entry[
                        "triple"
                    ][
                        1
                    ]
                    != anonymous_entry[
                        "triple"
                    ][
                        1
                    ]
                ):

                    structure_failures.append(
                        (
                            qid,
                            condition,
                            "relation_order",
                        )
                    )

                    break

    print(
        "Structure failures:",
        len(
            structure_failures
        ),
    )

    if structure_failures:

        static_ok = False

        print(
            "First failures:",
            structure_failures[
                :10
            ],
        )

    else:

        print(
            "PASS - evidence size/order/roles/relations preserved"
        )

    # ========================================================
    # STATIC SUMMARY
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "STATIC PREFLIGHT SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "STATIC CHECKS:",
        (
            "PASS"
            if static_ok
            else "FAIL"
        ),
    )

    if not static_ok:

        raise RuntimeError(
            "Static preflight failed. "
            "Do not start full inference."
        )

    # ========================================================
    # 9. MODEL SMOKE TEST
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "9. MODEL SMOKE TEST: ONE ITEM PER CONDITION"
    )

    print(
        "=" * 80
    )

    (
        tokenizer,
        model,
    ) = (
        load_model()
    )

    errors = []

    for spec in DATASETS:

        key = (
            spec[
                "condition"
            ],
            spec[
                "label_mode"
            ],
        )

        item = (
            loaded[
                key
            ][
                0
            ]
        )

        try:

            result = (
                generate_answer(
                    item,
                    tokenizer,
                    model,
                )
            )

            print(
                f"{key[0]:22} "
                f"{key[1]:11} | "
                f"qid={item['qid']} | "
                f"out={result['output_tokens']} | "
                f"max={result['hit_max_token_limit']} | "
                f"{result['generation_seconds']:.2f}s | "
                f"{result['raw_output'][:100]!r}"
            )

        except Exception as error:

            errors.append(
                (
                    key,

                    item[
                        "qid"
                    ],

                    type(
                        error
                    ).__name__,

                    str(
                        error
                    ),
                )
            )

            print(
                f"{key} | ERROR: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL PREFLIGHT"
    )

    print(
        "=" * 80
    )

    print(
        "Smoke errors:",
        len(
            errors
        ),
    )

    if errors:

        print(
            "FULL PIPELINE PREFLIGHT: FAIL"
        )

        print(
            "Errors:",
            errors,
        )

        raise RuntimeError(
            "Model smoke test failed. "
            "Do not start full inference."
        )

    print(
        "FULL PIPELINE PREFLIGHT: PASS"
    )

    print(
        "Safe to start full inference."
    )


if __name__ == "__main__":

    main()