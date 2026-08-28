from pathlib import Path
import json
import random


PATH = Path(
    "data/processed/"
    "metaqa_pilot_entity_substitution.jsonl"
)


def load_jsonl(path):
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


if __name__ == "__main__":

    items = load_jsonl(PATH)

    rng = random.Random(42)

    sample = rng.sample(
        items,
        20,
    )

    for i, item in enumerate(
        sample,
        start=1,
    ):

        manifest = item[
            "corruption_manifest"
        ]

        print()
        print("=" * 80)
        print(f"EXAMPLE {i}")
        print("=" * 80)

        print(
            "QID:",
            item["qid"],
        )

        print(
            "Hop:",
            item["hop"],
        )

        print(
            "Question:",
            item["question"],
        )

        print(
            "Gold:",
            item["gold_answers"],
        )

        print(
            "Target gold:",
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
            "Injected false answer:",
            manifest[
                "injected_false_answer"
            ],
        )