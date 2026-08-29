from pathlib import Path
import json
import random


PATH = Path(
    "data/processed/metaqa_pilot_contradiction.jsonl"
)


def load_jsonl(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                items.append(json.loads(line))

    return items


if __name__ == "__main__":

    items = load_jsonl(PATH)

    rng = random.Random(42)

    sample = rng.sample(
        items,
        min(20, len(items)),
    )

    for i, item in enumerate(sample, start=1):

        m = item["corruption_manifest"]

        print()
        print("=" * 80)
        print(f"EXAMPLE {i}")
        print("=" * 80)

        print("QID:", item["qid"])
        print("Hop:", item["hop"])
        print("Question:", item["question"])
        print("Gold:", item["gold_answers"])
        print("Target gold:", m["target_gold_answer"])
        print()

        print("TRUE:")
        print(" | ".join(m["true_triple"]))

        print()

        print("FALSE CONTRADICTION:")
        print(
            " | ".join(
                m["contradictory_triple"]
            )
        )

        print()

        print(
            "Injected false answer:",
            m["injected_false_answer"],
        )

        print(
            "Removed context:",
            " | ".join(
                m["removed_context_triple"]
            ),
        )