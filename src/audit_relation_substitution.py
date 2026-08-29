from pathlib import Path
import json
import random


PATH = Path(
    "data/processed/"
    "metaqa_pilot_relation_substitution.jsonl"
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
        print("Position:", m["corrupted_position"])
        print("Family:", m["relation_family"])

        print()
        print("ORIGINAL:")
        print(" | ".join(m["original_triple"]))

        print()
        print("CORRUPTED:")
        print(" | ".join(m["corrupted_triple"]))

        print()
        print(
            "Relation:",
            m["original_relation"],
            "->",
            m["replacement_relation"],
        )