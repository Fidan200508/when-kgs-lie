from data import load_kb, load_questions
from evidence import (
    build_adjacency,
    find_exact_hop_paths,
)


def print_unique_relations(triples):

    relations = sorted(
        set(
            triple["relation"]
            for triple in triples
        )
    )

    print("=" * 70)
    print("UNIQUE RELATIONS")
    print("=" * 70)

    for relation in relations:
        print(relation)

    print()

    print(
        "Total relations:",
        len(relations)
    )


def inspect_spike_lee(adjacency):

    source = "Spike Lee"

    answers = [
        "Jungle Fever",
        "Mo' Better Blues",
        "Girl 6",
        "She's Gotta Have It",
    ]

    print()
    print("=" * 70)
    print("SPIKE LEE ALL 1-HOP PATHS")
    print("=" * 70)

    for answer in answers:

        paths = find_exact_hop_paths(
            adjacency=adjacency,
            source=source,
            target=answer,
            num_hops=1,
            max_paths=20,
        )

        print()
        print(
            f"{source} -> {answer}"
        )

        if not paths:

            print("NO PATH")
            continue

        for path in paths:

            for step in path:

                direction = (
                    "REVERSE"
                    if step["reverse"]
                    else "FORWARD"
                )

                print(
                    f"relation={step['relation']}, "
                    f"direction={direction}, "
                    f"original_triple="
                    f"{step['triple']}"
                )


def inspect_missing_3hop(
    adjacency,
    n_sample=100,
    seed=45,
):

    import random

    items = load_questions(3)

    rng = random.Random(seed)

    sampled = rng.sample(
        items,
        n_sample,
    )

    print()
    print("=" * 70)
    print("3-HOP MISSING ITEMS")
    print("=" * 70)

    missing_questions = 0

    for item in sampled:

        missing_answers = []

        for answer in item["gold_answers"]:

            paths = find_exact_hop_paths(
                adjacency=adjacency,
                source=item["topic_entity"],
                target=answer,
                num_hops=3,
                max_paths=1,
            )

            if not paths:
                missing_answers.append(
                    answer
                )

        if missing_answers:

            missing_questions += 1

            print()
            print(
                "QID:",
                item["qid"],
            )

            print(
                "QUESTION:",
                item["question"],
            )

            print(
                "TOPIC:",
                item["topic_entity"],
            )

            print(
                "ALL GOLD:",
                item["gold_answers"],
            )

            print(
                "MISSING:",
                missing_answers,
            )

    print()
    print(
        "Questions with missing paths:",
        missing_questions,
    )


if __name__ == "__main__":

    print("Loading KG...")

    triples = load_kb()

    adjacency = build_adjacency(
        triples
    )

    print_unique_relations(
        triples
    )

    inspect_spike_lee(
        adjacency
    )

    inspect_missing_3hop(
        adjacency
    )