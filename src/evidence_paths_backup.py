from collections import defaultdict, deque

from data import load_kb, load_questions


def build_adjacency(triples):
    """
    KG üçün bidirectional adjacency index qurur.

    Original triple:
        head | relation | tail

    Traversal zamanı:
        head -> tail       reverse=False
        tail -> head       reverse=True

    Original triple orientation həmişə ayrıca saxlanılır.
    """

    adjacency = defaultdict(list)

    for triple in triples:
        head = triple["head"]
        relation = triple["relation"]
        tail = triple["tail"]

        original_triple = (head, relation, tail)

        # Forward traversal
        adjacency[head].append(
            {
                "next": tail,
                "relation": relation,
                "reverse": False,
                "triple": original_triple,
            }
        )

        # Reverse traversal
        adjacency[tail].append(
            {
                "next": head,
                "relation": relation,
                "reverse": True,
                "triple": original_triple,
            }
        )

    return adjacency


def find_exact_hop_paths(
    adjacency,
    source,
    target,
    num_hops,
    max_paths=1,
):
    """
    source-dan target-a EXACT num_hops uzunluğunda path tapır.

    Məsələn:
        2-hop question -> yalnız 2 edge-lik path qəbul edilir.
    """

    if source is None or target is None:
        return []

    paths_found = []

    # current_entity, path_so_far, visited_nodes
    queue = deque(
        [
            (
                source,
                [],
                {source},
            )
        ]
    )

    while queue:
        current, path, visited = queue.popleft()

        # Exact hop sayına çatmışıq
        if len(path) == num_hops:
            if current == target:
                paths_found.append(path)

                if len(paths_found) >= max_paths:
                    return paths_found

            continue

        for edge in adjacency.get(current, []):
            nxt = edge["next"]

            # Sadə cycle prevention
            if nxt in visited:
                continue

            step = {
                "from_entity": current,
                "to_entity": nxt,
                "relation": edge["relation"],
                "reverse": edge["reverse"],
                "triple": edge["triple"],
            }

            queue.append(
                (
                    nxt,
                    path + [step],
                    visited | {nxt},
                )
            )

    return paths_found


def find_support_paths(item, adjacency, max_paths_per_answer=1):
    """
    Hər gold answer üçün topic entity-dən həmin answer-a
    item['hop'] uzunluğunda support path tapır.
    """

    source = item["topic_entity"]
    hop = item["hop"]

    result = {}

    for answer in item["gold_answers"]:
        paths = find_exact_hop_paths(
            adjacency=adjacency,
            source=source,
            target=answer,
            num_hops=hop,
            max_paths=max_paths_per_answer,
        )

        result[answer] = paths

    return result


def format_path(source, path):
    """
    Path-i insan üçün rahat formatda göstərir.
    """

    parts = [source]

    for step in path:
        relation = step["relation"]

        if step["reverse"]:
            relation = f"{relation}^-1"

        parts.append(
            f"--[{relation}]--> {step['to_entity']}"
        )

    return " ".join(parts)


def test_support_path_extraction(n_per_hop=20):
    print("Loading MetaQA KG...")

    triples = load_kb()

    print(f"Triples: {len(triples)}")

    print("Building adjacency index...")

    adjacency = build_adjacency(triples)

    print(f"Entities in adjacency: {len(adjacency)}")
    print()

    for hop in [1, 2, 3]:

        items = load_questions(hop)[:n_per_hop]

        total_answers = 0
        found_answers = 0

        first_example = None

        for item in items:

            support_paths = find_support_paths(
                item,
                adjacency,
                max_paths_per_answer=1,
            )

            if first_example is None:
                first_example = (
                    item,
                    support_paths,
                )

            for answer in item["gold_answers"]:
                total_answers += 1

                if support_paths[answer]:
                    found_answers += 1

        coverage = (
            found_answers / total_answers
            if total_answers
            else 0
        )

        print("=" * 70)
        print(f"{hop}-HOP")
        print(
            f"Support-path coverage: "
            f"{found_answers}/{total_answers} "
            f"({coverage:.2%})"
        )

        item, support_paths = first_example

        print()
        print("Question:")
        print(item["question"])

        print()
        print("Topic entity:")
        print(item["topic_entity"])

        print()
        print("Gold answers:")
        print(item["gold_answers"])

        print()
        print("Support paths:")

        for answer, paths in support_paths.items():

            print(f"\nAnswer: {answer}")

            if not paths:
                print("  NOT FOUND")
                continue

            for path in paths:
                print(
                    " ",
                    format_path(
                        item["topic_entity"],
                        path,
                    ),
                )

        print()


if __name__ == "__main__":
    test_support_path_extraction(
        n_per_hop=20
    )