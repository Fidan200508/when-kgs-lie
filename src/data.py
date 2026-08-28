from pathlib import Path
import re


METAQA_ROOT = Path("data/raw/metaqa")


def load_kb():
    """
    MetaQA knowledge graph-i oxuyur.

    Hər sətir:
        head|relation|tail
    """

    path = METAQA_ROOT / "kb.txt"

    triples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("|")

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid KG triple: {line}"
                )

            head, relation, tail = parts

            triples.append(
                {
                    "head": head.strip(),
                    "relation": relation.strip(),
                    "tail": tail.strip(),
                }
            )

    return triples


def extract_topic_entity(question):
    """
    Question-da [ ... ] arasında verilən topic entity-ni tapır.
    """

    match = re.search(r"\[(.*?)\]", question)

    if match is None:
        return None

    return match.group(1).strip()


def clean_question(question):
    """
    [entity] -> entity
    """

    return question.replace("[", "").replace("]", "").strip()


def load_questions(hop):
    """
    1-hop / 2-hop / 3-hop qa_test.txt faylını oxuyur.
    """

    path = METAQA_ROOT / f"{hop}-hop" / "qa_test.txt"

    items = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid QA line at {path}, line {idx}: {line}"
                )

            question_raw, answers_raw = parts

            topic_entity = extract_topic_entity(question_raw)

            answers = [
                answer.strip()
                for answer in answers_raw.split("|")
                if answer.strip()
            ]

            items.append(
                {
                    "qid": f"{hop}hop_test_{idx}",
                    "hop": hop,
                    "question_raw": question_raw,
                    "question": clean_question(question_raw),
                    "topic_entity": topic_entity,
                    "gold_answers": answers,
                }
            )

    return items


if __name__ == "__main__":

    print("=== Loading MetaQA KG ===")

    triples = load_kb()

    print(f"Number of triples: {len(triples)}")
    print("Example triple:")
    print(triples[0])

    print()

    for hop in [1, 2, 3]:

        questions = load_questions(hop)

        print(f"=== {hop}-hop ===")
        print(f"Questions: {len(questions)}")

        print("Example:")
        print(questions[0])

        print()