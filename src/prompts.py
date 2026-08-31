# ============================================================
# FINAL CONTROLLED KG-QA PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are a knowledge-graph question answering system. "
    "Answer strictly using only the provided knowledge graph evidence."
)


# ============================================================
# SERIALIZE EVIDENCE
# ============================================================

def serialize_evidence(item):
    """
    Only raw triples are shown to the model.

    NEVER expose internal metadata such as:

        support
        context
        contradiction
        corruption type
        gold answers
        injected false answer
    """

    lines = []

    for entry in item[
        "evidence"
    ]:

        head, relation, tail = (
            entry[
                "triple"
            ]
        )

        lines.append(
            f"{head} | "
            f"{relation} | "
            f"{tail}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# BUILD USER PROMPT
# ============================================================

def build_user_prompt(item):
    """
    Exact same prompt template for every condition.

    Conditions:
        clean
        entity substitution
        relation substitution
        contradiction
        rerouting

    Label modes:
        natural
        anonymized
    """

    evidence_text = (
        serialize_evidence(
            item
        )
    )

    question = item[
        "question"
    ]

    prompt = f"""
Answer the question using only the provided knowledge graph evidence.

Return only the answer value.

If there are multiple answers, separate them with semicolons.

If the evidence is insufficient to determine the requested answer or answer set,
return exactly:
UNKNOWN

If conflicting evidence prevents determining the requested answer or answer set,
return exactly:
UNKNOWN

Do not use outside knowledge.
Do not explain your reasoning.

Knowledge graph evidence:
{evidence_text}

Question:
{question}

Answer:
""".strip()

    return prompt


# ============================================================
# CHAT FORMAT
# ============================================================

def build_messages(item):
    return [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },
        {
            "role":
                "user",

            "content":
                build_user_prompt(
                    item
                ),
        },
    ]


# ============================================================
# LOCAL LEAKAGE TEST
# ============================================================

if __name__ == "__main__":

    import json

    from pathlib import Path

    path = Path(
        "data/processed/metaqa_pilot_clean.jsonl"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        item = json.loads(
            next(f)
        )

    print(
        "=" * 80
    )

    print(
        "SYSTEM"
    )

    print(
        "=" * 80
    )

    print(
        SYSTEM_PROMPT
    )

    print()

    print(
        "=" * 80
    )

    print(
        "USER"
    )

    print(
        "=" * 80
    )

    prompt_text = (
        build_user_prompt(
            item
        )
    )

    print(
        prompt_text
    )

    print()

    print(
        "=" * 80
    )

    print(
        "LEAKAGE CHECK"
    )

    print(
        "=" * 80
    )

    forbidden_terms = [
        "SUPPORT |",
        "CONTEXT |",
        "CONTRADICTION |",
        "corruption_manifest",
        "gold_answers",
        "injected_false_answer",
    ]

    found = []

    for term in (
        forbidden_terms
    ):

        if term in prompt_text:

            found.append(
                term
            )

    if found:

        print(
            "FAIL - forbidden metadata found:",
            found,
        )

    else:

        print(
            "PASS - no internal metadata exposed"
        )