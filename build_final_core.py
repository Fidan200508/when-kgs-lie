from pathlib import Path
import json
import random

SEED = 42
N_PER_HOP = 14

BASE = Path("data/processed")
OUT = BASE / "final_core"
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


clean = load(
    BASE / "metaqa_pilot_clean.jsonl"
)

relation = load(
    BASE
    / "metaqa_pilot_relation_substitution.jsonl"
)

reroute = load(
    BASE
    / "metaqa_pilot_rerouting.jsonl"
)

relation_qids = {
    x["qid"]
    for x in relation
}

reroute_qids = {
    x["qid"]
    for x in reroute
}

eligible = {
    2: [],
    3: [],
}

for x in clean:

    qid = x["qid"]

    if qid not in relation_qids:
        continue

    if qid not in reroute_qids:
        continue

    if x["hop"] not in (2, 3):
        continue

    gold_count = len(
        x.get(
            "gold_answers",
            [],
        )
    )

    evidence_count = len(
        x.get(
            "evidence",
            [],
        )
    )

    if (
        gold_count <= 10
        and evidence_count <= 40
    ):
        eligible[
            x["hop"]
        ].append(
            x
        )


print(
    "Eligible before sampling:"
)
print(
    "2-hop:",
    len(
        eligible[2]
    ),
)
print(
    "3-hop:",
    len(
        eligible[3]
    ),
)

if (
    len(eligible[2])
    < N_PER_HOP
    or len(eligible[3])
    < N_PER_HOP
):
    raise RuntimeError(
        "Not enough eligible questions."
    )


rng = random.Random(
    SEED
)

eligible2 = sorted(
    eligible[2],
    key=lambda x:
        x["qid"],
)

eligible3 = sorted(
    eligible[3],
    key=lambda x:
        x["qid"],
)

selected2 = rng.sample(
    eligible2,
    N_PER_HOP,
)

selected3 = rng.sample(
    eligible3,
    N_PER_HOP,
)

selected = (
    selected2
    + selected3
)

selected_qids = {
    x["qid"]
    for x in selected
}


manifest = {
    "seed":
        SEED,

    "selection": {
        "required_relation_substitution_feasible":
            True,

        "required_rerouting_feasible":
            True,

        "max_gold_answers":
            10,

        "max_evidence_triples":
            40,

        "n_per_hop":
            N_PER_HOP,

        "hops":
            [2, 3],
    },

    "eligible_counts": {
        "2hop":
            len(
                eligible[2]
            ),

        "3hop":
            len(
                eligible[3]
            ),
    },

    "selected_count":
        len(
            selected_qids
        ),

    "selected": [
        {
            "qid":
                x["qid"],

            "hop":
                x["hop"],

            "gold_count":
                len(
                    x.get(
                        "gold_answers",
                        [],
                    )
                ),

            "evidence_count":
                len(
                    x.get(
                        "evidence",
                        [],
                    )
                ),
        }

        for x in sorted(
            selected,
            key=lambda z:
                (
                    z["hop"],
                    z["qid"],
                ),
        )
    ],
}


with open(
    OUT
    / "final_core_manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=2,
    )


FILES = [
    "metaqa_pilot_clean.jsonl",
    "metaqa_pilot_clean_anonymized.jsonl",

    "metaqa_pilot_entity_substitution.jsonl",
    "metaqa_pilot_entity_substitution_anonymized.jsonl",

    "metaqa_pilot_relation_substitution.jsonl",
    "metaqa_pilot_relation_substitution_anonymized.jsonl",

    "metaqa_pilot_contradiction.jsonl",
    "metaqa_pilot_contradiction_anonymized.jsonl",

    "metaqa_pilot_rerouting.jsonl",
    "metaqa_pilot_rerouting_anonymized.jsonl",
]


print()
print(
    "Writing final core files..."
)

for filename in FILES:

    source = (
        BASE
        / filename
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    rows = load(
        source
    )

    filtered = [
        x
        for x in rows
        if x["qid"]
        in selected_qids
    ]

    if len(filtered) != 28:
        raise RuntimeError(
            f"{filename}: "
            f"expected 28, "
            f"found {len(filtered)}"
        )

    destination = (
        OUT
        / filename
    )

    save(
        destination,
        filtered,
    )

    hops = {}

    for x in filtered:

        hop = x["hop"]

        hops[hop] = (
            hops.get(
                hop,
                0,
            )
            + 1
        )

    print(
        filename,
        ":",
        len(filtered),
        "| hops:",
        hops,
    )


print()
print(
    "FINAL CORE READY"
)
print(
    "Questions:",
    len(
        selected_qids
    ),
)
print(
    "2-hop:",
    sum(
        x["hop"] == 2
        for x in selected
    ),
)
print(
    "3-hop:",
    sum(
        x["hop"] == 3
        for x in selected
    ),
)
print(
    "Output:",
    OUT,
)
