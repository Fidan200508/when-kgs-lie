from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import random
import re

from data import load_kb, load_questions


# ============================================================
# CONFIG
# ============================================================

OUTPUT_PATH = Path(
    "data/processed/metaqa_pilot_clean.jsonl"
)

REJECTION_PATH = Path(
    "results/raw/clean_semantic_rejections.jsonl"
)

TARGET_AUDIT_PATH = Path(
    "results/raw/target_relation_audit.jsonl"
)

DEFAULT_SEED = 42
TARGET_PER_HOP = 100
CONTEXT_SIZE = 5
MAX_PATHS = 500


# ============================================================
# 1. REPRODUCIBLE SEED
# ============================================================

def stable_seed(text, seed=DEFAULT_SEED):
    value = f"{seed}:{text}"

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:16],
        16,
    )


# ============================================================
# 2. JSONL SAVE
# ============================================================

def save_jsonl(items, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for item in items:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# 3. NORMALIZE KG
# ============================================================

def normalize_kb_triples(raw_triples):
    """
    load_kb() dict və ya tuple/list qaytarsa da:

        (head, relation, tail)

    formatına çevirir.
    """

    normalized = []

    for item in raw_triples:

        if isinstance(item, dict):

            if all(
                key in item
                for key in (
                    "head",
                    "relation",
                    "tail",
                )
            ):
                head = item["head"]
                relation = item["relation"]
                tail = item["tail"]

            elif all(
                key in item
                for key in (
                    "subject",
                    "predicate",
                    "object",
                )
            ):
                head = item["subject"]
                relation = item["predicate"]
                tail = item["object"]

            elif all(
                key in item
                for key in (
                    "h",
                    "r",
                    "t",
                )
            ):
                head = item["h"]
                relation = item["r"]
                tail = item["t"]

            else:
                raise ValueError(
                    "Unknown KB dictionary format: "
                    f"{item}"
                )

        elif (
            isinstance(
                item,
                (tuple, list),
            )
            and len(item) == 3
        ):
            head, relation, tail = item

        else:
            raise ValueError(
                "Unknown KB triple format: "
                f"{item}"
            )

        normalized.append(
            (
                str(head),
                str(relation),
                str(tail),
            )
        )

    return normalized


# ============================================================
# 4. BUILD BIDIRECTIONAL ADJACENCY
# ============================================================

def build_adjacency(triples):
    """
    Original KG:

        head | relation | tail

    Traversal üçün:

        head -> tail
        tail -> head

    yaradılır.

    reverse=True original triple-ın əks istiqamətdə
    traverse edildiyini göstərir.
    """

    adjacency = defaultdict(list)

    for head, relation, tail in triples:

        original_triple = (
            head,
            relation,
            tail,
        )

        # Forward
        adjacency[head].append(
            {
                "to_entity": tail,
                "relation": relation,
                "reverse": False,
                "triple": original_triple,
            }
        )

        # Reverse
        adjacency[tail].append(
            {
                "to_entity": head,
                "relation": relation,
                "reverse": True,
                "triple": original_triple,
            }
        )

    for entity in adjacency:

        adjacency[entity] = sorted(
            adjacency[entity],
            key=lambda edge: (
                edge["relation"],
                edge["to_entity"],
                edge["reverse"],
                edge["triple"],
            ),
        )

    return adjacency


# ============================================================
# 5. QUESTION NORMALIZATION
# ============================================================

def normalize_question(question):
    return (
        question
        .lower()
        .replace("’", "'")
        .replace("–", "-")
        .strip()
    )


# ============================================================
# 6. QUESTION RELATION SCORES
# ============================================================

def get_question_relation_scores(question):
    """
    Metadata/debug üçün soft lexical scores.

    Support path selection strict profile ilə edilir;
    bu score əsas seçim mexanizmi deyil.
    """

    q = normalize_question(
        question
    )

    scores = {
        "directed_by": 0.0,
        "written_by": 0.0,
        "starred_actors": 0.0,
        "has_genre": 0.0,
        "in_language": 0.0,
        "release_year": 0.0,
        "has_imdb_rating": 0.0,
        "has_imdb_votes": 0.0,
        "has_tags": 0.0,
    }

    patterns = {
        "directed_by":
            r"\b(?:directed|director|directors)\b",

        "written_by":
            r"\b(?:written|wrote|writer|writers|"
            r"screenwriter|screenwriters)\b",

        "starred_actors":
            r"\b(?:actor|actors|actress|actresses|"
            r"act|acted|star|stars|starred|starring|"
            r"appear|appears|appeared)\b",

        "has_genre":
            r"\b(?:genre|genres|type|types|kind|kinds)\b",

        "in_language":
            r"\blanguages?\b",

        "release_year":
            r"\b(?:release|released|year|years|when)\b",

        "has_imdb_rating":
            r"\b(?:imdb\s+)?ratings?\b",

        "has_imdb_votes":
            r"\b(?:imdb\s+)?votes?\b",

        "has_tags":
            r"\btags?\b",
    }

    for relation, pattern in patterns.items():

        scores[relation] = float(
            len(
                re.findall(
                    pattern,
                    q,
                )
            )
        )

    return scores


# ============================================================
# 7. FIRST MATCH POSITION
# ============================================================

def first_match_position(text, pattern):
    match = re.search(
        pattern,
        text,
    )

    if match is None:
        return None

    return match.start()


# ============================================================
# 8. INFER TARGET / FINAL RELATION
# ============================================================

def infer_target_relation(question):
    """
    Question-un tələb etdiyi FINAL answer relation.

    Support path-in son traversal step-i bu relation
    olmalıdır.
    """

    q = normalize_question(
        question
    )

    # ========================================================
    # A. VALUE-TYPE ANSWERS
    # ========================================================

    if re.search(
        r"\b(?:imdb\s+)?votes?\b",
        q,
    ):
        return "has_imdb_votes"

    if re.search(
        r"\b(?:imdb\s+)?ratings?\b",
        q,
    ):
        return "has_imdb_rating"

    if re.search(
        r"\btags?\b",
        q,
    ):
        return "has_tags"

    if re.search(
        r"\blanguages?\b",
        q,
    ):
        return "in_language"

    if (
        re.search(
            r"^\s*when\b",
            q,
        )
        or re.search(
            r"\brelease\s+years?\b",
            q,
        )
        or re.search(
            r"\breleased\s+(?:in|when)\b",
            q,
        )
        or re.search(
            r"\bwhat\s+year\b",
            q,
        )
        or re.search(
            r"\bwhich\s+year\b",
            q,
        )
        or re.search(
            r"\byears?\s+(?:was|were|did)\b",
            q,
        )
    ):
        return "release_year"

    if (
        re.search(
            r"\bgenres?\b",
            q,
        )
        or re.search(
            r"\bwhat\s+(?:kind|type)\s+of\s+"
            r"(?:film|movie)\b",
            q,
        )
        or re.search(
            r"\bwhat\s+(?:kind|type)\b",
            q,
        )
        or re.search(
            r"\bwhat\s+types?\b",
            q,
        )
    ):
        return "has_genre"

    # ========================================================
    # B. STRONG DIRECTOR ANSWER PATTERNS
    # ========================================================

    if re.search(
        r"\bdirected\s+by\s+"
        r"(?:who|which\s+person)\s*\??$",
        q,
    ):
        return "directed_by"

    if re.search(
        r"\b(?:who|which\s+person)\s+"
        r"(?:is|was)?\s*"
        r"(?:listed\s+as\s+)?"
        r"(?:the\s+)?director\b",
        q,
    ):
        return "directed_by"

    # ========================================================
    # C. STRONG WRITER ANSWER PATTERNS
    # ========================================================

    if re.search(
        r"\bwritten\s+by\s+"
        r"(?:who|which\s+person|this\s+person)\s*\??$",
        q,
    ):
        return "written_by"

    if re.search(
        r"\b(?:who|which\s+person)\s+"
        r"(?:is|was)?\s*"
        r"(?:listed\s+as\s+)?"
        r"(?:the\s+)?"
        r"(?:writer|screenwriter)\b",
        q,
    ):
        return "written_by"

    # ========================================================
    # D. STRONG ACTOR ANSWER PATTERNS
    # ========================================================

    if re.search(
        r"\b(?:starred|stars|acted)\s+"
        r"(?:who|which\s+person)\s*\??$",
        q,
    ):
        return "starred_actors"

    # ========================================================
    # E. REVERSE MOVIE QUESTIONS
    # ========================================================

    # what does X star in
    if re.search(
        r"\bwhat\s+does\b.*\bstar\s+in\b",
        q,
    ):
        return "starred_actors"

    # what does X act in
    if re.search(
        r"\bwhat\s+does\b.*\bact\s+in\b",
        q,
    ):
        return "starred_actors"

    # what/which films did X act/star/appear in
    if (
        re.search(
            r"\b(?:films?|movies?)\b",
            q,
        )
        and re.search(
            r"\b(?:act|acted|star|starred|"
            r"appear|appears|appeared)\b",
            q,
        )
    ):
        return "starred_actors"

    # ========================================================
    # F. QUESTION PREFIX PERSON INTENT
    # ========================================================

    if re.search(
        r"^\s*(?:who|which\s+person|"
        r"what\s+(?:films?|movies?)|"
        r"which\s+(?:films?|movies?))\b",
        q,
    ):

        positions = {}

        directed_position = (
            first_match_position(
                q,
                r"\b(?:directed|director|directors)\b",
            )
        )

        written_position = (
            first_match_position(
                q,
                r"\b(?:written|wrote|writer|writers|"
                r"screenwriter|screenwriters)\b",
            )
        )

        actor_position = (
            first_match_position(
                q,
                r"\b(?:act|acted|star|stars|starred|starring|"
                r"actor|actors|appear|appears|appeared)\b",
            )
        )

        if directed_position is not None:
            positions[
                "directed_by"
            ] = directed_position

        if written_position is not None:
            positions[
                "written_by"
            ] = written_position

        if actor_position is not None:
            positions[
                "starred_actors"
            ] = actor_position

        if positions:

            return min(
                positions,
                key=positions.get,
            )

    # ========================================================
    # G. ONLY WRITER RELATION PRESENT
    # ========================================================

    if (
        re.search(
            r"\b(?:films?|movies?)\b",
            q,
        )
        and re.search(
            r"\b(?:writer|writers|screenwriter|screenwriters|"
            r"written|wrote)\b",
            q,
        )
        and not re.search(
            r"\b(?:directed|director|"
            r"act|acted|starred|actors?)\b",
            q,
        )
    ):
        return "written_by"

    # ========================================================
    # H. ONLY DIRECTOR RELATION PRESENT
    # ========================================================

    if (
        re.search(
            r"\b(?:films?|movies?)\b",
            q,
        )
        and re.search(
            r"\b(?:directed|director|directors)\b",
            q,
        )
        and not re.search(
            r"\b(?:written|writer|screenwriter|"
            r"act|acted|starred|actors?)\b",
            q,
        )
    ):
        return "directed_by"

    # ========================================================
    # I. ENDING CUES
    # ========================================================

    if re.search(
        r"\b(?:director|directors)\s*\??$",
        q,
    ):
        return "directed_by"

    if re.search(
        r"\b(?:writer|writers|"
        r"screenwriter|screenwriters)\s*\??$",
        q,
    ):
        return "written_by"

    if re.search(
        r"\b(?:actor|actors|"
        r"actress|actresses)\s*\??$",
        q,
    ):
        return "starred_actors"

    # ========================================================
    # J. FALLBACK IF ONLY ONE PERSON RELATION EXISTS
    # ========================================================

    person_relations = set()

    if re.search(
        r"\b(?:directed|director|directors)\b",
        q,
    ):
        person_relations.add(
            "directed_by"
        )

    if re.search(
        r"\b(?:written|wrote|writer|writers|"
        r"screenwriter|screenwriters)\b",
        q,
    ):
        person_relations.add(
            "written_by"
        )

    if re.search(
        r"\b(?:actor|actors|actress|actresses|"
        r"act|acted|star|stars|starred|starring|"
        r"appear|appears|appeared)\b",
        q,
    ):
        person_relations.add(
            "starred_actors"
        )

    if len(person_relations) == 1:

        return next(
            iter(
                person_relations
            )
        )

    return None


# ============================================================
# 9. COUNT PERSON-RELATION MENTIONS
# ============================================================

def count_person_relation_mentions(question):
    q = normalize_question(
        question
    )

    counts = Counter()

    directed_events = re.findall(
        r"\b(?:directed|director|directors)\b",
        q,
    )

    written_events = re.findall(
        r"\b(?:written|wrote|writer|writers|"
        r"screenwriter|screenwriters)\b",
        q,
    )

    actor_events = re.findall(
        r"\b(?:actor|actors|actress|actresses|"
        r"act|acted|star|stars|starred|starring|"
        r"appear|appears|appeared)\b",
        q,
    )

    if directed_events:
        counts[
            "directed_by"
        ] = len(
            directed_events
        )

    if written_events:
        counts[
            "written_by"
        ] = len(
            written_events
        )

    if actor_events:
        counts[
            "starred_actors"
        ] = len(
            actor_events
        )

    return counts


# ============================================================
# 10. INFER REQUIRED RELATION PROFILE
# ============================================================

def infer_required_relation_counts(
    question,
    hop,
    target_relation,
):
    """
    Example:

    who directed movies written by X

        written_by: 1
        directed_by: 1

    genres of films that share actors with X

        starred_actors: 2
        has_genre: 1
    """

    q = normalize_question(
        question
    )

    counts = (
        count_person_relation_mentions(
            question
        )
    )

    # ========================================================
    # VALUE RELATIONS
    # ========================================================

    if (
        target_relation == "has_genre"
        or re.search(
            r"\b(?:genres?|types?|kinds?)\b",
            q,
        )
    ):
        counts[
            "has_genre"
        ] = max(
            counts["has_genre"],
            1,
        )

    if (
        target_relation == "in_language"
        or re.search(
            r"\blanguages?\b",
            q,
        )
    ):
        counts[
            "in_language"
        ] = max(
            counts["in_language"],
            1,
        )

    if (
        target_relation == "release_year"
        or re.search(
            r"\brelease\s+years?\b",
            q,
        )
        or re.search(
            r"\breleased\b",
            q,
        )
        or re.search(
            r"^\s*when\b",
            q,
        )
    ):
        counts[
            "release_year"
        ] = max(
            counts["release_year"],
            1,
        )

    if (
        target_relation == "has_imdb_rating"
        or re.search(
            r"\b(?:imdb\s+)?ratings?\b",
            q,
        )
    ):
        counts[
            "has_imdb_rating"
        ] = max(
            counts["has_imdb_rating"],
            1,
        )

    if (
        target_relation == "has_imdb_votes"
        or re.search(
            r"\b(?:imdb\s+)?votes?\b",
            q,
        )
    ):
        counts[
            "has_imdb_votes"
        ] = max(
            counts["has_imdb_votes"],
            1,
        )

    if (
        target_relation == "has_tags"
        or re.search(
            r"\btags?\b",
            q,
        )
    ):
        counts[
            "has_tags"
        ] = max(
            counts["has_tags"],
            1,
        )

    # ========================================================
    # SHARED PROPERTY = RELATION USED TWICE
    # ========================================================

    shared_patterns = {
        "directed_by":
            r"\b(?:share|shares|shared|same)\b.*"
            r"\b(?:director|directors)\b",

        "written_by":
            r"\b(?:share|shares|shared|same)\b.*"
            r"\b(?:writer|writers|"
            r"screenwriter|screenwriters)\b",

        "starred_actors":
            r"\b(?:share|shares|shared|same)\b.*"
            r"\b(?:actor|actors|"
            r"actress|actresses)\b",
    }

    for relation, pattern in (
        shared_patterns.items()
    ):

        if re.search(
            pattern,
            q,
        ):
            counts[
                relation
            ] = max(
                counts[relation],
                2,
            )

    # acted/starred together with X
    if re.search(
        r"\b(?:acted|starred)\s+"
        r"together\s+with\b",
        q,
    ):
        counts[
            "starred_actors"
        ] = max(
            counts[
                "starred_actors"
            ],
            2,
        )

    # ========================================================
    # TARGET RELATION ALWAYS REQUIRED
    # ========================================================

    if target_relation is not None:

        counts[
            target_relation
        ] = max(
            counts[
                target_relation
            ],
            1,
        )

    # lexical mention cannot exceed hop count
    for relation in list(
        counts.keys()
    ):

        counts[
            relation
        ] = min(
            counts[
                relation
            ],
            hop,
        )

    return Counter(
        {
            relation: count
            for relation, count
            in counts.items()
            if count > 0
        }
    )


# ============================================================
# 11. TARGET RELATION AUDIT
# ============================================================

def audit_target_relation_coverage():
    rows = []

    print()
    print(
        "Auditing target-relation inference..."
    )

    for hop in (
        1,
        2,
        3,
    ):

        questions = list(
            load_questions(
                hop
            )
        )

        missing = []

        counts = Counter()

        for item in questions:

            relation = (
                infer_target_relation(
                    item["question"]
                )
            )

            if relation is None:

                missing.append(
                    item
                )

                rows.append(
                    {
                        "qid":
                            item[
                                "qid"
                            ],

                        "hop":
                            hop,

                        "question":
                            item[
                                "question"
                            ],

                        "status":
                            "target_relation_not_inferred",
                    }
                )

            else:

                counts[
                    relation
                ] += 1

        inferred = (
            len(questions)
            - len(missing)
        )

        coverage = (
            100.0
            * inferred
            / len(questions)
        )

        print(
            f"{hop}-hop target inference: "
            f"{inferred}/{len(questions)} "
            f"({coverage:.2f}%)"
        )

        print(
            "  Relations:",
            dict(
                counts
            ),
        )

        if missing:

            print(
                "  First unmatched examples:"
            )

            for item in missing[:5]:

                print(
                    "   -",
                    item[
                        "question"
                    ],
                )

    save_jsonl(
        rows,
        TARGET_AUDIT_PATH,
    )

    print(
        "Target audit saved:",
        TARGET_AUDIT_PATH,
    )


# ============================================================
# 12. EXACT-HOP WALK SEARCH
# ============================================================

def find_exact_hop_paths(
    adjacency,
    start_entity,
    target_entity,
    hops,
    relation_profile,
    max_paths=MAX_PATHS,
):
    """
    Exact-hop WALK.

    Node repetition allowed.

    Candidate path yalnız exact semantic relation profile
    istifadə edə bilər.
    """

    results = []

    path = []

    relation_counter = Counter()

    allowed_relations = set(
        relation_profile.keys()
    )

    def dfs(
        current_entity,
        depth,
    ):
        if len(results) >= max_paths:
            return

        if depth == hops:

            if (
                current_entity
                == target_entity
                and relation_counter
                == relation_profile
            ):
                results.append(
                    list(
                        path
                    )
                )

            return

        remaining_steps = (
            hops
            - depth
        )

        required_remaining = sum(
            max(
                0,
                relation_profile[relation]
                - relation_counter[
                    relation
                ],
            )
            for relation
            in relation_profile
        )

        if (
            required_remaining
            > remaining_steps
        ):
            return

        for edge in adjacency.get(
            current_entity,
            [],
        ):

            relation = (
                edge[
                    "relation"
                ]
            )

            if (
                relation
                not in allowed_relations
            ):
                continue

            if (
                relation_counter[
                    relation
                ]
                >= relation_profile[
                    relation
                ]
            ):
                continue

            step = {
                "from_entity":
                    current_entity,

                "to_entity":
                    edge[
                        "to_entity"
                    ],

                "relation":
                    relation,

                "reverse":
                    edge[
                        "reverse"
                    ],

                "triple":
                    edge[
                        "triple"
                    ],
            }

            path.append(
                step
            )

            relation_counter[
                relation
            ] += 1

            dfs(
                edge[
                    "to_entity"
                ],
                depth + 1,
            )

            relation_counter[
                relation
            ] -= 1

            if (
                relation_counter[
                    relation
                ]
                == 0
            ):
                del relation_counter[
                    relation
                ]

            path.pop()

    dfs(
        start_entity,
        0,
    )

    return results


# ============================================================
# 13. PATH VALIDATION
# ============================================================

def validate_path_semantics(
    path,
    hop,
    target_relation,
    relation_profile,
):
    if len(path) != hop:
        return False

    if not path:
        return False

    if (
        path[-1][
            "relation"
        ]
        != target_relation
    ):
        return False

    actual_profile = Counter(
        step[
            "relation"
        ]
        for step in path
    )

    if (
        actual_profile
        != relation_profile
    ):
        return False

    return True


# ============================================================
# 14. DETERMINISTIC PATH KEY
# ============================================================

def path_sort_key(path):
    return tuple(
        (
            step[
                "from_entity"
            ],
            step[
                "relation"
            ],
            step[
                "to_entity"
            ],
            step[
                "reverse"
            ],
        )
        for step in path
    )


# ============================================================
# 15. FIND CANONICAL SUPPORT PATH
# ============================================================

def find_support_paths(
    adjacency,
    question,
    topic_entity,
    gold_answer,
    hop,
):
    target_relation = (
        infer_target_relation(
            question
        )
    )

    if target_relation is None:
        return []

    relation_profile = (
        infer_required_relation_counts(
            question=
                question,

            hop=
                hop,

            target_relation=
                target_relation,
        )
    )

    # Strict semantic profile must explain every hop.
    if (
        sum(
            relation_profile.values()
        )
        != hop
    ):
        return []

    candidate_paths = (
        find_exact_hop_paths(
            adjacency=
                adjacency,

            start_entity=
                topic_entity,

            target_entity=
                gold_answer,

            hops=
                hop,

            relation_profile=
                relation_profile,

            max_paths=
                MAX_PATHS,
        )
    )

    valid_paths = [
        path
        for path
        in candidate_paths
        if validate_path_semantics(
            path=
                path,

            hop=
                hop,

            target_relation=
                target_relation,

            relation_profile=
                relation_profile,
        )
    ]

    if not valid_paths:
        return []

    valid_paths = sorted(
        valid_paths,
        key=
            path_sort_key,
    )

    return [
        valid_paths[0]
    ]


# ============================================================
# 16. POSITION LABEL
# ============================================================

def get_position_label(
    step_index,
    total_hops,
):
    if (
        step_index
        == total_hops - 1
    ):
        return "answer-adjacent"

    if step_index == 0:
        return "early"

    return "middle"


# ============================================================
# 17. SERIALIZE PATH
# ============================================================

def serialize_path(path):
    output = []

    total_hops = len(
        path
    )

    for i, step in enumerate(
        path
    ):

        output.append(
            {
                "hop_index":
                    i + 1,

                "position":
                    get_position_label(
                        i,
                        total_hops,
                    ),

                "from_entity":
                    step[
                        "from_entity"
                    ],

                "to_entity":
                    step[
                        "to_entity"
                    ],

                "relation":
                    step[
                        "relation"
                    ],

                "reverse":
                    step[
                        "reverse"
                    ],

                "triple":
                    list(
                        step[
                            "triple"
                        ]
                    ),
            }
        )

    return output


# ============================================================
# 18. COLLECT SUPPORT TRIPLES
# ============================================================

def collect_support_triples(
    support_paths,
):
    output = []

    seen = set()

    for answer in sorted(
        support_paths.keys()
    ):

        paths = (
            support_paths[
                answer
            ]
        )

        if not paths:
            continue

        for step in paths[0]:

            triple = tuple(
                step[
                    "triple"
                ]
            )

            if triple in seen:
                continue

            seen.add(
                triple
            )

            output.append(
                triple
            )

    return output


# ============================================================
# 19. COLLECT PATH ENTITIES
# ============================================================

def collect_path_entities(
    topic_entity,
    gold_answers,
    support_paths,
):
    entities = {
        topic_entity
    }

    gold_set = set(
        gold_answers
    )

    for paths in (
        support_paths.values()
    ):

        if not paths:
            continue

        for step in paths[0]:

            for entity in (
                step[
                    "from_entity"
                ],
                step[
                    "to_entity"
                ],
            ):

                if (
                    entity
                    not in gold_set
                ):
                    entities.add(
                        entity
                    )

    return entities


# ============================================================
# 20. COLLECT ANSWER-SLOT CONSTRAINTS
# ============================================================

def collect_answer_slot_constraints(
    support_paths,
):
    """
    Clean context-də competing answer fact-ləri bloklamaq üçün.

    Hər canonical support path-in final traversal step-i:

        source entity
        relation
        direction

    saxlanılır.

    Example forward:

        Movie -> directed_by -> Director

    source=Movie
    relation=directed_by
    reverse=False

    Example reverse:

        Actor -> starred_actors^-1 -> Movie

    source=Actor
    relation=starred_actors
    reverse=True
    """

    constraints = []

    seen = set()

    for paths in (
        support_paths.values()
    ):

        if not paths:
            continue

        path = paths[0]

        if not path:
            continue

        final_step = path[-1]

        constraint = (
            final_step[
                "from_entity"
            ],
            final_step[
                "relation"
            ],
            final_step[
                "reverse"
            ],
        )

        if constraint in seen:
            continue

        seen.add(
            constraint
        )

        constraints.append(
            constraint
        )

    return constraints


# ============================================================
# 21. VALID CLEAN CONTEXT TRIPLE
# ============================================================

def valid_context_triple(
    triple,
    support_set,
    gold_set,
    answer_slot_constraints,
):
    """
    Clean context safety rules.

    Reject:
    1. support triple itself
    2. triples touching gold answer entities
    3. any triple that would introduce a competing answer
       for the SAME final reasoning slot

    Example:

    Question:
        who directed Captain America?

    Support:
        Captain America | directed_by | Rod Holcomb

    Context candidate:
        Captain America | directed_by | Albert Pyun

    This is rejected because it creates clean-condition
    answer ambiguity.
    """

    if triple in support_set:
        return False

    head, relation, tail = triple

    if head in gold_set:
        return False

    if tail in gold_set:
        return False

    # ========================================================
    # COMPETING ANSWER PROTECTION
    # ========================================================

    for (
        source_entity,
        target_relation,
        reverse,
    ) in answer_slot_constraints:

        if (
            relation
            != target_relation
        ):
            continue

        # Forward traversal:
        #
        # source --relation--> answer
        if (
            not reverse
            and head
            == source_entity
        ):
            return False

        # Reverse traversal:
        #
        # source <--relation-- answer
        #
        # Original triple:
        # answer | relation | source
        if (
            reverse
            and tail
            == source_entity
        ):
            return False

    return True


# ============================================================
# 22. DETERMINISTIC CONTEXT ORDER
# ============================================================

def deterministic_triple_order(
    triples,
    qid,
    label,
    seed=DEFAULT_SEED,
):
    return sorted(
        triples,
        key=lambda triple:
            stable_seed(
                (
                    f"{qid}:"
                    f"{label}:"
                    f"{triple[0]}|"
                    f"{triple[1]}|"
                    f"{triple[2]}"
                ),
                seed,
            ),
    )


# ============================================================
# 23. BUILD FIXED CONTEXT
# ============================================================

def build_context_triples(
    item,
    triples,
    adjacency,
    support_paths,
    support_triples,
    target_size=CONTEXT_SIZE,
    seed=DEFAULT_SEED,
):
    qid = item[
        "qid"
    ]

    gold_set = set(
        item[
            "gold_answers"
        ]
    )

    support_set = set(
        support_triples
    )

    path_entities = (
        collect_path_entities(
            topic_entity=
                item[
                    "topic_entity"
                ],

            gold_answers=
                item[
                    "gold_answers"
                ],

            support_paths=
                support_paths,
        )
    )

    answer_slot_constraints = (
        collect_answer_slot_constraints(
            support_paths
        )
    )

    selected = []

    selected_set = set()

    # ========================================================
    # HELPER
    # ========================================================

    def add_candidates(
        candidates,
        label,
    ):
        ordered = (
            deterministic_triple_order(
                triples=
                    candidates,

                qid=
                    qid,

                label=
                    label,

                seed=
                    seed,
            )
        )

        for triple in ordered:

            triple = tuple(
                triple
            )

            if (
                triple
                in selected_set
            ):
                continue

            if not valid_context_triple(
                triple=
                    triple,

                support_set=
                    support_set,

                gold_set=
                    gold_set,

                answer_slot_constraints=
                    answer_slot_constraints,
            ):
                continue

            selected.append(
                triple
            )

            selected_set.add(
                triple
            )

            if (
                len(selected)
                >= target_size
            ):
                return True

        return False

    # ========================================================
    # LEVEL 1: PATH-LOCAL NEIGHBORHOOD
    # ========================================================

    level1 = set()

    neighbor_entities = set()

    for entity in path_entities:

        for edge in adjacency.get(
            entity,
            [],
        ):

            level1.add(
                tuple(
                    edge[
                        "triple"
                    ]
                )
            )

            neighbor_entities.add(
                edge[
                    "to_entity"
                ]
            )

    if add_candidates(
        level1,
        "level1",
    ):
        return selected

    # ========================================================
    # LEVEL 2: TWO-HOP NEIGHBORHOOD
    # ========================================================

    level2 = set()

    for entity in neighbor_entities:

        for edge in adjacency.get(
            entity,
            [],
        ):

            level2.add(
                tuple(
                    edge[
                        "triple"
                    ]
                )
            )

    if add_candidates(
        level2,
        "level2",
    ):
        return selected

    # ========================================================
    # LEVEL 3: GLOBAL FALLBACK
    # ========================================================

    if triples:

        start_index = (
            stable_seed(
                (
                    f"{qid}:"
                    "global-context-start"
                ),
                seed,
            )
            % len(triples)
        )

        for offset in range(
            len(triples)
        ):

            index = (
                start_index
                + offset
            ) % len(triples)

            triple = tuple(
                triples[
                    index
                ]
            )

            if (
                triple
                in selected_set
            ):
                continue

            if not valid_context_triple(
                triple=
                    triple,

                support_set=
                    support_set,

                gold_set=
                    gold_set,

                answer_slot_constraints=
                    answer_slot_constraints,
            ):
                continue

            selected.append(
                triple
            )

            selected_set.add(
                triple
            )

            if (
                len(selected)
                >= target_size
            ):
                break

    return selected


# ============================================================
# 24. BUILD ONE CLEAN ITEM
# ============================================================

def build_clean_evidence(
    question_item,
    triples,
    adjacency,
    context_size=CONTEXT_SIZE,
    seed=DEFAULT_SEED,
):
    question = (
        question_item[
            "question"
        ]
    )

    hop = (
        question_item[
            "hop"
        ]
    )

    topic_entity = (
        question_item[
            "topic_entity"
        ]
    )

    gold_answers = (
        question_item[
            "gold_answers"
        ]
    )

    target_relation = (
        infer_target_relation(
            question
        )
    )

    if target_relation is None:

        return (
            None,
            "target_relation_not_inferred",
        )

    relation_profile = (
        infer_required_relation_counts(
            question=
                question,

            hop=
                hop,

            target_relation=
                target_relation,
        )
    )

    # ========================================================
    # STRICT:
    # question semantics must explain every hop
    # ========================================================

    if (
        sum(
            relation_profile.values()
        )
        != hop
    ):

        return (
            None,
            "relation_profile_incomplete",
        )

    # ========================================================
    # SUPPORT PATH PER GOLD ANSWER
    # ========================================================

    support_paths_internal = {}

    for gold_answer in (
        gold_answers
    ):

        paths = find_support_paths(
            adjacency=
                adjacency,

            question=
                question,

            topic_entity=
                topic_entity,

            gold_answer=
                gold_answer,

            hop=
                hop,
        )

        if not paths:

            return (
                None,
                "no_semantic_support_path",
            )

        support_paths_internal[
            gold_answer
        ] = paths

    # ========================================================
    # SUPPORT TRIPLES
    # ========================================================

    support_triples = (
        collect_support_triples(
            support_paths_internal
        )
    )

    if not support_triples:

        return (
            None,
            "empty_support",
        )

    # ========================================================
    # SERIALIZE PATHS
    # ========================================================

    support_paths = {}

    for answer, paths in (
        support_paths_internal.items()
    ):

        support_paths[
            answer
        ] = [
            serialize_path(
                path
            )
            for path in paths
        ]

    # ========================================================
    # CLEAN CONTEXT
    # ========================================================

    context_triples = (
        build_context_triples(
            item=
                question_item,

            triples=
                triples,

            adjacency=
                adjacency,

            support_paths=
                support_paths_internal,

            support_triples=
                support_triples,

            target_size=
                context_size,

            seed=
                seed,
        )
    )

    if (
        len(
            context_triples
        )
        != context_size
    ):

        return (
            None,
            "insufficient_context",
        )

    # ========================================================
    # EVIDENCE PACKET
    # ========================================================

    evidence = []

    for triple in (
        support_triples
    ):

        evidence.append(
            {
                "triple":
                    list(
                        triple
                    ),

                "role":
                    "support",
            }
        )

    for triple in (
        context_triples
    ):

        evidence.append(
            {
                "triple":
                    list(
                        triple
                    ),

                "role":
                    "context",
            }
        )

    # Deterministic evidence order
    rng = random.Random(
        stable_seed(
            (
                question_item[
                    "qid"
                ]
                + ":evidence-order"
            ),
            seed,
        )
    )

    rng.shuffle(
        evidence
    )

    # ========================================================
    # RELATION SIGNATURE METADATA
    # ========================================================

    signatures = {}

    for answer, paths in (
        support_paths.items()
    ):

        signatures[
            answer
        ] = [
            step[
                "relation"
            ]
            for step
            in paths[0]
        ]

    output = {
        "qid":
            question_item[
                "qid"
            ],

        "hop":
            hop,

        "question":
            question,

        "question_raw":
            question_item.get(
                "question_raw",
                question,
            ),

        "topic_entity":
            topic_entity,

        "gold_answers":
            gold_answers,

        "target_relation":
            target_relation,

        "required_relation_counts":
            dict(
                relation_profile
            ),

        "relation_profile_exact":
            True,

        "selected_relation_signatures":
            signatures,

        "question_relation_scores":
            get_question_relation_scores(
                question
            ),

        "support_paths":
            support_paths,

        "support_triples":
            [
                list(
                    triple
                )
                for triple
                in support_triples
            ],

        "context_triples":
            [
                list(
                    triple
                )
                for triple
                in context_triples
            ],

        "evidence":
            evidence,

        "context_size":
            context_size,

        "seed":
            seed,
    }

    return (
        output,
        "ok",
    )


# ============================================================
# 25. VALIDATE CLEAN ITEM
# ============================================================

def validate_clean_item(item):
    hop = item[
        "hop"
    ]

    target_relation = (
        item[
            "target_relation"
        ]
    )

    required_profile = Counter(
        item[
            "required_relation_counts"
        ]
    )

    if (
        sum(
            required_profile.values()
        )
        != hop
    ):

        return (
            False,
            "incomplete_profile",
        )

    # ========================================================
    # SUPPORT PATH VALIDATION
    # ========================================================

    for answer, paths in (
        item[
            "support_paths"
        ].items()
    ):

        if not paths:

            return (
                False,
                "missing_path",
            )

        path = paths[0]

        if len(path) != hop:

            return (
                False,
                "wrong_hop_count",
            )

        if (
            path[-1][
                "to_entity"
            ]
            != answer
        ):

            return (
                False,
                "wrong_final_entity",
            )

        if (
            path[-1][
                "relation"
            ]
            != target_relation
        ):

            return (
                False,
                "wrong_final_relation",
            )

        actual_profile = Counter(
            step[
                "relation"
            ]
            for step in path
        )

        if (
            actual_profile
            != required_profile
        ):

            return (
                False,
                "relation_profile_mismatch",
            )

    # ========================================================
    # CONTEXT SIZE
    # ========================================================

    if (
        len(
            item[
                "context_triples"
            ]
        )
        != CONTEXT_SIZE
    ):

        return (
            False,
            "wrong_context_size",
        )

    # ========================================================
    # CLEAN ANSWER-CONFLICT CHECK
    # ========================================================

    support_paths = (
        item[
            "support_paths"
        ]
    )

    constraints = []

    for paths in (
        support_paths.values()
    ):

        if not paths:
            continue

        final_step = (
            paths[0][-1]
        )

        constraints.append(
            (
                final_step[
                    "from_entity"
                ],
                final_step[
                    "relation"
                ],
                final_step[
                    "reverse"
                ],
            )
        )

    for context_triple in (
        item[
            "context_triples"
        ]
    ):

        head, relation, tail = (
            context_triple
        )

        for (
            source_entity,
            target_relation,
            reverse,
        ) in constraints:

            if (
                relation
                != target_relation
            ):
                continue

            if (
                not reverse
                and head
                == source_entity
            ):

                return (
                    False,
                    "competing_context_answer",
                )

            if (
                reverse
                and tail
                == source_entity
            ):

                return (
                    False,
                    "competing_context_answer",
                )

    return (
        True,
        "ok",
    )


# ============================================================
# 26. BUILD PILOT DATASET
# ============================================================

def build_pilot_dataset(
    triples,
    adjacency,
    target_per_hop=TARGET_PER_HOP,
    seed=DEFAULT_SEED,
):
    """
    Exactly:
        100 x 1-hop
        100 x 2-hop
        100 x 3-hop

    Only semantically valid questions accepted.
    """

    pilot = []

    all_rejections = []

    for hop in (
        1,
        2,
        3,
    ):

        questions = list(
            load_questions(
                hop
            )
        )

        rng = random.Random(
            stable_seed(
                f"pilot-hop-{hop}",
                seed,
            )
        )

        rng.shuffle(
            questions
        )

        accepted = []

        rejection_counts = Counter()

        for item in questions:

            clean_item, reason = (
                build_clean_evidence(
                    question_item=
                        item,

                    triples=
                        triples,

                    adjacency=
                        adjacency,

                    context_size=
                        CONTEXT_SIZE,

                    seed=
                        seed,
                )
            )

            if clean_item is None:

                rejection_counts[
                    reason
                ] += 1

                all_rejections.append(
                    {
                        "qid":
                            item[
                                "qid"
                            ],

                        "hop":
                            hop,

                        "question":
                            item[
                                "question"
                            ],

                        "reason":
                            reason,
                    }
                )

                continue

            (
                valid,
                validation_reason,
            ) = validate_clean_item(
                clean_item
            )

            if not valid:

                reason = (
                    "validation:"
                    + validation_reason
                )

                rejection_counts[
                    reason
                ] += 1

                all_rejections.append(
                    {
                        "qid":
                            item[
                                "qid"
                            ],

                        "hop":
                            hop,

                        "question":
                            item[
                                "question"
                            ],

                        "reason":
                            reason,
                    }
                )

                continue

            accepted.append(
                clean_item
            )

            if (
                len(accepted)
                >= target_per_hop
            ):
                break

        pilot.extend(
            accepted
        )

        print(
            f"{hop}-hop clean items: "
            f"{len(accepted)}/"
            f"{target_per_hop}"
        )

        print(
            "  Rejections:",
            dict(
                rejection_counts
            ),
        )

    return (
        pilot,
        all_rejections,
    )


# ============================================================
# 27. PRINT EXAMPLE
# ============================================================

def print_example(item):
    print()
    print("=" * 80)
    print("SEMANTIC CLEAN EXAMPLE")
    print("=" * 80)

    print(
        "QID:",
        item[
            "qid"
        ],
    )

    print(
        "Hop:",
        item[
            "hop"
        ],
    )

    print(
        "Question:",
        item[
            "question"
        ],
    )

    print(
        "Topic:",
        item[
            "topic_entity"
        ],
    )

    print(
        "Gold:",
        item[
            "gold_answers"
        ],
    )

    print(
        "Target relation:",
        item[
            "target_relation"
        ],
    )

    print(
        "Required relation profile:",
        item[
            "required_relation_counts"
        ],
    )

    print()

    for answer, paths in (
        item[
            "support_paths"
        ].items()
    ):

        print(
            "Answer:",
            answer,
        )

        print(
            "Signature:",
            [
                step[
                    "relation"
                ]
                for step
                in paths[0]
            ],
        )

        for step in paths[0]:

            reverse_marker = (
                "^-1"
                if step[
                    "reverse"
                ]
                else ""
            )

            print(
                f"  "
                f"{step['from_entity']} "
                f"--["
                f"{step['relation']}"
                f"{reverse_marker}"
                f"]--> "
                f"{step['to_entity']}"
            )

        print()

    print(
        "EVIDENCE:"
    )

    for entry in (
        item[
            "evidence"
        ]
    ):

        triple = (
            entry[
                "triple"
            ]
        )

        print(
            f"{entry['role'].upper():8} | "
            f"{triple[0]} | "
            f"{triple[1]} | "
            f"{triple[2]}"
        )


# ============================================================
# 28. MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # TARGET RELATION COVERAGE
    # ========================================================

    audit_target_relation_coverage()

    # ========================================================
    # LOAD KG
    # ========================================================

    print()
    print(
        "Loading KG..."
    )

    raw_triples = (
        load_kb()
    )

    print(
        "Raw triple example:",
        raw_triples[0],
    )

    triples = (
        normalize_kb_triples(
            raw_triples
        )
    )

    print(
        "Triples:",
        len(
            triples
        ),
    )

    print(
        "Normalized triple example:",
        triples[0],
    )

    # ========================================================
    # ADJACENCY
    # ========================================================

    print(
        "Building adjacency..."
    )

    adjacency = (
        build_adjacency(
            triples
        )
    )

    print(
        "Entities:",
        len(
            adjacency
        ),
    )

    # Catch parsing bugs immediately
    if len(adjacency) < 1000:

        raise RuntimeError(
            "Adjacency entity count is unexpectedly small. "
            "KG parsing/normalization is broken."
        )

    # ========================================================
    # BUILD PILOT
    # ========================================================

    print()
    print(
        "Building strict semantic-clean pilot..."
    )
    print()

    pilot, rejections = (
        build_pilot_dataset(
            triples=
                triples,

            adjacency=
                adjacency,

            target_per_hop=
                TARGET_PER_HOP,

            seed=
                DEFAULT_SEED,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_jsonl(
        pilot,
        OUTPUT_PATH,
    )

    save_jsonl(
        rejections,
        REJECTION_PATH,
    )

    print()

    print(
        "Total pilot items:",
        len(
            pilot
        ),
    )

    print(
        "Saved clean pilot:",
        OUTPUT_PATH,
    )

    print(
        "Saved rejection log:",
        REJECTION_PATH,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    hop_counts = {
        1: 0,
        2: 0,
        3: 0,
    }

    semantic_violations = []

    for item in pilot:

        hop_counts[
            item[
                "hop"
            ]
        ] += 1

        (
            valid,
            reason,
        ) = validate_clean_item(
            item
        )

        if not valid:

            semantic_violations.append(
                {
                    "qid":
                        item[
                            "qid"
                        ],

                    "reason":
                        reason,
                }
            )

    if pilot:

        avg_support = (
            sum(
                len(
                    item[
                        "support_triples"
                    ]
                )
                for item
                in pilot
            )
            / len(pilot)
        )

        avg_context = (
            sum(
                len(
                    item[
                        "context_triples"
                    ]
                )
                for item
                in pilot
            )
            / len(pilot)
        )

    else:

        avg_support = 0.0
        avg_context = 0.0

    context_shortages = sum(
        1
        for item
        in pilot
        if (
            len(
                item[
                    "context_triples"
                ]
            )
            != CONTEXT_SIZE
        )
    )

    competing_context_violations = sum(
        1
        for violation
        in semantic_violations
        if (
            violation[
                "reason"
            ]
            == "competing_context_answer"
        )
    )

    print()

    print(
        "Hop counts:",
        hop_counts,
    )

    print(
        f"Average support triples: "
        f"{avg_support:.2f}"
    )

    print(
        f"Average context triples: "
        f"{avg_context:.2f}"
    )

    print(
        "Semantic violations:",
        len(
            semantic_violations
        ),
    )

    print(
        "Competing-context violations:",
        competing_context_violations,
    )

    print(
        "Context shortages:",
        context_shortages,
    )

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    pilot_ok = (
        len(pilot) == 300
        and hop_counts
        == {
            1: 100,
            2: 100,
            3: 100,
        }
    )

    semantic_ok = (
        len(
            semantic_violations
        )
        == 0
    )

    context_ok = (
        context_shortages
        == 0
    )

    print()

    print(
        "Pilot size check:",
        (
            "PASS"
            if pilot_ok
            else "FAIL"
        ),
    )

    print(
        "Semantic constraint check:",
        (
            "PASS"
            if semantic_ok
            else "FAIL"
        ),
    )

    print(
        "Clean answer-conflict check:",
        (
            "PASS"
            if (
                competing_context_violations
                == 0
            )
            else "FAIL"
        ),
    )

    print(
        "Fixed context check:",
        (
            "PASS"
            if context_ok
            else "FAIL"
        ),
    )

    # ========================================================
    # EXAMPLE PER HOP
    # ========================================================

    for hop in (
        1,
        2,
        3,
    ):

        example = next(
            (
                item
                for item
                in pilot
                if (
                    item[
                        "hop"
                    ]
                    == hop
                )
            ),
            None,
        )

        if (
            example
            is not None
        ):

            print_example(
                example
            )