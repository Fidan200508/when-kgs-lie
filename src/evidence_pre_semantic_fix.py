from collections import defaultdict, deque
from pathlib import Path
import hashlib
import json
import random
import re

from data import load_kb, load_questions


# ============================================================
# CONFIG
# ============================================================

PROCESSED_DIR = Path("data/processed")

DEFAULT_SEED = 42
DEFAULT_CONTEXT_SIZE = 5


# ============================================================
# 1. BUILD BIDIRECTIONAL KG INDEX
# ============================================================

def build_adjacency(triples):
    """
    MetaQA KG-ni bidirectional traversal index-ə çevirir.

    Original triple:
        head | relation | tail

    Traversal zamanı:
        head -> tail : reverse=False
        tail -> head : reverse=True

    Vacib:
    Original KG triple həmişə öz istiqamətində saxlanılır.
    """

    adjacency = defaultdict(list)

    for triple in triples:

        head = triple["head"]
        relation = triple["relation"]
        tail = triple["tail"]

        original_triple = (
            head,
            relation,
            tail,
        )

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


# ============================================================
# 2. EXACT-HOP WALK SEARCH
# ============================================================

def find_exact_hop_paths(
    adjacency,
    source,
    target,
    num_hops,
    max_paths=100,
    allowed_relations=None,
):
    """
    source-dan target-a EXACT num_hops uzunluğunda WALK tapır.

    Əvvəlki versiyada visited-node prevention var idi.
    Bu bəzi MetaQA 3-hop reasoning chain-lərini bloklayırdı.

    İndi node repetition mümkündür:

        A -> B -> A -> C

    kimi walk valid ola bilər.

    num_hops maksimum 3 olduğu üçün search manageable-dir.
    """

    if source is None or target is None:
        return []

    paths_found = []

    # queue item:
    # (
    #     current_entity,
    #     path_so_far
    # )

    queue = deque(
        [
            (
                source,
                [],
            )
        ]
    )

    while queue:

        current_entity, current_path = queue.popleft()

        # Exact hop limit-ə çatmışıq
        if len(current_path) == num_hops:

            if current_entity == target:

                paths_found.append(
                    current_path
                )

                if len(paths_found) >= max_paths:
                    return paths_found

            continue

        # Current entity-dən bütün edges
        for edge in adjacency.get(
            current_entity,
            [],
        ):

            relation = edge["relation"]

            # Semantic-restricted search verilmişdirsə
            if (
                allowed_relations is not None
                and relation not in allowed_relations
            ):
                continue

            next_entity = edge["next"]

            step = {
                "from_entity": current_entity,
                "to_entity": next_entity,
                "relation": relation,
                "reverse": edge["reverse"],
                "triple": edge["triple"],
            }

            queue.append(
                (
                    next_entity,
                    current_path + [step],
                )
            )

    return paths_found


# ============================================================
# 3. QUESTION -> RELATION SEMANTICS
# ============================================================

def get_question_relation_scores(question):
    """
    Question text-də hansı MetaQA relation-larının
    semantik olaraq relevant olduğunu score edir.

    MetaQA yalnız 9 relation istifadə edir.
    Ona görə lightweight deterministic mapping kifayətdir.
    """

    q = question.lower()

    scores = {
        "directed_by": 0,
        "has_genre": 0,
        "has_imdb_rating": 0,
        "has_imdb_votes": 0,
        "has_tags": 0,
        "in_language": 0,
        "release_year": 0,
        "starred_actors": 0,
        "written_by": 0,
    }

    keyword_map = {

        "directed_by": [
            "directed",
            "director",
            "directors",
        ],

        "written_by": [
            "wrote",
            "written",
            "writer",
            "writers",
            "screenwriter",
            "screenwriters",
        ],

        "starred_actors": [
            "starred",
            "stars in",
            "star in",
            "appear in",
            "appears in",
            "appeared in",
            "actor",
            "actors",
            "actress",
            "actresses",
            "cast",
        ],

        "in_language": [
            "language",
            "languages",
        ],

        "has_genre": [
            "genre",
            "genres",
        ],

        "release_year": [
            "release year",
            "released",
            "release",
            "year",
        ],

        "has_imdb_rating": [
            "imdb rating",
            "rating",
            "ratings",
        ],

        "has_imdb_votes": [
            "imdb votes",
            "votes",
            "vote",
        ],

        "has_tags": [
            "tags",
            "tag",
        ],
    }

    for relation, keywords in keyword_map.items():

        for keyword in keywords:

            if keyword in q:
                scores[relation] += 1

    return scores


# ============================================================
# 4. INFER FINAL/TARGET RELATION
# ============================================================

def infer_target_relation(question):
    """
    Question-un cavabına aparan FINAL relation-u
    mümkün olduğu qədər infer edir.

    Bu support-path ranking-i gücləndirir.

    Əgər confidence kifayət deyilsə None qaytarır.
    """

    q = question.lower().strip()

    # -------------------------------------
    # Language
    # -------------------------------------

    if (
        "which language" in q
        or "which languages" in q
        or "what language" in q
        or "what languages" in q
        or "in which language" in q
        or "in which languages" in q
    ):
        return "in_language"

    # -------------------------------------
    # Genre
    # -------------------------------------

    if (
        "which genre" in q
        or "which genres" in q
        or "what genre" in q
        or "what genres" in q
    ):
        return "has_genre"

    # -------------------------------------
    # Rating
    # -------------------------------------

    if "rating" in q:
        return "has_imdb_rating"

    # -------------------------------------
    # Votes
    # -------------------------------------

    if "votes" in q or "how many votes" in q:
        return "has_imdb_votes"

    # -------------------------------------
    # Tags
    # -------------------------------------

    if (
        "which tags" in q
        or "what tags" in q
    ):
        return "has_tags"

    # -------------------------------------
    # Release year
    # -------------------------------------

    if (
        "what year" in q
        or "which year" in q
        or "release year" in q
        or "when was" in q
    ):
        return "release_year"

    # -------------------------------------
    # Written by
    # -------------------------------------

    written_patterns = [
        r"^who wrote\b",
        r"^which person wrote\b",
        r"^who are the screenwriters\b",
        r"^who were the screenwriters\b",
        r"^which screenwriters\b",
        r"^which writers\b",
        r"^who are the writers\b",
        r"^who is the writer\b",
        r"^what person wrote\b",
    ]

    for pattern in written_patterns:

        if re.search(pattern, q):
            return "written_by"

    # -------------------------------------
    # Directed by
    # -------------------------------------

    directed_patterns = [
        r"^who directed\b",
        r"^which person directed\b",
        r"^who are the directors\b",
        r"^who were the directors\b",
        r"^which directors\b",
        r"^who is the director\b",
        r"^what person directed\b",
    ]

    for pattern in directed_patterns:

        if re.search(pattern, q):
            return "directed_by"

    # -------------------------------------
    # Starred actors / appear in
    # -------------------------------------

    starred_patterns = [
        r"^who starred\b",
        r"^which actors\b",
        r"^who are the actors\b",
        r"^who were the actors\b",
        r"^what films does .+ appear in\b",
        r"^what movies does .+ appear in\b",
        r"^which films does .+ appear in\b",
        r"^which movies does .+ appear in\b",
        r"^what does .+ appear in\b",
    ]

    for pattern in starred_patterns:

        if re.search(pattern, q):
            return "starred_actors"

    return None


# ============================================================
# 5. SCORE CANDIDATE PATH SEMANTICALLY
# ============================================================

def score_path_semantically(
    path,
    relation_scores,
    target_relation=None,
):
    """
    Candidate support path üçün semantic score.

    Relevant relation-lar score artırır.

    Əgər final edge question-un target relation-u ilə
    uyğun gəlirsə böyük bonus alır.
    """

    score = 0.0

    for step in path:

        relation = step["relation"]

        score += relation_scores.get(
            relation,
            0,
        )

    # Final relation xüsusi əhəmiyyətlidir
    if (
        target_relation is not None
        and path
        and path[-1]["relation"] == target_relation
    ):
        score += 5.0

    return score


def path_signature(path):
    """
    Deterministic tie-break üçün path representation.
    """

    return tuple(
        (
            step["relation"],
            step["reverse"],
            step["from_entity"],
            step["to_entity"],
        )
        for step in path
    )


# ============================================================
# 6. FIND SEMANTICALLY-ALIGNED SUPPORT PATHS
# ============================================================

def find_support_paths(
    item,
    adjacency,
    max_paths_per_answer=1,
):
    """
    Hər gold answer üçün support path tapır.

    Proses:

        Question
           ↓
    relevant relations
           ↓
    exact-hop candidate paths
           ↓
    semantic scoring
           ↓
    best path

    Beləliklə sadəcə "ilk tapılan path" seçilmir.
    """

    support_paths = {}

    relation_scores = get_question_relation_scores(
        item["question"]
    )

    target_relation = infer_target_relation(
        item["question"]
    )

    relevant_relations = {
        relation
        for relation, score
        in relation_scores.items()
        if score > 0
    }

    for answer in item["gold_answers"]:

        # ====================================================
        # FIRST TRY:
        # only semantically relevant relations
        # ====================================================

        candidate_paths = []

        if relevant_relations:

            candidate_paths = find_exact_hop_paths(
                adjacency=adjacency,
                source=item["topic_entity"],
                target=answer,
                num_hops=item["hop"],
                max_paths=100,
                allowed_relations=relevant_relations,
            )

        # ====================================================
        # FALLBACK:
        # unrestricted graph walk
        # ====================================================

        if not candidate_paths:

            candidate_paths = find_exact_hop_paths(
                adjacency=adjacency,
                source=item["topic_entity"],
                target=answer,
                num_hops=item["hop"],
                max_paths=100,
                allowed_relations=None,
            )

        if not candidate_paths:

            support_paths[answer] = []
            continue

        # ====================================================
        # Rank candidate paths
        # ====================================================

        ranked_paths = sorted(
            candidate_paths,
            key=lambda path: (
                -score_path_semantically(
                    path,
                    relation_scores,
                    target_relation,
                ),
                path_signature(path),
            ),
        )

        support_paths[answer] = ranked_paths[
            :max_paths_per_answer
        ]

    return support_paths


# ============================================================
# 7. CORRUPTED POSITION LABEL
# ============================================================

def get_position_label(
    step_index,
    total_hops,
):
    """
    Hop position metadata.

    1-hop:
        answer-adjacent

    2-hop:
        early
        answer-adjacent

    3-hop:
        early
        middle
        answer-adjacent
    """

    if step_index == total_hops - 1:
        return "answer-adjacent"

    if step_index == 0:
        return "early"

    return "middle"


# ============================================================
# 8. COLLECT SUPPORT TRIPLES
# ============================================================

def collect_support_triples(
    support_paths,
):
    """
    Bütün selected gold support paths-dəki
    original KG triples-ları union edir.

    Duplicate triple yalnız bir dəfə saxlanılır.
    """

    support_triples = []
    seen = set()

    for answer, paths in support_paths.items():

        if not paths:
            continue

        # Bir canonical path / answer
        path = paths[0]

        for step in path:

            triple = tuple(
                step["triple"]
            )

            if triple in seen:
                continue

            seen.add(triple)
            support_triples.append(
                triple
            )

    return support_triples


# ============================================================
# 9. COLLECT PATH ENTITIES
# ============================================================

def collect_path_entities(
    item,
    support_paths,
):
    """
    Context retrieval üçün topic və intermediate
    entities-i toplayır.

    Gold answer entities context anchor deyil.
    """

    entities = {
        item["topic_entity"]
    }

    gold_answers = set(
        item["gold_answers"]
    )

    for paths in support_paths.values():

        if not paths:
            continue

        path = paths[0]

        for step in path:

            from_entity = step[
                "from_entity"
            ]

            to_entity = step[
                "to_entity"
            ]

            if from_entity not in gold_answers:
                entities.add(
                    from_entity
                )

            if to_entity not in gold_answers:
                entities.add(
                    to_entity
                )

    return entities


# ============================================================
# 10. STABLE DETERMINISTIC SEED
# ============================================================

def stable_seed(
    text,
    base_seed=DEFAULT_SEED,
):
    """
    Python hash() run-lar arasında dəyişə bilər.

    Ona görə SHA256 əsaslı reproducible seed istifadə olunur.
    """

    value = (
        f"{base_seed}:{text}"
    )

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


# ============================================================
# 11. CONTEXT VALIDATION
# ============================================================

def is_valid_context_triple(
    triple,
    support_triples,
    gold_answers,
):
    """
    Context triple:

    - support triple ola bilməz;
    - gold answer entity-ni birbaşa göstərə bilməz.
    """

    if triple in support_triples:
        return False

    head, relation, tail = triple

    if (
        head in gold_answers
        or tail in gold_answers
    ):
        return False

    return True


# ============================================================
# 12. DETERMINISTIC POOL SAMPLING
# ============================================================

def add_from_candidate_pool(
    selected,
    selected_set,
    candidates,
    n_context,
    seed_value,
):
    """
    Candidate pool-u deterministic shuffle edib
    lazım olan qədər context əlavə edir.
    """

    candidates = sorted(
        set(candidates)
    )

    rng = random.Random(
        seed_value
    )

    rng.shuffle(
        candidates
    )

    for triple in candidates:

        if len(selected) >= n_context:
            break

        if triple in selected_set:
            continue

        selected.append(
            triple
        )

        selected_set.add(
            triple
        )


# ============================================================
# 13. BUILD FIXED-SIZE CONTEXT
# ============================================================

def build_context_triples(
    item,
    support_paths,
    adjacency,
    all_triples,
    n_context=DEFAULT_CONTEXT_SIZE,
    base_seed=DEFAULT_SEED,
):
    """
    Fixed-size contextual evidence qurur.

    Priority:

    LEVEL 1:
        support-path entities-in immediate neighborhood-u

    LEVEL 2:
        həmin neighborhood-un 2-hop expansion-u

    LEVEL 3:
        deterministic global KG fallback

    Məqsəd:
        mümkün olduğu qədər EXACTLY n_context triple.
    """

    support_triples = set(
        collect_support_triples(
            support_paths
        )
    )

    gold_answers = set(
        item["gold_answers"]
    )

    anchor_entities = collect_path_entities(
        item,
        support_paths,
    )

    selected = []
    selected_set = set()

    # ========================================================
    # LEVEL 1
    # Immediate local context
    # ========================================================

    level1_candidates = set()
    level1_entities = set()

    for entity in anchor_entities:

        for edge in adjacency.get(
            entity,
            [],
        ):

            triple = tuple(
                edge["triple"]
            )

            if not is_valid_context_triple(
                triple,
                support_triples,
                gold_answers,
            ):
                continue

            level1_candidates.add(
                triple
            )

            level1_entities.add(
                edge["next"]
            )

    add_from_candidate_pool(
        selected=selected,
        selected_set=selected_set,
        candidates=level1_candidates,
        n_context=n_context,
        seed_value=stable_seed(
            item["qid"] + ":context_level1",
            base_seed,
        ),
    )

    # ========================================================
    # LEVEL 2
    # 2-hop local expansion
    # ========================================================

    if len(selected) < n_context:

        level2_candidates = set()

        for entity in level1_entities:

            for edge in adjacency.get(
                entity,
                [],
            ):

                triple = tuple(
                    edge["triple"]
                )

                if not is_valid_context_triple(
                    triple,
                    support_triples,
                    gold_answers,
                ):
                    continue

                level2_candidates.add(
                    triple
                )

        add_from_candidate_pool(
            selected=selected,
            selected_set=selected_set,
            candidates=level2_candidates,
            n_context=n_context,
            seed_value=stable_seed(
                item["qid"] + ":context_level2",
                base_seed,
            ),
        )

    # ========================================================
    # LEVEL 3
    # Global deterministic fallback
    # ========================================================

    if len(selected) < n_context:

        global_candidates = []

        for triple_dict in all_triples:

            triple = (
                triple_dict["head"],
                triple_dict["relation"],
                triple_dict["tail"],
            )

            if not is_valid_context_triple(
                triple,
                support_triples,
                gold_answers,
            ):
                continue

            if triple in selected_set:
                continue

            global_candidates.append(
                triple
            )

        add_from_candidate_pool(
            selected=selected,
            selected_set=selected_set,
            candidates=global_candidates,
            n_context=n_context,
            seed_value=stable_seed(
                item["qid"] + ":context_global",
                base_seed,
            ),
        )

    return selected


# ============================================================
# 14. SERIALIZE SUPPORT PATHS
# ============================================================

def serialize_support_paths(
    item,
    support_paths,
):
    """
    Support paths JSON-compatible formata çevrilir.

    Corruption üçün lazım olacaq metadata:

    - hop_index
    - early/middle/answer-adjacent
    - from_entity
    - to_entity
    - relation
    - reverse
    - original triple
    """

    serialized = {}

    total_hops = item[
        "hop"
    ]

    for answer, paths in support_paths.items():

        serialized[
            answer
        ] = []

        for path in paths:

            serialized_path = []

            for step_index, step in enumerate(
                path
            ):

                serialized_path.append(
                    {
                        "hop_index":
                            step_index + 1,

                        "position":
                            get_position_label(
                                step_index,
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

            serialized[
                answer
            ].append(
                serialized_path
            )

    return serialized


# ============================================================
# 15. BUILD CLEAN EVIDENCE E
# ============================================================

def build_clean_evidence(
    item,
    adjacency,
    all_triples,
    n_context=DEFAULT_CONTEXT_SIZE,
    seed=DEFAULT_SEED,
):
    """
    Final frozen clean evidence:

        E = support triples + context triples
    """

    # ========================================================
    # A. Semantically aligned support paths
    # ========================================================

    support_paths = find_support_paths(
        item=item,
        adjacency=adjacency,
        max_paths_per_answer=1,
    )

    missing_answers = [
        answer
        for answer, paths
        in support_paths.items()
        if not paths
    ]

    # Bütün gold answers üçün support lazımdır
    if missing_answers:
        return None

    # ========================================================
    # B. Support triples
    # ========================================================

    support_triples = collect_support_triples(
        support_paths
    )

    # ========================================================
    # C. Fixed contextual triples
    # ========================================================

    context_triples = build_context_triples(
        item=item,
        support_paths=support_paths,
        adjacency=adjacency,
        all_triples=all_triples,
        n_context=n_context,
        base_seed=seed,
    )

    # ========================================================
    # D. Combined evidence
    # ========================================================

    evidence = []

    for triple in support_triples:

        evidence.append(
            {
                "triple":
                    list(triple),

                "role":
                    "support",
            }
        )

    for triple in context_triples:

        evidence.append(
            {
                "triple":
                    list(triple),

                "role":
                    "context",
            }
        )

    # Evidence order deterministic şəkildə shuffle olunur.
    # Beləliklə support həmişə prompt-un əvvəlində görünmür.

    rng = random.Random(
        stable_seed(
            item["qid"]
            + ":evidence_order",
            seed,
        )
    )

    rng.shuffle(
        evidence
    )

    # ========================================================
    # E. Metadata
    # ========================================================

    relation_scores = get_question_relation_scores(
        item["question"]
    )

    target_relation = infer_target_relation(
        item["question"]
    )

    clean_item = {

        "qid":
            item["qid"],

        "hop":
            item["hop"],

        "question":
            item["question"],

        "question_raw":
            item["question_raw"],

        "topic_entity":
            item["topic_entity"],

        "gold_answers":
            item["gold_answers"],

        "target_relation":
            target_relation,

        "question_relation_scores":
            relation_scores,

        "support_paths":
            serialize_support_paths(
                item,
                support_paths,
            ),

        "support_triples":
            [
                list(triple)
                for triple
                in support_triples
            ],

        "context_triples":
            [
                list(triple)
                for triple
                in context_triples
            ],

        "evidence":
            evidence,

        "num_support_triples":
            len(
                support_triples
            ),

        "num_context_triples":
            len(
                context_triples
            ),

        "seed":
            seed,
    }

    return clean_item


# ============================================================
# 16. BUILD PILOT DATASET
# ============================================================

def build_pilot_dataset(
    adjacency,
    all_triples,
    n_per_hop=100,
    n_context=DEFAULT_CONTEXT_SIZE,
    seed=DEFAULT_SEED,
):
    """
    Pilot dataset:

        100 x 1-hop
        100 x 2-hop
        100 x 3-hop

    Total target:
        300 VALID clean items.

    Əvvəlki versiyadan fərqli olaraq,
    ilk 100 sample-dan problemli item çıxsa belə
    növbəti item-lərlə əvəz edilir.
    """

    pilot = []

    for hop in [1, 2, 3]:

        all_items = load_questions(
            hop
        )

        rng = random.Random(
            seed + hop
        )

        shuffled_items = list(
            all_items
        )

        rng.shuffle(
            shuffled_items
        )

        built = 0
        rejected = 0

        for item in shuffled_items:

            if built >= n_per_hop:
                break

            clean_item = build_clean_evidence(
                item=item,
                adjacency=adjacency,
                all_triples=all_triples,
                n_context=n_context,
                seed=seed,
            )

            if clean_item is None:

                rejected += 1
                continue

            pilot.append(
                clean_item
            )

            built += 1

        print(
            f"{hop}-hop clean items: "
            f"{built}/{n_per_hop} "
            f"(rejected={rejected})"
        )

    return pilot


# ============================================================
# 17. SAVE JSONL
# ============================================================

def save_jsonl(
    items,
    path,
):
    """
    JSON Lines format:

        one sample = one line
    """

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

            line = json.dumps(
                item,
                ensure_ascii=False,
            )

            f.write(
                line + "\n"
            )


# ============================================================
# 18. PRINT SUPPORT PATH
# ============================================================

def print_support_path(
    answer,
    paths,
):
    """
    Debug/inspection üçün human-readable path.
    """

    print(
        f"Answer: {answer}"
    )

    if not paths:

        print(
            "  NO PATH"
        )

        return

    for path in paths:

        for step in path:

            reverse_mark = (
                "^-1"
                if step["reverse"]
                else ""
            )

            print(
                f"  hop={step['hop_index']} "
                f"position={step['position']} | "
                f"{step['from_entity']} "
                f"--[{step['relation']}"
                f"{reverse_mark}]--> "
                f"{step['to_entity']}"
            )


# ============================================================
# 19. MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Load KG
    # ========================================================

    print(
        "Loading KG..."
    )

    triples = load_kb()

    print(
        f"Triples: "
        f"{len(triples)}"
    )

    # ========================================================
    # Build graph index
    # ========================================================

    print(
        "Building adjacency..."
    )

    adjacency = build_adjacency(
        triples
    )

    print(
        f"Entities: "
        f"{len(adjacency)}"
    )

    # ========================================================
    # Build pilot
    # ========================================================

    print()
    print(
        "Building pilot clean evidence..."
    )
    print()

    pilot = build_pilot_dataset(
        adjacency=adjacency,
        all_triples=triples,
        n_per_hop=100,
        n_context=5,
        seed=42,
    )

    # ========================================================
    # Save frozen clean pilot
    # ========================================================

    output_path = (
        PROCESSED_DIR
        / "metaqa_pilot_clean.jsonl"
    )

    save_jsonl(
        pilot,
        output_path,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        f"Total pilot items: "
        f"{len(pilot)}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    if not pilot:
        raise RuntimeError(
            "Pilot dataset is empty."
        )

    avg_support = sum(
        item[
            "num_support_triples"
        ]
        for item in pilot
    ) / len(pilot)

    avg_context = sum(
        item[
            "num_context_triples"
        ]
        for item in pilot
    ) / len(pilot)

    context_shortages = sum(
        item[
            "num_context_triples"
        ] < DEFAULT_CONTEXT_SIZE
        for item in pilot
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
        f"Items with <"
        f"{DEFAULT_CONTEXT_SIZE} "
        f"context triples: "
        f"{context_shortages}"
    )

    # ========================================================
    # Example
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "EXAMPLE CLEAN EVIDENCE"
    )

    print(
        "=" * 70
    )

    example = pilot[0]

    print()

    print(
        "QID:",
        example["qid"],
    )

    print(
        "Hop:",
        example["hop"],
    )

    print(
        "Question:",
        example["question"],
    )

    print(
        "Topic:",
        example["topic_entity"],
    )

    print(
        "Gold answers:",
        example["gold_answers"],
    )

    print(
        "Target relation:",
        example["target_relation"],
    )

    print()

    print(
        "Evidence:"
    )

    print()

    for evidence_item in example[
        "evidence"
    ]:

        triple = evidence_item[
            "triple"
        ]

        role = evidence_item[
            "role"
        ]

        print(
            f"{role.upper():8} | "
            f"{triple[0]} | "
            f"{triple[1]} | "
            f"{triple[2]}"
        )

    print()
    print(
        "Support paths:"
    )
    print()

    for answer, paths in example[
        "support_paths"
    ].items():

        print_support_path(
            answer,
            paths,
        )

        print()

    # ========================================================
    # Final sanity checks
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "SANITY CHECKS"
    )

    print(
        "=" * 70
    )

    hop_counts = {
        1: 0,
        2: 0,
        3: 0,
    }

    for item in pilot:

        hop_counts[
            item["hop"]
        ] += 1

    print(
        "Hop counts:",
        hop_counts,
    )

    print(
        "Context shortages:",
        context_shortages,
    )

    if len(pilot) == 300:
        print(
            "Pilot size check: PASS"
        )
    else:
        print(
            "Pilot size check: WARNING"
        )

    if context_shortages == 0:
        print(
            "Fixed context check: PASS"
        )
    else:
        print(
            "Fixed context check: WARNING"
        )