from pathlib import Path
import csv
import json
import math
import re
import unicodedata

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ============================================================
# CONFIG
# ============================================================

MODEL_SHORT_NAME = "qwen3_4b"

CONDITIONS = [
    "clean",
    "entity_substitution",
    "relation_substitution",
    "contradiction",
    "rerouting",
]

LABEL_MODES = [
    "natural",
    "anonymized",
]

CORRUPTIONS = [
    "entity_substitution",
    "relation_substitution",
    "contradiction",
    "rerouting",
]

INFERENCE_DIR = Path(
    "results/raw/final_core_inference"
)

DATA_DIR = Path(
    "data/processed/final_core"
)

TABLE_DIR = Path(
    "results/tables"
)

FIGURE_DIR = Path(
    "results/figures"
)

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42

EXPECTED_QIDS = 28
EXPECTED_HOPS = {
    2: 14,
    3: 14,
}


# ============================================================
# FILE MAP
# ============================================================

def dataset_path(
    condition,
    label_mode,
):
    suffix = (
        "_anonymized"
        if label_mode == "anonymized"
        else ""
    )

    return (
        DATA_DIR
        / f"metaqa_pilot_{condition}{suffix}.jsonl"
    )


def inference_path(
    condition,
    label_mode,
):
    return (
        INFERENCE_DIR
        / (
            f"{MODEL_SHORT_NAME}_"
            f"{condition}_"
            f"{label_mode}.jsonl"
        )
    )


# ============================================================
# JSONL
# ============================================================

def load_jsonl(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSONL in {path} "
                    f"at line {line_number}: {error}"
                )

    return rows


def load_latest_successful_outputs(path):
    """
    Keep the latest status='ok' row per qid.
    Also retain audit information about all raw rows.
    """

    all_rows = load_jsonl(path)

    latest_ok = {}
    error_rows = []

    for row in all_rows:

        qid = row.get("qid")
        status = row.get("status")

        if status == "ok":
            latest_ok[qid] = row
        else:
            error_rows.append(row)

    return (
        latest_ok,
        all_rows,
        error_rows,
    )


# ============================================================
# NORMALIZATION / PARSING
# ============================================================

def normalize_space(text):
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )

    return " ".join(
        text.split()
    )


def normalize_answer(
    value,
    label_mode,
):
    value = normalize_space(value)

    if label_mode == "natural":
        return value.casefold()

    return value


def parse_prediction(
    raw_output,
    label_mode,
):
    raw = (
        ""
        if raw_output is None
        else str(raw_output)
    )

    normalized_raw = normalize_space(raw)

    abstained = (
        normalized_raw
        == "UNKNOWN"
    )

    mixed_unknown = False
    empty_segment = False

    if abstained:
        prediction_list = []
        prediction_set = set()

    else:
        pieces = raw.split(";")

        prediction_list = []

        for piece in pieces:

            cleaned = normalize_space(
                piece
            )

            if not cleaned:
                empty_segment = True
                continue

            if cleaned == "UNKNOWN":
                mixed_unknown = True

            prediction_list.append(
                normalize_answer(
                    cleaned,
                    label_mode,
                )
            )

        prediction_set = set(
            prediction_list
        )

    duplicate_output = (
        len(prediction_list)
        != len(prediction_set)
    )

    # Conservative answer-only format audit.
    #
    # For natural labels we intentionally do NOT require outputs to be
    # evidence nodes, because doing so would conflate formatting with
    # semantic/evidence-following behavior.
    obvious_explanation = False

    lower_raw = normalized_raw.casefold()

    explanation_markers = [
        "the answer is",
        "answer:",
        "according to",
        "based on",
        "knowledge graph",
        "evidence",
        "because ",
        "therefore ",
        " is the screenwriter",
        " are the ",
        "the films ",
        "the movies ",
    ]

    if any(
        marker in lower_raw
        for marker in explanation_markers
    ):
        obvious_explanation = True

    if "\n" in raw.strip():
        obvious_explanation = True

    if "```" in raw:
        obvious_explanation = True

    anonymized_syntax_violation = False

    if (
        label_mode == "anonymized"
        and not abstained
    ):
        entity_pattern = re.compile(
            r"^Entity_\d+$"
        )

        anonymized_syntax_violation = any(
            not entity_pattern.fullmatch(
                normalize_space(piece)
            )
            for piece in raw.split(";")
            if normalize_space(piece)
        )

    protocol_violation = any(
        [
            mixed_unknown,
            empty_segment,
            obvious_explanation,
            anonymized_syntax_violation,
        ]
    )

    return {
        "normalized_raw":
            normalized_raw,

        "prediction_list":
            prediction_list,

        "prediction_set":
            prediction_set,

        "abstained":
            abstained,

        "mixed_unknown":
            mixed_unknown,

        "empty_segment":
            empty_segment,

        "duplicate_output":
            duplicate_output,

        "obvious_explanation":
            obvious_explanation,

        "anonymized_syntax_violation":
            anonymized_syntax_violation,

        "protocol_violation":
            protocol_violation,
    }


# ============================================================
# SET METRICS
# ============================================================

def set_metrics(
    prediction_set,
    gold_set,
):
    exact_match = int(
        prediction_set
        == gold_set
    )

    if not prediction_set:
        precision = 0.0
    else:
        precision = (
            len(
                prediction_set
                & gold_set
            )
            / len(
                prediction_set
            )
        )

    if not gold_set:
        recall = 1.0
    else:
        recall = (
            len(
                prediction_set
                & gold_set
            )
            / len(
                gold_set
            )
        )

    if (
        precision + recall
        == 0
    ):
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    return {
        "exact_match":
            exact_match,

        "set_precision":
            precision,

        "set_recall":
            recall,

        "set_f1":
            f1,
    }


# ============================================================
# CORRUPTION MANIFEST HELPERS
# ============================================================

FALSE_KEY_PRIORITY = [
    "injected_false_answer",
    "false_answer",
    "injected_answer",
    "alternative_answer",
    "replacement_answer",
    "false_entity",
    "injected_entity",
    "alternative_entity",
    "replacement_entity",
    "corrupted_entity",
    "new_entity",
    "replacement_tail",
    "new_tail",
    "replacement_head",
    "new_head",
]


def recursive_key_values(
    obj,
    prefix="",
):
    """
    Yield (path, value) for scalar/list values in nested dicts.
    """

    if isinstance(
        obj,
        dict,
    ):
        for key, value in obj.items():

            new_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            yield from recursive_key_values(
                value,
                new_prefix,
            )

    elif isinstance(
        obj,
        list,
    ):
        # A list of scalar values can itself be useful.
        if all(
            not isinstance(
                x,
                (dict, list),
            )
            for x in obj
        ):
            yield (
                prefix,
                obj,
            )

        else:
            for index, value in enumerate(
                obj
            ):
                yield from recursive_key_values(
                    value,
                    f"{prefix}[{index}]",
                )

    else:
        yield (
            prefix,
            obj,
        )


def candidate_false_values_from_manifest(
    manifest,
):
    """
    Best-effort extraction of injected/replacement false answer/entity
    from corruption_manifest.

    The evaluator records extraction coverage so a missing schema never
    silently becomes a valid zero.
    """

    if not isinstance(
        manifest,
        dict,
    ):
        return []

    flattened = list(
        recursive_key_values(
            manifest
        )
    )

    # First pass: exact/high-priority key names.
    candidates = []

    for priority_key in FALSE_KEY_PRIORITY:

        for path, value in flattened:

            leaf = (
                path
                .split(".")[-1]
                .split("[")[0]
            )

            if leaf == priority_key:

                if isinstance(
                    value,
                    list,
                ):
                    candidates.extend(
                        value
                    )
                else:
                    candidates.append(
                        value
                    )

        if candidates:
            break

    # Second pass: looser path-name search.
    if not candidates:

        for path, value in flattened:

            low = path.casefold()

            if (
                any(
                    marker in low
                    for marker in [
                        "false",
                        "injected",
                        "replacement",
                        "corrupted",
                        "alternative",
                    ]
                )
                and not any(
                    bad in low
                    for bad in [
                        "original",
                        "old",
                        "reason",
                        "seed",
                        "index",
                    ]
                )
            ):
                if isinstance(
                    value,
                    list,
                ):
                    # Avoid automatically treating a whole triple as
                    # three false answers.
                    if len(value) == 1:
                        candidates.extend(
                            value
                        )
                elif isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                    ),
                ):
                    candidates.append(
                        value
                    )

    cleaned = []

    for value in candidates:

        if value is None:
            continue

        text = normalize_space(
            value
        )

        if not text:
            continue

        if text not in cleaned:
            cleaned.append(
                text
            )

    return cleaned


# ============================================================
# VALIDATE INPUT DATA
# ============================================================

def validate_final_core_dataset(
    rows,
    path,
):
    if len(rows) != EXPECTED_QIDS:
        raise RuntimeError(
            f"{path}: expected "
            f"{EXPECTED_QIDS} rows, "
            f"found {len(rows)}"
        )

    qids = [
        row["qid"]
        for row in rows
    ]

    if len(
        set(qids)
    ) != EXPECTED_QIDS:
        raise RuntimeError(
            f"{path}: duplicate qids found."
        )

    hop_counts = {}

    for row in rows:

        hop = row["hop"]

        hop_counts[hop] = (
            hop_counts.get(
                hop,
                0,
            )
            + 1
        )

    if hop_counts != EXPECTED_HOPS:
        raise RuntimeError(
            f"{path}: expected hops "
            f"{EXPECTED_HOPS}, "
            f"found {hop_counts}"
        )


# ============================================================
# ITEM-LEVEL EVALUATION
# ============================================================

def evaluate_all():
    item_rows = []

    completeness_rows = []

    dataset_cache = {}

    # Load all datasets first.
    for condition in CONDITIONS:
        for label_mode in LABEL_MODES:

            dpath = dataset_path(
                condition,
                label_mode,
            )

            if not dpath.exists():
                raise FileNotFoundError(
                    dpath
                )

            rows = load_jsonl(
                dpath
            )

            validate_final_core_dataset(
                rows,
                dpath,
            )

            dataset_cache[
                (
                    condition,
                    label_mode,
                )
            ] = {
                row["qid"]:
                    row
                for row in rows
            }

    # Ensure every dataset uses exactly the same qid set.
    reference_qids = set(
        dataset_cache[
            (
                "clean",
                "natural",
            )
        ].keys()
    )

    for key, mapping in dataset_cache.items():

        if set(
            mapping.keys()
        ) != reference_qids:
            raise RuntimeError(
                f"QID mismatch in dataset {key}"
            )

    # Evaluate raw inference.
    for condition in CONDITIONS:
        for label_mode in LABEL_MODES:

            ipath = inference_path(
                condition,
                label_mode,
            )

            if not ipath.exists():
                raise FileNotFoundError(
                    ipath
                )

            (
                outputs,
                all_raw_rows,
                error_rows,
            ) = load_latest_successful_outputs(
                ipath
            )

            missing_qids = sorted(
                reference_qids
                - set(
                    outputs.keys()
                )
            )

            extra_qids = sorted(
                set(
                    outputs.keys()
                )
                - reference_qids
            )

            completeness_rows.append(
                {
                    "condition":
                        condition,

                    "label_mode":
                        label_mode,

                    "raw_rows":
                        len(
                            all_raw_rows
                        ),

                    "unique_ok_qids":
                        len(
                            outputs
                        ),

                    "error_rows":
                        len(
                            error_rows
                        ),

                    "missing_qids":
                        "|".join(
                            missing_qids
                        ),

                    "extra_qids":
                        "|".join(
                            extra_qids
                        ),
                }
            )

            if missing_qids:
                raise RuntimeError(
                    f"{ipath}: missing successful qids: "
                    f"{missing_qids}"
                )

            if extra_qids:
                raise RuntimeError(
                    f"{ipath}: unexpected qids: "
                    f"{extra_qids}"
                )

            data_mapping = dataset_cache[
                (
                    condition,
                    label_mode,
                )
            ]

            for qid in sorted(
                reference_qids
            ):

                output_row = outputs[
                    qid
                ]

                item = data_mapping[
                    qid
                ]

                raw_output = (
                    output_row.get(
                        "raw_output",
                        "",
                    )
                )

                parsed = parse_prediction(
                    raw_output,
                    label_mode,
                )

                gold_list = [
                    normalize_answer(
                        answer,
                        label_mode,
                    )
                    for answer in item.get(
                        "gold_answers",
                        [],
                    )
                ]

                gold_set = set(
                    gold_list
                )

                metrics = set_metrics(
                    parsed[
                        "prediction_set"
                    ],
                    gold_set,
                )

                manifest = (
                    item.get(
                        "corruption_manifest"
                    )
                    or output_row.get(
                        "corruption_manifest"
                    )
                    or {}
                )

                false_candidates_raw = (
                    candidate_false_values_from_manifest(
                        manifest
                    )
                    if condition
                    in {
                        "entity_substitution",
                        "contradiction",
                    }
                    else []
                )

                false_candidates = {
                    normalize_answer(
                        value,
                        label_mode,
                    )
                    for value in false_candidates_raw
                }

                false_adopted = None

                if condition in {
                    "entity_substitution",
                    "contradiction",
                }:
                    if false_candidates:
                        false_adopted = int(
                            bool(
                                parsed[
                                    "prediction_set"
                                ]
                                & false_candidates
                            )
                        )

                gold_retained_any = int(
                    bool(
                        parsed[
                            "prediction_set"
                        ]
                        & gold_set
                    )
                )

                gold_retained_all = int(
                    gold_set.issubset(
                        parsed[
                            "prediction_set"
                        ]
                    )
                )

                contradiction_category = ""

                if condition == "contradiction":

                    if parsed[
                        "abstained"
                    ]:
                        contradiction_category = (
                            "UNKNOWN"
                        )

                    elif false_adopted == 1:

                        if gold_retained_any:
                            contradiction_category = (
                                "both"
                            )
                        else:
                            contradiction_category = (
                                "false_only"
                            )

                    elif (
                        parsed[
                            "prediction_set"
                        ]
                        and parsed[
                            "prediction_set"
                        ].issubset(
                            gold_set
                        )
                    ):
                        contradiction_category = (
                            "gold_only"
                        )

                    else:
                        contradiction_category = (
                            "other"
                        )

                item_rows.append(
                    {
                        "qid":
                            qid,

                        "hop":
                            item["hop"],

                        "condition":
                            condition,

                        "label_mode":
                            label_mode,

                        "raw_output":
                            raw_output,

                        "prediction":
                            " || ".join(
                                parsed[
                                    "prediction_list"
                                ]
                            ),

                        "gold_answers":
                            " || ".join(
                                gold_list
                            ),

                        "gold_count":
                            len(
                                gold_set
                            ),

                        "prediction_count":
                            len(
                                parsed[
                                    "prediction_set"
                                ]
                            ),

                        "exact_match":
                            metrics[
                                "exact_match"
                            ],

                        "set_precision":
                            metrics[
                                "set_precision"
                            ],

                        "set_recall":
                            metrics[
                                "set_recall"
                            ],

                        "set_f1":
                            metrics[
                                "set_f1"
                            ],

                        "abstained":
                            int(
                                parsed[
                                    "abstained"
                                ]
                            ),

                        "mixed_unknown":
                            int(
                                parsed[
                                    "mixed_unknown"
                                ]
                            ),

                        "duplicate_output":
                            int(
                                parsed[
                                    "duplicate_output"
                                ]
                            ),

                        "protocol_violation":
                            int(
                                parsed[
                                    "protocol_violation"
                                ]
                            ),

                        "hit_max_token_limit":
                            int(
                                bool(
                                    output_row.get(
                                        "hit_max_token_limit",
                                        False,
                                    )
                                )
                            ),

                        "input_tokens":
                            output_row.get(
                                "input_tokens"
                            ),

                        "output_tokens":
                            output_row.get(
                                "output_tokens"
                            ),

                        "generation_seconds":
                            output_row.get(
                                "generation_seconds"
                            ),

                        "peak_gpu_memory_gb":
                            output_row.get(
                                "peak_gpu_memory_gb"
                            ),

                        "false_candidate_count":
                            len(
                                false_candidates
                            ),

                        "false_candidates":
                            " || ".join(
                                sorted(
                                    false_candidates
                                )
                            ),

                        "false_adopted":
                            (
                                ""
                                if false_adopted
                                is None
                                else false_adopted
                            ),

                        "gold_retained_any":
                            gold_retained_any,

                        "gold_retained_all":
                            gold_retained_all,

                        "contradiction_category":
                            contradiction_category,
                    }
                )

    return (
        item_rows,
        completeness_rows,
    )


# ============================================================
# SUMMARIES
# ============================================================

def mean(values):
    values = list(
        values
    )

    if not values:
        return float(
            "nan"
        )

    return sum(
        values
    ) / len(
        values
    )


def rows_for(
    item_rows,
    condition,
    label_mode,
    hop=None,
):
    rows = [
        row
        for row in item_rows
        if (
            row["condition"]
            == condition
            and row["label_mode"]
            == label_mode
        )
    ]

    if hop is not None:
        rows = [
            row
            for row in rows
            if row["hop"] == hop
        ]

    return rows


def condition_summary(
    item_rows,
):
    summary = []

    for condition in CONDITIONS:
        for label_mode in LABEL_MODES:

            rows = rows_for(
                item_rows,
                condition,
                label_mode,
            )

            false_rows = [
                row
                for row in rows
                if row[
                    "false_adopted"
                ] != ""
            ]

            summary.append(
                {
                    "condition":
                        condition,

                    "label_mode":
                        label_mode,

                    "n":
                        len(rows),

                    "accuracy":
                        mean(
                            row[
                                "exact_match"
                            ]
                            for row in rows
                        ),

                    "mean_set_f1":
                        mean(
                            row[
                                "set_f1"
                            ]
                            for row in rows
                        ),

                    "unknown_rate":
                        mean(
                            row[
                                "abstained"
                            ]
                            for row in rows
                        ),

                    "coverage":
                        1
                        - mean(
                            row[
                                "abstained"
                            ]
                            for row in rows
                        ),

                    "max_token_rate":
                        mean(
                            row[
                                "hit_max_token_limit"
                            ]
                            for row in rows
                        ),

                    "max_token_count":
                        sum(
                            row[
                                "hit_max_token_limit"
                            ]
                            for row in rows
                        ),

                    "protocol_violation_rate":
                        mean(
                            row[
                                "protocol_violation"
                            ]
                            for row in rows
                        ),

                    "protocol_violation_count":
                        sum(
                            row[
                                "protocol_violation"
                            ]
                            for row in rows
                        ),

                    "duplicate_output_rate":
                        mean(
                            row[
                                "duplicate_output"
                            ]
                            for row in rows
                        ),

                    "false_metadata_coverage":
                        (
                            len(
                                false_rows
                            )
                            / len(
                                rows
                            )
                            if condition
                            in {
                                "entity_substitution",
                                "contradiction",
                            }
                            else ""
                        ),

                    "false_adoption_rate":
                        (
                            mean(
                                int(
                                    row[
                                        "false_adopted"
                                    ]
                                )
                                for row
                                in false_rows
                            )
                            if false_rows
                            else ""
                        ),

                    "gold_retained_any_rate":
                        mean(
                            row[
                                "gold_retained_any"
                            ]
                            for row in rows
                        ),

                    "gold_retained_all_rate":
                        mean(
                            row[
                                "gold_retained_all"
                            ]
                            for row in rows
                        ),
                }
            )

    return summary


# ============================================================
# PAIRED EFFECTS
# ============================================================

def keyed(
    rows,
):
    return {
        row["qid"]:
            row
        for row in rows
    }


def paired_effects(
    item_rows,
):
    output = []

    for corruption in CORRUPTIONS:
        for label_mode in LABEL_MODES:

            clean = keyed(
                rows_for(
                    item_rows,
                    "clean",
                    label_mode,
                )
            )

            corrupt = keyed(
                rows_for(
                    item_rows,
                    corruption,
                    label_mode,
                )
            )

            qids = sorted(
                set(
                    clean
                )
                & set(
                    corrupt
                )
            )

            clean_acc = mean(
                clean[qid][
                    "exact_match"
                ]
                for qid in qids
            )

            corrupt_acc = mean(
                corrupt[qid][
                    "exact_match"
                ]
                for qid in qids
            )

            delta = (
                clean_acc
                - corrupt_acc
            )

            clean_correct = [
                qid
                for qid in qids
                if clean[qid][
                    "exact_match"
                ] == 1
            ]

            flip_rate = (
                mean(
                    int(
                        corrupt[qid][
                            "exact_match"
                        ] == 0
                    )
                    for qid
                    in clean_correct
                )
                if clean_correct
                else float(
                    "nan"
                )
            )

            robustness_ratio = (
                corrupt_acc
                / clean_acc
                if clean_acc > 0
                else float(
                    "nan"
                )
            )

            output.append(
                {
                    "condition":
                        corruption,

                    "label_mode":
                        label_mode,

                    "n":
                        len(
                            qids
                        ),

                    "paired_clean_accuracy":
                        clean_acc,

                    "corrupt_accuracy":
                        corrupt_acc,

                    "delta_accuracy":
                        delta,

                    "robustness_ratio":
                        robustness_ratio,

                    "clean_correct_n":
                        len(
                            clean_correct
                        ),

                    "flip_rate_among_clean_correct":
                        flip_rate,

                    "corrupt_mean_set_f1":
                        mean(
                            corrupt[qid][
                                "set_f1"
                            ]
                            for qid in qids
                        ),

                    "corrupt_unknown_rate":
                        mean(
                            corrupt[qid][
                                "abstained"
                            ]
                            for qid in qids
                        ),

                    "corrupt_coverage":
                        1
                        - mean(
                            corrupt[qid][
                                "abstained"
                            ]
                            for qid in qids
                        ),
                }
            )

    return output


# ============================================================
# BOOTSTRAP
# ============================================================

def percentile_ci(
    samples,
):
    return (
        float(
            np.percentile(
                samples,
                2.5,
            )
        ),
        float(
            np.percentile(
                samples,
                97.5,
            )
        ),
    )


def bootstrap_mean_effect(
    effects,
    rng,
):
    effects = np.asarray(
        effects,
        dtype=float,
    )

    n = len(
        effects
    )

    if n == 0:
        return (
            float("nan"),
            float("nan"),
        )

    bootstrap_means = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=float,
    )

    for b in range(
        BOOTSTRAP_SAMPLES
    ):

        indexes = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_means[b] = (
            effects[
                indexes
            ].mean()
        )

    return percentile_ci(
        bootstrap_means
    )


def bootstrap_tables(
    item_rows,
):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    effect_rows = []

    hop_rows = []

    pmg_rows = []

    # Overall corruption deltas.
    for corruption in CORRUPTIONS:
        for label_mode in LABEL_MODES:

            clean = keyed(
                rows_for(
                    item_rows,
                    "clean",
                    label_mode,
                )
            )

            corrupt = keyed(
                rows_for(
                    item_rows,
                    corruption,
                    label_mode,
                )
            )

            qids = sorted(
                set(clean)
                & set(corrupt)
            )

            effects = [
                (
                    clean[qid][
                        "exact_match"
                    ]
                    - corrupt[qid][
                        "exact_match"
                    ]
                )
                for qid in qids
            ]

            ci_low, ci_high = (
                bootstrap_mean_effect(
                    effects,
                    rng,
                )
            )

            effect_rows.append(
                {
                    "condition":
                        corruption,

                    "label_mode":
                        label_mode,

                    "n":
                        len(qids),

                    "delta_accuracy":
                        mean(
                            effects
                        ),

                    "ci95_low":
                        ci_low,

                    "ci95_high":
                        ci_high,
                }
            )

            # Hop-specific effects.
            for hop in [
                2,
                3,
            ]:

                hop_qids = [
                    qid
                    for qid in qids
                    if clean[qid][
                        "hop"
                    ] == hop
                ]

                hop_effects = [
                    (
                        clean[qid][
                            "exact_match"
                        ]
                        - corrupt[qid][
                            "exact_match"
                        ]
                    )
                    for qid
                    in hop_qids
                ]

                hop_ci_low, hop_ci_high = (
                    bootstrap_mean_effect(
                        hop_effects,
                        rng,
                    )
                )

                hop_rows.append(
                    {
                        "condition":
                            corruption,

                        "label_mode":
                            label_mode,

                        "hop":
                            hop,

                        "n":
                            len(
                                hop_qids
                            ),

                        "delta_accuracy":
                            mean(
                                hop_effects
                            ),

                        "ci95_low":
                            hop_ci_low,

                        "ci95_high":
                            hop_ci_high,
                    }
                )

    # PMG:
    # (clean_anon - corrupt_anon)
    # -
    # (clean_nat - corrupt_nat)
    for corruption in CORRUPTIONS:

        clean_nat = keyed(
            rows_for(
                item_rows,
                "clean",
                "natural",
            )
        )

        corrupt_nat = keyed(
            rows_for(
                item_rows,
                corruption,
                "natural",
            )
        )

        clean_anon = keyed(
            rows_for(
                item_rows,
                "clean",
                "anonymized",
            )
        )

        corrupt_anon = keyed(
            rows_for(
                item_rows,
                corruption,
                "anonymized",
            )
        )

        qids = sorted(
            set(
                clean_nat
            )
            & set(
                corrupt_nat
            )
            & set(
                clean_anon
            )
            & set(
                corrupt_anon
            )
        )

        item_pmg = [
            (
                (
                    clean_anon[qid][
                        "exact_match"
                    ]
                    - corrupt_anon[qid][
                        "exact_match"
                    ]
                )
                -
                (
                    clean_nat[qid][
                        "exact_match"
                    ]
                    - corrupt_nat[qid][
                        "exact_match"
                    ]
                )
            )
            for qid in qids
        ]

        ci_low, ci_high = (
            bootstrap_mean_effect(
                item_pmg,
                rng,
            )
        )

        pmg_rows.append(
            {
                "condition":
                    corruption,

                "n":
                    len(
                        qids
                    ),

                "pmg":
                    mean(
                        item_pmg
                    ),

                "ci95_low":
                    ci_low,

                "ci95_high":
                    ci_high,
            }
        )

    return (
        effect_rows,
        hop_rows,
        pmg_rows,
    )


# ============================================================
# CONTRADICTION BREAKDOWN
# ============================================================

def contradiction_breakdown(
    item_rows,
):
    output = []

    categories = [
        "gold_only",
        "false_only",
        "both",
        "UNKNOWN",
        "other",
    ]

    for label_mode in LABEL_MODES:

        rows = rows_for(
            item_rows,
            "contradiction",
            label_mode,
        )

        counts = {
            category:
                0
            for category
            in categories
        }

        unknown_false_metadata = 0

        for row in rows:

            category = row[
                "contradiction_category"
            ]

            if (
                row[
                    "false_adopted"
                ] == ""
                and not row[
                    "abstained"
                ]
            ):
                unknown_false_metadata += 1

            if category in counts:
                counts[
                    category
                ] += 1

        record = {
            "label_mode":
                label_mode,

            "n":
                len(
                    rows
                ),

            "false_metadata_missing_nonabstain":
                unknown_false_metadata,
        }

        for category in categories:

            record[
                f"{category}_count"
            ] = counts[
                category
            ]

            record[
                f"{category}_rate"
            ] = (
                counts[
                    category
                ]
                / len(
                    rows
                )
                if rows
                else float(
                    "nan"
                )
            )

        output.append(
            record
        )

    return output


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            f"No rows for {path}"
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# FIGURES
# ============================================================

def make_main_robustness_figure(
    effect_rows,
):
    if plt is None:
        print(
            "matplotlib unavailable; "
            "skipping figures."
        )
        return

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions = CORRUPTIONS

    natural = {
        row["condition"]:
            row
        for row in effect_rows
        if row[
            "label_mode"
        ] == "natural"
    }

    anonymized = {
        row["condition"]:
            row
        for row in effect_rows
        if row[
            "label_mode"
        ] == "anonymized"
    }

    x = np.arange(
        len(
            conditions
        )
    )

    width = 0.36

    nat_values = np.array(
        [
            natural[c][
                "delta_accuracy"
            ]
            * 100
            for c in conditions
        ]
    )

    anon_values = np.array(
        [
            anonymized[c][
                "delta_accuracy"
            ]
            * 100
            for c in conditions
        ]
    )

    nat_yerr = np.array(
        [
            [
                (
                    natural[c][
                        "delta_accuracy"
                    ]
                    - natural[c][
                        "ci95_low"
                    ]
                )
                * 100
                for c in conditions
            ],
            [
                (
                    natural[c][
                        "ci95_high"
                    ]
                    - natural[c][
                        "delta_accuracy"
                    ]
                )
                * 100
                for c in conditions
            ],
        ]
    )

    anon_yerr = np.array(
        [
            [
                (
                    anonymized[c][
                        "delta_accuracy"
                    ]
                    - anonymized[c][
                        "ci95_low"
                    ]
                )
                * 100
                for c in conditions
            ],
            [
                (
                    anonymized[c][
                        "ci95_high"
                    ]
                    - anonymized[c][
                        "delta_accuracy"
                    ]
                )
                * 100
                for c in conditions
            ],
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            7.0,
            4.2,
        )
    )

    ax.bar(
        x - width / 2,
        nat_values,
        width,
        yerr=nat_yerr,
        capsize=3,
        label="Natural",
    )

    ax.bar(
        x + width / 2,
        anon_values,
        width,
        yerr=anon_yerr,
        capsize=3,
        label="Anonymized",
    )

    labels = [
        "Entity\nsubstitution",
        "Relation\nsubstitution",
        "Contradiction",
        "Rerouting",
    ]

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Accuracy degradation (percentage points)"
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "main_robustness.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR
        / "main_robustness.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_hop_figure(
    hop_rows,
):
    if plt is None:
        return

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep this figure compact: average Natural/Anon separately,
    # grouped by condition and hop.
    labels = []

    values = []

    ci_low = []

    ci_high = []

    for condition in CORRUPTIONS:
        for hop in [
            2,
            3,
        ]:

            selected = [
                row
                for row in hop_rows
                if (
                    row[
                        "condition"
                    ]
                    == condition
                    and row[
                        "label_mode"
                    ]
                    == "natural"
                    and row[
                        "hop"
                    ]
                    == hop
                )
            ]

            row = selected[
                0
            ]

            labels.append(
                (
                    condition
                    .replace(
                        "_substitution",
                        "",
                    )
                    .replace(
                        "_",
                        " ",
                    )
                    + f"\n{hop}-hop"
                )
            )

            values.append(
                row[
                    "delta_accuracy"
                ]
                * 100
            )

            ci_low.append(
                (
                    row[
                        "delta_accuracy"
                    ]
                    - row[
                        "ci95_low"
                    ]
                )
                * 100
            )

            ci_high.append(
                (
                    row[
                        "ci95_high"
                    ]
                    - row[
                        "delta_accuracy"
                    ]
                )
                * 100
            )

    x = np.arange(
        len(
            labels
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            4.3,
        )
    )

    ax.bar(
        x,
        values,
        yerr=np.array(
            [
                ci_low,
                ci_high,
            ]
        ),
        capsize=3,
    )

    ax.set_xticks(
        x,
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        "Natural-label accuracy degradation (pp)"
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "hop_degradation.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR
        / "hop_degradation.pdf",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# TERMINAL REPORT
# ============================================================

def pct(value):
    if (
        value is None
        or (
            isinstance(
                value,
                float,
            )
            and math.isnan(
                value
            )
        )
    ):
        return "NA"

    return (
        f"{100 * value:.1f}%"
    )


def print_report(
    summaries,
    paired,
    effects,
    pmg,
    contradiction,
    item_rows,
):
    print()
    print(
        "=" * 92
    )
    print(
        "FINAL CORE EVALUATION COMPLETE"
    )
    print(
        "=" * 92
    )

    print()
    print(
        "CONDITION SUMMARY"
    )

    for row in summaries:

        print(
            f"{row['condition']:22s} "
            f"{row['label_mode']:10s} "
            f"N={row['n']:2d} | "
            f"Acc={pct(row['accuracy'])} | "
            f"F1={pct(row['mean_set_f1'])} | "
            f"UNK={pct(row['unknown_rate'])} | "
            f"MAXTOK={row['max_token_count']} | "
            f"ProtoViol={row['protocol_violation_count']}"
        )

    print()
    print(
        "PAIRED DEGRADATION"
    )

    ci_map = {
        (
            row["condition"],
            row["label_mode"],
        ):
            row
        for row in effects
    }

    for row in paired:

        ci = ci_map[
            (
                row["condition"],
                row["label_mode"],
            )
        ]

        print(
            f"{row['condition']:22s} "
            f"{row['label_mode']:10s} | "
            f"clean={pct(row['paired_clean_accuracy'])} | "
            f"corrupt={pct(row['corrupt_accuracy'])} | "
            f"Delta={100 * row['delta_accuracy']:.1f} pp | "
            f"95% CI "
            f"[{100 * ci['ci95_low']:.1f}, "
            f"{100 * ci['ci95_high']:.1f}] | "
            f"flip={pct(row['flip_rate_among_clean_correct'])}"
        )

    print()
    print(
        "PARAMETRIC MASKING GAP (PMG)"
    )

    for row in pmg:

        print(
            f"{row['condition']:22s} | "
            f"PMG={100 * row['pmg']:.1f} pp | "
            f"95% CI "
            f"[{100 * row['ci95_low']:.1f}, "
            f"{100 * row['ci95_high']:.1f}]"
        )

    print()
    print(
        "CONTRADICTION BREAKDOWN"
    )

    for row in contradiction:

        print(
            f"{row['label_mode']:10s} | "
            f"gold_only={row['gold_only_count']} | "
            f"false_only={row['false_only_count']} | "
            f"both={row['both_count']} | "
            f"UNKNOWN={row['UNKNOWN_count']} | "
            f"other={row['other_count']} | "
            f"false-metadata-missing="
            f"{row['false_metadata_missing_nonabstain']}"
        )

    total_maxtok = sum(
        row[
            "hit_max_token_limit"
        ]
        for row in item_rows
    )

    total_protocol = sum(
        row[
            "protocol_violation"
        ]
        for row in item_rows
    )

    print()
    print(
        "OUTPUT INTEGRITY"
    )

    print(
        "Total successful evaluated generations:",
        len(
            item_rows
        ),
    )

    print(
        "MAXTOK generations:",
        total_maxtok,
    )

    print(
        "Conservative protocol violations:",
        total_protocol,
        f"({100 * total_protocol / len(item_rows):.1f}%)",
    )

    print()
    print(
        "Saved tables:",
        TABLE_DIR,
    )

    print(
        "Saved figures:",
        FIGURE_DIR,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        item_rows,
        completeness_rows,
    ) = evaluate_all()

    summaries = condition_summary(
        item_rows
    )

    paired = paired_effects(
        item_rows
    )

    (
        effects,
        hop_rows,
        pmg_rows,
    ) = bootstrap_tables(
        item_rows
    )

    contradiction_rows = (
        contradiction_breakdown(
            item_rows
        )
    )

    write_csv(
        TABLE_DIR
        / "item_level_metrics.csv",
        item_rows,
    )

    write_csv(
        TABLE_DIR
        / "condition_summary.csv",
        summaries,
    )

    write_csv(
        TABLE_DIR
        / "paired_effects.csv",
        paired,
    )

    write_csv(
        TABLE_DIR
        / "bootstrap_effects.csv",
        effects,
    )

    write_csv(
        TABLE_DIR
        / "hop_effects.csv",
        hop_rows,
    )

    write_csv(
        TABLE_DIR
        / "pmg.csv",
        pmg_rows,
    )

    write_csv(
        TABLE_DIR
        / "contradiction_breakdown.csv",
        contradiction_rows,
    )

    write_csv(
        TABLE_DIR
        / "inference_completeness.csv",
        completeness_rows,
    )

    make_main_robustness_figure(
        effects
    )

    make_hop_figure(
        hop_rows
    )

    print_report(
        summaries,
        paired,
        effects,
        pmg_rows,
        contradiction_rows,
        item_rows,
    )


if __name__ == "__main__":
    main()
