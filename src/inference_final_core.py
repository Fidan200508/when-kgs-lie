from pathlib import Path
import gc
import json
import time

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from prompts import (
    build_messages,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = (
    "Qwen/Qwen3-4B"
)

MODEL_SHORT_NAME = (
    "qwen3_4b"
)

# Final matched-core evaluation:
# each selected item has at most 10 gold answers and 40 evidence triples.
# We use one fixed cap for ALL final-core generations.
MAX_NEW_TOKENS = 128

EXPECTED_ITEMS_PER_DATASET = 28

EXPECTED_HOP_COUNTS = {
    2: 14,
    3: 14,
}

DEFAULT_SEED = 42

PROMPT_VERSION = (
    "kgqa_evidence_only_v1"
)


# ============================================================
# DATASETS
# ============================================================

DATASETS = [

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    {
        "condition":
            "clean",

        "label_mode":
            "natural",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_clean.jsonl"
            ),
    },

    {
        "condition":
            "clean",

        "label_mode":
            "anonymized",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_clean_anonymized.jsonl"
            ),
    },

    # --------------------------------------------------------
    # ENTITY SUBSTITUTION
    # --------------------------------------------------------

    {
        "condition":
            "entity_substitution",

        "label_mode":
            "natural",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_entity_substitution.jsonl"
            ),
    },

    {
        "condition":
            "entity_substitution",

        "label_mode":
            "anonymized",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_entity_substitution_anonymized.jsonl"
            ),
    },

    # --------------------------------------------------------
    # RELATION SUBSTITUTION
    # --------------------------------------------------------

    {
        "condition":
            "relation_substitution",

        "label_mode":
            "natural",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_relation_substitution.jsonl"
            ),
    },

    {
        "condition":
            "relation_substitution",

        "label_mode":
            "anonymized",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_relation_substitution_anonymized.jsonl"
            ),
    },

    # --------------------------------------------------------
    # CONTRADICTION
    # --------------------------------------------------------

    {
        "condition":
            "contradiction",

        "label_mode":
            "natural",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_contradiction.jsonl"
            ),
    },

    {
        "condition":
            "contradiction",

        "label_mode":
            "anonymized",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_contradiction_anonymized.jsonl"
            ),
    },

    # --------------------------------------------------------
    # REROUTING
    # --------------------------------------------------------

    {
        "condition":
            "rerouting",

        "label_mode":
            "natural",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_rerouting.jsonl"
            ),
    },

    {
        "condition":
            "rerouting",

        "label_mode":
            "anonymized",

        "input":
            Path(
                "data/processed/final_core/"
                "metaqa_pilot_rerouting_anonymized.jsonl"
            ),
    },
]


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = Path(
    "results/raw/final_core_inference"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "inference_config.json"
)


# ============================================================
# JSONL LOADING
# ============================================================

def load_jsonl(path):
    items = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = (
                line.strip()
            )

            if not line:
                continue

            items.append(
                json.loads(
                    line
                )
            )

    return items


# ============================================================
# APPEND JSONL
# ============================================================

def append_jsonl(
    item,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )

        # Save immediately so the run is resumable.
        f.flush()


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_completed_qids(path):
    """
    Reads already completed outputs.

    If inference stops halfway, rerunning inference.py will skip
    qids that already have status == "ok".
    """

    completed = set()

    if not path.exists():

        return completed

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = (
                line.strip()
            )

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

            except json.JSONDecodeError:

                # Ignore potentially incomplete final line.
                continue

            if (
                row.get(
                    "status"
                )
                == "ok"
            ):

                completed.add(
                    row[
                        "qid"
                    ]
                )

    return completed


# ============================================================
# OUTPUT FILE NAME
# ============================================================

def get_output_path(
    condition,
    label_mode,
):

    filename = (
        f"{MODEL_SHORT_NAME}_"
        f"{condition}_"
        f"{label_mode}.jsonl"
    )

    return (
        OUTPUT_DIR
        / filename
    )


# ============================================================
# SAVE EXPERIMENT CONFIG
# ============================================================

def save_config():

    CONFIG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {

        "model":
            MODEL_NAME,

        "model_short_name":
            MODEL_SHORT_NAME,

        "quantization":
            "bitsandbytes_nf4_4bit",

        "double_quant":
            True,

        "compute_dtype":
            "float16",

        "thinking":
            False,

        "do_sample":
            False,

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "prompt_version":
            PROMPT_VERSION,

        "experiment_scope":
            "final_matched_core",

        "expected_items_per_dataset":
            EXPECTED_ITEMS_PER_DATASET,

        "expected_hop_counts":
            EXPECTED_HOP_COUNTS,

        "selection_constraints": {
            "relation_substitution_feasible":
                True,
            "rerouting_feasible":
                True,
            "max_gold_answers":
                10,
            "max_evidence_triples":
                40,
        },

        "seed":
            DEFAULT_SEED,

        "torch_version":
            torch.__version__,

        "cuda_version":
            torch.version.cuda,

        "gpu":
            (
                torch.cuda.get_device_name(
                    0
                )
                if torch.cuda.is_available()
                else None
            ),

        "gpu_vram_gb":
            (
                round(
                    torch.cuda.get_device_properties(
                        0
                    ).total_memory
                    / 1024**3,
                    2,
                )
                if torch.cuda.is_available()
                else None
            ),
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Saved experiment config:",
        CONFIG_PATH,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for this experiment."
        )

    print()

    print(
        "=" * 80
    )

    print(
        "LOADING MODEL"
    )

    print(
        "=" * 80
    )

    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "VRAM GB:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / 1024**3,
            2,
        ),
    )

    # ========================================================
    # 4-BIT NF4
    # ========================================================

    quantization_config = (
        BitsAndBytesConfig(

            load_in_4bit=
                True,

            bnb_4bit_quant_type=
                "nf4",

            bnb_4bit_use_double_quant=
                True,

            bnb_4bit_compute_dtype=
                torch.float16,
        )
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    torch.cuda.empty_cache()

    # ========================================================
    # MODEL
    # ========================================================

    start = (
        time.time()
    )

    model = (
        AutoModelForCausalLM.from_pretrained(

            MODEL_NAME,

            quantization_config=
                quantization_config,

            device_map=
                "auto",

            dtype=
                torch.float16,

            low_cpu_mem_usage=
                True,
        )
    )

    model.eval()

    # ========================================================
    # GREEDY / DETERMINISTIC
    # ========================================================

    model.generation_config.temperature = (
        None
    )

    model.generation_config.top_p = (
        None
    )

    model.generation_config.top_k = (
        None
    )

    elapsed = (
        time.time()
        - start
    )

    print(
        "Model loaded in:",
        round(
            elapsed,
            2,
        ),
        "seconds",
    )

    print(
        "GPU allocated GB:",
        round(
            torch.cuda.memory_allocated(
                0
            )
            / 1024**3,
            2,
        ),
    )

    return (
        tokenizer,
        model,
    )


# ============================================================
# GENERATE ONE ANSWER
# ============================================================

def generate_answer(
    item,
    tokenizer,
    model,
):

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    messages = (
        build_messages(
            item
        )
    )

    text = (
        tokenizer.apply_chat_template(

            messages,

            tokenize=
                False,

            add_generation_prompt=
                True,

            enable_thinking=
                False,
        )
    )

    # ========================================================
    # TOKENIZE
    # ========================================================

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    model_device = next(
        model.parameters()
    ).device

    inputs = {
        key:
            value.to(
                model_device
            )

        for key, value
        in inputs.items()
    }

    input_tokens = int(
        inputs[
            "input_ids"
        ].shape[1]
    )

    torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # GENERATE
    # ========================================================

    start = (
        time.time()
    )

    with torch.inference_mode():

        outputs = (
            model.generate(

                **inputs,

                max_new_tokens=
                    MAX_NEW_TOKENS,

                do_sample=
                    False,

                use_cache=
                    True,

                pad_token_id=
                    tokenizer.eos_token_id,

                eos_token_id=
                    tokenizer.eos_token_id,
            )
        )

    generation_seconds = (
        time.time()
        - start
    )

    # ========================================================
    # REMOVE PROMPT TOKENS
    # ========================================================

    generated_ids = (
        outputs[
            0,
            input_tokens:
        ]
    )

    output_tokens = int(
        generated_ids.shape[
            0
        ]
    )

    # ========================================================
    # DECODE
    # ========================================================

    raw_output = (
        tokenizer.decode(

            generated_ids,

            skip_special_tokens=
                True,
        )
        .strip()
    )

    # ========================================================
    # MEMORY
    # ========================================================

    peak_memory_gb = (
        round(
            torch.cuda.max_memory_allocated(
                0
            )
            / 1024**3,
            3,
        )
    )

    # ========================================================
    # TRUNCATION FLAG
    # ========================================================

    hit_max_token_limit = (
        output_tokens
        >= MAX_NEW_TOKENS
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    del inputs
    del outputs
    del generated_ids

    return {

        "raw_output":
            raw_output,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "generation_seconds":
            round(
                generation_seconds,
                4,
            ),

        "peak_gpu_memory_gb":
            peak_memory_gb,

        "hit_max_token_limit":
            hit_max_token_limit,
    }


# ============================================================
# FINAL-CORE VALIDATION
# ============================================================

def validate_final_core(
    items,
    input_path,
):
    if len(items) != EXPECTED_ITEMS_PER_DATASET:
        raise RuntimeError(
            f"{input_path}: expected "
            f"{EXPECTED_ITEMS_PER_DATASET} items, "
            f"found {len(items)}"
        )

    hop_counts = {}

    for item in items:
        hop = item["hop"]

        hop_counts[hop] = (
            hop_counts.get(
                hop,
                0,
            )
            + 1
        )

        gold_count = len(
            item.get(
                "gold_answers",
                [],
            )
        )

        evidence_count = len(
            item.get(
                "evidence",
                [],
            )
        )

        if gold_count > 10:
            raise RuntimeError(
                f"{input_path}: "
                f"{item['qid']} has "
                f"{gold_count} gold answers; "
                f"expected <= 10."
            )

        if evidence_count > 40:
            raise RuntimeError(
                f"{input_path}: "
                f"{item['qid']} has "
                f"{evidence_count} evidence triples; "
                f"expected <= 40."
            )

    if hop_counts != EXPECTED_HOP_COUNTS:
        raise RuntimeError(
            f"{input_path}: expected hop counts "
            f"{EXPECTED_HOP_COUNTS}, "
            f"found {hop_counts}"
        )

    print(
        "Final-core validation: PASS | "
        f"items={len(items)} | "
        f"hops={hop_counts}"
    )


# ============================================================
# RUN ONE CONDITION
# ============================================================

def run_dataset(
    spec,
    tokenizer,
    model,
):

    condition = (
        spec[
            "condition"
        ]
    )

    label_mode = (
        spec[
            "label_mode"
        ]
    )

    input_path = (
        spec[
            "input"
        ]
    )

    output_path = (
        get_output_path(
            condition,
            label_mode,
        )
    )

    items = (
        load_jsonl(
            input_path
        )
    )

    validate_final_core(
        items,
        input_path,
    )

    completed_qids = (
        load_completed_qids(
            output_path
        )
    )

    remaining = [

        item

        for item
        in items

        if item[
            "qid"
        ]
        not in completed_qids
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        f"CONDITION: "
        f"{condition} / "
        f"{label_mode}"
    )

    print(
        "=" * 80
    )

    print(
        "Input items:",
        len(
            items
        ),
    )

    print(
        "Already complete:",
        len(
            completed_qids
        ),
    )

    print(
        "Remaining:",
        len(
            remaining
        ),
    )

    print(
        "Output:",
        output_path,
    )

    if not remaining:

        print(
            "Nothing to do."
        )

        return

    dataset_start = (
        time.time()
    )

    successful = 0

    failed = 0

    truncated = 0

    # ========================================================
    # LOOP
    # ========================================================

    for index, item in enumerate(
        remaining,
        start=1,
    ):

        qid = item[
            "qid"
        ]

        try:

            generation = (
                generate_answer(
                    item,
                    tokenizer,
                    model,
                )
            )

            if generation[
                "hit_max_token_limit"
            ]:

                truncated += 1

            # =================================================
            # OUTPUT RECORD
            # =================================================

            record = {

                "qid":
                    qid,

                "hop":
                    item[
                        "hop"
                    ],

                "condition":
                    condition,

                "label_mode":
                    label_mode,

                "model":
                    MODEL_NAME,

                "prompt_version":
                    PROMPT_VERSION,

                "gold_answers":
                    item[
                        "gold_answers"
                    ],

                "question":
                    item[
                        "question"
                    ],

                "raw_output":
                    generation[
                        "raw_output"
                    ],

                "input_tokens":
                    generation[
                        "input_tokens"
                    ],

                "output_tokens":
                    generation[
                        "output_tokens"
                    ],

                "generation_seconds":
                    generation[
                        "generation_seconds"
                    ],

                "peak_gpu_memory_gb":
                    generation[
                        "peak_gpu_memory_gb"
                    ],

                "hit_max_token_limit":
                    generation[
                        "hit_max_token_limit"
                    ],

                "status":
                    "ok",
            }

            # =================================================
            # INTERNAL METADATA SAVED FOR EVALUATION ONLY
            #
            # NEVER exposed to model.
            # =================================================

            if (
                "corruption_manifest"
                in item
            ):

                record[
                    "corruption_manifest"
                ] = (
                    item[
                        "corruption_manifest"
                    ]
                )

            append_jsonl(
                record,
                output_path,
            )

            successful += 1

            # =================================================
            # TERMINAL PREVIEW
            # =================================================

            preview = (
                generation[
                    "raw_output"
                ]
                .replace(
                    "\n",
                    " ",
                )
            )

            if (
                len(
                    preview
                )
                > 100
            ):

                preview = (
                    preview[
                        :100
                    ]
                    + "..."
                )

            token_flag = (
                " MAXTOK"

                if generation[
                    "hit_max_token_limit"
                ]

                else ""
            )

            print(
                f"[{index}/"
                f"{len(remaining)}] "
                f"{qid} | "
                f"{generation['generation_seconds']:.2f}s | "
                f"out={generation['output_tokens']}"
                f"{token_flag} | "
                f"{preview}"
            )

        # ====================================================
        # CUDA OOM
        # ====================================================

        except torch.cuda.OutOfMemoryError as error:

            failed += 1

            torch.cuda.empty_cache()

            gc.collect()

            append_jsonl(

                {
                    "qid":
                        qid,

                    "hop":
                        item[
                            "hop"
                        ],

                    "condition":
                        condition,

                    "label_mode":
                        label_mode,

                    "model":
                        MODEL_NAME,

                    "prompt_version":
                        PROMPT_VERSION,

                    "status":
                        "error",

                    "error_type":
                        "cuda_out_of_memory",

                    "error":
                        str(
                            error
                        ),
                },

                output_path,
            )

            print(
                f"[{index}/"
                f"{len(remaining)}] "
                f"{qid} | "
                f"CUDA OOM"
            )

        # ====================================================
        # OTHER ERRORS
        # ====================================================

        except Exception as error:

            failed += 1

            append_jsonl(

                {
                    "qid":
                        qid,

                    "hop":
                        item[
                            "hop"
                        ],

                    "condition":
                        condition,

                    "label_mode":
                        label_mode,

                    "model":
                        MODEL_NAME,

                    "prompt_version":
                        PROMPT_VERSION,

                    "status":
                        "error",

                    "error_type":
                        type(
                            error
                        ).__name__,

                    "error":
                        str(
                            error
                        ),
                },

                output_path,
            )

            print(
                f"[{index}/"
                f"{len(remaining)}] "
                f"{qid} | ERROR: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    total_seconds = (
        time.time()
        - dataset_start
    )

    print()

    print(
        "Dataset complete."
    )

    print(
        "Successful:",
        successful,
    )

    print(
        "Failed:",
        failed,
    )

    print(
        "Hit max token limit:",
        truncated,
    )

    print(
        "Runtime minutes:",
        round(
            total_seconds
            / 60,
            2,
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for this experiment."
        )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    torch.manual_seed(
        DEFAULT_SEED
    )

    torch.cuda.manual_seed_all(
        DEFAULT_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_config()

    # ========================================================
    # MODEL LOAD ONCE
    # ========================================================

    (
        tokenizer,
        model,
    ) = load_model()

    print()

    print(
        "=" * 80
    )

    print(
        "STARTING FINAL CORE EXPERIMENT"
    )

    print(
        "=" * 80
    )

    total_start = (
        time.time()
    )

    # ========================================================
    # ALL 10 CONDITIONS
    # ========================================================

    for spec in DATASETS:

        run_dataset(
            spec,
            tokenizer,
            model,
        )

    total_seconds = (
        time.time()
        - total_start
    )

    print()

    print(
        "=" * 80
    )

    print(
        "ALL CONDITIONS COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Total runtime hours:",
        round(
            total_seconds
            / 3600,
            2,
        ),
    )


if __name__ == "__main__":

    main()