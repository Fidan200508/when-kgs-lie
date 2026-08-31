# When Knowledge Graphs Lie

**Controlled Robustness Evaluation of KG-Grounded Large Language Model Question Answering**

This repository contains the code, processed evaluation data, raw model generations, evaluation tables, and figures for the WISE 2026 student-paper project **“When Knowledge Graphs Lie: Controlled Robustness Evaluation of KG-Grounded Large Language Model Question Answering.”**

The project studies a simple but important question: **what happens when an LLM is grounded on structured evidence that is itself wrong, misleading, or internally conflicting?**

Rather than changing retrieval, the experiment freezes the selected evidence for each question and perturbs only the knowledge-graph evidence shown to the generator. This makes it possible to measure how the model responds to specific evidence corruptions under matched conditions.

---

## Research Questions

The study addresses three questions:

1. **RQ1 — Corruption robustness:**  
   How do semantic and structural knowledge-graph corruptions affect KG-grounded question answering, and how does the effect vary with hop length?

2. **RQ2 — Natural vs. anonymized entities:**  
   To what extent does entity anonymization change the model's sensitivity to corrupted evidence?

3. **RQ3 — Reliability under conflict:**  
   When the supplied evidence is misleading or contradictory, does the model abstain, retain valid answers, or adopt the injected alternative?

---

## Experimental Principle

For a question \(q\), the clean and corrupted conditions use the same:

- question
- gold-answer set
- prompt template
- model
- evidence format
- decoding configuration

Only the evidence packet changes.

This is a **post-retrieval intervention**: retrieval/evidence selection is frozen before corruption. The goal is to isolate the generator's response to the evidence intervention rather than mixing corruption effects with retrieval errors.

---

## Dataset

**Dataset:** MetaQA

The reference knowledge graph contains:

- **134,741 triples**
- **43,234 nodes** in the bidirectional traversal index

A semantically validated pool of **300 questions** was constructed:

| Hop length | Questions |
|---|---:|
| 1-hop | 100 |
| 2-hop | 100 |
| 3-hop | 100 |
| **Total** | **300** |

Clean support paths were required to match the relation pattern implied by the MetaQA question template.

A second packet-level validation checked that the complete evidence packet still produced exactly the benchmark gold-answer set under the intended reasoning pattern.

During this filtering stage:

- 54 candidate 2-hop questions were rejected
- 15 candidate 3-hop questions were rejected

They were replaced by subsequent deterministic valid samples.

---

## Corruption Operators

A KG fact is represented as a triple:

```text
(head, relation, tail)
```

or mathematically as \((h,r,t)\), where:

- \(h\) = head entity
- \(r\) = relation
- \(t\) = tail entity

Synthetic triples are considered **false with respect to the reference MetaQA KG**. Their absence from MetaQA is not a claim about real-world truth.

### 1. Entity Substitution

An answer-adjacent endpoint is replaced while the relation is preserved:

```text
(h, r, t) -> (h, r, t')
```

Constraints:

- relation remains unchanged
- replacement comes from a relation-compatible endpoint pool
- the generated triple must not already exist in the reference KG
- evidence size is preserved

Final feasibility:

- **300 / 300**

---

### 2. Relation Substitution

The endpoint entities remain fixed while the predicate changes:

```text
(h, r, t) -> (h, r', t)
```

Replacement relations are restricted to the compatible person-valued family:

```text
directed_by
written_by
starred_actors
```

Final feasibility:

| Hop | Feasible |
|---|---:|
| 1-hop | 74 |
| 2-hop | 43 |
| 3-hop | 39 |
| **Total** | **156 / 300** |

---

### 3. Contradictory Insertion

The original support fact is retained and a type-compatible alternative using the same relation is inserted.

To keep evidence size fixed, one non-support context triple is replaced.

Final feasibility:

- **300 / 300**

This condition creates an explicit conflict while keeping the original valid support fact visible.

---

### 4. Intermediate-Node Rerouting

For a multi-hop fragment:

```text
A --r1--> B --r2--> C
```

the intermediate node is replaced:

```text
A --r1--> B' --r2--> C
```

Both adjacent support triples are rewritten while relation labels and traversal directions remain unchanged.

This operator is only defined for multi-hop questions.

Final feasibility:

| Hop | Feasible |
|---|---:|
| 1-hop | not applicable |
| 2-hop | 100 / 100 |
| 3-hop | 97 / 100 |
| **Total eligible multi-hop** | **197 / 200** |

---

## Entity Anonymization

Each question is also evaluated with deterministic anonymized entity labels.

Example:

```text
Daniel de Oliveira
```

may become:

```text
Entity_0042
```

For a given question ID, the same natural entity receives the same anonymous identifier across all clean and corrupted conditions.

Mappings are independent across different question IDs.

Anonymized elements include:

- topic entities
- gold answers
- evidence head/tail nodes
- intermediate nodes
- replacement entities
- literals such as years, genres, and languages

Preserved elements include:

- relation labels
- graph connectivity
- evidence order
- evidence size
- corruption structure

The anonymized condition is used as a **diagnostic**. It does not prove that all parametric knowledge has been removed from the model.

---

## Final Matched Evaluation Core

The final generator experiment uses only questions for which **all four corruption operators are feasible**.

Additional constraints:

- gold-answer set size <= 10
- evidence packet size <= 40 triples
- seed = 42

Eligible questions before final sampling:

- 2-hop: 39
- 3-hop: 14

Final matched core:

| Hop | Questions |
|---|---:|
| 2-hop | 14 |
| 3-hop | 14 |
| **Total** | **28** |

No 1-hop question is included because intermediate-node rerouting is undefined for 1-hop paths.

The same 28 question IDs are evaluated under:

- clean evidence
- entity substitution
- relation substitution
- contradiction
- rerouting

and under both:

- natural entity labels
- anonymized entity labels

Therefore:

```text
28 questions × 5 evidence conditions × 2 label modes = 280 generations
```

Final-core data:

```text
data/processed/final_core/
```

Manifest:

```text
data/processed/final_core/final_core_manifest.json
```

SHA-256 record:

```text
results/raw/final_core_sha256.txt
```

---

## Model and Inference

**Model:** `Qwen/Qwen3-4B`

> Note: the final reported experiment uses **Qwen3-4B**, not Qwen3-8B.

Configuration:

- NF4 4-bit quantization
- bitsandbytes
- double quantization enabled
- FP16 compute
- Qwen3 thinking mode disabled
- deterministic greedy decoding
- `do_sample=False`
- seed = 42
- `max_new_tokens=128`

Hardware used for the reported experiment:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- 4 GB VRAM

The model receives only the evidence packet and the fixed instruction.

Triples are shown as:

```text
head | relation | tail
```

The model is instructed to return:

- one answer
- semicolon-separated answers
- or exactly `UNKNOWN` when evidence is insufficient or conflicting

No corruption labels, evidence-role metadata, or gold answers are exposed to the model.

Final inference script:

```text
src/inference_final_core.py
```

Final raw generations:

```text
results/raw/final_core_inference/
```

Inference configuration:

```text
results/raw/final_core_inference/inference_config.json
```

### Inference integrity

- successful generations: **280 / 280**
- inference errors: **0**
- CUDA OOM failures: **0**
- generations reaching the 128-token cap: **5**

All five capped generations correspond to the same 3-hop question under anonymized labels across the five evidence conditions. They were retained rather than rerun.

---

## Evaluation

Evaluation script:

```text
src/eval.py
```

Primary metrics:

- set exact-match accuracy
- set F1
- paired accuracy degradation
- clean-correct flip rate
- `UNKNOWN` rate
- false-alternative adoption
- contradiction behavior

For corruption type \(c\):

```text
Delta Acc_c = Acc_clean - Acc_c
```

where:

- `Acc_clean` = exact-match accuracy on clean evidence
- `Acc_c` = exact-match accuracy under corruption `c`

To compare natural and anonymized conditions, the project uses the **Parametric Masking Gap (PMG)**:

```text
PMG_c =
(Acc_clean^anon - Acc_c^anon)
-
(Acc_clean^nat - Acc_c^nat)
```

A positive PMG would indicate a larger measured corruption effect after anonymization.

PMG is treated as a diagnostic, not as a direct measurement of the model's internal parametric memory.

Uncertainty estimates use:

- question-level paired bootstrap
- 10,000 resamples
- seed = 42
- 95% confidence intervals

---

## Main Results

### Condition-level performance

| Condition | Labels | Accuracy | Set F1 | UNKNOWN |
|---|---|---:|---:|---:|
| Clean | Natural | 50.0% | 82.5% | 7.1% |
| Clean | Anonymized | 46.4% | 75.2% | 7.1% |
| Entity substitution | Natural | 0.0% | 40.1% | 14.3% |
| Entity substitution | Anonymized | 0.0% | 39.5% | 7.1% |
| Relation substitution | Natural | 7.1% | 50.3% | 32.1% |
| Relation substitution | Anonymized | 3.6% | 49.2% | 28.6% |
| Contradiction | Natural | 3.6% | 55.7% | 17.9% |
| Contradiction | Anonymized | 7.1% | 65.9% | 3.6% |
| Rerouting | Natural | 35.7% | 68.7% | 10.7% |
| Rerouting | Anonymized | 42.9% | 66.5% | 7.1% |

### Paired exact-match degradation

| Corruption | Natural | Anonymized |
|---|---:|---:|
| Entity substitution | 50.0 pp | 46.4 pp |
| Relation substitution | 42.9 pp | 42.9 pp |
| Contradiction | 46.4 pp | 39.3 pp |
| Rerouting | 14.3 pp | 3.6 pp |

Entity substitution is the strongest tested intervention and reduces exact-match accuracy to **0% in both label modes**.

---

## Natural vs. Anonymized Labels

PMG estimates:

| Corruption | PMG | 95% CI |
|---|---:|---:|
| Entity substitution | -3.6 pp | [-17.9, 10.7] |
| Relation substitution | 0.0 pp | [-17.9, 17.9] |
| Contradiction | -7.1 pp | [-25.0, 10.7] |
| Rerouting | -10.7 pp | [-35.7, 14.3] |

All confidence intervals include zero.

The final matched experiment therefore does **not** provide statistically clear evidence that anonymization increases corruption sensitivity.

This is a limited empirical conclusion rather than a claim that entity familiarity or parametric knowledge never matters.

---

## Contradictory Evidence Behavior

Under contradictory evidence, the model rarely resolves the conflict by abstaining.

### Natural labels

- gold only: 4 / 28
- false only: 0 / 28
- both gold and injected alternative: 17 / 28
- `UNKNOWN`: 5 / 28
- other: 2 / 28

### Anonymized labels

- gold only: 3 / 28
- false only: 0 / 28
- both gold and injected alternative: 21 / 28
- `UNKNOWN`: 1 / 28
- other: 3 / 28

The dominant behavior is therefore to return **both alternatives**, rather than exclusively choosing the injected false value or abstaining.

---

## Repository Structure

```text
when-kgs-lie/
├── README.md
├── requirements.txt
├── build_final_core.py
│
├── configs/
│
├── data/
│   └── processed/
│       └── final_core/
│           ├── final_core_manifest.json
│           └── ...
│
├── paper/
│   └── research_log.md
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── raw/
│       ├── final_core_inference/
│       └── final_core_sha256.txt
│
└── src/
    ├── eval.py
    ├── inference_final_core.py
    ├── preflight_inference.py
    ├── plot_extra_results.py
    ├── plot_contradiction_clean.py
    └── ...
```

---

## Reproducing the Reported Experiment

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

The committed `data/processed/final_core/` directory contains the final matched evaluation core used in the paper.

Before inference, run the preflight checks:

```powershell
python src/preflight_inference.py
```

Run final-core inference:

```powershell
python src/inference_final_core.py
```

Evaluate the resulting generations:

```powershell
python src/eval.py
```

Generate additional result figures:

```powershell
python src/plot_extra_results.py
python src/plot_contradiction_clean.py
```

The evaluation outputs are written under:

```text
results/tables/
```

and figures under:

```text
results/figures/
```

---

## Important Reproducibility Note

The directory:

```text
results/raw/inference/
```

contains an **older interrupted full-pool inference run** retained only for audit purposes.

It must **not** be mixed with the final reported results.

The reported paper results use only:

```text
results/raw/final_core_inference/
```

with the matched 28-question core.

---

## Main Result Files

Tables:

```text
results/tables/condition_summary.csv
results/tables/paired_effects.csv
results/tables/bootstrap_effects.csv
results/tables/hop_effects.csv
results/tables/pmg.csv
results/tables/contradiction_breakdown.csv
results/tables/inference_completeness.csv
results/tables/item_level_metrics.csv
```

Figures:

```text
results/figures/main_robustness.png
results/figures/contradiction_behavior.png
results/figures/hop_degradation.png
results/figures/pmg_with_ci.png
```

---

## Limitations

The reported results should be interpreted within the scope of the experiment:

- MetaQA is a synthetic movie-domain benchmark
- the final generator study contains 28 matched multi-hop questions
- only one 4B model is tested
- the model is evaluated under 4-bit quantization
- corruptions are synthetic
- anonymization removes entity surface forms but leaves relation labels and question templates visible
- a correct answer after corruption does not necessarily imply faithful use of the displayed reasoning path
- `UNKNOWN` behavior is instruction-conditioned because abstention is explicitly allowed by the prompt

---

## Summary

The experiment shows that clean-answer accuracy alone is not sufficient evidence of faithful graph grounding.

In the matched Qwen3-4B evaluation:

- entity substitution is the most damaging corruption
- relation substitution and contradiction also produce large accuracy losses
- intermediate-node rerouting is less damaging in aggregate
- anonymization does not reveal a statistically clear increase in corruption sensitivity
- contradictory evidence frequently leads the model to return both valid and injected alternatives rather than abstain

The repository preserves the processed final core, model outputs, evaluation code, result tables, and figures used to support these conclusions.

---

## Research Log

A detailed chronological record of dataset construction, semantic audits, corruption generation, final-core selection, inference, evaluation, and reporting decisions is available at:

```text
paper/research_log.md
```

---

## Author

**Fidan Allahverdiyeva**  
French-Azerbaijani University (UFAZ)

Project repository:

https://github.com/Fidan200508/when-kgs-lie
