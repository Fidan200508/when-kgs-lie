# Research Log

## Project

**Title:** When Knowledge Graphs Lie: Controlled Robustness Evaluation of KG-Grounded Large Language Model Question Answering

**Dataset:** MetaQA  
**Model:** Qwen/Qwen3-4B  
**Primary seed:** 42  
**Repository:** https://github.com/Fidan200508/when-kgs-lie

### Research questions

- **RQ1:** How do semantic and structural graph corruptions affect KG-grounded QA, and how does the effect vary with hop length?
- **RQ2:** To what extent does entity anonymization change the model's sensitivity to evidence corruption?
- **RQ3:** When the supplied triples are misleading or contradictory, does the generator abstain, retain valid targets, or adopt the injected alternative?

### Core experimental principle

Retrieval/evidence selection is frozen before corruption. For a given question, the question, gold-answer set, prompt template, model, evidence format, and decoding configuration are held fixed. Only the evidence packet is changed.

This isolates the generator's response to the evidence intervention rather than mixing corruption effects with retrieval changes.

---

## 1. Initial clean benchmark — superseded

> Historical development stage. This version was not used for the final reported experiment. It was superseded after the semantic support-path audit described below.

Dataset: MetaQA

Initial pilot size: 300 questions

- 1-hop: 100
- 2-hop: 100
- 3-hop: 100

Initial context budget: 5 context triples per question

Seed: 42

Initial clean evidence file:

`data/processed/metaqa_pilot_clean.jsonl`

Initial support-path strategy:

- exact-hop KG walks
- soft semantic ranking
- deterministic path selection

This first version was sufficient for pipeline development, but later manual inspection showed that exact-hop walks could still be semantically inconsistent with the relation pattern implied by the question.

---

## 2. Initial corruption development — superseded

> The following early corruption counts are kept only as an audit trail. They were regenerated after the clean evidence construction was corrected, so they are not the final feasibility numbers used in the paper.

### Entity substitution — initial version

Status: completed during early pipeline development

- successful corruptions: 300/300
- failures: 0

Primary position: answer-adjacent

Method:

- change one answer-adjacent support triple
- keep the relation unchanged
- replace the endpoint with an entity drawn from a relation-compatible pool
- reject a generated triple if it already exists in the reference KG
- preserve evidence size

Manual audit:

- 20 random examples inspected
- no obvious type-invalid corruptions observed

### Relation substitution — initial version

Status: completed during early pipeline development

Early feasibility:

- successful: 152/300
- infeasible: 148

Early feasible hop distribution:

- 1-hop: 52
- 2-hop: 57
- 3-hop: 43

Early failure reasons:

- `no_eligible_support_relation`: 145
- `no_valid_relation_replacement`: 3

Compatibility rule:

Only the relation was changed. Head and tail entities remained fixed. Replacement relations were restricted to the person-valued family:

- `directed_by`
- `written_by`
- `starred_actors`

Synthetic triples already present in the reference KG were rejected.

Manual audit:

- 20 randomly sampled corruptions
- 20/20 preserved the head and tail entities
- only the relation changed
- no obvious schema-invalid substitutions observed

### Contradictory insertion — initial version

Status: completed during early pipeline development

- successful corruptions: 300/300
- failures: 0

Method:

- retain the original true support fact
- insert a type-compatible alternative using the same relation
- replace one non-support context triple so the evidence count does not increase
- require the inserted triple to be absent from the reference KG

Manual audit:

- 20 randomly sampled examples
- each inspected item contained both the original support fact and a type-compatible conflicting alternative
- no obvious type-invalid contradictory fact observed

### Intermediate-node rerouting — initial version

Status: completed during early pipeline development

Early feasibility:

- successful reroutings: 198/300
- 1-hop: not applicable
- 2-hop: 100
- 3-hop: 98

Early failure reasons:

- `no_intermediate_node`: 100
- `no_valid_distinct_edge_target`: 2

Method:

For a multi-hop fragment

`A --r1--> B --r2--> C`

replace the intermediate node `B` with a structurally compatible `B'`, producing

`A --r1--> B' --r2--> C`.

Both adjacent support triples are rewritten while relation labels and traversal directions are preserved. Both newly generated triples must be absent from the reference KG.

---

## 3. Semantic support-path audit

Manual inspection of rerouting examples revealed a more important issue in the clean benchmark itself.

Several exact-hop support walks reached the benchmark answer through relation sequences that did not match the semantics of the question. Examples included:

- paths using `written_by` where the question required `directed_by`
- repeated `release_year` relations where the question required a shared-director chain

Cause:

The early evidence builder ranked exact-hop KG walks with a soft semantic score rather than strictly enforcing the relation pattern implied by each MetaQA question template.

Decision:

- pause model inference
- correct clean evidence construction
- require question-consistent relation signatures
- regenerate all corruption conditions from the corrected clean benchmark
- treat all earlier corruption counts as historical only

---

## 4. Final semantic-clean MetaQA pool

Status: **Frozen**

Validated pool size: 300 questions

- 1-hop: 100
- 2-hop: 100
- 3-hop: 100

Seed: 42

Reference KG:

- triples: 134,741
- nodes in bidirectional traversal index: 43,234

### Support-path construction

For each MetaQA question:

1. infer the relation profile implied by the question template
2. search the reference KG for an exact-hop walk matching that profile
3. allow repeated nodes where required by valid MetaQA reasoning patterns
4. constrain the final support relation to the inferred answer relation
5. retain one deterministic canonical support path per gold answer
6. merge the resulting support triples
7. add five deterministic context triples

### Packet-level validation

Support-path validity alone is not enough because context triples or recombinations of valid support paths can introduce unintended answers.

For the complete evidence packet `E`, the item is retained only if the answers reachable under the intended relation profile are exactly equal to the benchmark gold-answer set.

Validation outcome:

- semantic support-path failures in the frozen pool: 0
- packet-level answer-set failures in the frozen pool: 0
- context shortages: 0
- fixed-size pilot check: PASS
- semantic constraint check: PASS
- clean answer-conflict check: PASS

During candidate filtering, packet-level ambiguity caused rejection of:

- 54 candidate 2-hop questions
- 15 candidate 3-hop questions

Sampling continued deterministically until 100 valid questions were obtained for each hop length.

Average number of support triples in the final clean pool: 10.72

Target-relation inference coverage on the full MetaQA test split:

- 1-hop: 72.99%
- 2-hop: 100.00%
- 3-hop: 100.00%

Only semantically valid questions were eligible for the frozen 100-per-hop pool.

---

## 5. Final corruption operators and feasibility

All final corruptions were regenerated from the frozen semantic-clean 300-question pool.

We represent a KG fact as `(h, r, t)`, where:

- `h` = head entity
- `r` = relation
- `t` = tail entity

A synthetic triple is called "false" only with respect to the reference MetaQA KG. Non-membership in MetaQA is not a claim about real-world falsity.

### 5.1 Entity substitution

Status: **Complete**

Feasible:

- total: 300/300
- 1-hop: 100
- 2-hop: 100
- 3-hop: 100

Transformation:

`(h, r, t) -> (h, r, t')`

Rules:

- use an answer-adjacent support triple
- keep `h` and `r` fixed
- draw `t'` from the same relation-endpoint pool
- require `(h, r, t')` to be absent from the reference KG
- preserve evidence size and order
- deterministic seed: 42

### 5.2 Relation substitution

Status: **Complete**

Final feasibility:

- total: 156/300
- 1-hop: 74
- 2-hop: 43
- 3-hop: 39

Transformation:

`(h, r, t) -> (h, r', t)`

Rules:

- keep the endpoint entities fixed
- change only the predicate
- restrict `r'` to the compatible person-valued relation family:
  - `directed_by`
  - `written_by`
  - `starred_actors`
- require the new triple to be absent from the reference KG
- deterministic seed: 42

The lower feasibility is intentional. The generator rejects substitutions that cannot preserve the intended type/schema constraints.

### 5.3 Contradictory insertion

Status: **Complete**

Feasible:

- total: 300/300
- 1-hop: 100
- 2-hop: 100
- 3-hop: 100

Method:

- retain the original answer-adjacent support fact
- insert a type-compatible alternative with the same relation
- replace one non-support context triple so the evidence count remains unchanged
- require the inserted alternative to be absent from the reference KG
- deterministic seed: 42

This condition creates an explicit conflict while keeping the original true support fact visible.

### 5.4 Intermediate-node rerouting

Status: **Complete**

This operator is defined only for multi-hop questions.

Final feasibility among eligible multi-hop questions:

- total: 197/200
- 2-hop: 100/100
- 3-hop: 97/100
- 1-hop: not applicable

Transformation:

`A --r1--> B --r2--> C`

becomes

`A --r1--> B' --r2--> C`

Rules:

- replace the intermediate node with a structurally compatible alternative
- preserve both relation labels
- preserve traversal directions
- rewrite both adjacent support triples
- require both rewritten triples to be absent from the reference KG
- preserve evidence size and order
- deterministic seed: 42

---

## 6. Entity-label anonymization

Status: **Complete**

Purpose:

Create a paired diagnostic condition in which recognizable node surface forms are removed while graph structure and relation labels remain unchanged.

For each question ID, a deterministic mapping is constructed from the union of entities appearing across the clean item and all feasible corrupted variants for that same question.

Entity representation:

`Entity_XXXX`

The same natural entity receives the same anonymous identifier across all conditions for one question ID. Mappings are independent across different question IDs.

Anonymized:

- topic entities
- gold answers
- evidence head/tail nodes
- intermediate nodes
- replacement entities
- literal KG nodes such as years, languages, genres, and tags

Preserved:

- relation labels
- graph connectivity
- traversal structure
- evidence order
- evidence size
- corruption structure
- question semantics except for the topic entity surface form

Validation:

- clean: 300/300
- entity substitution: 300/300
- relation substitution: 156/156
- contradiction: 300/300
- intermediate-node rerouting: 197/197
- total anonymization validation failures: 0

Important:

Internal evidence-role metadata such as support/context/contradiction is never shown to the LLM.

Anonymization is treated as a diagnostic rather than proof that all parametric knowledge has been removed.

---

## 7. Final matched evaluation core

Status: **Frozen**

The final generator experiment uses a matched subset so that every selected question is available under all four corruption operators and both label modes.

Selection occurs **before model inference**.

Additional constraints:

- number of gold answers <= 10
- evidence packet size <= 40 triples
- deterministic seed: 42

Eligible common questions before final sampling:

- 2-hop: 39
- 3-hop: 14

Final matched core:

- total: 28 questions
- 2-hop: 14
- 3-hop: 14

No 1-hop questions are included because intermediate-node rerouting is undefined for 1-hop paths.

The same 28 question IDs are used for:

- clean
- entity substitution
- relation substitution
- contradictory insertion
- intermediate-node rerouting

and for both:

- natural entity labels
- anonymized entity labels

Total matched conditions:

- 5 evidence conditions
- 2 label modes
- 28 questions per condition
- 280 final generations

Final-core directory:

`data/processed/final_core/`

Manifest:

`data/processed/final_core/final_core_manifest.json`

SHA-256 record:

`results/raw/final_core_sha256.txt`

---

## 8. Model and final inference protocol

Model:

`Qwen/Qwen3-4B`

Hardware:

- NVIDIA GeForce RTX 3050 Ti Laptop GPU
- 4 GB VRAM

Model loading / quantization:

- bitsandbytes
- NF4 4-bit quantization
- double quantization enabled
- FP16 compute

Decoding:

- Qwen3 thinking mode disabled
- deterministic greedy decoding
- `do_sample=False`
- seed: 42
- `max_new_tokens=128`

Prompt policy:

- answer only from the supplied evidence
- return only answer values separated by semicolons
- return exactly `UNKNOWN` when the evidence is insufficient or conflicting
- no explanation
- no use of outside knowledge
- corruption metadata, evidence roles, and gold answers are hidden

Evidence display format:

`head | relation | tail`

Final inference script:

`src/inference_final_core.py`

Final raw outputs:

`results/raw/final_core_inference/`

Inference configuration:

`results/raw/final_core_inference/inference_config.json`

### Final inference outcome

- total expected generations: 280
- successful generations: 280/280
- inference errors: 0
- CUDA OOM failures: 0
- failed generations rerun: 0

Five outputs reached the 128-token cap.

All five correspond to the same question:

`3hop_test_10307`

under anonymized labels across all five evidence conditions.

These outputs were retained as observed behavior and were not rerun.

### Historical interrupted run

An earlier full-pool inference attempt was interrupted and is **not used for final metrics**.

Historical audit directory:

`results/raw/inference/`

It contains only:

- clean natural: 300
- clean anonymized: 262

That run used an older token cap and must never be mixed with the final matched-core results.

---

## 9. Evaluation protocol

Evaluation script:

`src/eval.py`

### Answer parsing

- raw model generations are preserved unchanged
- Unicode normalization and trimming are applied
- natural labels are compared case-insensitively
- multiple answers are split only on semicolons
- abstention is counted only when the full normalized response is exactly `UNKNOWN`
- explanatory or mixed outputs are not repaired

### Primary metrics

**Set exact-match accuracy (`Acc`)**

A prediction is correct only when the predicted answer set exactly matches the benchmark gold-answer set.

Notation:

- `Acc_clean` = exact-match accuracy on clean evidence
- `Acc_c` = exact-match accuracy under corruption type `c`
- superscript `nat` = natural-label condition
- superscript `anon` = anonymized-label condition

Paired degradation for corruption `c`:

`Delta Acc_c = Acc_clean - Acc_c`

A positive value means accuracy decreased after corruption.

**Set F1**

Provides partial credit for multi-answer predictions when the predicted and gold sets overlap but are not identical.

**Flip rate**

Among questions answered correctly in the paired clean condition, the fraction whose exact-match correctness is lost after corruption.

**UNKNOWN rate**

Fraction of outputs that are exactly `UNKNOWN`.

**False-alternative adoption**

For corruption operators with a specific injected alternative, whether that injected value appears in the model output.

### Parametric Masking Gap (PMG)

For corruption `c`:

`PMG_c = (Acc_clean^anon - Acc_c^anon) - (Acc_clean^nat - Acc_c^nat)`

Interpretation:

- positive PMG: corruption causes a larger drop after anonymization
- zero PMG: no difference in measured degradation
- negative PMG: measured degradation is smaller after anonymization

PMG is a diagnostic comparison, not a direct measurement of internal parametric memory.

### Uncertainty estimation

- paired question-level bootstrap
- 10,000 resamples
- seed: 42
- 95% confidence intervals

---

## 10. Final condition-level results

| Condition | Labels | N | Accuracy | Set F1 | UNKNOWN |
|---|---|---:|---:|---:|---:|
| Clean | Natural | 28 | 50.0% | 82.5% | 7.1% |
| Clean | Anonymized | 28 | 46.4% | 75.2% | 7.1% |
| Entity substitution | Natural | 28 | 0.0% | 40.1% | 14.3% |
| Entity substitution | Anonymized | 28 | 0.0% | 39.5% | 7.1% |
| Relation substitution | Natural | 28 | 7.1% | 50.3% | 32.1% |
| Relation substitution | Anonymized | 28 | 3.6% | 49.2% | 28.6% |
| Contradiction | Natural | 28 | 3.6% | 55.7% | 17.9% |
| Contradiction | Anonymized | 28 | 7.1% | 65.9% | 3.6% |
| Rerouting | Natural | 28 | 35.7% | 68.7% | 10.7% |
| Rerouting | Anonymized | 28 | 42.9% | 66.5% | 7.1% |

---

## 11. Paired robustness results

### Natural labels

- entity substitution:
  - clean accuracy: 50.0%
  - corrupted accuracy: 0.0%
  - degradation: 50.0 percentage points
  - 95% CI: [32.1, 67.9]
  - flip rate: 100.0%

- relation substitution:
  - clean accuracy: 50.0%
  - corrupted accuracy: 7.1%
  - degradation: 42.9 percentage points
  - 95% CI: [25.0, 60.7]
  - flip rate: 85.7%

- contradiction:
  - clean accuracy: 50.0%
  - corrupted accuracy: 3.6%
  - degradation: 46.4 percentage points
  - 95% CI: [28.6, 64.3]
  - flip rate: 92.9%

- rerouting:
  - clean accuracy: 50.0%
  - corrupted accuracy: 35.7%
  - degradation: 14.3 percentage points
  - 95% CI: [0.0, 32.1]
  - flip rate: 35.7%

### Anonymized labels

- entity substitution:
  - clean accuracy: 46.4%
  - corrupted accuracy: 0.0%
  - degradation: 46.4 percentage points
  - 95% CI: [28.6, 64.3]
  - flip rate: 100.0%

- relation substitution:
  - clean accuracy: 46.4%
  - corrupted accuracy: 3.6%
  - degradation: 42.9 percentage points
  - 95% CI: [25.0, 60.7]
  - flip rate: 92.3%

- contradiction:
  - clean accuracy: 46.4%
  - corrupted accuracy: 7.1%
  - degradation: 39.3 percentage points
  - 95% CI: [17.9, 60.7]
  - flip rate: 92.3%

- rerouting:
  - clean accuracy: 46.4%
  - corrupted accuracy: 42.9%
  - degradation: 3.6 percentage points
  - 95% CI: [-14.3, 21.4]
  - flip rate: 30.8%

Main RQ1 observation:

Local semantic corruptions are much more damaging than intermediate-node rerouting on this matched core. Entity substitution is the strongest intervention and reduces exact-match accuracy to zero in both label modes.

A correct answer after rerouting must not be interpreted as proof that the model faithfully followed the rerouted path; the model may still recover the benchmark endpoint from other cues.

---

## 12. Hop-length analysis

Natural-label degradation:

| Corruption | 2-hop | 3-hop |
|---|---:|---:|
| Entity substitution | 71.4 pp | 28.6 pp |
| Relation substitution | 71.4 pp | 14.3 pp |
| Contradiction | 71.4 pp | 21.4 pp |
| Rerouting | 28.6 pp | 0.0 pp |

Anonymized-label degradation:

| Corruption | 2-hop | 3-hop |
|---|---:|---:|
| Entity substitution | 71.4 pp | 21.4 pp |
| Relation substitution | 71.4 pp | 14.3 pp |
| Contradiction | 64.3 pp | 14.3 pp |
| Rerouting | 14.3 pp | -7.1 pp |

Important interpretation:

These numbers are **not evidence that deeper paths are intrinsically more robust**.

Clean accuracy already differs substantially by hop length:

- natural 2-hop clean accuracy: 71.4%
- natural 3-hop clean accuracy: 28.6%
- anonymized 2-hop clean accuracy: 71.4%
- anonymized 3-hop clean accuracy: 21.4%

The three-hop subset therefore has much less exact-match headroom available for further decline.

---

## 13. Natural-versus-anonymized diagnostic

PMG estimates:

- entity substitution: -3.6 pp
  - 95% CI: [-17.9, 10.7]

- relation substitution: 0.0 pp
  - 95% CI: [-17.9, 17.9]

- contradiction: -7.1 pp
  - 95% CI: [-25.0, 10.7]

- rerouting: -10.7 pp
  - 95% CI: [-35.7, 14.3]

All PMG point estimates are non-positive and every confidence interval includes zero.

Final RQ2 interpretation:

The matched experiment does **not** provide statistically clear evidence that anonymization increases corruption sensitivity.

This should not be converted into a stronger claim that entity familiarity or parametric knowledge never matters. The sample is small, relation labels and natural-language templates remain visible, and anonymization removes surface forms rather than internal model knowledge.

---

## 14. Reliability and false-evidence adoption

### Entity substitution

Natural labels:

- false-alternative adoption: 82.1%
- at least one gold answer retained: 60.7%
- all gold answers retained: 0.0%
- UNKNOWN: 14.3%

Anonymized labels:

- false-alternative adoption: 78.6%
- at least one gold answer retained: 67.9%
- all gold answers retained: 0.0%
- UNKNOWN: 7.1%

Interpretation:

Entity substitution often causes the model to follow the injected replacement value.

### Contradictory insertion

Natural labels:

- gold only: 4/28
- injected false only: 0/28
- both gold and injected alternative: 17/28
- UNKNOWN: 5/28
- other: 2/28

Anonymized labels:

- gold only: 3/28
- injected false only: 0/28
- both gold and injected alternative: 21/28
- UNKNOWN: 1/28
- other: 3/28

Equivalent percentages for the dominant "both" behavior:

- natural: 60.7%
- anonymized: 75.0%

False-only adoption:

- natural: 0%
- anonymized: 0%

Final RQ3 interpretation:

Under contradiction, the dominant failure mode is not exclusive adoption of the injected false answer. The model usually returns **both** the benchmark answer and the injected alternative, while explicit abstention remains uncommon.

---

## 15. Output integrity

Final-core generations evaluated: 280

- successful generations: 280
- inference errors: 0
- CUDA OOM failures: 0
- MAXTOK outputs: 5
- conservative protocol violations: 7/280 = 2.5%

The five token-capped outputs all correspond to:

`3hop_test_10307`

under anonymized labels across all five evidence conditions.

The remaining two protocol violations are natural-label responses that mix a normal answer with `UNKNOWN`.

No failed generation was rerun and no output was manually repaired.

---

## 16. Qualitative audit examples

### Example A — entity substitution and contradiction

Question ID:

`2hop_test_11457`

Clean evidence supports:

`Daniel de Oliveira`

Entity substitution changes an answer-adjacent triple from:

`Boca --starred_actors--> Daniel de Oliveira`

to the same relation ending in:

`Kirina Mano`

Observed behavior:

- under entity substitution, the model follows the injected entity
- under contradiction, both the true and injected alternatives are visible
- the model returns both alternatives rather than `UNKNOWN`
- this behavior occurs in both label modes

Interpretation:

The case shows direct susceptibility to a local evidence edit and illustrates why contradiction behavior cannot be summarized by exact-match accuracy alone.

### Example B — intermediate-node rerouting

Question ID:

`3hop_test_438`

Rerouting replaces the intermediate movie:

`Bright Leaves`

with:

`The Page Turner`

in the two adjacent support triples.

Observed behavior:

The model still returns the benchmark answer:

`Ross McElwee`

in both natural and anonymized label modes.

Interpretation:

A correct final answer after a structural intervention does not by itself prove faithful use of the displayed reasoning path.

---

## 17. Main output files

### Final data

`data/processed/final_core/`

`data/processed/final_core/final_core_manifest.json`

### Final inference

`results/raw/final_core_inference/`

`results/raw/final_core_inference/inference_config.json`

`results/raw/final_core_sha256.txt`

### Evaluation tables

`results/tables/item_level_metrics.csv`

`results/tables/condition_summary.csv`

`results/tables/paired_effects.csv`

`results/tables/bootstrap_effects.csv`

`results/tables/hop_effects.csv`

`results/tables/pmg.csv`

`results/tables/contradiction_breakdown.csv`

`results/tables/inference_completeness.csv`

### Figures

`results/figures/main_robustness.png`

`results/figures/hop_degradation.png`

`results/figures/contradiction_behavior.png`

`results/figures/pmg_with_ci.png`

### Main scripts

`build_final_core.py`

`src/inference_final_core.py`

`src/eval.py`

`src/plot_extra_results.py`

`src/plot_contradiction_clean.py`

`src/preflight_inference.py`

### Historical run retained only for audit

`results/raw/inference/`

Do not use the historical interrupted run for final metrics.

---

## 18. Final reporting decisions

The paper reports the matched 28-question experiment rather than mixing different feasible subsets across corruption operators.

Main paper emphasis:

1. **RQ1:** paired exact-match degradation and flip behavior across corruption types
2. **RQ2:** natural-versus-anonymized PMG with paired-bootstrap confidence intervals
3. **RQ3:** abstention, false-alternative adoption, and contradiction-output categories

Main figures:

- corruption-operator illustration
- paired robustness degradation with 95% confidence intervals
- contradiction behavior

Hop-length and PMG plots are retained as research artifacts but are not required as main-paper figures because:

- hop-level degradation is strongly affected by different clean baselines
- PMG values and confidence intervals can be reported more directly in text

---

## 19. Final conclusions recorded for the project

The final experiment supports the following limited claims:

- KG-grounded QA can be highly sensitive to local semantic corruption even when retrieval is held fixed.
- Entity substitution is the strongest tested intervention on the matched core, reducing exact-match accuracy to 0% in both natural and anonymized conditions.
- Relation substitution and contradiction also produce large paired losses.
- Intermediate-node rerouting is less damaging in aggregate, but retained correctness does not establish faithful path use.
- Anonymization does not reveal a statistically clear increase in corruption sensitivity in this matched sample; every PMG confidence interval includes zero.
- Under contradiction, the model usually returns both the valid and injected alternatives rather than abstaining or selecting only the false alternative.
- Clean-answer accuracy alone is therefore insufficient to establish faithful graph grounding.

These conclusions are restricted to the validated MetaQA subset, the 28-question matched core, the tested Qwen3-4B configuration, and the specified corruption protocol.
