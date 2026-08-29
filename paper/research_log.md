# Research Log

## Frozen clean benchmark

Dataset: MetaQA
Pilot size: 300 questions
1-hop: 100
2-hop: 100
3-hop: 100
Context triples per question: 5
Seed: 42

Clean evidence file:
data/processed/metaqa_pilot_clean.jsonl

Support paths:
Question-semantically aligned exact-hop KG walks.

## Entity substitution

Status: Complete

Successful corruptions: 300/300
Failures: 0

Primary position:
answer-adjacent

Replacement:
relation-compatible entity pool

Validation:
- evidence size unchanged
- only one support triple changed
- relation unchanged
- corrupted triple absent from clean KG
- deterministic seed

Manual audit:
20 random examples inspected; no obvious type-invalid corruptions observed.




## Relation substitution

Status: Complete

Eligible / successful: 152 / 300
Infeasible: 148

Hop distribution among feasible cases:
- 1-hop: 52
- 2-hop: 57
- 3-hop: 43

Failure reasons:
- no_eligible_support_relation: 145
- no_valid_relation_replacement: 3

Compatibility rule:
Relation substitution was restricted to the person-valued relation family:
directed_by, written_by, starred_actors.

The head and tail entities were kept fixed and only the predicate was changed.
Corrupted triples already present in the original KG were rejected.

Primary corruption position:
answer-adjacent

Seed:
42


## Relation substitution manual audit

Manual audit size: 20 randomly sampled corruptions
Seed: 42

Result:
20/20 inspected examples preserved the head and tail entities and changed
only the relation within the compatible person-valued relation family
(directed_by, written_by, starred_actors).

No obvious schema-invalid relation substitutions were observed.



## Contradictory insertion

Status: Complete

Successful corruptions: 300/300
Failures: 0

Hop distribution:
- 1-hop: 100
- 2-hop: 100
- 3-hop: 100

Primary position:
answer-adjacent

Method:
The original true support triple was retained.
A type-compatible false triple with the same relation was inserted.
To preserve the evidence budget, one non-support context triple was
deterministically replaced by the contradictory triple.

Evidence size:
unchanged between clean and contradiction conditions.

Seed:
42

Validation:
- true support remains present
- exactly one context slot is replaced
- relation remains unchanged
- contradictory triple is absent from the original KG
- false replacement is type-compatible


## Contradictory insertion manual audit

Manual audit size: 20 randomly sampled examples
Seed: 42

Result:
20 inspected examples showed a valid conflict structure:
the original true support fact was retained and a type-compatible false
fact using the same relation was introduced.

The evidence budget was preserved by replacing one non-support context
triple with the contradictory triple.

No obvious type-invalid contradictory fact was observed.




## Intermediate-node rerouting

Status: Complete

Successful reroutings: 198 / 300
Failures: 102

Successful hop distribution:
- 1-hop: 0
- 2-hop: 100
- 3-hop: 98

Failure reasons:
- no_intermediate_node: 100
- no_valid_distinct_edge_target: 2

Feasibility among multi-hop examples:
198 / 200 = 99%

Method:
For a multi-hop support path A -> B -> C, the intermediate node B was
replaced with a structurally compatible node B', producing A -> B' -> C.

Both adjacent support triples were changed while preserving their relation
labels. Replacement nodes were restricted to entities observed in the same
structural relation slots.

Both newly generated triples were required to be absent from the original KG.

Evidence size and evidence order were preserved.

Seed:
42



## Semantic support-path audit

During manual inspection of rerouting examples, several clean canonical
support paths were found to reach the correct answer through semantically
misaligned KG walks.

Examples included paths using written_by where the question required
directed_by, and repeated release_year relations where the question required
a shared-director chain.

Cause:
Exact-hop KG walks were ranked with a soft semantic score, which did not
strictly enforce the relation pattern implied by the question.

Decision:
Model inference was paused. Clean evidence selection will be corrected to
prefer/require question-consistent relation signatures before regenerating
all corruption conditions.




## Final semantic-clean MetaQA pilot

Status: Frozen

Pilot size: 300
- 1-hop: 100
- 2-hop: 100
- 3-hop: 100

Seed: 42
Context triples per item: 5

KG:
- triples: 134,741
- entities in bidirectional adjacency: 43,234

Support-path construction:
- exact-hop KG walks
- node repetition permitted where required by MetaQA
- exact question-derived relation profile required
- final support relation constrained to the inferred answer relation
- one deterministic canonical path selected per gold answer

Validation:
- semantic violations: 0
- competing clean-context answer violations: 0
- context shortages: 0
- pilot size check: PASS
- semantic constraint check: PASS
- clean answer-conflict check: PASS
- fixed context check: PASS

Average support triples: 10.72

Target-relation inference coverage on full MetaQA test split:
- 1-hop: 72.99%
- 2-hop: 100.00%
- 3-hop: 100.00%

Only semantically valid questions were eligible for the 100-per-hop pilot.




