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






