# When Knowledge Graphs Lie

Controlled robustness evaluation of KG-grounded LLM question answering.

## Research Questions

- How do semantic and structural KG corruptions affect factual accuracy?
- Does parametric memory mask corrupted KG evidence?
- Can LLMs detect conflicting evidence and abstain?

## Primary Setup

- Dataset: MetaQA
- Model: Qwen3-8B
- Hops: 1-hop, 2-hop, 3-hop
- Labels: Natural / Anonymized
- Core corruptions:
  - Entity substitution
  - Relation substitution
  - Contradiction
  - Intermediate-node rerouting