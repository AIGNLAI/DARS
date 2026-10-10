# Reported DARS results

The values below are transcribed from Table 1 of [arXiv:2606.06924v2](https://arxiv.org/html/2606.06924v2#S5.T1), revised September 4, 2026. They are the paper's reported results, not outputs of a new local run. [CSV](paper_results.csv) contains all 42 comparisons.

Each cell shows **single-shot → DARS**. Utilities use λ = 0.05; higher is better. Single-shot results average 100 independently sampled training sets. DARS changes supervision while keeping each router's architecture and training protocol fixed. Rewrite utility evaluates three rewritten-query observations; decoding utility evaluates three decoding-strategy observations on the original query.

| Router | DROP rewrite | DROP decoding | GPQA rewrite | GPQA decoding | MATH rewrite | MATH decoding |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| MLP | 0.661 → 0.675 | 0.709 → 0.726 | 0.382 → 0.399 | 0.400 → 0.413 | 0.682 → 0.687 | 0.698 → 0.696 |
| kNN | 0.713 → 0.749 | 0.759 → 0.791 | 0.441 → 0.461 | 0.469 → 0.489 | 0.720 → 0.730 | 0.733 → 0.742 |
| EmbedLLM | 0.739 → 0.763 | 0.785 → 0.805 | 0.469 → 0.473 | 0.496 → 0.480 | 0.706 → 0.719 | 0.728 → 0.747 |
| AvengersPro | 0.731 → 0.762 | 0.775 → 0.807 | 0.464 → 0.480 | 0.490 → 0.496 | 0.720 → 0.734 | 0.736 → 0.742 |
| GraphRouter | 0.762 → 0.765 | 0.805 → 0.807 | 0.462 → 0.496 | 0.487 → 0.516 | 0.746 → 0.757 | 0.755 → 0.758 |
| MIRT | 0.687 → 0.705 | 0.734 → 0.751 | 0.409 → 0.444 | 0.433 → 0.468 | 0.697 → 0.759 | 0.714 → 0.760 |
| RM-Softmax | 0.740 → 0.761 | 0.787 → 0.804 | 0.453 → 0.477 | 0.475 → 0.512 | 0.721 → 0.731 | 0.746 → 0.749 |

DARS improves 40 of 42 comparisons. The exceptions are MLP on MATH-500 decoding (0.698 → 0.696) and EmbedLLM on GPQA decoding (0.496 → 0.480).

## Reproduction

Use the [released scored generations](https://huggingface.co/datasets/AIGNLAI/DARS) and the [repository commands](../README.md#reproduce-the-routing-experiments). Match the encoder, hyperparameters, seeds, and observation protocol in [Appendix C.1](https://arxiv.org/html/2606.06924v2#A3.SS1) for a comparison with paper numbers. The TF-IDF quick start verifies the pipeline and can produce different utility values.

The paper also reports label-instability diagnostics (Section 3.3), component ablations (Table 2), sampling and risk sensitivity (Figure 4), and cost–quality curves (Figure 5). These analyses should be cited with their particular setting rather than generalized to every router or deployment.
