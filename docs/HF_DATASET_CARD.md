---
pretty_name: 'DARS: Distribution-Aware Routing Supervision'
license: other
task_categories:
- question-answering
language:
- en
size_categories:
- 100K<n<1M
tags:
- arxiv:2606.06924
- llm-routing
- model-routing
- uncertainty
- distribution-aware-supervision
- risk-aware-routing
- emnlp-2026
- gpqa
- math
- drop
configs:
- config_name: gpqa
  data_files:
  - split: train
    path: gpqa/train_scored_generations.jsonl
  - split: test
    path: gpqa/test_scored_generations.jsonl
- config_name: math-500
  data_files:
  - split: train
    path: math-500/train_scored_generations.jsonl
  - split: test
    path: math-500/test_scored_generations.jsonl
- config_name: drop-800
  data_files:
  - split: train
    path: drop-800/train_scored_generations.jsonl
  - split: test
    path: drop-800/test_scored_generations.jsonl
dataset_info:
- config_name: gpqa
  features:
  - name: generation_key
    dtype: string
  - name: dataset
    dtype: string
  - name: query_id
    dtype: string
  - name: model
    dtype: string
  - name: prompt_variant_id
    dtype: int64
  - name: prompt_variant_source
    dtype: string
  - name: decode_id
    dtype: int64
  - name: input_question
    dtype: string
  - name: original_question
    dtype: string
  - name: context
    dtype: 'null'
  - name: choices
    list: string
  - name: answer
    dtype: string
  - name: answer_index
    dtype: int64
  - name: answer_text
    dtype: string
  - name: category
    dtype: string
  - name: task_type
    dtype: string
  - name: score_type
    dtype: string
  - name: output_text
    dtype: string
  - name: finish_reason
    dtype: string
  - name: openrouter_response_id
    dtype: string
  - name: usage
    struct:
    - name: prompt_tokens
      dtype: int64
    - name: completion_tokens
      dtype: int64
    - name: total_tokens
      dtype: int64
    - name: cost
      dtype: float64
  - name: request_config
    struct:
    - name: temperature
      dtype: float64
    - name: top_p
      dtype: float64
    - name: max_tokens
      dtype: int64
  - name: created_at
    dtype: timestamp[s]
  - name: predicted_answer
    dtype: string
  - name: gold_answer
    dtype: string
  - name: score
    dtype: float64
  - name: cost
    dtype: float64
  - name: parse_success
    dtype: bool
  - name: split
    dtype: string
  - name: observation_type
    dtype: string
  - name: observation_id
    dtype: string
  - name: prompt_variant_local_index
    dtype: int64
  - name: decoding_strategy_id
    dtype: string
- config_name: math-500
  features:
  - name: generation_key
    dtype: string
  - name: dataset
    dtype: string
  - name: query_id
    dtype: string
  - name: model
    dtype: string
  - name: prompt_variant_id
    dtype: int64
  - name: prompt_variant_source
    dtype: string
  - name: decode_id
    dtype: int64
  - name: input_question
    dtype: string
  - name: original_question
    dtype: string
  - name: context
    dtype: 'null'
  - name: choices
    dtype: 'null'
  - name: answer
    dtype: string
  - name: answer_index
    dtype: 'null'
  - name: answer_text
    dtype: string
  - name: category
    dtype: string
  - name: task_type
    dtype: string
  - name: score_type
    dtype: string
  - name: output_text
    dtype: string
  - name: finish_reason
    dtype: string
  - name: openrouter_response_id
    dtype: string
  - name: usage
    struct:
    - name: prompt_tokens
      dtype: int64
    - name: completion_tokens
      dtype: int64
    - name: total_tokens
      dtype: int64
    - name: cost
      dtype: float64
  - name: request_config
    struct:
    - name: temperature
      dtype: float64
    - name: top_p
      dtype: float64
    - name: max_tokens
      dtype: int64
  - name: created_at
    dtype: timestamp[s]
  - name: predicted_answer
    dtype: string
  - name: gold_answer
    dtype: string
  - name: score
    dtype: float64
  - name: output_text_extracted
    dtype: string
  - name: parse_success
    dtype: bool
  - name: cost
    dtype: float64
  - name: split
    dtype: string
  - name: scoring_version
    dtype: string
  - name: observation_type
    dtype: string
  - name: observation_id
    dtype: string
  - name: prompt_variant_local_index
    dtype: int64
  - name: decoding_strategy_id
    dtype: string
- config_name: drop-800
  features:
  - name: generation_key
    dtype: string
  - name: dataset
    dtype: string
  - name: query_id
    dtype: string
  - name: model
    dtype: string
  - name: prompt_variant_id
    dtype: int64
  - name: prompt_variant_source
    dtype: string
  - name: decode_id
    dtype: int64
  - name: input_question
    dtype: string
  - name: original_question
    dtype: string
  - name: context
    dtype: string
  - name: choices
    dtype: 'null'
  - name: answer
    list: string
  - name: answer_index
    dtype: 'null'
  - name: answer_text
    dtype: string
  - name: category
    dtype: string
  - name: task_type
    dtype: string
  - name: score_type
    dtype: string
  - name: output_text
    dtype: string
  - name: finish_reason
    dtype: string
  - name: openrouter_response_id
    dtype: string
  - name: usage
    struct:
    - name: prompt_tokens
      dtype: int64
    - name: completion_tokens
      dtype: int64
    - name: total_tokens
      dtype: int64
    - name: cost
      dtype: float64
  - name: request_config
    struct:
    - name: temperature
      dtype: float64
    - name: top_p
      dtype: float64
    - name: max_tokens
      dtype: int64
  - name: created_at
    dtype: timestamp[s]
  - name: predicted_answer
    dtype: string
  - name: gold_answer
    list: string
  - name: em
    dtype: float64
  - name: score
    dtype: float64
  - name: cost
    dtype: float64
  - name: parse_success
    dtype: bool
  - name: split
    dtype: string
  - name: observation_type
    dtype: string
  - name: observation_id
    dtype: string
  - name: prompt_variant_local_index
    dtype: int64
  - name: decoding_strategy_id
    dtype: string
---

# DARS Routing Benchmark

**EMNLP 2026 · Main Conference**

**From Sampled Outcomes to Capability Distributions: Rethinking Supervision for LLM Routing**

Guannan Lai · Haoran Hu · Long Chen · Zhenguo Li · Han-Jia Ye

**[Paper](https://arxiv.org/abs/2606.06924) · [PDF](https://arxiv.org/pdf/2606.06924) · [Code & Reproduction](https://github.com/AIGNLAI/DARS) · [Project Page](https://aignlai.github.io/DARS/)**

DARS provides repeated, scored observations of each query–model pair so that routing supervision can capture expected quality, expected cost, and performance variability. The benchmark supports comparing single-shot labels against distribution-aware supervision across scientific QA, mathematical reasoning, and discrete reasoning over text.

![DARS framework: query rewrites and stochastic decoding produce capability estimates for risk-aware supervision.](assets/framework.png)

The paper reports improvements in **40 of 42** router–task–evaluation settings across seven routers. See [Table 1 and its protocol](https://github.com/AIGNLAI/DARS/blob/main/docs/RESULTS.md). These are paper results; the dataset is a collection of observations, not a pretrained router.

## Quick Start

```bash
pip install datasets
```

```python
from datasets import load_dataset

gpqa = load_dataset("AIGNLAI/DARS", "gpqa")
math = load_dataset("AIGNLAI/DARS", "math-500")
drop = load_dataset("AIGNLAI/DARS", "drop-800")

example = gpqa["train"][0]
print(example["query_id"], example["model"], example["score"])
```

Choose one of the **three task configurations** explicitly. GPQA and MATH use string answers; DROP uses lists of answer strings. Train and test also carry different observation-specific columns; the declared schema makes columns absent from a split nullable. Combining all raw files into a single JSON builder without a compatible schema can cause type or column conflicts.

For routing experiments, download the original JSONL layout using the [repository downloader](https://github.com/AIGNLAI/DARS/blob/main/dataset/download_hf_data.py):

```bash
git clone https://github.com/AIGNLAI/DARS.git
cd DARS
pip install -r requirements.txt
python dataset/download_hf_data.py --output-dir data
```

This uses the released observations and makes no paid generation calls. Pin a Hub commit with `revision="<commit-sha>"` in `load_dataset`, or `--revision <commit-sha>` in the downloader, for archival runs.

## Dataset Statistics

| Configuration | Task / response score | Train queries | Train rows | Test queries | Test rows |
|:--|:--|--:|--:|--:|--:|
| `gpqa` | Graduate-level science QA / binary accuracy | 200 | 30,000 | 248 | 8,928 |
| `math-500` | Mathematical reasoning / binary answer correctness | 200 | 30,000 | 300 | 10,800 |
| `drop-800` | Discrete reasoning over text / DROP F1 | 200 | 30,000 | 600 | 21,600 |
| **Total** | | **600** | **90,000** | **1,148** | **41,328** |

One row is one scored generation. All observations derived from an original query share a `query_id`. The training and test query IDs are disjoint within each configuration.

## Collection and Observation Protocol

- **Sources and splits:** [GPQA main](https://huggingface.co/datasets/Idavidrein/gpqa) (448 source queries), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (500 source queries), and an 800-query subset of [DROP](https://huggingface.co/datasets/ucinlp/drop). The released routing splits reserve 200 training queries per task, with seed 42 and stratification by `category`. They are routing-study splits, not the upstream datasets' original splits.
- **Training observations:** five semantics-preserving rewrites generated with GPT-4o, five stochastic decodes per rewrite, and six candidate models. Each query–model pair has 25 observations, giving 150 rows per training query. `prompt_variant_id` 0 is the first rewrite; it is not an original-query marker in the training files.
- **Test observations:** three rewritten prompts (`observation_type="prompt_variation"`) and three decoding strategies on the original query (`observation_type="decoding_variation"`). Each query–model pair has six observations. The original-query rows use `prompt_variant_id=-1`.
- **Decoding settings:** recorded per observation in `request_config`. The collection code's default training temperature is 0.7; its test decoding strategies use temperatures 0.2, 0.7, and 1.0. See the [collection code](https://github.com/AIGNLAI/DARS/tree/main/dataset) and [paper Appendix B](https://arxiv.org/html/2606.06924v2#A2) for the protocol.
- **Scoring:** `score` is binary for GPQA and MATH, and F1 in [0, 1] for DROP. DROP additionally records `em`. MATH records a `scoring_version`; see the [scorer implementation](https://github.com/AIGNLAI/DARS/blob/main/dataset/score_generations.py) when regenerating scores.

### Candidate Model Pool

| Model | Identifier in the data |
|:--|:--|
| Gemma-3-12B-IT | `google/gemma-3-12b-it` |
| Mistral-Small-3.2-24B-Instruct | `mistralai/mistral-small-3.2-24b-instruct` |
| Qwen3-32B | `qwen/qwen3-32b` |
| Llama-3.3-70B-Instruct | `meta-llama/llama-3.3-70b-instruct` |
| Gemini-2.5-Flash-Lite | `google/gemini-2.5-flash-lite` |
| DeepSeek-Chat-V3.1 | `deepseek/deepseek-chat-v3.1` |

## Data Fields

The following names are present in the released JSONL files. Fields that do not apply to a task or observation view may be null.

| Field | Meaning |
|:--|:--|
| `generation_key`, `dataset`, `split` | Observation identifier, task configuration, and routing split. |
| `query_id`, `model` | Original-query identifier and candidate-model identifier. |
| `input_question`, `original_question`, `context` | Observed query wording, original query, and passage (DROP). |
| `choices`, `answer_index`, `answer_text` | GPQA answer options and zero-based answer index; canonical answer text. |
| `answer`, `gold_answer` | String reference for GPQA/MATH; list of reference strings for DROP. |
| `output_text`, `predicted_answer`, `parse_success` | Generated response, parsed answer, and parser success flag. |
| `score`, `score_type` | Task-specific score and its convention. |
| `cost`, `usage` | Recorded raw generation cost and provider usage metadata, including token counts. |
| `request_config`, `created_at` | Recorded decoding parameters and generation timestamp. |
| `task_type`, `category` | Task format and subject or answer category. |
| `finish_reason`, `openrouter_response_id` | Provider completion status and response identifier. |
| `prompt_variant_id` | Rewrite index; `-1` for original-query test observations. |
| `prompt_variant_source`, `decode_id` | Training-only rewrite source and independent decoding index. |
| `observation_type`, `observation_id` | Test-only observation group and identifier. |
| `prompt_variant_local_index`, `decoding_strategy_id` | Test-only rewrite index and decoding-strategy identifier. |
| `output_text_extracted`, `scoring_version` | MATH-only extracted answer text and scoring version. |
| `em` | DROP-only exact-match score. |

### Cost Convention

`cost` is the **raw recorded cost**, not a pre-normalized routing cost. The scorer reads it from the generation record or `usage.cost`. In the released router code, `routing_cost` is computed with min–max normalization over train and test observations **within each task**:

`routing_cost = (cost - task_min_cost) / (task_max_cost - task_min_cost)`

When all costs are equal, the normalized value is zero. This is the released benchmark evaluation convention; do not treat test-derived cost ranges as an available deployment statistic. Model/provider pricing can change, so the snapshot costs are not current price quotes.

## Constructing DARS Targets

The paper defines `U(x, m) = mean_quality - λ mean_cost - β quality_std`, with defaults `λ=0.05` and `β=0.2`. Construct targets from **training observations**. The example below follows the released router's task-level cost normalization and sample standard deviation (`ddof=1`):

```python
import pandas as pd

train = gpqa["train"].to_pandas()
test = gpqa["test"].to_pandas()
all_costs = pd.concat([train["cost"], test["cost"]], ignore_index=True)
lo, hi = all_costs.min(), all_costs.max()
train["routing_cost"] = (train["cost"] - lo) / (hi - lo) if hi > lo else 0.0

targets = (
    train.groupby(["dataset", "query_id", "model"], dropna=False)
    .agg(
        mean_quality=("score", "mean"),
        quality_std=("score", "std"),
        mean_cost=("routing_cost", "mean"),
        num_observations=("score", "size"),
    )
    .reset_index()
)
targets["quality_std"] = targets["quality_std"].fillna(0.0)
targets["utility"] = (
    targets["mean_quality"] - 0.05 * targets["mean_cost"]
    - 0.2 * targets["quality_std"]
)
```

For single-shot supervision, randomly select one **training** observation per query–model pair. The paper averages 100 independently sampled training sets. At inference, a learned router selects a model from the original query; it does not need to repeat the offline sampling protocol.

## Evaluation and Leakage

Keep all rewrites and decodes of a `query_id` in one split. Train with the released training split, then evaluate the selected model separately on the two test observation groups. Test scores must not enter supervision construction. Report the encoder, router settings, sampling seeds, scoring version, cost normalization, risk coefficient, and evaluation view when comparing results. The [GitHub quick start](https://github.com/AIGNLAI/DARS#quick-start) verifies the pipeline; matching paper numbers requires the paper's full experimental configuration.

## Files and Provenance

Each task directory contains `train_scored_generations.jsonl`, `test_scored_generations.jsonl`, and `metadata.json`. Only the scored observations needed for routing experiments are included; intermediate rewrite files, unscored generations, and run logs are omitted. `metadata.json` describes **source preparation and splits**, and also mentions intermediate filenames that are not part of this scored-data release. Generation settings are in each row's `request_config`.

The schema metadata declares nullable fields across train/test; it does not change the stored observations, scores, costs, or split membership.

## Intended Uses and Limitations

The benchmark supports routing research, supervision comparisons, cost–quality–risk analysis, and studies of prompt or decoding variability. It covers three English reasoning tasks and six fixed candidate LLMs. It does not establish performance for multilingual, multimodal, interactive, or agentic routing. Automatic scoring and rewrite validity can introduce measurement errors. Provider-side model changes can affect regenerated observations. Repeated sampling incurs additional offline collection cost; the released data lets researchers study routing without repeating those paid API calls.

## Licensing and Source Attribution

The accompanying **code** is [MIT licensed](https://github.com/AIGNLAI/DARS/blob/main/LICENSE). This dataset retains `license: other` because it contains derived material from multiple sources; the code license does not replace their licenses or access terms.

- **GPQA:** [source card and conditions](https://huggingface.co/datasets/Idavidrein/gpqa); its card lists CC BY 4.0 and gated source access. Accept the upstream access conditions before regenerating from GPQA.
- **MATH-500:** [source dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and [MATH source repository](https://github.com/hendrycks/math). Consult the upstream terms for reuse.
- **DROP:** [source dataset](https://huggingface.co/datasets/ucinlp/drop); its card lists CC BY-SA 4.0.

Cite the source benchmarks as well as DARS when using their derived material. Model outputs may additionally be subject to the relevant providers' terms.

## Citation

```bibtex
@inproceedings{lai2026dars,
  title     = {From Sampled Outcomes to Capability Distributions: Rethinking Supervision for {LLM} Routing},
  author    = {Lai, Guannan and Hu, Haoran and Chen, Long and Li, Zhenguo and Ye, Han-Jia},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.06924}
}
```

The paper is accepted to EMNLP 2026 Main; the URL links to its arXiv version.

## Contact

For questions, use [GitHub Issues](https://github.com/AIGNLAI/DARS/issues), the [dataset Community tab](https://huggingface.co/datasets/AIGNLAI/DARS/discussions), or contact Guannan Lai at [laign@lamda.nju.edu.cn](mailto:laign@lamda.nju.edu.cn).
