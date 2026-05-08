# LLM Eval Wrapper

Small wrapper around EleutherAI LM Evaluation Harness with a project-level API:

Install from Git:

```powershell
pip install git+https://github.com/jilidhalle/manifold_eval.git@<version>
uv add git+https://github.com/jilidhalle/manifold_eval.git@<version>
```

```python
from manifold_eval import Benchmark, Eval, LLMModel

model = LLMModel.from_hf(
    "distilgpt2",
    dtype="float32",
    batch_size=1,
    device="cpu",
)

benchmark = Benchmark.from_hf(
    "gsm8k",
    num_fewshot=0,
    limit=1,
)

result = Eval(
    model,
    benchmark,
    output_dir="output",  # defaults to "output"
).run(log_samples=True)  # log_samples defaults to False
```

Results are saved under:

```text
<output_dir>/<model_name>/<benchmark_name>/<timestamp>.json
```

Open the Streamlit viewer for the same result directory:

```powershell
uv run manifold-eval-viewer --results-dir <output>
```

## Saved JSON

Current structure:

```json
{
  "model": {
    "name": "distilgpt2",
    "parameters": {
      "model": "hf",
      "model_args": {
        "pretrained": "distilgpt2",
        "dtype": "float32"
      },
      "batch_size": 1,
      "device": "cpu"
    }
  },
  "benchmark": {
    "name": "gsm8k",
    "source": "hf",
    "parameters": {
      "tasks": ["gsm8k"],
      "num_fewshot": 0,
      "limit": 1
    }
  },
  "run": {
    "backend": "lm_eval_harness",
    "log_samples": true
  },
  "scores": {
    "gsm8k": {
      "alias": "gsm8k",
      "exact_match,flexible-extract": 1.0,
      "exact_match,strict-match": 0.0,
      "exact_match_stderr,flexible-extract": "N/A",
      "exact_match_stderr,strict-match": "N/A"
    }
  },
  "examples": [
    {
      "benchmark": "gsm8k",
      "doc_id": 0,
      "question": "Janets ducks lay 16 eggs per day...",
      "expected_answer": "Janet sells 16 - 3 - 4 = ... #### 18",
      "model_answer": "Janet has 16 eggs, uses 7, and sells 9 for $18.",
      "scores": [
        {
          "filter": "strict-match",
          "metrics": {
            "exact_match": 0.0
          },
          "filtered_responses": ["[invalid]"]
        },
        {
          "filter": "flexible-extract",
          "metrics": {
            "exact_match": 1.0
          },
          "filtered_responses": ["$18"]
        }
      ]
    }
  ]
}
```
