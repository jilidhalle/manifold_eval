"""Evaluation runner domain object.

This is the first thin runner layer for the public API shape:

    model = LLMModel.from_hf("gpt2", dtype="float32")
    benchmark = Benchmark("hellaswag", limit=10)

    result = Eval(model, benchmark).run()

For now this runner only targets EleutherAI LM Evaluation Harness. It
deconstructs our domain objects into the keyword arguments expected by
``lm_eval.simple_evaluate``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from manifold_eval.domain.benchmark import Benchmark
from manifold_eval.domain.llm_model import LLMModel


@dataclass(frozen=True)
class Eval:
    """Run one model against one benchmark definition."""

    model: LLMModel
    benchmark: Benchmark
    output_dir: str | Path = "output"

    def getforlmevalharness(self) -> dict[str, Any]:
        """Return merged kwargs for ``lm_eval.simple_evaluate``."""
        return get_lm_eval_harness_kwargs(self.model, self.benchmark)

    def get_for_lm_eval_harness(self) -> dict[str, Any]:
        """PEP 8 alias for ``getforlmevalharness``."""
        return self.getforlmevalharness()

    def run(
        self,
        *,
        log_samples: bool = False,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the evaluation through EleutherAI LM Evaluation Harness."""
        return run(
            self.model,
            self.benchmark,
            output_dir=output_dir if output_dir is not None else self.output_dir,
            log_samples=log_samples,
        )


def run(
    model: LLMModel,
    benchmark: Benchmark,
    *,
    output_dir: str | Path = "output",
    log_samples: bool = False,
) -> dict[str, Any]:
    """Run ``lm_eval.simple_evaluate`` using our domain objects.

    A compact JSON summary is written under:

        <output_dir>/<model_name>/<benchmark_name>/<timestamp>.json

    When ``output_dir`` is omitted, results are written under ``output``.
    """
    import lm_eval

    result = lm_eval.simple_evaluate(
        **get_lm_eval_harness_kwargs(model, benchmark),
        log_samples=log_samples,
    )

    result_path = build_result_path(output_dir, model, benchmark)
    save_result(result_path, model, benchmark, result, log_samples=log_samples)

    return result


def build_result_path(
    output_dir: str | Path,
    model: LLMModel,
    benchmark: Benchmark,
    *,
    timestamp: str | None = None,
) -> Path:
    """Return the package-standard JSON result path."""
    return (
        Path(output_dir)
        / safe_path_name(model.name)
        / safe_path_name(benchmark.name)
        / f"{timestamp or timestamp_name()}.json"
    )


def timestamp_name() -> str:
    """Return a sortable timestamp for output filenames."""
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def safe_path_name(value: str) -> str:
    """Return a filesystem-safe folder name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def save_result(
    path: str | Path,
    model: LLMModel,
    benchmark: Benchmark,
    result: dict[str, Any],
    *,
    log_samples: bool = False,
) -> None:
    """Save a readable JSON summary of an evaluation result."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": {
            "name": model.name,
            "parameters": model.getforlmevalharness(),
        },
        "benchmark": {
            "name": benchmark.name,
            "source": benchmark.source,
            "parameters": benchmark.getforlmevalharness(),
        },
        "run": {
            "backend": "lm_eval_harness",
            "log_samples": log_samples,
        },
        "scores": result.get("results", {}),
        "examples": _extract_examples(result),
    }

    output_path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    """Convert common harness result values into JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _extract_examples(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract readable question/answer pairs from logged harness samples."""
    examples_by_key: dict[tuple[str, Any], dict[str, Any]] = {}

    for benchmark_name, samples in result.get("samples", {}).items():
        for sample in samples:
            doc_id = sample.get("doc_id")
            key = (benchmark_name, doc_id)

            if key not in examples_by_key:
                doc = sample.get("doc", {})
                examples_by_key[key] = {
                    "benchmark": benchmark_name,
                    "doc_id": doc_id,
                    "question": doc.get("question"),
                    "expected_answer": doc.get("answer"),
                    "model_answer": _first_response(sample.get("resps")),
                    "scores": [],
                }

            examples_by_key[key]["scores"].append(_extract_sample_score(sample))

    return list(examples_by_key.values())


def _extract_sample_score(sample: dict[str, Any]) -> dict[str, Any]:
    """Extract metric scores for one logged sample/filter record."""
    metrics = sample.get("metrics", [])

    return {
        "filter": sample.get("filter"),
        "metrics": {
            metric: sample.get(metric)
            for metric in metrics
            if metric in sample
        },
        "filtered_responses": sample.get("filtered_resps", []),
    }


def _first_response(responses: Any) -> Any:
    """Return the first model response from the harness nested response shape."""
    if not responses:
        return None

    first = responses[0]
    if isinstance(first, list) and first:
        return first[0]

    return first


def get_lm_eval_harness_kwargs(
    model: LLMModel,
    benchmark: Benchmark,
) -> dict[str, Any]:
    """Deconstruct model and benchmark objects into harness kwargs."""
    if not isinstance(model, LLMModel):
        raise TypeError("model must be an LLMModel instance.")

    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must be a Benchmark instance.")

    return {
        **model.getforlmevalharness(),
        **benchmark.getforlmevalharness(),
    }
