"""Domain model for benchmarks.

The public API keeps our framework language while adapting to backend terms:

    benchmark = Benchmark.from_hf("gsm8k", limit=100)
    lm_eval_kwargs = benchmark.getforlmevalharness()

EleutherAI LM Evaluation Harness calls benchmarks ``tasks``. This object owns
the task/benchmark-side arguments and returns the keyword arguments expected by
``lm_eval.simple_evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Benchmark:
    """One or more named benchmarks selected from a backend registry.

    Args:
        benchmark: A benchmark registry name such as ``"gsm8k"``, multiple
            names such as ``["hellaswag", "arc_easy"]``.
        source: Benchmark provider or registry. The current runner supports
            ``"hf"`` benchmarks through LM Evaluation Harness.
        num_fewshot: Number of few-shot examples to include in each prompt.
        limit: Optional example cap per task. Integers mean an exact count;
            floats from 0.0 to 1.0 mean a fraction of each task.
        samples: Optional mapping of task names to exact sample indices. This
            is mutually exclusive with ``limit`` in LM Evaluation Harness.
        check_integrity: Ask the harness to run task integrity checks.

    Examples:
        Named benchmark:

            benchmark = Benchmark.from_hf("gsm8k", limit=100)

        Multiple named benchmarks:

            benchmark = Benchmark.from_hf(
                ["hellaswag", "arc_easy"],
                num_fewshot=5,
            )
    """

    benchmark: str | list[str] | tuple[str, ...]
    num_fewshot: int | None = None
    limit: int | float | None = None
    samples: dict[str, list[int]] | None = None
    check_integrity: bool = False
    source: str = "hf"

    @classmethod
    def from_hf(
        cls,
        benchmark: str | list[str] | tuple[str, ...],
        *,
        num_fewshot: int | None = None,
        limit: int | float | None = None,
        samples: dict[str, list[int]] | None = None,
        check_integrity: bool = False,
    ) -> "Benchmark":
        """Create a benchmark from the current HF/lm-eval-supported registry."""
        return cls(
            benchmark=benchmark,
            source="hf",
            num_fewshot=num_fewshot,
            limit=limit,
            samples=samples,
            check_integrity=check_integrity,
        )

    def __post_init__(self) -> None:
        """Validate benchmark-side options that the harness treats specially."""
        if not isinstance(self.benchmark, (str, list, tuple)):
            raise TypeError("benchmark must be a string or a list/tuple of strings.")

        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty benchmark source name.")

        if isinstance(self.benchmark, (list, tuple)) and not self.benchmark:
            raise ValueError("benchmark list cannot be empty.")

        if isinstance(self.benchmark, (list, tuple)) and not all(
            isinstance(task, str) for task in self.benchmark
        ):
            raise TypeError("all benchmark names must be strings.")

        if self.limit is not None and self.samples is not None:
            raise ValueError("LM Evaluation Harness does not allow both limit and samples.")

        if isinstance(self.limit, float) and not 0.0 < self.limit <= 1.0:
            raise ValueError("Float limit must be in the range (0.0, 1.0].")

        if self.num_fewshot is not None and self.num_fewshot < 0:
            raise ValueError("num_fewshot must be greater than or equal to 0.")

    @property
    def name(self) -> str:
        """Return a stable display name for the benchmark."""
        if isinstance(self.benchmark, str):
            return self.benchmark

        if isinstance(self.benchmark, (list, tuple)):
            return ",".join(self.benchmark)

    @property
    def tasks(self) -> list[str]:
        """Return normalized harness task names."""
        if isinstance(self.benchmark, str):
            return [self.benchmark]

        return list(self.benchmark)

    def getforlmevalharness(self) -> dict[str, Any]:
        """Return benchmark kwargs expected by EleutherAI LM Evaluation Harness.

        ``lm_eval.simple_evaluate`` expects named tasks through its ``tasks``
        argument. For a registry benchmark such as ``Benchmark("gsm8k")``, this
        method returns:

            {"tasks": ["gsm8k"]}

        Optional benchmark-side controls are included only when explicitly set,
        so callers can pass the result directly:

            lm_eval.simple_evaluate(model="hf", model_args="...", **kwargs)
        """
        kwargs: dict[str, Any] = {
            "tasks": self.tasks,
        }

        if self.num_fewshot is not None:
            kwargs["num_fewshot"] = self.num_fewshot

        if self.limit is not None:
            kwargs["limit"] = self.limit

        if self.samples is not None:
            kwargs["samples"] = self.samples

        if self.check_integrity:
            kwargs["check_integrity"] = self.check_integrity

        return kwargs

    def get_for_lm_eval_harness(self) -> dict[str, Any]:
        """PEP 8 alias for ``getforlmevalharness``."""
        return self.getforlmevalharness()
