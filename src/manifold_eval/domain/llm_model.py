"""Domain model for LLM backends.

The public API keeps our framework language while adapting to backend terms:

    model = LLMModel.from_hf("gpt2", dtype="float32", device="cuda:0")
    lm_eval_kwargs = model.getforlmevalharness()

EleutherAI LM Evaluation Harness expects model-side settings as keyword
arguments to ``lm_eval.simple_evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMModel:
    """A named model backend and its runtime options.

    Args:
        model: Harness model backend name, such as ``"hf"``, ``"vllm"``,
            ``"openai-chat-completions"``, or ``"local-chat-completions"``.
        model_args: Backend constructor arguments. The harness accepts either a
            comma-separated string or a dictionary.
        batch_size: Evaluation batch size. The harness accepts an integer,
            ``"auto"``, or ``"auto:N"``.
        max_batch_size: Optional ceiling used when ``batch_size`` is automatic.
        device: Runtime device, such as ``"cuda"``, ``"cuda:0"``, ``"cpu"``,
            or ``"mps"``.
        gen_kwargs: Optional generation arguments, such as temperature, top_p,
            max tokens, or stop sequences.
        apply_chat_template: Whether to apply the tokenizer chat template.
            Some harness versions also accept a string template name.
        system_instruction: Optional system prompt for chat-style models.
        fewshot_as_multiturn: Whether few-shot examples should be formatted as
            multiple chat turns.

    Examples:
        HuggingFace model:

            model = LLMModel.from_hf("gpt2", dtype="float32")

        Explicit harness backend:

            model = LLMModel(
                model="hf",
                model_args="pretrained=gpt2,dtype=float32",
                batch_size=8,
                device="cuda:0",
            )
    """

    model: str = "hf"
    model_args: str | dict[str, Any] | None = None
    batch_size: int | str | None = None
    max_batch_size: int | None = None
    device: str | None = None
    gen_kwargs: dict[str, Any] | None = None
    apply_chat_template: bool | str | None = None
    system_instruction: str | None = None
    fewshot_as_multiturn: bool = False

    @classmethod
    def from_hf(
        cls,
        pretrained: str,
        *,
        dtype: str | None = None,
        tokenizer: str | None = None,
        revision: str | None = None,
        batch_size: int | str | None = None,
        max_batch_size: int | None = None,
        device: str | None = None,
        model_args: dict[str, Any] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        apply_chat_template: bool | str | None = None,
        system_instruction: str | None = None,
        fewshot_as_multiturn: bool = False,
    ) -> "LLMModel":
        """Create a HuggingFace-backed harness model."""
        args = {
            "pretrained": pretrained,
            **(model_args or {}),
        }

        if dtype is not None:
            args["dtype"] = dtype
        if tokenizer is not None:
            args["tokenizer"] = tokenizer
        if revision is not None:
            args["revision"] = revision

        return cls(
            model="hf",
            model_args=args,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
            gen_kwargs=gen_kwargs,
            apply_chat_template=apply_chat_template,
            system_instruction=system_instruction,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    def __post_init__(self) -> None:
        """Validate model-side options before they reach a backend adapter."""
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty harness backend name.")

        if self.model_args is not None and not isinstance(
            self.model_args,
            (str, dict),
        ):
            raise TypeError("model_args must be a string, a dict, or None.")

        if isinstance(self.batch_size, int) and self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if isinstance(self.batch_size, str) and not self.batch_size:
            raise ValueError("batch_size string cannot be empty.")

        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0.")

    @property
    def name(self) -> str:
        """Return a stable display name for the model."""
        if isinstance(self.model_args, dict) and "pretrained" in self.model_args:
            return str(self.model_args["pretrained"])

        return self.model

    def getforlmevalharness(self) -> dict[str, Any]:
        """Return model kwargs expected by EleutherAI LM Evaluation Harness.

        Optional model-side controls are included only when explicitly set, so
        callers can pass the result directly:

            lm_eval.simple_evaluate(**model_kwargs, **benchmark_kwargs)
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
        }

        if self.model_args is not None:
            kwargs["model_args"] = self._format_model_args()

        if self.batch_size is not None:
            kwargs["batch_size"] = self.batch_size

        if self.max_batch_size is not None:
            kwargs["max_batch_size"] = self.max_batch_size

        if self.device is not None:
            kwargs["device"] = self.device

        if self.gen_kwargs is not None:
            kwargs["gen_kwargs"] = self.gen_kwargs

        if self.apply_chat_template is not None:
            kwargs["apply_chat_template"] = self.apply_chat_template

        if self.system_instruction is not None:
            kwargs["system_instruction"] = self.system_instruction

        if self.fewshot_as_multiturn:
            kwargs["fewshot_as_multiturn"] = self.fewshot_as_multiturn

        return kwargs

    def get_for_lm_eval_harness(self) -> dict[str, Any]:
        """PEP 8 alias for ``getforlmevalharness``."""
        return self.getforlmevalharness()

    def _format_model_args(self) -> str | dict[str, Any]:
        """Return model args in a harness-compatible shape."""
        if not isinstance(self.model_args, dict):
            return self.model_args

        return {
            key: value
            for key, value in self.model_args.items()
            if value is not None
        }
