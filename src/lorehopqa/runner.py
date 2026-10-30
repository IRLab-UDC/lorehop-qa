"""In-process vLLM engine used by the validation experiments.

One model is loaded per process and every request that model owns is submitted to vLLM in
a single scheduled sweep, so model loads scale with the number of models rather than the
number of requests. Decoding is greedy and constrained by a grammar compiled from the
output schema, so a reply cannot be parsed against a schema it was not decoded against.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .llm import _format_error

T = TypeVar("T", bound=BaseModel)


class VLLMModelConfig(BaseModel):
    """One model the batch pipeline may load. `name` is the handle used by `--model`.

    `name` is deliberately separate from `model`: tasks are pinned to a name at init time
    and must survive a later change of checkpoint path or a local/remote swap.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192
    seed: int | None = None

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int | None = None
    # Concurrent sequences. Left unset vLLM defaults to 1024, which hybrid Mamba models
    # cannot honour: each decode sequence holds one Mamba cache block, and exceeding the
    # available blocks aborts CUDA graph capture at startup.
    max_num_seqs: int | None = None
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    trust_remote_code: bool = False
    enforce_eager: bool = False

    # Append the output schema to the end of every user message this model is sent. The
    # grammar already forces a parsable object, so this buys the two things the grammar
    # cannot express: pydantic's cross-field validators, and the field descriptions that
    # carry the task's semantics. Costs a few hundred prompt tokens per request; turn it off
    # for a model that follows the schema without it, or when context is tight.
    schema_in_prompt: bool = True

    # Qwen-style hybrid reasoning models emit a thinking block that structured decoding
    # would have to fight; pass {"enable_thinking": false} here to turn it off.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Escape hatch for LLM(...) arguments this config does not name explicitly.
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Environment for this model's engine, set before it is created so its worker processes
    # inherit it. Each phase is its own process, so a setting here touches no other model.
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_window(self) -> VLLMModelConfig:
        """The reply budget must leave room for the prompt inside the context window.

        `VLLMRunner._oversized` refuses to run otherwise, but only once the engine has loaded —
        probe job 11751 spent seven minutes loading Mistral Small 4 to find out that
        max_tokens == max_model_len. Here it fails when the config is read, offline.
        """
        if self.max_model_len is not None and self.max_tokens >= self.max_model_len:
            raise ValueError(
                f"{self.name}: max_tokens={self.max_tokens} leaves no room for the prompt inside "
                f"max_model_len={self.max_model_len}"
            )
        return self




@dataclass(frozen=True)
class ChatRequest:
    """One structured-output request. `output_model` drives both the decoding grammar and
    the parse, so a request cannot be decoded against one schema and validated against
    another.

    `temperature` overrides the model config for this request alone. The generation phases do
    not use it — one model, one role, one setting — but the induction loop runs generation,
    judging and editing through a single loaded engine, and those want 0.9, 0.0 and 0.4.
    """

    system: str
    user: str
    output_model: type[BaseModel]
    temperature: float | None = None


@dataclass(frozen=True)
class Generation:
    """One decoded response, and why decoding stopped.

    Grammar-constrained decoding guarantees a syntactically valid *prefix*, never a complete
    object: a response that runs out of token budget mid-object arrives here as ordinary
    malformed JSON. `finish_reason` is the only thing that separates the two, and the
    difference matters — a truncated response is a budget failure, not something the prompt
    did wrong, and charging it to the prompt is how a search loop ends up optimising against
    its own harness.
    """

    text: str
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def _structured_output_kwargs(output_model: type[BaseModel]) -> dict[str, Any]:
    """vLLM renamed guided decoding to structured outputs in 0.11; accept either.

    `disable_any_whitespace` is the argument that matters here. Left at its default the
    compiled grammar admits `[ \\n\\t]*` between every pair of JSON tokens, which is
    unbounded — and a sampler at temperature 0.9 that steps into a run of newlines has
    nothing in the grammar pulling it back out, so it stays there until `max_tokens` is spent
    and the response comes back as a valid JSON prefix tens of thousands of lines long. That
    is a hole in the grammar, not a formatting habit the prompt can correct.
    """
    schema = output_model.model_json_schema()
    try:
        from vllm.sampling_params import StructuredOutputsParams as Params  # noqa: PLC0415

        key = "structured_outputs"
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams as Params  # noqa: PLC0415

        key = "guided_decoding"

    names = {field.name for field in dataclasses.fields(Params)}
    extra = {"disable_any_whitespace": True} if "disable_any_whitespace" in names else {}
    return {key: Params(json=schema, **extra)}


SCHEMA_HEADER = "=== REQUIRED OUTPUT SCHEMA ==="

_SCHEMA_BLOCKS: dict[type[BaseModel], str] = {}


def _without_class_docs(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop the class-level `description` keys, keep the per-field ones.

    A model's top-level description is its Python docstring, written for whoever maintains
    the class: it explains design decisions, names other classes, and in the judge's case
    recounts the failure mode the field order was chosen to avoid — which is exactly the sort
    of thing that should not be in front of a judge scoring an item. `Field(description=...)`
    is the half written for the model, and it stays.
    """
    schema.pop("description", None)
    for definition in schema.get("$defs", {}).values():
        if isinstance(definition, dict):
            definition.pop("description", None)
    return schema


def schema_block(output_model: type[BaseModel]) -> str:
    """The output schema, rendered to go at the end of the user message.

    The grammar already guarantees the response parses against this schema, so this is not
    there to fix syntax. It is there for everything the grammar cannot say:

    * pydantic's cross-field validators — that the question ends in '?', that titles are
      unique, that a `sentence_index` points at a sentence that exists. Grammar-constrained
      decoding cannot express a constraint that spans two fields, so these are checked only
      after the whole object is written, and a violation costs the entire generation.
    * the `description` on every field, which is where the semantics of the task actually
      live and which the model otherwise never sees.

    Rendered once per model class and reused: a sweep submits hundreds of requests sharing a
    handful of schemas, and serialising the same object each time is pure waste.
    """
    block = _SCHEMA_BLOCKS.get(output_model)
    if block is None:
        schema = orjson.dumps(
            _without_class_docs(output_model.model_json_schema()), option=orjson.OPT_INDENT_2
        ).decode()
        block = (
            f"{SCHEMA_HEADER}\n"
            "Your reply must be one JSON object valid against this schema. Read the field "
            "descriptions: they carry requirements the schema alone does not state, and some "
            "constraints hold across fields rather than within one.\n\n"
            f"{schema}"
        )
        _SCHEMA_BLOCKS[output_model] = block
    return block


class VLLMRunner:
    """Loads one model and answers every request handed to it in a single sweep."""

    def __init__(self, config: VLLMModelConfig) -> None:
        from vllm import LLM

        self.config = config
        print(f"[batch] loading {config.name} ({config.model})", flush=True)
        if config.env:
            os.environ.update(config.env)
            print(f"[batch] {config.name} environment: {config.env}", flush=True)

        engine_kwargs: dict[str, Any] = {
            "model": config.model,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "enforce_eager": config.enforce_eager,
        }
        # Omitted rather than passed as None: vLLM types several of these as plain ints,
        # so an explicit None overrides the engine default and fails validation. "Unset"
        # has to mean "absent" here, not "None".
        for field in ("seed", "max_num_seqs"):
            value = getattr(config, field)
            if value is not None:
                engine_kwargs[field] = value
        engine_kwargs.update(config.engine_kwargs)

        self.llm = LLM(**engine_kwargs)

    def run(self, requests: Sequence[ChatRequest], *, progress: bool = True) -> list[str]:
        return [item.text for item in self.generate(requests, progress=progress)]

    def _user_message(self, request: ChatRequest) -> str:
        if not self.config.schema_in_prompt:
            return request.user
        return f"{request.user}\n\n{schema_block(request.output_model)}"

    def generate(
        self, requests: Sequence[ChatRequest], *, progress: bool = True
    ) -> list[Generation]:
        if not requests:
            return []

        from vllm import SamplingParams

        # One grammar per distinct schema (and per temperature it is requested at), reused
        # across requests: compiling an xgrammar matcher is not free and this phase submits
        # hundreds of requests sharing a handful of schemas.
        params_by_model: dict[tuple[type[BaseModel], float], Any] = {}
        for request in requests:
            key = (request.output_model, request.temperature if request.temperature is not None
                   else self.config.temperature)
            if key not in params_by_model:
                params_by_model[key] = SamplingParams(
                    temperature=key[1],
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed,
                    **_structured_output_kwargs(request.output_model),
                )

        # The schema goes at the *end* of the user message, after the material the request is
        # about. Every prompt on both paths is built lore-or-source first and per-item detail
        # last, so appending here leaves those shared prefixes intact for prefix caching, and
        # puts the output contract in the position a model weights most heavily.
        conversations = [
            [
                {"role": "system", "content": request.system},
                {"role": "user", "content": self._user_message(request)},
            ]
            for request in requests
        ]
        sampling_params = [
            params_by_model[
                (
                    request.output_model,
                    request.temperature if request.temperature is not None else self.config.temperature,
                )
            ]
            for request in requests
        ]

        extra: dict[str, Any] = {}
        if self.config.chat_template_kwargs:
            extra["chat_template_kwargs"] = self.config.chat_template_kwargs

        # A prompt longer than the context window raises for the whole batch, not just for the
        # offending request — and because the caller resumes by item, the same chunk would be
        # resubmitted and fail again forever. Short-context readers therefore have their
        # over-long prompts held back and reported rather than sent. The count is small but has
        # to be visible: silently truncating would drop documents from the context, which is the
        # one thing an evidence-based measurement cannot survive.
        oversized = self._oversized(conversations)
        if oversized:
            print(
                f"[batch] {self.config.name}: {len(oversized)} of {len(conversations)} prompts "
                f"exceed the {self.config.max_model_len}-token window and are recorded unanswered",
                flush=True,
            )

        sendable = [index for index in range(len(conversations)) if index not in oversized]
        outputs = (
            self.llm.chat(
                [conversations[index] for index in sendable],
                sampling_params=[sampling_params[index] for index in sendable],
                use_tqdm=progress,
                **extra,
            )
            if sendable
            else []
        )

        # vLLM restores submission order, so result j corresponds to sendable[j].
        results: list[Generation] = [Generation("", "prompt_too_long")] * len(conversations)
        for position, output in zip(sendable, outputs, strict=True):
            results[position] = Generation(output.outputs[0].text, output.outputs[0].finish_reason)
        return results

    def _oversized(self, conversations: Sequence[list[dict[str, str]]]) -> set[int]:
        """Indices whose rendered prompt leaves no room for the reply.

        Measured with the model's own tokeniser through its own chat template, because a word or
        character estimate is exactly the kind of approximation that is wrong at the tail — and
        the tail is the only place this matters.

        Two things here are deliberate. A character prefilter keeps the cost proportional to the
        risk: no tokeniser produces fewer than one token per two characters of English, so a
        conversation shorter than twice the budget cannot overflow it, and for readers with
        generous windows that skips every prompt. And a tokeniser that cannot be loaded is a hard
        failure rather than a silent pass. An earlier version returned an empty set when the
        tokeniser lookup raised, which disabled the guard without saying so and let a 4,097-token
        prompt reach a 4,096-token model twenty minutes into a sweep.
        """
        if not self.config.max_model_len:
            return set()
        budget = self.config.max_model_len - self.config.max_tokens
        if budget <= 0:
            raise SystemExit(
                f"{self.config.name}: max_tokens={self.config.max_tokens} leaves no prompt room "
                f"inside max_model_len={self.config.max_model_len}"
            )

        # No tokeniser emits fewer than one token per two characters, so anything shorter than
        # this cannot exceed the budget and never needs measuring.
        threshold = budget * 2
        candidates = [
            index
            for index, conversation in enumerate(conversations)
            if sum(len(message["content"]) for message in conversation) > threshold
        ]
        if not candidates:
            return set()

        tokenizer = self._tokenizer()
        oversized: set[int] = set()
        for index in candidates:
            conversation = conversations[index]
            length = self._prompt_tokens(tokenizer, conversation)
            chars = sum(len(message["content"]) for message in conversation)
            # A plausibility check on our own measurement. No tokeniser produces fewer than one
            # token per twenty characters, so a count below that is not a short prompt — it is a
            # measurement that went wrong, and it must not be allowed to read as "fits". The
            # first version of this guard counted the two keys of a BatchEncoding and reported
            # 18,000 characters as two tokens, which passed every prompt straight through.
            if length * 20 < chars:
                raise SystemExit(
                    f"{self.config.name}: prompt-length check produced {length} tokens for "
                    f"{chars} characters, which cannot be right. Refusing to run rather than "
                    "silently disable the context-window guard."
                )
            if length > budget:
                oversized.add(index)
        return oversized

    @staticmethod
    def _prompt_tokens(tokenizer: Any, conversation: list[dict[str, str]]) -> int:
        """Token count for one rendered conversation.

        Rendered to text first and encoded second, rather than asking the template to tokenise.
        `apply_chat_template` returns a bare list of ids in some versions, a nested list in
        others, and a BatchEncoding mapping in transformers 5 — and `len()` is meaningful for
        only one of the three. The text form is unambiguous in every version.
        """
        try:
            text = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            text = "\n".join(message["content"] for message in conversation)
        if not isinstance(text, str):
            text = "\n".join(message["content"] for message in conversation)
        # The template has already inserted the model's own special tokens.
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _tokenizer(self) -> Any:
        """The tokeniser for this checkpoint, loaded once and kept."""
        cached = getattr(self, "_tokenizer_cache", None)
        if cached is not None:
            return cached

        errors: list[str] = []
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            cached = AutoTokenizer.from_pretrained(
                self.config.model, trust_remote_code=self.config.trust_remote_code
            )
        except Exception as exc:
            errors.append(f"AutoTokenizer: {type(exc).__name__}: {exc}")
            try:
                cached = self.llm.get_tokenizer()
            except Exception as inner:
                errors.append(f"LLM.get_tokenizer: {type(inner).__name__}: {inner}")
                raise SystemExit(
                    f"{self.config.name}: cannot load a tokeniser to check prompt lengths "
                    f"against max_model_len={self.config.max_model_len}. "
                    + "; ".join(errors)
                ) from inner

        self._tokenizer_cache = cached
        return cached


def parse_structured(text: str, output_model: type[T]) -> tuple[T | None, str | None]:
    """Decode one structured response. Grammar-constrained output is always syntactically
    valid JSON, but the grammar does not enforce pydantic's value constraints (min lengths,
    cross-field validators), so a parse failure here is normal and becomes feedback."""
    try:
        return output_model.model_validate_json(text), None
    except ValidationError as exc:
        return None, _format_error(exc)
    except Exception as exc:  # malformed JSON from a truncated or unconstrained response
        return None, _format_error(exc)


# --------------------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------------------


