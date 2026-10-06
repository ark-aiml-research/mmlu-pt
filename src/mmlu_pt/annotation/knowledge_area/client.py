"""OpenAI-compatible transport, strict final-output validation and bounded retries."""

import asyncio
import json
import re
import time
from dataclasses import dataclass

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .config import UNCERTAIN, RunConfig, request_seed


class Annotation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    subject: str
    alternative_subject: str | None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    justification: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def concise_final_output(self):
        if not self.justification.strip() or re.search(r"</?think\b", self.justification, re.I):
            raise ValueError("Justification must be a concise final explanation")
        if self.alternative_subject == self.subject:
            raise ValueError("Alternative must differ from subject")
        return self


class FormatFailure(ValueError):
    """An invalid final result; never contains raw model text."""


class FatalClientError(RuntimeError):
    """A global server/configuration failure with a sanitized message."""


@dataclass(frozen=True)
class Outcome:
    result: dict | None
    error: str | None


def response_schema(candidates: list[str]) -> dict:
    schema = Annotation.model_json_schema()
    schema["properties"]["subject"]["enum"] = [*candidates, UNCERTAIN]
    schema["properties"]["alternative_subject"] = {
        "anyOf": [{"type": "string", "enum": candidates}, {"type": "null"}]}
    return {"type": "json_schema", "json_schema": {
        "name": "subject_annotation", "strict": True, "schema": schema}}


def final_content(content: str | None) -> str:
    if not isinstance(content, str) or not content.strip():
        raise FormatFailure("missing_final_content")
    # Qwen3.5 may return only a closing tag because the opening tag is in the prompt.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    if re.search(r"</?think\b", content, re.I):
        raise FormatFailure("incomplete_or_invalid_thinking")
    content = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.S)
    return fenced.group(1).strip() if fenced else content


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise FormatFailure("duplicate_json_key")
        result[key] = value
    return result


def reject_constant(value: str):
    raise FormatFailure("nonfinite_json")


def usage_metadata(usage) -> dict | None:
    if usage is None:
        return None
    values = {name: getattr(usage, name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    for name in ("prompt_tokens_details", "completion_tokens_details"):
        details = getattr(usage, name, None)
        if details is not None:
            values[name] = {key: value for key, value in details.model_dump().items()
                            if isinstance(value, int) and not isinstance(value, bool)}
    return values


def parse_annotation(content: str | None, candidates: list[str]) -> dict:
    try:
        value = json.loads(final_content(content), object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
        result = Annotation.model_validate(value).model_dump()
    except (json.JSONDecodeError, ValidationError, TypeError):
        raise FormatFailure("invalid_final_schema") from None
    if result["subject"] not in candidates and result["subject"] != UNCERTAIN:
        raise FormatFailure("subject_outside_candidates")
    if result["alternative_subject"] is not None and result["alternative_subject"] not in candidates:
        raise FormatFailure("alternative_outside_candidates")
    return result


def make_client(config: RunConfig, api_key: str, transport=None, trust_env: bool = True) -> AsyncOpenAI:
    http_client = httpx.AsyncClient(transport=transport, timeout=config.timeout, trust_env=trust_env,
                                  limits=httpx.Limits(max_connections=config.concurrency,
                                                     max_keepalive_connections=config.concurrency))
    return AsyncOpenAI(base_url=config.endpoint, api_key=api_key, timeout=config.timeout,
                       max_retries=0, http_client=http_client)


def request_arguments(config: RunConfig, messages: list[dict], candidates: list[str], seed: int) -> dict:
    arguments = {"model": config.model, "messages": messages, "seed": seed,
                 "temperature": config.temperature, "top_p": config.top_p,
                 "presence_penalty": config.presence_penalty, "max_tokens": config.max_tokens,
                 "extra_body": {"top_k": config.top_k, "min_p": config.min_p,
                                "repetition_penalty": config.repetition_penalty,
                                "chat_template_kwargs": {"enable_thinking": config.thinking}}}
    if config.structured_output == "json_schema":
        arguments["response_format"] = response_schema(candidates)
    return arguments


async def classify(client: AsyncOpenAI, config: RunConfig, semaphore: asyncio.Semaphore,
                   messages: list[dict], candidates: list[str], annotation_id: str,
                   kind: str, pass_number: int, attempt_offset: int, persist_attempt,
                   sleep=asyncio.sleep, start_attempt=None) -> Outcome:
    format_failures = 0
    for local_attempt in range(1, config.max_attempts + 1):
        attempt = attempt_offset + local_attempt
        seed = request_seed(config.seed, annotation_id, kind, pass_number, local_attempt)
        started = time.monotonic()
        error, result, response_metadata, transient, fatal = None, None, {}, False, False
        format_error = False
        try:
            async with semaphore:
                if start_attempt is not None:
                    start_attempt(attempt, seed)
                response = await asyncio.wait_for(
                    client.chat.completions.create(**request_arguments(config, messages, candidates, seed)),
                    timeout=config.timeout)
            response_metadata = {
                "model": response.model, "request_id": response.id,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
                "usage": usage_metadata(response.usage),
                "system_fingerprint": response.system_fingerprint}
            if response.model != config.model:
                raise FatalClientError("response_model_mismatch")
            if not response.choices or response.choices[0].finish_reason != "stop":
                raise FormatFailure("incomplete_generation")
            result = parse_annotation(response.choices[0].message.content, candidates)
        except FormatFailure as exc:
            error = str(exc)  # Only static codes produced by our validation functions.
            format_failures += 1
            format_error = True
        except (APIConnectionError, APITimeoutError, TimeoutError):
            error, transient = "connection_or_timeout", True
        except APIStatusError as exc:
            error = f"http_{exc.status_code}"
            transient = exc.status_code in (408, 409, 429) or exc.status_code >= 500
            if exc.status_code == 400:
                # Inspect in memory only: server messages may echo prompts or model output.
                text = str(exc).lower()
                context_error = any(s in text for s in ("maximum context", "context length", "too many tokens"))
                error = "context_overflow" if context_error else "http_400_configuration"
                fatal = not context_error
            elif not transient:
                fatal = True
        except FatalClientError as exc:
            error, fatal = str(exc), True
        persist_attempt({"attempt": attempt, "seed": seed, "result": result, "error": error,
                         "metadata": response_metadata, "duration_seconds": time.monotonic() - started})
        if fatal:
            raise FatalClientError(error)
        if result is not None:
            return Outcome(result, None)
        if local_attempt == config.max_attempts:
            return Outcome(None, error)
        if transient:
            delay = min(config.backoff_max, config.backoff_initial * 2 ** (local_attempt - 1))
            await sleep(delay)
            continue
        if format_error and format_failures <= config.format_retries:
            continue
        return Outcome(None, error)
    raise AssertionError("Unreachable retry state")
