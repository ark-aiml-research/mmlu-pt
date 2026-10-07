"""Strict validation of final annotations; raw reasoning never leaves the worker."""

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .config import UNCERTAIN


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


def response_schema(candidates: list[str]) -> dict:
    schema = Annotation.model_json_schema()
    schema["properties"]["subject"]["enum"] = [*candidates, UNCERTAIN]
    schema["properties"]["alternative_subject"] = {
        "anyOf": [{"type": "string", "enum": candidates}, {"type": "null"}]}
    return schema


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

