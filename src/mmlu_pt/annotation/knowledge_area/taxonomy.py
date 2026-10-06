"""Authoritative subject definitions and deterministic candidate resolution."""

import json
from dataclasses import dataclass
from pathlib import Path

from .config import UNCERTAIN, digest

CONTEXT_FIELDS = frozenset({"exam_edition", "course", "specialization", "block"})


@dataclass(frozen=True)
class Taxonomy:
    document: dict
    aliases: dict[str, str]
    definitions: dict[str, str]
    macro_areas: dict[str, str]

    @property
    def version(self) -> str:
        return str(self.document["version"])

    @property
    def checksum(self) -> str:
        return digest(self.document)

    def exam_key(self, exam: str) -> str | None:
        key = self.aliases.get(exam, exam)
        return key if key in self.document["exams"] else None

    def resolve_candidates(self, exam: str, metadata: dict) -> list[str]:
        key = self.exam_key(exam)
        if key is None:
            raise ValueError(f"Unmapped exam: {exam!r}")
        entry = self.document["exams"][key]
        matches = [rule for rule in entry.get("context_rules", [])
                   if all(metadata.get(field) == value for field, value in rule["match"].items())]
        if len(matches) > 1:
            raise ValueError(f"Conflicting context rules for exam {exam!r}")
        return list(matches[0]["allowed_subjects"] if matches else entry["allowed_subjects"])


def validate_candidates(candidates: object, definitions: dict, location: str) -> None:
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{location}: allowed_subjects must be a nonempty list")
    if any(not isinstance(s, str) for s in candidates) or len(set(candidates)) != len(candidates):
        raise ValueError(f"{location}: invalid or duplicate subject names")
    unknown = set(candidates) - definitions.keys()
    if unknown:
        raise ValueError(f"{location}: undefined subjects {sorted(unknown)}")


def load_taxonomy(path: Path, aliases_path: Path) -> Taxonomy:
    document = json.loads(path.read_text(encoding="utf-8"))
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
    if not document.get("version") or not isinstance(document.get("annotation_policy"), dict):
        raise ValueError("Taxonomy requires version and annotation_policy")
    definitions, macro_areas = {}, {}
    area_names = set()
    for area in document.get("macro_areas", []):
        name = area["name"]
        if not isinstance(name, str) or not name.strip() or name in area_names:
            raise ValueError("Invalid or duplicate macro area")
        area_names.add(name)
        for subject in area["subjects"]:
            label, description = subject["name"], subject["description"]
            if not isinstance(label, str) or not label.strip() or label == UNCERTAIN or label in definitions:
                raise ValueError(f"Invalid or duplicate global subject: {label!r}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Missing definition for {label!r}")
            definitions[label], macro_areas[label] = description, name
    exams = document.get("exams")
    if not definitions or not isinstance(exams, dict) or not exams:
        raise ValueError("Taxonomy requires subjects and exams")
    for exam, entry in exams.items():
        validate_candidates(entry["allowed_subjects"], definitions, exam)
        for rule in entry.get("context_rules", []):
            match = rule.get("match")
            if not isinstance(match, dict) or not match or set(match) - CONTEXT_FIELDS:
                raise ValueError(f"{exam}: invalid context selector; answer is forbidden")
            if any(not isinstance(v, str) or not v.strip() for v in match.values()):
                raise ValueError(f"{exam}: context values must be nonempty strings")
            validate_candidates(rule["allowed_subjects"], definitions, exam)
            if set(rule["allowed_subjects"]) - set(entry["allowed_subjects"]):
                raise ValueError(f"{exam}: context rules may only narrow exam candidates")
    if not isinstance(aliases, dict):
        raise ValueError("Aliases must be a JSON object")
    for source, target in aliases.items():
        if not isinstance(source, str) or not isinstance(target, str) or target not in exams:
            raise ValueError(f"Invalid exam alias: {source!r}")
        if source in exams and source != target:
            raise ValueError(f"Alias overrides an authoritative exam: {source!r}")
    return Taxonomy(document, aliases, definitions, macro_areas)
