"""Taxonomia única para o prompt, a geração restrita e a macroárea."""

import hashlib
import json
from functools import cache
from importlib.resources import files

LEVELS = ("high_school", "undergraduate")


@cache
def load_taxonomy() -> dict:
    taxonomy = json.loads(resource_text("taxonomy.json"))
    if tuple(taxonomy["levels"]) != LEVELS:
        raise ValueError("Níveis inválidos na taxonomia.")
    for level, groups in taxonomy["levels"].items():
        subjects = [subject for group in groups.values() for subject in group]
        if len(subjects) != len(set(subjects)) or not subjects:
            raise ValueError(f"Subjects duplicados ou ausentes em {level}.")
        for subject in subjects:
            definition = taxonomy["subjects"][subject]
            if not all(definition.get(key) for key in ("definition", "inclusions", "exclusions")):
                raise ValueError(f"Definição incompleta: {subject}.")
    if "Law" in allowed_subjects_from(taxonomy, "high_school"):
        raise ValueError("Law não é permitido em high_school.")
    if taxonomy["levels"]["undergraduate"].get("Law") != ["Law"]:
        raise ValueError("Law deve ser uma macroárea exclusiva no superior.")
    return taxonomy


def resource_text(name: str) -> str:
    return files("mmlu_pt.classification").joinpath(name).read_text(encoding="utf-8")


def resource_hash(name: str) -> str:
    return hashlib.sha256(resource_text(name).encode("utf-8")).hexdigest()


def allowed_subjects_from(taxonomy: dict, level: str) -> list[str]:
    return [subject for group in taxonomy["levels"][level].values() for subject in group]


def allowed_subjects(level: str) -> list[str]:
    if level not in LEVELS:
        raise ValueError(f"academic_level inválido: {level!r}.")
    return allowed_subjects_from(load_taxonomy(), level)


def macro_area(level: str, subject: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"academic_level inválido: {level!r}.")
    for area, subjects in load_taxonomy()["levels"][level].items():
        if subject in subjects:
            return area
    raise ValueError(f"Subject {subject!r} não permitido em {level}.")


@cache
def system_prompt(level: str) -> str:
    taxonomy = load_taxonomy()
    descriptions = []
    for subject in allowed_subjects(level):
        entry = taxonomy["subjects"][subject]
        descriptions.append(
            f"- {subject}: {entry['definition']} "
            f"Inclui: {'; '.join(entry['inclusions'])}. "
            f"Exclui: {'; '.join(entry['exclusions'])}."
        )
    return resource_text("prompt.md").format(
        academic_level=level,
        taxonomy="\n".join(descriptions),
        boundary_rules="\n".join(f"- {rule}" for rule in taxonomy["boundary_rules"]),
    )


def messages_for(record: dict) -> list[dict[str, str]]:
    payload = {
        "question": record["question"],
        "choices": [{"index": i, "text": text} for i, text in enumerate(record["choices"])],
    }
    return [
        {"role": "system", "content": system_prompt(record["academic_level"])},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
