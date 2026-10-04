"""Generate the figures and verification summary for the knowledge-area annotation appendix."""

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from mmlu_pt.classification.taxonomy import LEVELS, allowed_subjects, load_taxonomy, macro_area

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = Path(__file__).resolve().parent
FIGURE_DIR = ABLATION_DIR / "figures"

OUTPUT_DIR = REPO_ROOT / "output" / "knowledge-classification"
DATASET_DIR = OUTPUT_DIR / "dataset"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = REPO_ROOT / "knowledge_classification.log"

SEED = 42
BOUNDARY_SAMPLE_SIZE = 2
ENADE_GENERAL_MAX_NUM = 10
FIGURE_MIN_GROUP = 30
GENERAL_MIN_GROUP = 10
TABLE_MIN_GROUP = 140

LEVEL_LABELS = {"high_school": "High school", "undergraduate": "Undergraduate"}
EXAM_LABELS = {"RESIDENCIA_USP_UNICAMP": "Res. USP/Unicamp"}
MACRO_AREA_LABELS = {
    "STEM": "STEM",
    "Languages and Arts": "Languages\nand Arts",
    "Humanities and Social Sciences": "Humanities and\nSocial Sciences",
    "Education": "Education",
    "Law": "Law",
    "Business and Economics": "Business and\nEconomics",
    "Health Sciences": "Health\nSciences",
    "Agricultural and Veterinary Sciences": "Agricultural and\nVeterinary Sci.",
    "Services": "Services",
}

# Exams whose scope is one discipline or a small, known set of disciplines.
EXAM_EXPECTED_SUBJECTS = {
    "OAB": ("Law",),
    "ENAM": ("Law",),
    "CFCES": ("Accounting", "Law"),
    "OBI": ("Logic", "Mathematics", "Computer Science"),
    "POSCOMP": ("Computer Science", "Mathematics", "Statistics", "Logic"),
    "REVALIDA": ("Medicine", "Public Health"),
    "RESIDENCIA_USP_UNICAMP": ("Medicine", "Public Health"),
}

# ENADE course families: (regex on the course slug, family label, expected subjects).
# The first match wins, so specific patterns precede generic ones. Health-profession
# courses also expect Public Health; licenciaturas also expect Education (below).
HEALTH = ("Public Health",)
ENADE_COURSE_FAMILIES = [
    (r"medicina_veterinaria", "Veterinary medicine", ("Veterinary Medicine",)),
    (r"^medicina", "Medicine", ("Medicine",) + HEALTH),
    (r"enfermagem", "Nursing", ("Nursing",) + HEALTH),
    (r"odontologia", "Dentistry", ("Dentistry",) + HEALTH),
    (r"farmacia", "Pharmacy", ("Pharmacy",) + HEALTH),
    (r"fisioterapia", "Physical therapy", ("Physical Therapy",) + HEALTH),
    (r"fonoaudiologia", "Speech therapy", ("Speech and Language Therapy",) + HEALTH),
    (r"terapia_ocupacional", "Occupational therapy", ("Occupational Therapy",) + HEALTH),
    (r"nutricao", "Nutrition", ("Nutrition",) + HEALTH),
    (r"biomedicina", "Biomedicine", ("Biomedical Sciences",) + HEALTH),
    (r"educacao_fisica", "Physical education", ("Physical Education",) + HEALTH),
    (r"zootecnia", "Animal science", ("Animal Science",)),
    (r"agronomia|engenharia_florestal", "Agronomy and forestry", ("Agronomy",)),
    (r"^direito", "Law", ("Law",)),
    (r"psicologia", "Psychology", ("Psychology",)),
    (r"servico_social", "Social work", ("Social Work",)),
    (r"ciencias_sociais", "Social sciences", ("Sociology",)),
    (r"relacoes_internacionais", "International relations", ("International Relations",)),
    (r"^historia", "History", ("History",)),
    (r"^geografia", "Geography", ("Geography",)),
    (r"^filosofia", "Philosophy", ("Philosophy",)),
    (r"teologia", "Theology", ("Religious Studies",)),
    (r"comunicacao_social|jornalismo", "Communication", ("Communication",)),
    (r"arquivologia|biblioteconomia", "Archival and library science", ("Library and Archival Science",)),
    (r"ciencias_contabeis", "Accounting", ("Accounting",)),
    (r"ciencias_economicas", "Economics", ("Economics",)),
    (r"administracao_publica|gestao_publica", "Public administration", ("Public Administration",)),
    (r"gestao_financeira", "Financial management", ("Finance",)),
    (r"administracao|secretariado|marketing|recursos_humanos|logistica|processos_gerenciais|"
     r"gestao_comercial|gestao_da_producao|gestao_d[ae]_qualidade|comercio_exterior",
     "Business and management", ("Business Administration",)),
    (r"computacao|sistemas?_de_informacao|redes_de_computadores|analise_e_desenvolvimento|tecnologia_da_informacao",
     "Computing", ("Computer Science",)),
    (r"engenharia_de_alimentos|tecnologia_em_alimentos|agroindustria", "Food technology",
     ("Food Science and Technology",)),
    (r"engenharia_ambiental|gestao_ambiental|saneamento", "Environmental engineering",
     ("Environmental Sciences", "Engineering")),
    (r"processos_quimicos", "Chemical processes", ("Chemistry", "Engineering")),
    (r"engenharia|fabricacao_mecanica|manutencao_industrial|automacao_industrial|construcao_de_edificios",
     "Engineering", ("Engineering",)),
    (r"arquitetura|design_de_interiores", "Architecture", ("Architecture and Urbanism",)),
    (r"design", "Design", ("Design",)),
    (r"^pedagogia|^normal_superior", "Pedagogy", ("Education",)),
    (r"^matematica", "Mathematics", ("Mathematics",)),
    (r"^estatistica", "Statistics", ("Statistics",)),
    (r"^fisica", "Physics", ("Physics",)),
    (r"^quimica", "Chemistry", ("Chemistry",)),
    (r"biologia|ciencias_biologicas", "Biological sciences", ("Biology",)),
    (r"letras.*ingles", "Letters (English)", ("Portuguese", "Literature", "Linguistics", "English")),
    (r"letras.*espanhol", "Letters (Spanish)", ("Portuguese", "Literature", "Linguistics", "Spanish")),
    (r"^letras", "Letters", ("Portuguese", "Literature", "Linguistics")),
    (r"artes_visuais|musica|teatro", "Arts", ("Arts",)),
    (r"turismo", "Tourism", ("Tourism and Hospitality",)),
    (r"gastronomia", "Gastronomy", ("Gastronomy",)),
    (r"estetica", "Aesthetics", ("Aesthetics and Cosmetology",)),
    (r"seguranca_do_trabalho", "Occupational safety", ("Occupational Health and Safety",)),
]
LICENCIATURA_PATTERN = re.compile(r"licenciatura|^pedagogia|^normal_superior")

# Boundary groups: (name, exams, listed subjects, sampled categories); "other" samples the
# questions of those exams whose subject is not among the listed ones.
BOUNDARY_GROUPS = [
    ("Portuguese / Literature", ("ENEM", "FUVEST"), ("Portuguese", "Literature"), ("Portuguese", "Literature")),
    ("Logic / Mathematics / Computer Science", ("OBI",), ("Logic", "Mathematics", "Computer Science"),
     ("Logic", "Mathematics", "Computer Science")),
    ("Computer Science / Mathematics / Logic / Statistics", ("POSCOMP",),
     ("Computer Science", "Mathematics", "Logic", "Statistics"),
     ("Computer Science", "Mathematics", "Logic", "Statistics")),
    ("Accounting / Law", ("CFCES",), ("Accounting", "Law"), ("Accounting", "Law")),
    ("Medicine / Public Health", ("REVALIDA", "RESIDENCIA_USP_UNICAMP"), ("Medicine", "Public Health"),
     ("Medicine", "Public Health")),
    ("Education / taught discipline (ENADE licenciaturas)", ("ENADE",), ("Education",), ("Education", "other")),
    ("Non-Law labels in OAB", ("OAB",), ("Law",), ("other",)),
]

# ACL page geometry: 7.7 cm columns inside a 16 cm text block.
COLUMN_WIDTH = 7.7 / 2.54
TEXT_WIDTH = 16.0 / 2.54

# First two slots of the reference categorical palette (validated for all pairs).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("sequential", ["#f7f9fc", *BLUE_RAMP])

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.width": 0.4,
    "xtick.minor.size": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    share = successes / total
    center = (share + z**2 / (2 * total)) / (1 + z**2 / total)
    margin = z * np.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return center - margin, center + margin


# -----------------------------------------------------------------------------
# Loading


def load_frame() -> pd.DataFrame:
    """The classified dataset, with the ENADE course slug and the numeric question number."""
    frame = load_from_disk(str(DATASET_DIR)).to_pandas()
    frame["num_int"] = pd.to_numeric(frame["num"], errors="coerce")
    frame["course"] = np.where(frame["exam"].eq("ENADE"), frame["exam_edition"].str.split(" - ", n=1).str[1], None)
    frame["year"] = pd.to_numeric(frame["exam_edition"].str.extract(r"(\d{4})")[0], errors="coerce")
    levels = frame.groupby("exam")["academic_level"].nunique()
    if (levels != 1).any():
        raise ValueError("Every exam must have a single academic level.")
    return frame


def load_run() -> tuple[dict, dict, dict]:
    """Manifest, identity and runtime of the production run."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_dir = CHECKPOINT_DIR / json.loads((CHECKPOINT_DIR / "latest.json").read_text(encoding="utf-8"))["run"]
    identity = json.loads((run_dir / "identity.json").read_text(encoding="utf-8"))
    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("The production run is not complete.")
    return manifest, identity, runtime


def load_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Checkpoint rows, checked against the dataset and the taxonomy."""
    run_dir = CHECKPOINT_DIR / json.loads((CHECKPOINT_DIR / "latest.json").read_text(encoding="utf-8"))["run"]
    predictions = pd.concat(
        [pd.read_json(path, lines=True) for path in sorted(run_dir.glob("rank-*.jsonl"))], ignore_index=True
    ).sort_values("index", ignore_index=True)
    if not predictions["index"].eq(range(len(frame))).all():
        raise ValueError("Checkpoint indices do not cover the dataset.")
    same_subjects = predictions["subject"].eq(frame["subject"]).all()
    same_labels = same_subjects and predictions["macro_area"].eq(frame["macro_area"]).all()
    if not same_labels:
        raise ValueError("Checkpoint labels differ from the published dataset.")
    derived = [macro_area(level, subject)
               for level, subject in zip(frame["academic_level"], frame["subject"], strict=True)]
    if not frame["macro_area"].eq(derived).all():
        raise ValueError("Stored macro-areas differ from the taxonomy.")
    predictions["academic_level"] = frame["academic_level"].to_numpy()
    return predictions


def parse_log() -> dict:
    """Engine facts and the generation window recorded in the run log."""
    text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "weights_seconds": r"Loading weights took ([\d.]+) seconds",
        "weights_gib": r"Model loading took ([\d.]+) GiB memory",
        "init_seconds": r"init engine .*? took ([\d.]+) seconds",
        "warmup_seconds": r"Initial profiling/warmup run took ([\d.]+) s",
        "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
        "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
        "prefix_caching": r"enable_prefix_caching=(\w+)",
        "chunked_prefill": r"enable_chunked_prefill=(\w+)",
        "quantization": r"quantization=(\w+)",
        "dtype": r"dtype=torch\.(\w+)",
    }
    facts = {key: (match.group(1) if (match := re.search(pattern, text)) else None)
             for key, pattern in patterns.items()}
    stamps = re.findall(r"^(\S+ \S+) INFO Classificadas", text, flags=re.MULTILINE)
    first, last = (datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f") for stamp in (stamps[0], stamps[-1]))
    facts["generation_window_seconds"] = (last - first).total_seconds()
    return facts


# -----------------------------------------------------------------------------
# Computations


def level_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """Subjects in the taxonomy, questions and share per (level, macro-area)."""
    taxonomy = load_taxonomy()
    rows = []
    for level in LEVELS:
        subset = frame[frame["academic_level"] == level]
        for area, subjects in taxonomy["levels"][level].items():
            count = int(subset["macro_area"].eq(area).sum())
            rows.append({"level": level, "macro_area": area, "subjects": len(subjects), "questions": count,
                         "share": 100 * count / len(subset)})
    return pd.DataFrame(rows)


def subject_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Questions per subject and level, in taxonomy order, with the macro-area of each subject."""
    taxonomy = load_taxonomy()
    counts = pd.crosstab(frame["subject"], frame["academic_level"])
    rows = []
    for area in taxonomy["levels"]["undergraduate"]:
        subjects = list(dict.fromkeys(
            taxonomy["levels"]["high_school"].get(area, []) + taxonomy["levels"]["undergraduate"][area]
        ))
        for subject in subjects:
            row = {"macro_area": area, "subject": subject}
            for level in LEVELS:
                allowed = subject in allowed_subjects(level)
                counted = allowed and subject in counts.index
                row[level] = int(counts.at[subject, level]) if counted else (0 if allowed else None)
            row["total"] = sum(value for value in (row[level] for level in LEVELS) if value is not None)
            rows.append(row)
    return pd.DataFrame(rows)


def exam_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Size, label diversity and the two most frequent subjects of each exam."""
    rows = []
    for exam, subset in frame.groupby("exam"):
        ranked = subset["subject"].value_counts()
        rows.append({
            "level": subset["academic_level"].iat[0], "exam": exam, "n": len(subset),
            "macro_areas": subset["macro_area"].nunique(), "subjects": len(ranked),
            "top": ranked.index[0], "top_share": 100 * ranked.iat[0] / len(subset),
            "second": ranked.index[1] if len(ranked) > 1 else "--",
            "second_share": 100 * ranked.iat[1] / len(subset) if len(ranked) > 1 else np.nan,
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["level", "n"], ascending=[True, False], ignore_index=True)


def token_stats(predictions: pd.DataFrame) -> pd.DataFrame:
    """Prompt-token statistics overall and per level."""
    groups = {"all": predictions, **{level: predictions[predictions["academic_level"] == level] for level in LEVELS}}
    rows = []
    for name, subset in groups.items():
        tokens = subset["prompt_tokens"]
        rows.append({"group": name, "items": len(subset), "total": int(tokens.sum()), "mean": tokens.mean(),
                     "median": tokens.median(), "min": int(tokens.min()), "p95": tokens.quantile(0.95),
                     "p99": tokens.quantile(0.99), "max": int(tokens.max()),
                     "output_total": int(subset["output_tokens"].sum())})
    return pd.DataFrame(rows).set_index("group")


def consistency_row(label: str, subset: pd.DataFrame, expected: str) -> dict:
    """Share of a group's labels inside its expected set (column `inside`), with Wilson interval and top other."""
    inside = subset["inside"].astype(bool)
    others = subset.loc[~inside, "subject"].value_counts()
    low, high = wilson_interval(int(inside.sum()), len(subset)) if len(subset) else (np.nan, np.nan)
    return {"group": label, "n": len(subset), "expected": expected, "inside": int(inside.sum()),
            "share": 100 * inside.mean() if len(subset) else np.nan, "low": 100 * low, "high": 100 * high,
            "top_other": others.index[0] if len(others) else "--",
            "top_other_n": int(others.iat[0]) if len(others) else 0}


def exam_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    """Consistency of the exams with a known disciplinary scope."""
    rows = []
    for exam, expected in EXAM_EXPECTED_SUBJECTS.items():
        subset = frame[frame["exam"] == exam].assign(inside=lambda part: part["subject"].isin(expected))
        rows.append(consistency_row(exam, subset, ", ".join(expected)))
    return pd.DataFrame(rows)


def course_family(slug: str) -> tuple[str, tuple[str, ...]] | None:
    """Family label and expected subjects of an ENADE course slug, or None when the course is not audited."""
    for pattern, family, expected in ENADE_COURSE_FAMILIES:
        if re.search(pattern, slug):
            if LICENCIATURA_PATTERN.search(slug) and "Education" not in expected:
                expected = expected + ("Education",)
            return family, expected
    return None


def enade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """ENADE records with their family, expected subjects, component and whether the label is expected."""
    enade = frame[frame["exam"] == "ENADE"].copy()
    families = enade["course"].map(course_family)
    enade["family"] = families.map(lambda value: value[0] if value else None)
    enade["expected"] = families.map(lambda value: value[1] if value else None)
    enade["component"] = np.where(enade["num_int"] <= ENADE_GENERAL_MAX_NUM, "general", "specific")
    enade["inside"] = [expected is not None and subject in expected
                       for subject, expected in zip(enade["subject"], enade["expected"], strict=True)]
    return enade


def expected_label(subset: pd.DataFrame) -> str:
    """Expected subjects shared by every course of a family, plus those added for its licenciaturas."""
    sets = list(dict.fromkeys(subset["expected"]))
    core = [subject for subject in sets[0] if all(subject in other for other in sets)]
    extra = [subject for subject in dict.fromkeys(sum(sets, ())) if subject not in core]
    return ", ".join(core) + (f" (+ {', '.join(extra)} for licenciaturas)" if extra else "")


def enade_consistency(enade: pd.DataFrame) -> pd.DataFrame:
    """Consistency per course family for the course-specific and the general component."""
    mapped = enade[enade["family"].notna()]
    rows = []
    for family, subset in mapped.groupby("family"):
        label = expected_label(subset)
        for component in ("specific", "general"):
            row = consistency_row(family, subset[subset["component"] == component], label)
            row["component"] = component
            rows.append(row)
    table = pd.DataFrame(rows)
    order = table[table["component"] == "specific"].sort_values(["n"], ascending=False)["group"]
    table["group"] = pd.Categorical(table["group"], categories=order, ordered=True)
    return table.sort_values(["component", "group"], ascending=[False, True], ignore_index=True)


def enade_component_totals(enade: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consistency over all mapped courses, per component, using each record's own expected set."""
    mapped = enade[enade["family"].notna()]
    return pd.DataFrame([
        consistency_row(component, mapped[mapped["component"] == component], "per-course sets")
        for component in ("specific", "general")
    ])


def boundary_examples(frame: pd.DataFrame, enade: pd.DataFrame) -> pd.DataFrame:
    """Fixed-seed samples from the groups where the boundary rules apply."""
    licenciatura = enade.index[enade["course"].str.contains(LICENCIATURA_PATTERN, regex=True)]
    rows = []
    for name, exams, listed, sampled in BOUNDARY_GROUPS:
        pool = frame[frame["exam"].isin(exams)]
        if name.startswith("Education"):
            pool = pool.loc[pool.index.intersection(licenciatura)]
        for category in sampled:
            selected = ~pool["subject"].isin(listed) if category == "other" else pool["subject"] == category
            candidates = pool[selected]
            sample = candidates.sample(min(BOUNDARY_SAMPLE_SIZE, len(candidates)), random_state=SEED)
            for _, record in sample.iterrows():
                rows.append({"boundary": name, "exam": record["exam"], "edition": record["exam_edition"],
                             "num": record["num"], "subject": record["subject"],
                             "question": " ".join(record["question"].split())[:160]})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Reporting


def print_frame(frame: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 200,
                           "display.max_colwidth", 60):
        print(frame.to_string(index=False))


def print_summary(frame: pd.DataFrame, predictions: pd.DataFrame, manifest: dict, identity: dict, runtime: dict,
                  facts: dict, enade: pd.DataFrame) -> None:
    taxonomy = load_taxonomy()
    print("== Input and run identity")
    print("Records:", len(frame), "| by level:", frame["academic_level"].value_counts().to_dict())
    print("Exams:", frame["exam"].nunique(), "| editions:", frame["exam_edition"].nunique())
    print("Input sha256:", identity["input"]["splits"][0]["records_sha256"])
    print("Model:", identity["model"]["id"], identity["model"]["resolved_revision"])
    print("Parameters:", identity["parameters"])
    print("Generation:", identity["generation"])
    print("Versions:", identity["versions"])
    print("Taxonomy:", identity["taxonomy"], "| prompt sha256:", identity["prompt_sha256"][:8],
          "| implementation sha256:", identity["implementation_sha256"][:8])
    print("Checkpoint sha256:", manifest["checkpoint_sha256"])
    print("Counts:", manifest["counts"])
    print("Started:", manifest["started_at"], "| completed:", manifest["completed_at"])

    print("\n== Taxonomy and label distribution")
    pairs = {(level, subject) for level in LEVELS for subject in allowed_subjects(level)}
    used = set(zip(frame["academic_level"], frame["subject"], strict=True))
    print("Allowed (level, subject) pairs:", len(pairs), "| used:", len(used & pairs),
          "| outside taxonomy:", len(used - pairs))
    print("Boundary rules:", len(taxonomy["boundary_rules"]))
    overview = level_overview(frame)
    print_frame(overview.round(2))
    overall = frame["macro_area"].value_counts()
    print_frame(pd.DataFrame({"macro_area": overall.index, "questions": overall.to_numpy(),
                              "share": (100 * overall / len(frame)).round(2).to_numpy()}))
    subjects = subject_table(frame)
    print_frame(subjects)
    print("Top subjects overall:", frame["subject"].value_counts().head(8).to_dict())
    cells = subjects.melt(id_vars=["macro_area", "subject"], value_vars=list(LEVELS),
                          var_name="academic_level", value_name="n").dropna()
    print("Pairs with fewer than 100 questions:", int((cells["n"] < 100).sum()),
          "| with fewer than 10:", int((cells["n"] < 10).sum()))
    print("Smallest cells:", cells.nsmallest(5, "n")[["subject", "academic_level", "n"]].to_dict("records"))

    print("\n== Per exam")
    exams = exam_table(frame)
    print_frame(exams.round(1))
    print("Macro-area shares per exam (%):")
    shares = 100 * pd.crosstab(frame["exam"], frame["macro_area"], normalize="index")
    print_frame(shares.round(1).reset_index())

    print("\n== Prompt and output tokens")
    print_frame(token_stats(predictions).round(1).reset_index())
    print("Output-token histogram:", predictions["output_tokens"].value_counts().sort_index().to_dict())
    spans = predictions.groupby(frame["subject"])["output_tokens"].agg(["min", "max"])
    print("Subjects whose output length varies:", spans[spans["min"] != spans["max"]].to_dict("index"))
    print("Prompt tokens of the longest item:", int(predictions["prompt_tokens"].max()),
          "| max_model_len:", identity["parameters"]["max_model_len"])

    print("\n== Consistency audit (exams with a known scope)")
    print_frame(exam_consistency(frame).round(1))
    for exam, expected in EXAM_EXPECTED_SUBJECTS.items():
        outside = frame[(frame["exam"] == exam) & ~frame["subject"].isin(expected)]["subject"].value_counts()
        print(f"  {exam}: labels outside the expected set:", outside.head(4).to_dict())

    print("\n== Consistency audit (ENADE)")
    print("ENADE records:", len(enade), "| courses:", enade["course"].nunique(),
          "| editions:", enade["exam_edition"].nunique())
    print("Question numbers: min", int(enade["num_int"].min()), "max", int(enade["num_int"].max()),
          "| components:", enade["component"].value_counts().to_dict())
    general_by_year = enade.groupby("year")["component"].apply(lambda values: 100 * values.eq("general").mean())
    print("Share of general-component records per year (%):", general_by_year.round(1).to_dict())
    smallest = enade.groupby("exam_edition")["num_int"].min()
    general = enade[enade["component"] == "general"]
    per_edition = general.groupby("exam_edition").size()
    print("Editions whose smallest question number is <= 10:", int((smallest <= ENADE_GENERAL_MAX_NUM).sum()),
          "of", len(smallest), "| general-component records per such edition: median", per_edition.median(),
          "max", per_edition.max())
    print("Editions with >= 8 general-component records:", int((per_edition >= 8).sum()))
    print("General component by subject:", general["subject"].value_counts().head(8).to_dict())
    mapped = enade["family"].notna()
    unmapped = enade.loc[~mapped, "course"].value_counts()
    print("Mapped courses:", enade.loc[mapped, "course"].nunique(), "of", enade["course"].nunique(),
          "| mapped records:", int(mapped.sum()), f"({100 * mapped.mean():.1f}%)",
          "| families:", enade["family"].nunique())
    print("Not audited courses:", unmapped.to_dict())
    print("Families (specific component):")
    table = enade_consistency(enade)
    print_frame(table[table["component"] == "specific"].drop(columns="component").round(1))
    print("Families (general component):")
    print_frame(table[table["component"] == "general"].drop(columns="component").round(1))
    print("Totals over mapped courses:")
    print_frame(enade_component_totals(enade).round(1))
    small = table[(table["component"] == "specific") & (table["n"] < TABLE_MIN_GROUP)]
    low, high = wilson_interval(int(small["inside"].sum()), int(small["n"].sum()))
    print(f"Families with n < {TABLE_MIN_GROUP} (specific component): {len(small)} families, "
          f"n={int(small['n'].sum())}, inside={int(small['inside'].sum())}, "
          f"share={100 * small['inside'].sum() / small['n'].sum():.1f} [{100 * low:.1f}, {100 * high:.1f}]")
    licenciatura = enade["course"].str.contains(LICENCIATURA_PATTERN, regex=True)
    print("Specific component, licenciaturas: Education share",
          round(100 * enade[(enade["component"] == "specific") & licenciatura]["subject"].eq("Education").mean(), 1))

    print("\n== Boundary examples (seed 42)")
    print_frame(boundary_examples(frame, enade))

    print("\n== Cost and runtime")
    seconds = runtime["runtime"]["seconds"]
    print("Devices:", runtime["devices"])
    memory = runtime["runtime"]["gpu_memory_after_load"]
    print("Free memory after load (GB):",
          [round(entry["free_bytes_after_load"] / 1e9, 2) for entries in memory.values() for entry in entries])
    print(f"End-to-end inference: {seconds:.1f} s = {seconds / 60:.1f} min | rounds: {runtime['runtime']['rounds']} "
          f"| {runtime['runtime']['generated'] / seconds:.2f} questions/s")
    print(f"Generation window (first to last batch logged): {facts['generation_window_seconds']:.0f} s = "
          f"{facts['generation_window_seconds'] / 60:.1f} min | "
          f"{runtime['runtime']['generated'] / facts['generation_window_seconds']:.2f} questions/s")
    print("Engine facts:", {key: value for key, value in facts.items() if key != "generation_window_seconds"})


# -----------------------------------------------------------------------------
# Figures


def exam_rows(exams: pd.DataFrame) -> list[tuple[str, str | None]]:
    """Exams grouped by level and sorted by size, with one header row per level."""
    rows = []
    for level, label in LEVEL_LABELS.items():
        rows.append((label, None))
        for exam in exams.loc[exams["level"] == level, "exam"]:
            rows.append((EXAM_LABELS.get(exam, exam), exam))
    return rows


def style_row_axis(ax: plt.Axes, rows: list[tuple[str, str | None]]) -> None:
    """Label grouped rows and set the header rows in bold."""
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    for tick_label, (_, key) in zip(ax.get_yticklabels(), rows, strict=True):
        if key is None:
            tick_label.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)


def plot_macro_area_by_exam(frame: pd.DataFrame) -> Figure:
    """Figure: share of each exam's questions in every macro-area."""
    areas = list(load_taxonomy()["levels"]["undergraduate"])
    rows = exam_rows(exam_table(frame))
    shares = 100 * pd.crosstab(frame["exam"], frame["macro_area"], normalize="index")
    shares = shares.reindex(columns=areas, fill_value=0)
    values = shares.reindex([exam for _, exam in rows]).to_numpy(dtype=float)
    values[values == 0] = np.nan

    figure, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.4), layout="constrained")
    colormap = SEQUENTIAL.copy()
    colormap.set_bad("none")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=colormap, norm=Normalize(vmin=0, vmax=100), aspect="auto")
    for (row, col), value in np.ndenumerate(values):
        if not np.isnan(value):
            ax.text(col, row, f"{value:.1f}" if value >= 1 else "<1", ha="center", va="center",
                    fontsize=5, color="white" if value >= 45 else INK)
    ax.set_xticks(range(len(areas)), [MACRO_AREA_LABELS[area] for area in areas])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    style_row_axis(ax, rows)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=ax, shrink=0.6, aspect=25, pad=0.02, ticks=range(0, 101, 25))
    colorbar.set_label("Share of the exam's questions (%)")
    colorbar.outline.set_visible(False)
    return figure


def subject_rows(table: pd.DataFrame, level: str) -> list[tuple[str, str | None]]:
    """Subjects of a level in taxonomy order, with one header row per macro-area."""
    rows = []
    for area in load_taxonomy()["levels"][level]:
        rows.append((area, None))
        subjects = table.loc[(table["macro_area"] == area) & table[level].notna(), "subject"]
        rows.extend((subject, subject) for subject in subjects)
    return rows


def plot_subjects(frame: pd.DataFrame, level: str) -> Figure:
    """Figure: questions per subject at one level, grouped by macro-area, on a linear scale."""
    table = subject_table(frame)
    rows = subject_rows(table, level)
    counts = table.set_index("subject")[level]
    positions = [row for row, (_, subject) in enumerate(rows) if subject is not None]
    values = [int(counts[subject]) for _, subject in rows if subject is not None]
    limit = max(values) * 1.18

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.115 * len(rows) + 0.55), layout="constrained")
    ax.barh(positions, values, height=0.68, color=BLUE, edgecolor="white", linewidth=0.4, zorder=2)
    for position, value in zip(positions, values, strict=True):
        ax.text(value + 0.012 * limit, position, f"{value:,}", va="center", ha="left", fontsize=6, color=MUTED)
    style_row_axis(ax, rows)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, limit)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"{LEVEL_LABELS[level]}: questions per subject")
    return figure


def plot_enade_consistency(table: pd.DataFrame) -> Figure:
    """Figure: share of ENADE labels inside the expected set per course family and component."""
    specific = table[(table["component"] == "specific") & (table["n"] >= FIGURE_MIN_GROUP)]
    specific = specific.sort_values("share", ascending=False)
    general = table[table["component"] == "general"].set_index("group").reindex(specific["group"])
    rows = range(len(specific))

    figure, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.105 * len(specific) + 0.75), layout="constrained")
    errors = [specific["share"] - specific["low"], specific["high"] - specific["share"]]
    ax.errorbar(specific["share"], rows, xerr=errors, fmt="o", markersize=3, color=BLUE, ecolor=MUTED,
                elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3, label="Course-specific component (questions 11+)")
    shown = general[general["n"] >= GENERAL_MIN_GROUP]
    ax.plot(shown["share"], [list(specific["group"]).index(group) for group in shown.index], "o", markersize=3,
            markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=0.8, zorder=3,
            label=f"General component (questions 1–10), $n \\geq {GENERAL_MIN_GROUP}$")
    for row, (n_specific, n_general) in enumerate(zip(specific["n"], general["n"], strict=True)):
        ax.text(103, row, f"{int(n_specific)} / {int(n_general)}", va="center", ha="left", fontsize=5, color=MUTED)
    ax.set_yticks(list(rows), list(specific["group"]))
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Labels inside the expected set (%, 95% Wilson interval)")
    ax.grid(axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    figure.legend(loc="outside upper center", ncol=1)
    return figure


def save_figure(figure: Figure, name: str) -> None:
    figure.savefig(FIGURE_DIR / name, metadata={"CreationDate": None}, dpi=600)
    plt.close(figure)


def main() -> int:
    inputs = (DATASET_DIR, MANIFEST_PATH, CHECKPOINT_DIR / "latest.json", LOG_PATH)
    missing = [path for path in inputs if not path.exists()]
    if missing:
        print(f"Missing inputs: {missing}. Run mmlu_pt.classification.cli first.")
        return 1

    frame = load_frame()
    manifest, identity, runtime = load_run()
    predictions = load_predictions(frame)
    facts = parse_log()
    enade = enade_frame(frame)
    print_summary(frame, predictions, manifest, identity, runtime, facts, enade)

    plt.rcParams.update(STYLE)
    FIGURE_DIR.mkdir(exist_ok=True)
    save_figure(plot_macro_area_by_exam(frame), "macro_area_by_exam.pdf")
    save_figure(plot_subjects(frame, "high_school"), "subjects_high_school.pdf")
    save_figure(plot_subjects(frame, "undergraduate"), "subjects_undergraduate.pdf")
    save_figure(plot_enade_consistency(enade_consistency(enade)), "enade_course_consistency.pdf")
    print(f"Figures written to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
