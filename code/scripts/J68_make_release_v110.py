"""J68 -- build release v1.2.0 of the repository bearing-onset-rule-audit (resubmission of Access-2026-51094).

v1.2.0 (amendment A9) supersedes v1.1.1: it adds J70 (the matched false-alarm calibration validated on independent
null series) and J71 (a scoring set that does not depend on the labelers, the alignment of training and scored
records, and the reversal test with the elapsed record count as input), J72 (the error of constant predictions on each
scoring set), their result tables and run logs, amendments A9 to A11, and the figures and README map of the shortened
manuscript. No earlier result table changed.

v1.1.1 (amendment A8) superseded v1.1.0, built by the first version of this script on the same day: it added the
exporters and supplementary tables of the planned secondary outputs that an internal review found unreported (J66),
Fig. 6 in two rows (J67), amendment A8, and the README map of the final supplementary numbering.

The Data Availability statement of the resubmitted manuscript points to
https://github.com/anhnvm-ai/bearing-onset-rule-audit, release v1.1.0. This script writes the working tree of that
release. Version 1.0.0 was the package J40 built for the first submission; 1.1.0 keeps its content (all analysis
scripts, the library, the derived data and the result tables) and adds the analyses of the resubmission (J57-J69),
their run logs, the pre-specified analysis plan with its digest and amendments, and the figures of the resubmitted
paper and its supplementary material.

Not released:
  - the four raw datasets (third-party; data/README.md says where to get them) and their windowed form (.npy);
  - the manuscript and supplementary sources and PDFs, and the authors' photographs (they go to the journal);
  - quick-run outputs ("_quick"), which are not evidence, and the out-of-fold prediction cache of J62 (46 MB,
    rewritten by J62);
  - the scripts that only produced manuscript files for earlier submissions and revisions (SKIP below). None of them
    writes a result table. J40, which built the 1.0.0 package, is released because this script reuses its
    derived-data writers.

In four scripts that read the raw acquisitions, the local dataset folder is replaced on release by the environment
variable BEARING_DATASETS (default ./datasets). Comments that refer to earlier manuscripts are shortened by the
replacements listed in paper/access_resub/release_sanitize.txt (not released), and review comments quoted word for
word in docstrings keep their identifier (R1.5, ...) but lose the quotation; the syntax tree of every edited script,
docstrings removed, must stay identical. Nothing else in the code differs from the code that produced the results.
Run logs are released with the local project folder written as <repo>.

Layout of the release (repository root):
    README.md  LICENSE (MIT, code)  LICENSE-DATA (CC BY 4.0)  CITATION.cff  MANIFEST.csv  .gitattributes  .gitignore
    code/          README.md, make_workspace.py, scripts/J*.py, src/*.py
    data/          README.md, derived/*.csv
    plan/          ANALYSIS_PLAN_2026-10-24.md and .sha256, ANALYSIS_PLAN_AMENDMENTS.md
    results/       tables/*.csv, logs/j57..j69 run logs, figures/ (every figure of the paper and its supplementary)
    environment/   requirements.txt, python_version.txt

Checks before anything is written into the release folder's MANIFEST:
  - the SHA-256 of the plan equals its .sha256 file and the digest printed in the supplementary (Section S10);
  - every result table that J66 and J67 read, and every script that J57-J69 import, is released;
  - every figure the two PDFs use is released;
  - no released file is raw, a cached array, a manuscript source, a photograph or larger than 20 MB, and no text
    file holds an absolute local path or an e-mail address.

The README's table of where each table and figure of the paper comes from is built from the sources: J66 and J67
(which tables each exporter reads), the supplementary body (which file each table inputs) and the .aux files of the
two documents (their numbers). Build both documents before running this script.

Usage:
    python scripts/J68_make_release_v110.py                       # into release/bearing-onset-rule-audit
    python scripts/J68_make_release_v110.py --out DIR
The release folder is a git working tree; rebuilding keeps its .git folder.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS, SRC = ROOT / "scripts", ROOT / "src"
TABLES, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"
MS = ROOT / "paper" / "access_resub"          # read to index the release; never released
VERSION, RELEASED = "1.2.0", "2026-10-29"
REPO = "https://github.com/anhnvm-ai/bearing-onset-rule-audit"
NAME = "bearing-onset-rule-audit"
TITLE = ("Effect of Onset Rules and Their Implementation on Degradation Labels and Remaining-Useful-Life "
         "Evaluation in Run-to-Failure Bearing Data")
AUTHORS = ["Anh Nguyen Vu Minh", "Cao-Phuc Ha", "Nguyen Le Thai", "Huu-Tho Nguyen", "Phu-Nguyen Le"]
PLAN = "ANALYSIS_PLAN_2026-10-24.md"
PLAN_SHA = "898407948fca88d93de586c4411d4173b655067a5f4f7c3b629876c91fcd9576"
PLAN_LOCKED = "24 October 2026, 11:11:16 (UTC+7)"
# Scripts not released, by number: each only produced manuscript files for earlier submissions or revisions, and
# none writes a result table.
SKIP = {"J33": "a graphical abstract that an earlier submission required",
        "J34": "a highlights file that an earlier submission required",
        "J50": "Overleaf projects of earlier manuscript trees",
        "J51": "the manuscript tree of an earlier submission, from a local template",
        "J52": "the figures of an earlier submission, restyled",
        "J53": "the submission package of an earlier submission",
        "J54": "Fig. 1 of an earlier revision, and two figures restyled from J14 and J25",
        "J55": "a redrawing of Fig. 1 of an earlier revision",
        "J56": "Fig. 1 of an earlier revision as a draw.io file"}


def skipped(name: str) -> bool:
    return name.split("_", 1)[0] in SKIP
SANITIZE = MS / "release_sanitize.txt"          # comment edits for the public copies; not released
DATASETS_ENV = "BEARING_DATASETS"
RUN_LOG = re.compile(r"^j(5[7-9]|6\d|7[0-2])_run\.log$")
TEMPLATE_IMAGES = {"bullet.png", "logo.png", "notaglinelogo.png"}

# Figures of the two documents that J67 does not draw: the script that writes each one. Two of them were restyled
# for an earlier revision by J54, which is not released; J14 and J25 draw them from the same data.
FIG_SCRIPT = {"fig2_onset_matrix_column": "J14_journal_figures.py", "fig8_design_sweep": "J25_reviewer_figures.py",
              "fig3_shift_at_alarm": "J21_fig_shift.py", "fig6_label_agreement": "J25_reviewer_figures.py",
              "fig7_rank_heatmap": "J25_reviewer_figures.py", "fig5_convention_grid": "J14_journal_figures.py",
              "j60_warmup_first100": "J60_baseline_protocols.py"}
# Short descriptions of the main paper's tables and figures, by label.
MAIN_ITEMS = {"tab:datasets": "Datasets and trajectories", "tab:rules": "Rules, baseline protocols and onset scaling",
              "tab:nonfiring": "Treatment and weighting of non-firing trajectories",
              "fig:framework": "Framework of the study (schematic)", "fig:protocol": "Detector-minus-chart gap per protocol",
              "fig:fraction": "Onset fraction against trajectory length", "fig:simulation": "Simulated gradual degradation",
              "fig:labels": "Label disagreement and target distance", "fig:rul": "RUL evaluation on the common reference",
              "tab:ord": "ORD-4 reporting items"}
# Table bodies that J66 writes for the main paper, and the J66 function that writes each one.
MAIN_TABLE_FN = {"resub_tab_rules.tex": "table_rules", "resub_tab_nonfiring.tex": "table_nonfiring"}
SUPP_TABLE_FN = {"supp_tab_configs.tex": "supp_table_configs", "supp_tab_models.tex": "supp_table_models"}

PIPELINE_NEW = [  # step, what it does, script, needs
    ("18", "Treatment x weighting of non-firing trajectories; Kaplan-Meier identity", "J57_censoring_decomposition.py", "derived"),
    ("19", "Onset fraction and onset scaling per rule and operating condition", "J58_onset_fraction.py", "derived"),
    ("20", "Rule configurations and model settings, read from the code", "J59_configurations.py", "derived"),
    ("21", "Baseline protocols P-retro to P-prospective; onsets, gaps, warm-up test, contamination",
     "J60_baseline_protocols.py", "derived"),
    ("22", "False-alarm calibration on an AR(1) null; matched onsets, ARL0, matched gaps", "J61_far_calibration.py", "derived"),
    ("23", "RUL evaluation on the common retrospective reference: C-rank, C-common, reversals, main effects "
           "(GRU and TCN need torch)", "J62_common_reference_rul.py", "derived"),
    ("24", "Simulated gradual degradation under each protocol", "J63_gradual_simulation.py", "derived"),
    ("25", "Design-parameter models M0-M4, in and out of sample", "J64_design_parameters.py", "derived"),
    ("26", "Linear mixed models with a random intercept per bearing (sensitivity)", "J65_mixed_effects.py", "derived"),
    ("27", "Amendment A6 (post hoc): ratio of labeler to model effects", "J69_labeler_model_contrast.py", "derived"),
    ("28", "Amendment A9: the calibrated multipliers of J61 on independent null series", "J70_far_validation.py", "derived"),
    ("29", "Amendment A9 (post hoc): fixed scoring set, alignment of training and scored records, reversal test with "
           "the elapsed record count (re-scores the prediction cache of J62)", "J71_fixed_scoring_set.py", "derived"),
    ("30", "Amendment A11 (post hoc, descriptive): constant predictions on each scoring set", "J72_constant_reference.py",
     "derived"),
    ("31", "Every number and table body of the resubmitted paper (writes into paper/access_resub/)",
     "J66_resub_export_numbers.py", "derived"),
    ("32", "The figures of the resubmitted paper and two of its supplementary material (writes into "
           "paper/access_resub/figures/)", "J67_resub_figures.py", "derived"),
    ("33", "This release", "J68_make_release_v110.py", "derived"),
]
RUN_ORDER = ["J57_censoring_decomposition.py", "J58_onset_fraction.py", "J59_configurations.py",
             "J60_baseline_protocols.py", "J61_far_calibration.py", "J62_common_reference_rul.py",
             "J63_gradual_simulation.py", "J64_design_parameters.py", "J65_mixed_effects.py",
             "J69_labeler_model_contrast.py", "J70_far_validation.py", "J71_fixed_scoring_set.py",
             "J72_constant_reference.py", "J66_resub_export_numbers.py", "J67_resub_figures.py"]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


J40 = load("J40_make_zenodo_package")      # derived-data writers, licence texts, manifest, environment


def sha256(path: Path) -> str:
    return J40.sha256(path)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ------------------------------------------------------------------ index: where each table and figure comes from
def functions(path: Path) -> dict[str, str]:
    """Top-level functions of a script: name -> source text, up to the end of the function's indented block."""
    src = path.read_text(encoding="utf-8")
    out = {}
    for part in re.split(r"(?m)^def ", src)[1:]:
        lines = part.splitlines()
        body = [lines[0]]
        for line in lines[1:]:
            if line.strip() and not line[0].isspace() and not line.startswith(("#", ")")):
                break
            body.append(line)
        out[part.split("(", 1)[0]] = "\n".join(body)
    return out


def reads(fns: dict[str, str], name: str, seen: set[str] | None = None) -> list[str]:
    """Result tables read by a function of J66/J67, including through the helpers it calls."""
    seen = set() if seen is None else seen
    if name in seen or name not in fns:
        return []
    seen.add(name)
    body = fns[name]
    out = re.findall(r'read\("([^"]+)"\)', body)
    for callee in re.findall(r"\b([A-Za-z_]\w*)\(", body):
        if callee != name and callee in fns:
            out += reads(fns, callee, seen)
    return list(dict.fromkeys(out))


def j66_supp_functions(src: str, fns: dict[str, str]) -> dict[str, str]:
    """supp_tab_<key>.tex -> J66 function, from the SUPP_TABLES dictionaries (entries whose value is a function)."""
    out = {}
    for key, lam, fn in re.findall(r'"(supp_\w+)":\s*(?:lambda:\s*(\w+)\(|(\w+))', src):
        if (lam or fn) in fns:
            out[f"supp_tab_{key[5:]}.tex"] = lam or fn
    return out


def aux_numbers(aux: Path) -> dict[str, str]:
    return dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux.read_text(encoding="utf-8", errors="replace")))


def environments(tex: str) -> list[dict]:
    """Every table, long table and figure of a LaTeX body, in order: label, input file and graphics."""
    out = []
    for m in re.finditer(r"\\begin\{(table\*?|longtable|figure\*?)\}(.*?)\\end\{\1\}", tex, re.S):
        body = m.group(2)
        lab = re.search(r"\\label\{([^}]+)\}", body)
        out.append({"kind": "figure" if m.group(1).startswith("figure") else "table", "label": lab.group(1) if lab else "",
                    "inputs": re.findall(r"\\(?:tabinput|input)\{([^}]+)\}", body),
                    "graphics": [Path(g).name for g in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", body)],
                    "start": m.start()})
    return out


_SOURCES: dict[str, str] = {}


def producer(table: str, released: set[str]) -> str:
    """The script that writes a result table: the one whose to_csv names the file, else J<prefix>_*.py by the naming
    convention of this project (tables whose name is built at run time)."""
    if not _SOURCES:
        _SOURCES.update({s: (SCRIPTS / s).read_text(encoding="utf-8") for s in released})
    pat = re.compile(r"to_csv\([^)\n]*[\"/]" + re.escape(table) + r"\.csv")
    writers = sorted(s for s, src in _SOURCES.items() if pat.search(src) and s not in (Path(__file__).name, "J40_make_zenodo_package.py"))
    prefix = table.split("_", 1)[0].lower() + "_"
    if writers:
        return next((s for s in writers if s.lower().startswith(prefix)), writers[0])
    cands = sorted(s for s in released if s.lower().startswith(prefix))
    return cands[0] if cands else ""


def build_index(released: set[str]) -> tuple[list[dict], list[str], list[str]]:
    """Rows of the README table (item, what, script, result tables with their writers); tables read; tables
    whose writer the naming convention does not give."""
    f66, f67 = functions(SCRIPTS / "J66_resub_export_numbers.py"), functions(SCRIPTS / "J67_resub_figures.py")
    supp_fn = {**j66_supp_functions((SCRIPTS / "J66_resub_export_numbers.py").read_text(encoding="utf-8"), f66),
               **SUPP_TABLE_FN}
    fig67 = {}
    for fn, body in f67.items():
        for name in re.findall(r'save\(fig,\s*"([^"]+)"\)', body):
            fig67[name] = reads(f67, fn)
    needed, orphans = set(), set()
    rows = []

    def tables_cell(tabs: list[str]) -> str:
        needed.update(tabs)
        cells = []
        for t in tabs:
            w = producer(t, released)
            orphans.update([] if w else [t])
            cells.append(f"`{t}`" + (f" ({w.split('_')[0]})" if w else ""))
        return ", ".join(cells)

    # main paper
    num = aux_numbers(MS / "main.aux")
    main_tex = (MS / "main.tex").read_text(encoding="utf-8")
    body = "".join((MS / f"{n}.tex").read_text(encoding="utf-8") for n in re.findall(r"\\input\{(sec[^}]+)\}", main_tex))
    for e in environments(body):
        if e["label"] not in MAIN_ITEMS:
            continue
        item = ("Table " if e["kind"] == "table" else "Fig. ") + num.get(e["label"], "?")
        what = MAIN_ITEMS[e["label"]]
        if e["kind"] == "table":
            ins = [i for i in e["inputs"] if i in MAIN_TABLE_FN]
            if ins:
                rows.append(dict(part="main", item=item, what=what, script="`J66_resub_export_numbers.py`",
                                 tables=tables_cell(reads(f66, MAIN_TABLE_FN[ins[0]]))))
            elif e["label"] == "tab:datasets" or e["label"] == "tab:ord":
                rows.append(dict(part="main", item=item, what=what, script="`J66_resub_export_numbers.py` (numbers)",
                                 tables="the tables named in the comments of the exported macro file"))
        else:
            g = e["graphics"][0]
            rows.append(dict(part="main", item=item, what=what, script="`J67_resub_figures.py`",
                             tables=tables_cell(fig67.get(g, [])) or "none (schematic)"))
    rows.append(dict(part="main", item="Text", what="Every other number of the paper", script="`J66_resub_export_numbers.py`",
                     tables="the tables named in the comments of the exported macro file"))
    # supplementary
    snum = aux_numbers(MS / "supplementary.aux")
    sbody = (MS / "supp_body.tex").read_text(encoding="utf-8")
    sections = [(m.start(), m.group(1)) for m in re.finditer(r"\\section\{([^}]+)\}", sbody)]
    for e in environments(sbody):
        sec_i = max(i for i, (pos, _) in enumerate(sections) if pos < e["start"])
        sec = f"S{sec_i + 1} {sections[sec_i][1]}"
        item = ("Table " if e["kind"] == "table" else "Fig. ") + snum.get(e["label"], "?")
        if e["kind"] == "figure":
            g = e["graphics"][0]
            script = "J67_resub_figures.py" if g in fig67 else FIG_SCRIPT.get(g, "")
            rows.append(dict(part="supp", item=item, what=sec, script=f"`{script}`" if script else "", graphic=g,
                             tables=tables_cell(fig67.get(g, [])) if g in fig67 else "see the script"))
            continue
        ins = [Path(i).name if i.endswith(".tex") else Path(i).name + ".tex" for i in e["inputs"]]
        ins = [i for i in ins if i.startswith("supp_tab_")]
        if not ins:
            rows.append(dict(part="supp", item=item, what=sec, script="text", tables="the plan and its amendments"))
        elif ins[0] == "supp_tab_defect.tex":
            rows.append(dict(part="supp", item=item, what=sec, script="earlier version (`supp_tables.py` of the first submission)",
                             tables=tables_cell(["j20_scored_check"])))
        else:
            fn = supp_fn.get(ins[0], "")
            rows.append(dict(part="supp", item=item, what=sec, script="`J66_resub_export_numbers.py` (rows)",
                             tables=tables_cell(reads(f66, fn)) if fn else "?"))
    # every table any exporter reads, whether or not it feeds a table body
    for fn in f66:
        needed.update(reads(f66, fn))
    for fn in f67:
        needed.update(reads(f67, fn))
    return rows, sorted(needed), sorted(orphans)


# ------------------------------------------------------------------ release content
RAW_PATH = re.compile(r'Path\(r?"[A-Za-z]:[/\\]Users[/\\][^/\\"]+[/\\]Bearings[/\\]datasets([^"]*)"\)')


def released_source(text: str) -> tuple[str, int]:
    """The local dataset folder becomes the environment variable BEARING_DATASETS (default ./datasets)."""
    def sub(m: re.Match) -> str:
        rest = m.group(1).replace("\\", "/").lstrip("/")
        base = f'Path(os.environ.get("{DATASETS_ENV}", "datasets"))'
        return f'{base} / "{rest}"' if rest else base
    new, n = RAW_PATH.subn(sub, text)
    if n and not re.search(r"(?m)^import os\b|^import [\w, ]*\bos\b", new):
        if "from pathlib import Path" not in new:
            raise SystemExit("FAIL: a script with a dataset path does not import Path the expected way")
        new = new.replace("from pathlib import Path", "import os\nfrom pathlib import Path", 1)
    return new, n


def sanitize_rules() -> tuple[list[tuple[str, str]], list[str]]:
    reps, terms = [], []
    for line in SANITIZE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("!"):
            terms.append(line[1:].strip())
        else:
            old, new = (x.strip() for x in line.split("=>"))
            reps.append((old, new))
    return reps, terms


def syntax(text: str) -> str:
    """The syntax tree of a script with its docstrings removed: equal trees differ only in comments and docstrings."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


# A review comment quoted word for word in a docstring: its identifier (R1.5, optionally after "Reviewer 1,"), a colon
# and the quotation in double quotes, possibly over several lines. The review reports are confidential, so the public
# copies keep the identifier and drop the quotation.
REVIEW_QUOTE = re.compile(r'(?m)^((?:Reviewer \d, )?R\d\.\d+): "[^"]*"[ \t]*$')


def copy_code(pkg: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    sc = pkg / "code" / "scripts"
    sc.mkdir(parents=True, exist_ok=True)
    reps, _ = sanitize_rules()
    released, changed, edited = [], [], []
    for f in sorted(SCRIPTS.glob("J*.py")):
        if skipped(f.name):
            continue
        text, n = released_source(f.read_text(encoding="utf-8"))
        clean = REVIEW_QUOTE.sub(r"\1: a comment of the review reports, answered in the response letter.", text)
        for old, new in reps:
            clean = clean.replace(old, new)
        if clean != text:
            if syntax(clean) != syntax(text):
                raise SystemExit(f"FAIL: shortening comments of {f.name} would change its code")
            edited.append(f.name)
        write(sc / f.name, clean)
        released.append(f.name)
        if n:
            changed.append(f"{f.name} ({n})")
    lib = []
    (pkg / "code" / "src").mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.glob("*.py")):
        shutil.copy2(f, pkg / "code" / "src" / f.name)
        lib.append(f.name)
    write(pkg / "code" / "make_workspace.py", MAKE_WORKSPACE)
    return released, lib, changed, edited


def copy_tables(pkg: Path) -> list[str]:
    t = pkg / "results" / "tables"
    t.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(TABLES.glob("*.csv")):
        if (f.name.startswith(("j", "fig")) or f.name == "e2_raw_counts_h5.csv") and "_quick" not in f.name:
            shutil.copy2(f, t / f.name)
            out.append(f.stem)
    return out


def scrub(text: str) -> str:
    """The local project folder written as <repo>; any other local home folder as <home>."""
    for root in {str(ROOT), ROOT.as_posix()}:
        text = re.sub(re.escape(root), "<repo>", text, flags=re.I)
    text = re.sub(r"(?i)\b[a-z]:[\\/]+users[\\/]+[^\\/\s]+", "<home>", text)
    return text


def copy_logs(pkg: Path) -> list[str]:
    out = []
    for f in sorted(LOGS.glob("*.log")):
        if RUN_LOG.match(f.name):
            write(pkg / "results" / "logs" / f.name, scrub(f.read_text(encoding="utf-8", errors="replace")))
            out.append(f.name)
    return out


def copy_figures(pkg: Path) -> list[str]:
    """Every graphics file the two PDFs use, as their build logs list them, with its PNG twin when there is one."""
    g = pkg / "results" / "figures"
    g.mkdir(parents=True, exist_ok=True)
    out = []
    for log in ("main.log", "supplementary.log"):
        for used in re.findall(r"<use ([^>]+)>", (MS / log).read_text(encoding="utf-8", errors="replace")):
            f = (MS / used).resolve()
            if f.suffix.lower() in (".jpg", ".jpeg") or f.name in TEMPLATE_IMAGES:
                continue
            for p in (f, f.with_suffix(".png")):
                if p.exists() and not (g / p.name).exists():
                    shutil.copy2(p, g / p.name)
            out.append(f.stem)
    return sorted(set(out))


def copy_plan(pkg: Path) -> None:
    src_plan, src_sha = MS / PLAN, MS / PLAN.replace(".md", ".sha256")
    got = sha256(src_plan)
    listed = src_sha.read_text(encoding="utf-8").split()[0]
    printed = PLAN_SHA in (MS / "supp_body.tex").read_text(encoding="utf-8")
    if not (got == listed == PLAN_SHA and printed):
        raise SystemExit(f"FAIL: plan digest {got} / .sha256 {listed} / supplementary {printed} disagree")
    (pkg / "plan").mkdir(parents=True, exist_ok=True)
    for f in (src_plan, src_sha, MS / "ANALYSIS_PLAN_AMENDMENTS.md"):
        shutil.copyfile(f, pkg / "plan" / f.name)      # contents only: the locked original is read-only
    if sha256(pkg / "plan" / PLAN) != PLAN_SHA:
        raise SystemExit("FAIL: the released copy of the plan has a different digest")


def derived(pkg: Path) -> dict:
    d = pkg / "data" / "derived"
    d.mkdir(parents=True, exist_ok=True)
    lab = J40.onset_labels(d)
    des = J40.rule_designs(d)
    met = J40.bearing_metadata(d, lab)
    tgt = J40.rul_targets(d, lab)
    idx = J40.record_index(d)
    copies = {"onsets_by_protocol.csv": "j60_onsets.csv", "onsets_calibrated.csv": "j61_matched_onsets.csv",
              "rule_configurations.csv": "j59_rule_configurations.csv", "model_settings.csv": "j59_model_settings.csv"}
    n = {}
    for dst, src in copies.items():
        shutil.copy2(TABLES / src, d / dst)
        n[dst] = len(pd.read_csv(TABLES / src))
    return dict(onsets=len(lab), designs=len(des), n_designs=des.design_id.nunique(), trajectories=len(met),
                targets=len(tgt), record_index=idx, copies=copies, copy_rows=n)


def environment(pkg: Path) -> None:
    import importlib.metadata as md
    lines = ["# versions the released results were produced with"]
    for pkgname, note in (("numpy", ""), ("pandas", ""), ("scipy", ""), ("scikit-learn", ""), ("statsmodels", "J65"),
                          ("matplotlib", ""), ("pyarrow", "J62 prediction cache"), ("torch", "GRU and TCN only (J47, J62)")):
        try:
            v = f"{pkgname}=={md.version(pkgname)}"
        except md.PackageNotFoundError:
            v = pkgname
        lines.append(v + (f"   # {note}" if note else ""))
    write(pkg / "environment" / "requirements.txt", "\n".join(lines) + "\n")
    write(pkg / "environment" / "python_version.txt", sys.version.split()[0] + "\n")


# ------------------------------------------------------------------ text files
MAKE_WORKSPACE = '''"""Lay this repository out so the analysis scripts resolve their relative paths.

Every script computes the project root as the parent of its own directory and reads results/tables, data/ and
colab_bundle/ from there. This copies code/scripts and code/src to the repository root, writes the record-index files
where the scripts expect them, and creates the folders into which the exporters write.

    python code/make_workspace.py            # into the repository root
    python code/make_workspace.py --dest DIR # into DIR (results/, data/ and plan/ are copied there too)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=REPO)
    dest = ap.parse_args().dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("scripts", "src"):
        shutil.copytree(HERE / name, dest / name, dirs_exist_ok=True)
        print(f"  {name}/ -> {dest / name}")
    if dest != REPO:
        for name in ("results", "data", "plan", "environment"):
            shutil.copytree(REPO / name, dest / name, dirs_exist_ok=True)
            print(f"  {name}/ -> {dest / name}")
    # the exporters write LaTeX files and figures into the manuscript folders, which are not released
    for name in ("paper/access_resub/figures", "paper/access_rev/figures", "paper/ieee", "paper/journal"):
        (dest / name).mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / "data" / "derived").glob("record_index_*.csv")):
        out = dest / "colab_bundle" / f.stem.replace("record_index_", "")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / "records.csv")
        print(f"  {f.name} -> {out / 'records.csv'}")
    print(f"\\nworkspace ready in {dest}")
    print("The raw windows (colab_bundle/<dataset>/windows.npy) are not released; data/README.md says how to rebuild them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def readme(c: dict, rows: list[dict]) -> str:
    def table(part: str) -> str:
        return "\n".join(f"| {r['item']} | {r['what']} | {r['script']} | {r['tables']} |" for r in rows if r["part"] == part)
    authors = ", ".join(AUTHORS[:-1]) + " and " + AUTHORS[-1]
    order = "\n".join(f"python scripts/{s}" for s in RUN_ORDER)
    figs = [r["item"].replace("Fig. ", "") for r in rows if r.get("graphic") in ("fig2_onset_matrix_column", "fig8_design_sweep")]
    restyled = f"Figs. {' and '.join(figs)} are" if len(figs) > 1 else f"Fig. {figs[0]} is"
    defect = next(r["item"] for r in rows if r["script"].startswith("earlier version"))
    return f"""# {NAME}

Code and derived data for the article

> {authors}, "{TITLE}," manuscript under review, 2026.

Run-to-failure bearing datasets record no degradation onset, so an onset rule creates one. This repository holds the
onsets that ten published rules and {c['n_designs']} rule designs place on {c['trajectories']} trajectories from four public datasets,
the labels and remaining-useful-life (RUL) targets derived from them, and the code that produced every number, table
and figure of the article and its supplementary material. Release {VERSION} accompanies the resubmitted manuscript.

## What is here

```
{NAME}/
  README.md         this file
  LICENSE           MIT, for everything under code/
  LICENSE-DATA      CC BY 4.0, for everything under data/, plan/ and results/
  CITATION.cff      how to cite this repository
  MANIFEST.csv      every file with its size and SHA-256
  code/             {c['n_scripts']} analysis scripts, a {c['n_lib']}-module library and the pipeline map (code/README.md)
  data/             the derived data ({c['n_derived']} files); no raw dataset is redistributed (data/README.md)
  plan/             the pre-specified analysis plan, its SHA-256 at locking, and its amendments
  results/tables/   {c['n_tables']} result tables
  results/logs/     the run logs of the analyses added for the resubmission (J57-J72), with their self-checks
  results/figures/  every figure of the article and its supplementary material
  environment/      package versions the results were produced with
```

## Where each table and figure comes from

Each script writes the result tables named after it (`j60_*` by `J60_*.py`, and so on). The writer of each table is
shown in brackets. `J66_resub_export_numbers.py` writes every number of the article as a LaTeX macro whose comment
names the table it comes from, and writes the rows of every data table; `J67_resub_figures.py` draws the figures
of the article and two figures of its supplementary material.

### Article

| Item | What | Script | Result tables |
|---|---|---|---|
{table('main')}

### Supplementary material

| Item | Section | Script | Result tables |
|---|---|---|---|
{table('supp')}

{restyled} the figures of `J14_journal_figures.py` and `J25_reviewer_figures.py`, restyled for an earlier
revision of the manuscript; the restyling script is not released. {defect} was written by the table script of the
first submission from `j20_scored_check`.

## The pre-specified analysis plan

The protocol, calibration, simulation, design-parameter and RUL analyses added for the resubmission were specified in
`plan/{PLAN}` before they were run. The plan was locked on {PLAN_LOCKED}; its SHA-256 at locking is

```
{PLAN_SHA}
```

Check it with `sha256sum plan/{PLAN}` (Linux, macOS) or `Get-FileHash plan/{PLAN}` (Windows PowerShell).
Deviations were recorded in the append-only file `plan/ANALYSIS_PLAN_AMENDMENTS.md` (amendments A1-A11), each before
any result it affects was reported. A6, A9 and A11 add analyses after related results had been seen and are labeled
post hoc; A9 also validates the calibration on new null series. A7 and A8 concern the release and the reporting of
secondary outputs, and A10 corrects the writing time stated in A9. The digest was recorded by the authors at locking
and first published with release 1.1.0, after the planned analyses had been run.

## Reproducing

The scripts resolve the project root as the parent of their own directory. One command lays them out that way:

```
python code/make_workspace.py
{order}
```

These steps read only the derived data and result tables released here; J62 needs `torch` for its GRU and TCN models
and runs for several hours on a laptop. J71 re-scores the out-of-fold predictions that J62 caches in
`results/cache_j62/`, which are not released, so it runs after J62. The full pipeline, including the earlier steps, is in `code/README.md`. The
steps that read the raw acquisitions need the datasets rebuilt first (`data/README.md`); set the environment variable
`{DATASETS_ENV}` to the folder that holds them.

Every script that wrote a result table of the resubmission checks itself before writing (for example, J62 reproduces
the submitted onsets and the earlier RUL models, J69 reproduces the labeler main effects of J62 within 1e-9, J70
reproduces every calibration of J61 and J71 every value and reversal of J62); the checks and their outcomes are in
`results/logs/`.

## Licences

Code under `code/` is MIT (`LICENSE`). The derived data, the plan, the result tables, logs and figures are CC BY 4.0
(`LICENSE-DATA`). The four run-to-failure datasets are third-party, are **not** redistributed here, and keep the terms
of their own providers; `data/README.md` names each source.

## How to cite

Cite the article and this repository:

> {authors}, "{NAME}: code and derived data, release {VERSION}," {REPO}, 2026.

## Versions

| Version | Date | Content |
|---|---|---|
| 1.0.0 | 2026-10-02 | Code and derived data of the first submission |
| 1.1.0 | 2026-10-26 | Adds the analyses of the resubmission (J57-J69), their logs, the pre-specified plan and the figures of the resubmitted paper |
| 1.1.1 | 2026-10-26 | Amendment A8: adds the supplementary tables of the planned secondary outputs (C-rank reversals, rank agreement, Spearman correlations, simulation delays, median run lengths), Fig. 6 in two rows, and the map of the final supplementary numbering; no result table changed |
| {VERSION} | {RELEASED} | Cited by the resubmission (amendment A9): adds the calibration validated on independent null series (J70), the fixed scoring set, the alignment of training and scored records and the reversal test with the elapsed record count (J71), constant predictions on each scoring set (J72), amendments A9 to A11, and the figures and map of the shortened manuscript; no earlier result table changed |
"""


def code_readme(c: dict) -> str:
    rows = "\n".join(f"| {s} | {what.replace('this deposit', 'this repository')} | `{scr}` | "
                     f"{'raw datasets' if need == 'raw' else 'this repository'} |"
                     for s, what, scr, need in list(J40.PIPELINE) + PIPELINE_NEW)
    changed = "; ".join(c["changed"]) if c["changed"] else "none"
    not_released = "\n".join(f"- {k}: {v}." for k, v in SKIP.items())
    return f"""# Code

{c['n_scripts']} scripts and the {c['n_lib']}-module library they import. File names are the ones the article, its
supplementary material and the logs refer to, so they are kept as they are rather than renumbered.

```
code/
  make_workspace.py   lays the scripts out so that their relative paths resolve
  scripts/            the analyses: J3-J49 (first submission) and J57-J72 (resubmission)
  src/                the library the scripts import (features, metrics, thresholds, splits)
```

## Running

```
python code/make_workspace.py
python scripts/<script>.py
```

`make_workspace.py` copies `code/scripts` and `code/src` to the repository root, next to `results/` and `data/`,
because every script resolves the project root as the parent of its own directory. It writes
`colab_bundle/<dataset>/records.csv` from `data/derived/record_index_*.csv`, and it creates the manuscript folders
into which the exporters write (the manuscript itself is not released).

## The pipeline

| Step | What it does | Script | Needs |
|---|---|---|---|
{rows}

Steps 01, 02 and 12 read the raw acquisitions; every other step runs from the tables in `results/tables/`.

## Differences from the code that produced the results

In the scripts that read the raw acquisitions, the local dataset folder was replaced by the environment variable
`{DATASETS_ENV}` (default `./datasets`) when this release was built: {changed}. In {len(c['edited'])} scripts, comments
that referred to an earlier manuscript or quoted a review comment were shortened; their code is unchanged, which the
release build checks by comparing syntax trees. Nothing else differs.

Not released, because each only produced manuscript files for earlier submissions or revisions and none writes a
result table:
{not_released}

## Conventions the scripts share

Bearings, not records or designs, are the independent units: bootstrap intervals resample whole bearings within each
operating condition (20 000 resamples), and cross-validation holds out whole bearings. A label set "fails the
prevalence screen" when its positive rate falls outside [0.01, 0.50]. Scripts that write a `_quick` output were run
on a reduced grid as a code check; such outputs are not evidence and are not released.
"""


def data_readme(c: dict) -> str:
    rows = "\n".join(f"| {v['name']} | {v['bearings_used']} | {v['record']} | {v['stop_criterion']} | {v['source']} |"
                     for v in J40.DATASETS.values())
    copies = "\n".join(f"- `{dst}` -- a copy of `results/tables/{src}` ({c['copy_rows'][dst]:,} rows)."
                       for dst, src in c["copies"].items())
    return f"""# Data

## The raw datasets are not redistributed

The original datasets are public, each carries its own terms, and each should be obtained from its provider:

| Dataset | Bearings used | Record | Test stopped when | Source |
|---|---|---|---|---|
{rows}

- **XJTU-SY**: https://biaowang.tech/xjtu-sy-bearing-datasets/ (the tutorial of Lei et al., 2019,
  doi: 10.3901/JME.2019.16.001, describes the tests).
- **FEMTO (PRONOSTIA, IEEE PHM 2012 challenge)**: NASA Prognostics Data Repository,
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **IMS (NSF I/UCRC Center for Intelligent Maintenance Systems)**: NASA Prognostics Data Repository, as above.
- **NSK 6205**: Mendeley Data, doi: 10.17632/5hcdd3tdvb.6 (version 6), released with Jung et al. (2024).

## What this repository holds

`data/derived/` holds only quantities computed from those datasets: onset indices, targets and metadata. It holds no
raw acquisition.

- `onset_labels.csv` -- one row per trajectory and rule under the retrospective protocol P-retro ({c['onsets']:,} rows).
  `onset_index` is the first record labeled degraded, empty when the rule never fired (`fired` false, `censored`
  true). Records `onset_index` to `n_records - 1` are degraded, earlier records healthy.
- `rule_designs.csv` -- the {c['n_designs']} designs of the design sweep, per dataset, with the size of change at their alarms.
- `bearing_metadata.csv` -- one row per trajectory ({c['trajectories']} rows): length, operating condition, acquisition
  and stopping criterion of its dataset.
- `rul_targets.csv` -- the RUL target of the article for every record of every (trajectory, rule) pair on which the
  rule fires: 1 before the onset, falling linearly to 0 at the last record ({c['targets']:,} rows).
- `record_index_{{xjtu,ims,nsk}}.csv` -- the row order of `results/tables/j7_features_*.csv`, one row per record.
  FEMTO carries its own identifiers inside `results/tables/j7c_femto_features.csv`.
{copies}

`onsets_by_protocol.csv` gives every onset under each baseline protocol (`arm`), with the baseline window `W`, the
records removed by iterative cleaning, the onset `i_star`, the onset fraction `u` and whether the onset falls inside the
baseline. `onsets_calibrated.csv` gives the onsets of the rules calibrated to a false-alarm rate (`level`), with the
calibrated `multiplier`.

A trajectory on which a rule never fires has no labels under that rule. Do not label it healthy without saying so:
the article shows that the treatment and weighting of such trajectories change the positive rate.

## If you rebuild the raw datasets

The code reads preprocessed acquisitions from `colab_bundle/<dataset>/` next to the scripts (`records.csv`, one row per
record, and `windows.npy`, the windowed vibration samples). `python code/make_workspace.py` writes the `records.csv`
files; the `windows.npy` arrays are rebuilt from the downloaded datasets with steps 01 and 02 of `code/README.md`.
Set the environment variable `{DATASETS_ENV}` to the folder that holds the downloaded datasets.
"""


def citation() -> str:
    authors = "\n".join(f'  - name: "{a}"' for a in AUTHORS)
    return f"""cff-version: 1.2.0
message: "If you use this code or data, please cite the article and this repository."
type: software
title: "{NAME}: code and derived data"
abstract: >-
  Code and derived data for the article "{TITLE}".
authors:
{authors}
version: "{VERSION}"
date-released: "{RELEASED}"
repository-code: "{REPO}"
license:
  - MIT
  - CC-BY-4.0
keywords:
  - bearing prognostics
  - degradation onset
  - first prediction time
  - remaining useful life
  - run-to-failure data
  - statistical process control
"""


GITATTRIBUTES = ("# Every file is stored byte for byte, so that MANIFEST.csv and the SHA-256 of the plan hold in every clone.\n"
                 "* -text\n")
GITIGNORE = """# written by code/make_workspace.py and by running the scripts
/scripts/
/src/
/colab_bundle/
/paper/
/datasets/
results/cache_*/
results/tables/*_quick*
results/logs/*_quick*
__pycache__/
*.pyc
"""

DATA_LICENCE_HEADER = """Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2026 {authors}

This licence covers the files of this repository that are derived from the four run-to-failure datasets, and the
analysis plan: everything under data/, plan/ and results/. It does NOT cover the datasets themselves, which are not
part of this repository and stay under the terms of their own providers (see data/README.md).

Attribution: cite the article and this repository (see README.md, "How to cite").

The full legal code follows.

=======================================================================

"""


# ------------------------------------------------------------------ checks
EMAIL = re.compile(r"(?i)[a-z0-9._%+-]+@[a-z0-9-]+(\.[a-z0-9-]+)*\.(com|org|net|edu|gov|info|io|ac|vn|uk|de|fr|cn|jp|kr)\b")
LOCAL = re.compile(r"(?i)\b[a-z]:[\\/]+(users|documents and settings)[\\/]|/[a-z]/users/|appdata[\\/]")
TEXT = {".py", ".md", ".csv", ".txt", ".log", ".cff", ".sha256", ""}


def guard(pkg: Path) -> list[str]:
    bad = []
    _, terms = sanitize_rules()
    term_re = [re.compile(r"(?i)(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])") for t in terms]
    for p in pkg.rglob("*"):
        if not p.is_file() or ".git" in p.relative_to(pkg).parts:
            continue
        rel = p.relative_to(pkg).as_posix()
        suf = p.suffix.lower()
        if suf in (".npy", ".npz", ".pt", ".pth", ".h5", ".parquet", ".pkl", ".zip"):
            bad.append(f"raw or cached data: {rel}")
        if suf in (".tex", ".bib", ".cls", ".sty", ".bst", ".aux", ".jpg", ".jpeg"):
            bad.append(f"manuscript source or photograph: {rel}")
        if suf == ".pdf" and not rel.startswith("results/figures/"):
            bad.append(f"unexpected PDF: {rel}")
        if "_quick" in p.name:
            bad.append(f"quick-run output: {rel}")
        if p.stat().st_size > 20 * 1024 * 1024:
            bad.append(f"larger than 20 MB: {rel}")
        if suf in TEXT or p.name.startswith((".git", "LICENSE")):
            txt = p.read_text(encoding="utf-8", errors="replace")
            if LOCAL.search(txt):
                bad.append(f"absolute local path: {rel}: {LOCAL.search(txt).group(0)}")
            if rel != "LICENSE-DATA" and EMAIL.search(txt):
                bad.append(f"e-mail address: {rel}: {EMAIL.search(txt).group(0)}")
            for r in term_re:
                if r.search(txt):
                    bad.append(f"term listed in {SANITIZE.name}: {rel}: {r.pattern}")
            if REVIEW_QUOTE.search(txt) or re.search(r'R\d\.\d+: "', txt):
                bad.append(f"review comment quoted word for word: {rel}")
    return bad


def completeness(pkg: Path, needed: list[str], figures: list[str], released: list[str]) -> list[str]:
    bad = [f"table read by J66/J67 not released: {t}" for t in needed if not (pkg / "results" / "tables" / f"{t}.csv").exists()]
    for s in released:                         # every script another released script loads is released
        if s == Path(__file__).name:           # this script names unreleased scripts in SKIP
            continue
        src = (pkg / "code" / "scripts" / s).read_text(encoding="utf-8")
        deps = re.findall(r'"(J\d+[a-z]?_\w+\.py)"', src)
        deps += [m + ".py" for m in re.findall(r"(?m)^\s*(?:import|from)\s+(J\d+[a-z]?_\w+)", src)]
        for dep in sorted(set(deps)):
            if dep not in released and (SCRIPTS / dep).exists():
                bad.append(f"{s} loads {dep}, which is not released")
    for f in figures:
        if not any((pkg / "results" / "figures" / f"{f}{e}").exists() for e in (".pdf", ".png")):
            bad.append(f"figure not released: {f}")
    return bad


def manifest(pkg: Path) -> int:
    files = sorted(p for p in pkg.rglob("*") if p.is_file() and p.name != "MANIFEST.csv"
                   and ".git" not in p.relative_to(pkg).parts)
    with (pkg / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "bytes", "sha256"])
        for p in files:
            w.writerow([p.relative_to(pkg).as_posix(), p.stat().st_size, sha256(p)])
    return len(files)


# ------------------------------------------------------------------ build
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "release" / NAME)
    pkg = ap.parse_args().out.resolve()
    pkg.mkdir(parents=True, exist_ok=True)
    def force(func, path, _exc):              # the locked plan is read-only on disk
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for p in sorted(pkg.iterdir()):           # empty it, keeping the git history
        if p.name == ".git":
            continue
        if p.is_dir():
            shutil.rmtree(p, onerror=force)
        else:
            force(os.unlink, p, None)

    released, lib, changed, edited = copy_code(pkg)
    tables = copy_tables(pkg)
    logs = copy_logs(pkg)
    figures = copy_figures(pkg)
    copy_plan(pkg)
    c = derived(pkg)
    environment(pkg)
    rows, needed, orphans = build_index(set(released))

    c.update(n_scripts=len(released), n_lib=len(lib), n_tables=len(tables), changed=changed, edited=edited,
             n_derived=len(list((pkg / "data" / "derived").glob("*.csv"))))
    write(pkg / "README.md", readme(c, rows))
    write(pkg / "code" / "README.md", code_readme(c))
    write(pkg / "data" / "README.md", data_readme(c))
    write(pkg / "CITATION.cff", citation())
    write(pkg / "LICENSE", J40.MIT.replace(J40.AUTHORS, ", ".join(AUTHORS)))
    cc = (SCRIPTS / "cc_by_4.0.txt").read_text(encoding="utf-8")
    write(pkg / "LICENSE-DATA", DATA_LICENCE_HEADER.format(authors=", ".join(AUTHORS)) + cc)
    write(pkg / ".gitattributes", GITATTRIBUTES)
    write(pkg / ".gitignore", GITIGNORE)

    bad = guard(pkg) + completeness(pkg, needed, figures, released)
    for b in bad:
        print("FAIL:", b)
    if bad:
        raise SystemExit("the release would be incomplete or would carry files it must not; MANIFEST not written")
    n = manifest(pkg)
    total = sum(p.stat().st_size for p in pkg.rglob("*") if p.is_file() and ".git" not in p.relative_to(pkg).parts)
    print(f"scripts {len(released)} (dataset path replaced in: {', '.join(changed) or 'none'}), library {len(lib)}")
    print(f"tables {len(tables)}, logs {len(logs)}, figures {len(figures)}, derived files {c['n_derived']}")
    print(f"README rows: {sum(r['part'] == 'main' for r in rows)} article items, {sum(r['part'] == 'supp' for r in rows)} supplementary items")
    if orphans:
        print("note: tables whose writer the naming convention does not give:", ", ".join(orphans))
    print(f"plan digest {PLAN_SHA[:16]}... verified against its .sha256 file and the supplementary")
    print(f"{n} files, {total / 1e6:.1f} MB in {pkg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
