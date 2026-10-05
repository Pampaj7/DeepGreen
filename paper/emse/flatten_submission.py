#!/usr/bin/env python3
"""Flatten the EMSE front end into the single-directory source Springer wants.

    flatten_submission.py stage <out-dir>
        Write <out-dir>/main.tex (one file: preamble, abstract, statements, body
        and every generated macro and table file inlined, comments stripped, no
        \\input left), <out-dir>/bibliography.bib (non-ASCII as TeX accents),
        sn-jnl.cls, sn-basic.bst and the figures the body includes, all at top
        level. Pointers into Online Resource 1 (\\esmsec, \\esmtab, \\esmfig and
        any bare \\ref to one of its labels) become the literal numbers recorded
        in esm.aux, so the package needs neither xr-hyper nor esm.aux.

    flatten_submission.py compare <reference.pdf> <candidate.pdf>
        Fail unless the two PDFs' text (pdftotext) is the same once whitespace,
        line-end hyphenation and ligatures are normalised.

Only paper/emse/main.tex and the shared sources are read; nothing in the source
tree is written. Every step that does not find exactly what it expects stops
with an error rather than shipping a package that differs from main.pdf.
"""
import difflib
import re
import shutil
import subprocess
import sys
from pathlib import Path

EMSE = Path(__file__).resolve().parent
PAPER = EMSE.parent

ESM_PREFIX = {"sec": "Sect.", "tab": "Table", "fig": "Fig."}
ESM_NAME = "Online Resource~1"

# Raw characters Springer asks to be given as TeX. The manuscript's .tex sources
# are ASCII today; the table is here so that one typed later is converted (or,
# if it is not listed, stops the build) instead of reaching the package raw.
# \ensuremath so a symbol converts correctly in text and in math alike.
TEX_SYMBOLS = {
    "\u00d7": r"\ensuremath{\times}",  # ×
    "\u2013": "--",                    # en dash
    "\u2014": "---",                   # em dash
    "\u2018": "`",
    "\u2019": "'",
    "\u201c": "``",
    "\u201d": "''",
    "\u00b5": r"\ensuremath{\mu}",     # micro sign
    "\u03bc": r"\ensuremath{\mu}",     # Greek mu
    "\u2264": r"\ensuremath{\le}",
    "\u2265": r"\ensuremath{\ge}",
    "\u00b1": r"\ensuremath{\pm}",
    "\u2212": r"\ensuremath{-}",       # minus sign
    "\u2192": r"\ensuremath{\rightarrow}",
    "\u2248": r"\ensuremath{\approx}",
    "\u2026": r"\dots{}",
    "\u00a0": "~",                     # no-break space
    "\u00a7": r"\S{}",
}
# Accented letters, braced so BibTeX treats each as one letter when it sorts
# and abbreviates names; the same form is valid in running text.
ACCENTS = {
    "\u00e1": r"{\'a}", "\u00e0": r"{\`a}", "\u00e2": r"{\^a}", "\u00e3": r"{\~a}",
    "\u00e4": r'{\"a}', "\u00e5": r"{\aa}", "\u00e7": r"{\c{c}}",
    "\u00e9": r"{\'e}", "\u00e8": r"{\`e}", "\u00ea": r"{\^e}", "\u00eb": r'{\"e}',
    "\u00ed": r"{\'\i}", "\u00ec": r"{\`\i}", "\u00ee": r"{\^\i}", "\u00ef": r'{\"\i}',
    "\u00f1": r"{\~n}",
    "\u00f3": r"{\'o}", "\u00f2": r"{\`o}", "\u00f4": r"{\^o}", "\u00f5": r"{\~o}",
    "\u00f6": r'{\"o}', "\u00f8": r"{\o}",
    "\u00fa": r"{\'u}", "\u00f9": r"{\`u}", "\u00fb": r"{\^u}", "\u00fc": r'{\"u}',
    "\u00df": r"{\ss}",
    "\u00c1": r"{\'A}", "\u00c9": r"{\'E}", "\u00d6": r'{\"O}', "\u00dc": r'{\"U}',
    "\u00c4": r'{\"A}', "\u00d8": r"{\O}", "\u00c5": r"{\AA}",
    "\u0107": r"{\'c}", "\u010d": r"{\v{c}}", "\u0161": r"{\v{s}}", "\u017e": r"{\v{z}}",
    "\u0141": r"{\L}", "\u0142": r"{\l}", "\u0144": r"{\'n}", "\u015b": r"{\'s}",
}
TO_TEX = {**TEX_SYMBOLS, **ACCENTS}


def die(msg):
    sys.exit(f"flatten_submission: {msg}")


def to_ascii(text, what):
    """Replace every non-ASCII character by its TeX code; report what changed."""
    found = {}
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        if ch not in TO_TEX:
            die(f"{what}: no TeX code for U+{ord(ch):04X} {ch!r}; add it to TO_TEX")
        found[ch] = found.get(ch, 0) + 1
        out.append(TO_TEX[ch])
    for ch, n in sorted(found.items()):
        print(f"    {what}: {n} x U+{ord(ch):04X} {ch} -> {TO_TEX[ch]}")
    if not found:
        print(f"    {what}: no non-ASCII characters")
    return "".join(out)


def comment_start(line):
    """Index of the % that starts a comment, or None (\\% is a percent sign)."""
    for i, ch in enumerate(line):
        if ch == "%":
            j = i
            while j > 0 and line[j - 1] == "\\":
                j -= 1
            if (i - j) % 2 == 0:
                return i
    return None


def strip_comments(text):
    """Drop comments, keeping TeX's reading of the file unchanged: a line that
    is only a comment contributes nothing and goes; a trailing comment keeps
    its % so the end of line still produces no space. (No verbatim material
    in these sources: checked by the caller.)"""
    out = []
    for line in text.split("\n"):
        i = comment_start(line)
        if i is None:
            out.append(line)
        elif line[:i].strip() == "":
            continue
        else:
            out.append(line[:i] + "%")
    return "\n".join(out)


INPUT_RE = re.compile(r"\\input\s*\{([^{}]*)\}")


def resolve_input(arg):
    arg = re.sub(r"\\gendir(?![A-Za-z@])\s*", (PAPER / "generated").as_posix() + "/", arg)
    arg = re.sub(r"\\paperroot(?![A-Za-z@])\s*", PAPER.as_posix() + "/", arg)
    if "\\" in arg:
        die(f"\\input{{{arg}}}: unknown macro in path")
    path = Path(arg)
    if not path.is_absolute():
        path = EMSE / path
    if path.suffix != ".tex":
        path = path.with_name(path.name + ".tex")
    if not path.is_file():
        die(f"\\input{{{arg}}}: {path} not found")
    return path


def inline(path, depth=0, seen=None):
    if depth > 5:
        die(f"\\input nested too deep at {path}")
    text = path.read_text(encoding="utf-8")
    if re.search(r"\\(verb|begin\{(verbatim|lstlisting|comment)\})", text):
        die(f"{path}: verbatim material; comment stripping would not be safe")
    text = strip_comments(text)
    if seen is not None:
        seen.append(path)

    def repl(m):
        sub = inline(resolve_input(m.group(1)), depth + 1, seen)
        # A file read by \input ends with an end of line; keep it, so the text
        # after the \input on the same line is read exactly as before.
        return sub if sub.endswith("\n") else sub + "\n"

    return INPUT_RE.sub(repl, text)


def replace_exact(text, old, new, what):
    n = text.count(old)
    if n != 1:
        die(f"{what}: expected once in the flattened source, found {n} times")
    return text.replace(old, new)


def remove_exact(text, needle, what):
    return replace_exact(text, needle, "", what)


def esm_labels():
    labels = {}
    for m in re.finditer(r"^\\newlabel\{([^{}]*)\}\{\{([^{}]*)\}", (EMSE / "esm.aux").read_text(), re.M):
        labels[m.group(1)] = m.group(2)
    if not labels:
        die("esm.aux holds no labels: build esm.tex first (paper/build.sh)")
    return labels


def stage(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = []
    tex = inline(EMSE / "main.tex", seen=inputs)
    print(f"    inlined {len(inputs) - 1} files into main.tex:",
          ", ".join(p.relative_to(PAPER).as_posix() for p in inputs[1:]))

    # Paths: everything is now in one directory beside main.tex.
    tex = remove_exact(tex, "\\newcommand{\\paperroot}{../}\n", "\\paperroot definition")
    tex = remove_exact(tex, "\\newcommand{\\gendir}{\\paperroot generated/}\n", "\\gendir definition")
    tex = remove_exact(tex, "\\graphicspath{{\\paperroot images/}{\\paperroot}}\n", "\\graphicspath")
    tex = replace_exact(tex, "\\bibliography{\\paperroot bibliography}",
                        "\\bibliography{bibliography}", "\\bibliography line")

    figures = []

    def fig(m):
        opts, path = m.group(1) or "", m.group(2)
        if not path.startswith("figures/"):
            die(f"\\includegraphics{{{path}}}: expected a path under figures/")
        name = path[len("figures/"):]
        src = PAPER / "figures" / name
        if not src.is_file():
            die(f"{src} not found")
        figures.append(src)
        return f"\\includegraphics{opts}{{{Path(name).stem}}}"

    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^{}]*)\}", fig, tex)

    # xr-hyper and everything that exists only for it: the package, the label
    # import and remote-link patch installed after it, \externaldocument, and
    # the four pointer macros, which become literal text below.
    start = tex.find("\\usepackage{xr-hyper}\n")
    end_marker = "\\externaldocument[][nocite]{esm}[esm.pdf]\n"
    end = tex.find(end_marker)
    if start < 0 or end < start:
        die("xr-hyper block not found where expected in main.tex")
    block = tex[start:end + len(end_marker)]
    if block.count("\n") > 15 or "\\begin{document}" in block:
        die("xr-hyper block is larger than expected; check main.tex")
    tex = tex[:start] + tex[end + len(end_marker):]
    for line in ("\\newcommand{\\ESMname}{Online Resource~1}\n",
                 "\\newcommand{\\esmsec}[1]{Online Resource~1, Sect.~\\ref{#1}}\n",
                 "\\newcommand{\\esmtab}[1]{Online Resource~1, Table~\\ref{#1}}\n",
                 "\\newcommand{\\esmfig}[1]{Online Resource~1, Fig.~\\ref{#1}}\n"):
        tex = remove_exact(tex, line, line.strip())

    esm = esm_labels()
    local = set(re.findall(r"\\label\{([^{}]*)\}", tex))
    clash = local & set(esm)
    if clash:
        die(f"labels defined in both documents: {sorted(clash)}")
    resolved = []

    def number(label, how):
        if label not in esm:
            die(f"{how}{{{label}}}: not a label of Online Resource 1 (esm.aux)")
        num = esm[label]
        if not re.fullmatch(r"[A-Z0-9][A-Za-z0-9.]*", num):
            die(f"esm.aux: label {label} has number {num!r}, not a plain number")
        resolved.append(f"{how}{{{label}}} -> {num}")
        return num

    tex = re.sub(r"\\esm(sec|tab|fig)\s*\{([^{}]*)\}",
                 lambda m: f"{ESM_NAME}, {ESM_PREFIX[m.group(1)]}~{number(m.group(2), '\\esm' + m.group(1))}",
                 tex)
    # \ESMname, like any control word, swallows the spaces after it.
    tex = re.sub(r"\\ESMname(?![A-Za-z@])(\{\})?[ \t]*", lambda m: ESM_NAME, tex)

    def ref(m):
        cmd, label = m.group(1), m.group(2)
        if label in local:
            return m.group(0)
        if cmd != "ref":
            die(f"\\{cmd}{{{label}}}: refers outside the article")
        return number(label, "\\ref")

    tex = re.sub(r"\\(ref|pageref|autoref|nameref|eqref|cref|Cref)\{([^{}]*)\}", ref, tex)
    print(f"    {len(resolved)} pointers into Online Resource 1 set as literal numbers")

    # Tidy the blank lines the stripped comments left (several empty lines
    # read as one \par).
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    header = ("% Empirical Software Engineering submission: Deep Green AI.\n"
              "% Single-file source generated by paper/emse/make_submission.sh from the\n"
              "% DeepGreen repository; every number is written by its analysis pipeline.\n")
    tex = header + tex.lstrip("\n")

    tex = to_ascii(tex, "main.tex")

    # The checks the package must pass, on the file as written.
    body = "\n".join(l[:comment_start(l)] if comment_start(l) is not None else l
                     for l in tex.split("\n"))
    for pattern, what in [
        (r"\\(input|include|InputIfFileExists|@@input)(?![A-Za-z@])", "\\input or \\include"),
        (r"\\(paperroot|gendir|graphicspath|externaldocument|ESMname|esm(sec|tab|fig))(?![A-Za-z@])",
         "a path or xr macro"),
        (r"xr-hyper|esm\.aux|esm\.pdf|figures/|generated/", "a reference to another file"),
    ]:
        m = re.search(pattern, body)
        if m:
            die(f"flattened main.tex still contains {what}: {m.group(0)!r}")
    if body.count("\\bibliography{bibliography}") != 1:
        die("flattened main.tex: \\bibliography{bibliography} not found once")

    (out / "main.tex").write_text(tex, encoding="ascii")
    bib = to_ascii((PAPER / "bibliography.bib").read_text(encoding="utf-8"), "bibliography.bib")
    (out / "bibliography.bib").write_text(bib, encoding="ascii")
    for f in ("sn-jnl.cls", "sn-basic.bst"):
        shutil.copy2(EMSE / f, out / f)
    for src in sorted(set(figures)):
        shutil.copy2(src, out / src.name)
    print(f"    {len(set(figures))} figures at top level:", ", ".join(sorted(p.name for p in set(figures))))


LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def pdf_words(pdf):
    text = subprocess.run(["pdftotext", str(pdf), "-"], check=True,
                          capture_output=True, text=True).stdout
    for lig, s in LIGATURES.items():
        text = text.replace(lig, s)
    text = text.replace("\u00ad", "")                  # soft hyphen
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # line-end hyphenation
    return text.split()


def compare(ref, cand):
    a, b = pdf_words(ref), pdf_words(cand)
    if a == b:
        print(f"    PDF text identical to {Path(ref).name} ({len(a)} words, whitespace,"
              " hyphenation and ligatures normalised)")
        return
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    shown = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        print(f"    {op}: {' '.join(a[max(0, i1 - 4):i2 + 4])!r}\n"
              f"      -> {' '.join(b[max(0, j1 - 4):j2 + 4])!r}", file=sys.stderr)
        shown += 1
        if shown >= 20:
            break
    die(f"text of {cand} differs from {ref} (first differences above)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "stage":
        stage(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        die(__doc__)
