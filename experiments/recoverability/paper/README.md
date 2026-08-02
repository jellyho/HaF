# AHA paper — ICML 2026 LaTeX project (Overleaf-ready)

**"Regularizing Vision-Language-Action Models by Asking Hard Questions"** — method = AHA.
Uses the **official ICML 2026 template** (`icml2026.sty`, `icml2026.bst`, `fancyhdr.sty`, `algorithm(ic).sty`).

## Build
- **Overleaf:** upload the whole `paper/` folder. Compiler: **pdfLaTeX**. (Bibliography = bibtex + `icml2026.bst`.)
- **CLI:** `pdflatex main; bibtex main; pdflatex main; pdflatex main`
- Verified: compiles to a **6-page two-column ICML paper**, no undefined citations/references.
- Camera-ready: change `\usepackage{icml2026}` → `\usepackage[accepted]{icml2026}` (shows authors + removes line numbers).

### One note on the running header
The inner-page running header renders correctly on Overleaf/pdfLaTeX (verified against the official
`example_paper.pdf`). If you compile with **Tectonic** locally it may print "Title Suppressed Due to Excessive
Size" in the header — a Tectonic font-metric quirk in ICML's one-line height check, **not** a template error. It
disappears on Overleaf. The `.tex` already sets a short `\icmltitlerunning{...}`.

## Files
- `main.tex` — the paper (§1–§10, 3 tables, 2 full-width figures).
- `icml2026.sty`, `icml2026.bst`, `fancyhdr.sty`, `algorithm.sty`, `algorithmic.sty` — official ICML 2026 style.
- `references.bib` — 22 refs (all web-verified).
- `figs/` — `fig_measures_frac4k.pdf` (§4), `fig_deepen.pdf` (§5–6), `fig_mixki.pdf` (spare).

## Provisional values — READ THIS
Numbers wrapped in `\prov{...}` render **gray italic** and are PLACEHOLDERS pending the running experiments
(20k closed-loop + full-VLA M2). Everything black is **measured** on the mini-VLA (RT-1/fractal, 3 seeds).
Written "as if results are in" per request. Provisional: Abstract (last line), Table 3 (`tab:closed`, all but the
2k proxy), the M2 paragraph in §8 (+ footnote), and the §9 hedge. Replace each `\prov{}` on completion.

## Section ↔ source
Sections mirror PAPER_S1…S10. Full prose: `../PAPER_DRAFT.md`. Citations + threats: `../CITATIONS_THREATS.md`.
