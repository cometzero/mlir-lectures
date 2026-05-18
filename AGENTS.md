# Repository Guidelines

## Project Structure & Module Organization

This repository contains a 20-lecture MLIR/NPU compiler study course. Root
content is organized for teaching and validation:

- `lectures/lecture-XX/`: per-lecture materials. Each lecture should keep
  `README.md`, `LESSON.md`, handouts, quizzes, slides, worksheets, `manifest.json`,
  `examples/`, and optional `assets/` together.
- `lectures/lecture-XX/examples/`: MLIR snippets, Python helpers, shell runners,
  YAML policies, and skeleton C++/TableGen files for that lecture.
- `docs/`: course-wide syllabus, teaching guide, and lab environment setup.
- `course-package/`: bundled course overview, quick reference, generated package
  artifacts, and package-level examples.
- `scripts/`: repository validation and stock-safe lesson runners.

## Build, Test, and Development Commands

- `python3 scripts/verify_materials.py`: verifies that all 20 lecture directories
  and required materials are present and that manifests parse as JSON.
- `scripts/run_lesson.sh 04`: lists Lecture 04 materials and examples.
- `scripts/run_lesson.sh 04 --try-mlir`: runs stock-safe MLIR checks and Python
  helpers for one lecture. Set `MLIR_OPT=/path/to/mlir-opt` when needed.
- `scripts/run_all_lessons.sh`: runs strict stock-safe checks across all lectures;
  requires `mlir-opt` on `PATH`.

## Coding Style & Naming Conventions

Use existing naming patterns. Lecture directories are `lecture-01` through
`lecture-20`; generated files use `MLIR_NPU_Lecture_XX_*`; examples should be
number-prefixed, such as `01_tensor_vs_memref_boundary.mlir`. Keep Markdown
headings sentence-case and concise. Shell scripts use Bash with `set -u`; Python
helpers should stay Python 3 compatible and prefer `pathlib` for paths.

## Testing Guidelines

Before submitting changes, run `python3 scripts/verify_materials.py`. If examples
or runners changed, also run the relevant `scripts/run_lesson.sh XX --try-mlir`.
Use `scripts/run_all_lessons.sh` for broad validation after cross-lecture changes.
Pseudo or custom NPU dialect examples may be skipped by design; do not convert
them into stock MLIR unless the lesson intent changes.

## Lecture Maintenance Policy

This is an MLIR lecture repository, so treat exercise findings as course
material feedback. When a lab exposes stale commands, missing setup notes,
unclear explanations, broken examples, or inconsistent assets, update the
relevant lecture files in the same topic: `README.md`, `LESSON.md`,
handouts, `examples/`, `assets/`, or `manifest.json`.

## Commit & Pull Request Guidelines

Git history uses Conventional Commits, for example `docs: add MLIR NPU compiler
lecture materials` and `test: add stock-safe MLIR lesson runner`. Keep commits
atomic and signed off with `git commit -s`. Commit material updates after
validation; do not leave lecture, docs, example, or asset changes uncommitted
unless a blocker is explicitly documented. Pull requests should describe changed
lectures or docs, list validation commands and results, link related issues when
available, and include screenshots only when visual assets or slides changed.
