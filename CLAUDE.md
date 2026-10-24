# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A practice set of 78 ML components to implement from scratch (NumPy and PyTorch). Every stub in `src/drills/<module>.py` raises `NotImplementedError`, and `tests/test_<module>.py` fails until the stub is implemented. The docstring on each stub is the spec (shapes, conventions, gotchas, hints). `README.md` holds the full drill index (number, symbols, tier, target minutes, test count) and a suggested order.

The repo is public. The user works the drills as interview practice, so the stubs and tests are the product: do not fill in stubs on your own initiative.

## Coaching mode

When the user is working a drill, act as a coach: never hand over the answer, give one small nudge at a time, use plain words. Reference solutions are kept outside this repo and must not be shown or copied in unless the user asks. Do not read them to answer questions about a drill.

Keep every reply brief: a few short lines, no long explanations or recaps.

## Commands

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"   # or python -m venv + pip install -e ".[dev]"
pytest tests/test_losses.py -k drill01 -q    # one drill
pytest tests/test_losses.py                  # one module
pytest -x -vv -k drill02                     # stop at first failure, full detail
```

`pytest` config (in `pyproject.toml`) sets `pythonpath = ["src"]` and `-q`. There is no linter or build step configured.

## Gotchas

- Test names zero-pad the drill number: `test_drill01_...`. A bare `-k drill1` also matches `drill10` to `drill19`; use `drill01`, or `drill1_` style only if the name supports it.
- Drill numbers are not grouped by module in order: for example #36 to #41 and #45 live in `structure_losses`, #42 to #44 and #46 to #49 in `structure_layers`, and #68 to #78 in `extras`. Use the README index to find the module for a drill.
- A fresh clone fails every test by design; judge progress per drill, not by the whole suite.
- Some drills also ship helpers you do not write (for example #35 `make_debug_data` and `buggy_train` in `training.py`).
