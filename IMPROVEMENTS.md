# IMPROVEMENTS.md

*Analysis date: 2026-07-11*

This repo is a Qiskit research prototype exploring quantum acceleration of p-adic linear regression (the "optimal hyperplane passes through n+1 points" theorem, documented well in `ALGORITHM_SUMMARY.md`). Current state: ~466 lines of Python across six small scripts (`twoadic.py` is the core, with `increment_by_one_no_control.py`, `initialise.py`, `linearA.py`, and two ad-hoc test scripts). It was recently migrated to `uv` (commit 8108dcc) but the CI and CLAUDE.md were not updated to match, so CI is broken. There is no automated test suite and the quantum circuits contain at least one likely ancilla-uncompute bug.

## Bugs & Fixes

1. **CI is broken**: `.github/workflows/main.yml` runs `pip install -r requirements.txt`, but `requirements.txt` was deleted in the uv migration (8108dcc). Every push to main fails. Fix: use `astral-sh/setup-uv` + `uv sync`, and bump the ancient `actions/checkout@v2` / `setup-python@v2` (deprecated) while you're there. Also make CI actually *run* something (e.g. `uv run test1.py`, `uv run pytest`).
2. **Ancilla not uncomputed in `stop_if_bit_is_1` (`twoadic.py`)**: the sequence is `ccx(still_zero, bit, tmp)`, `cx(tmp, still_zero)`, `ccx(still_zero, bit, tmp)`. When `still_zero=1` and `bit=1`, the `cx` flips `still_zero` to 0 *before* the second `ccx`, so the second Toffoli is a no-op and `tmp` is left in |1⟩. Dirty ancillas will corrupt any later subroutine that assumes `anc_reg` is |0⟩ (e.g. repeated calls inside `count_trailing_zeros_inplace`). Uncompute using the *original* control values (compute-copy-uncompute with a fresh ancilla, or restructure so `still_zero` is flipped last).
3. **Confused control logic in `increment_by_one_if` (`twoadic.py`)**: the X/MCX/X construction of `all_ones` carries inline comments literally saying the logic is "the opposite of what we want... Let's fix that logic below". Trace it carefully, delete the contradictory comments, and add a unit test over all control basis states (2–3 controls is cheap on Aer).
4. **`linearA.py` (12 lines) and `initialise.py`** look like stubs/scaffolding. Either finish them or state their purpose in a module docstring — right now it's unclear whether `linearA.py` is dangling work.

## Testing

- `test1.py`/`test2.py` are scripts, not tests. Convert to `pytest` (add via `uv add --dev pytest`) with assertions that compare circuit measurement results on `qiskit-aer` against classical expectations:
  - exhaustive basis-state tests for `increment_by_one_no_control`, `increment_by_one_if` (verify ancillas return to |0⟩ — this would catch bug #2),
  - `count_trailing_zeros_inplace` vs Python's `(x & -x).bit_length()-1` for all values of a 4–5 qubit register, including the all-zero edge case.
- Add a classical reference implementation of 2-adic valuation/regression so quantum results can be cross-checked end-to-end.

## Documentation

- **CLAUDE.md is stale**: it still instructs `pip install -r requirements.txt` and describes CI that installs from requirements.txt. Update to `uv sync` / `uv run <script>.py`, and list the real dependencies (qiskit, qiskit-aer, matplotlib, pylatexenc).
- README.md is one sentence. Add: how to run each script, what `twoadic.py` demonstrates, and a pointer to `ALGORITHM_SUMMARY.md` (which is genuinely good — link it prominently).
- `ALGORITHM_SUMMARY.md` references `../papers/...` paths outside the repo; note that dependency or vendor the relevant citations.
- Fill in `description = "Add your description here"` in `pyproject.toml`.

## Improvements

- `count_trailing_zeros_inplace` uses a linear chain of multi-controlled increments — fine for a prototype, but document the ancilla budget as a function (currently prose assumptions). A helper that allocates/validates register sizes would prevent silent off-by-one ancilla shortages.
- Replace the naive X/MCX/X AND-computation in `increment_by_one_if` with a single `MCXGate` onto a zeroed ancilla — simpler and manifestly correct.
- `sweep.yaml` is a leftover from the defunct Sweep AI experiment (commits 8f9314b/2ba66db); delete it.

## Housekeeping / Modernization

- The uv migration is done (good — matches owner preference: `pyproject.toml` + `uv add`, run with `uv run twoadic.py`); finish it by fixing CI and CLAUDE.md as above.
- `__pycache__/` is present in the working tree — confirm it's in `.gitignore` (it isn't committed, but check).
- No secrets or credentials found in the repo.

## Quick Wins

1. Fix the CI workflow to use uv (5 minutes; unbreaks every push).
2. Update CLAUDE.md's setup section (2 minutes).
3. Delete `sweep.yaml`.
4. Add a pytest that round-trips `count_trailing_zeros_inplace` and checks ancillas — highest insight-per-line test available, and it will adjudicate bug #2.
