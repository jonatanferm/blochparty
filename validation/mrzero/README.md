# Cross-validation against MRzero

`compare_mrzero.py` simulates the same `.seq` files in blochparty and MRzero (MRzeroCore) and compares k-space and images; it writes [RESULTS.md](RESULTS.md). `tests/validation/test_crosscheck_mrzero.py` runs a 16 x 16 version (`pytest -m crosscheck`).

    uv sync --extra mrzero
    uv run python validation/mrzero/compare_mrzero.py
