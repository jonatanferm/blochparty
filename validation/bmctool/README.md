# Cross-validation against BMCTool

`compare_bmctool.py` compares Z-spectra of a water pool with a bound pool (and free CEST pools) with BMCTool's Bloch-McConnell solver; results in [RESULTS.md](RESULTS.md).

    uv run --with bmctool==1.0.3 python validation/bmctool/compare_bmctool.py [--quick]
    BLOCHPARTY_BACKEND=gpu uv run --with bmctool==1.0.3 python validation/bmctool/compare_bmctool.py   # on the GPU
