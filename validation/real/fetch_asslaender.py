"""Download the data of benchmark 4 (inversion recovery NMR of BSA and MnCl2 with rectangular
inversion pulses of 22.8-912 us) into validation/real/data/asslaender/ (~240 MB of tracked files).

Asslaender J, Gultekin C, Flassbeck S, Glaser SJ, Sodickson DK. Generalized Bloch model: a
theory for pulsed magnetization transfer. MRM 2022;87:2003-2017 (arXiv 2107.11000). Data:
github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData (MIT), at the commit the paper cites.

    uv run python validation/real/fetch_asslaender.py
"""

import subprocess
from pathlib import Path

REPO = "https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData"
COMMIT = "77950c4"  # cited in the paper; later commits only edit the README
OUT = Path(__file__).with_name("data") / "asslaender"


def main():
    if not (OUT / ".git").exists():
        subprocess.run(["git", "clone", "-q", REPO, str(OUT)], check=True)
    subprocess.run(["git", "-C", str(OUT), "checkout", "-q", COMMIT], check=True)
    print("data at", OUT, "commit", subprocess.run(["git", "-C", str(OUT), "rev-parse", "--short", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
