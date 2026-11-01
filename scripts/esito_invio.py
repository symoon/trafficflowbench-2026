"""Aspetta la valutazione dell'ultimo invio e riporta punteggio pubblico e posizione.

    uv run python scripts/esito_invio.py
"""
import glob
import io
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd

C = "2026-ieee-big-data-traffic-flow-bench"
UTENTE = "simonezappatini"
CARTELLA = Path(__file__).resolve().parents[1] / "outputs" / "classifica"


def kaggle(*argomenti: str) -> str:
    out = subprocess.run(["kaggle", "competitions", *argomenti], capture_output=True, text=True, check=True).stdout
    # la CLI antepone un avviso quando esce una versione nuova
    return "".join(r for r in out.splitlines(keepends=True) if not r.startswith("Warning:"))


def main() -> None:
    for _ in range(120):
        righe = pd.read_csv(io.StringIO(kaggle("submissions", C, "--csv")))
        ultimo = righe.iloc[0]
        if "COMPLETE" in str(ultimo.status) or "ERROR" in str(ultimo.status):
            break
        time.sleep(15)
    print(f"invio: {ultimo.fileName} · {ultimo.description}")
    print(f"stato: {ultimo.status} · punteggio pubblico: {ultimo.publicScore}")
    CARTELLA.mkdir(parents=True, exist_ok=True)
    kaggle("leaderboard", C, "-d", "-p", str(CARTELLA))
    for z in glob.glob(str(CARTELLA / "*.zip")):
        with zipfile.ZipFile(z) as f:
            f.extractall(CARTELLA)
        Path(z).unlink()
    d = pd.read_csv(sorted(glob.glob(str(CARTELLA / "*.csv")))[-1])
    d.columns = [c.lstrip("﻿") for c in d.columns]
    noi = d[d.TeamMemberUserNames.astype(str).str.contains(UTENTE)]
    print(f"posizione: {int(noi.Rank.iloc[0])} su {len(d)} (miglior punteggio della squadra {noi.Score.iloc[0]})")


if __name__ == "__main__":
    main()
