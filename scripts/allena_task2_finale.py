"""Task 2: allena i modelli finali (onset e ongoing) su tutte le finestre ricostruite del train.

Righe e caratteristiche sono quelle di cv_task2.py: la frequenza storica di ogni finestra
esclude il suo blocco, così in allenamento ha la stessa natura fuori campione che avrà sui
mesi valutati. Soglie dalla curva per fold (massimo del fold peggiore, cv.log del 19/10).

Onset: le finestre attuali, che coincidono con la selezione ufficiale (prima origine ammessa).
Ongoing: unione delle finestre attuali e di quelle scelte a caso tra i candidati
(finestre.ongoing_casuali), che riproducono la difficoltà della selezione ufficiale
(cv_task2_selezione.py, 19/10: 0,857 contro 0,835 sulle 40 ongoing ufficiali fuori campione).
Con --semi N le casuali sono quelle dei semi 0 … N−1 (cv_task2_piu_finestre.py, 20/10: con 3 semi
+0,010 sulle ongoing casuali fuori campione); il default 1 è il modello di v8.

    uv run python scripts/allena_task2_finale.py [--thread 8] [--semi 3]
"""
import argparse
import json
import time
import warnings

import lightgbm as lgb
import pandas as pd

from cv_task2 import PANNELLI, PARAMETRI, USCITA, costruisci_righe
from tfb.finestre import ongoing_casuali

warnings.filterwarnings("ignore", category=RuntimeWarning)
ALBERI = 400
SOGLIE = {"queue_onset": 0.40, "queue_ongoing": 0.45}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", type=int, default=8)
    ap.add_argument("--semi", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    attuali = pd.read_parquet(USCITA / "finestre_train.parquet")[["pannello", "g", "t", "condition"]]
    casuali = pd.concat([ongoing_casuali(p, s) for s in range(a.semi) for p in PANNELLI], ignore_index=True)
    righe = {"queue_onset": costruisci_righe(attuali[attuali.condition == "queue_onset"])[0],
             "queue_ongoing": pd.concat([costruisci_righe(attuali[attuali.condition == "queue_ongoing"])[0],
                                         costruisci_righe(casuali)[0]], ignore_index=True)}
    colonne = [c for c in righe["queue_onset"].columns if c not in ("y", "ele", "fid", "l", "blocco", "condition")]
    for cond in SOGLIE:
        d = righe[cond]
        al = d.ele.to_numpy()
        m = lgb.train(dict(PARAMETRI, num_threads=a.thread), lgb.Dataset(d.loc[al, colonne], d.y[al]), num_boost_round=ALBERI)
        m.save_model(str(USCITA / f"finale_{cond}.txt"))
        print(f"{cond}: {al.sum():,} righe ({time.time() - t0:.0f} s)", flush=True)
    (USCITA / "finale_parametri.json").write_text(json.dumps(dict(PARAMETRI, num_threads=a.thread, alberi=ALBERI, soglie=SOGLIE,
                                                                  ongoing="attuali + casuali" if a.semi == 1 else f"attuali + casuali dei semi 0–{a.semi - 1}",
                                                                  semi=a.semi, colonne=colonne), indent=1))


if __name__ == "__main__":
    main()
