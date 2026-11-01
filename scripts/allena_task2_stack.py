"""Task 2: modelli finali dello stack (stack_motori di cv_task2_stack.py): onset per v16, ongoing per v18.

Righe e caratteristiche sono quelle di allena_task2_finale.py: onset sulle finestre attuali a T+30,
ongoing sulle attuali più le casuali dei semi 0–5 (ricetta di v15). Primo livello: LightGBM (come
finale_queue_*), XGBoost e CatBoost con le configurazioni di cv_task2_stack.py. Le righe di
allenamento del secondo livello ricevono le probabilità fuori fold dei tre motori (tre blocchi,
ognuno previsto dai modelli allenati sugli altri due) e il contesto della probabilità di LightGBM
nella finestra. All'invio il primo livello sono i tre modelli allenati su tutto il train.

--righe-salvate legge le righe di cv_task2_decoder.py e cv_task2_piu_finestre.py invece di
ricostruirle (sono le stesse); --oof-da prende le probabilità fuori fold da cv_task2_stack.py primo
(stessi modelli e fold) invece di ricalcolarle. Il pacchetto finale rifà tutto senza le due opzioni.

    uv run --group xgb --group catboost python scripts/allena_task2_stack.py --condizione onset
    uv run --group xgb --group catboost python scripts/allena_task2_stack.py --condizione ongoing --semi 6
"""
import argparse
import json
import time
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from cv_task2 import PARAMETRI, USCITA, costruisci_righe
from cv_task2_decoder import META, percorso
from cv_task2_stack import ALBERI, CAT, CONDIZIONI, SOGLIA, XGB, allenamento, finestre_insieme, motore
from tfb.task2_modello import contesto

warnings.filterwarnings("ignore", category=RuntimeWarning)
MOTORI = ("lgb", "xgb", "cat")
ESTENSIONE = {"lgb": "txt", "xgb": "json", "cat": "cbm"}


def file_modello(cond: str, nome: str) -> str:
    return f"finale_stack_{cond}_{nome}.{ESTENSIONE.get(nome, 'txt')}"


def file_parametri(cond: str) -> str:
    # v16 è nato con il solo onset e un nome senza condizione
    return "finale_stack_parametri.json" if cond == "onset" else f"finale_stack_{cond}_parametri.json"


def righe(cond: str, semi: int, salvate: bool) -> pd.DataFrame:
    parti = []
    for i, n in enumerate(allenamento(cond, semi)):
        d = pd.read_parquet(percorso(n, "righe")) if salvate else costruisci_righe(finestre_insieme(n, cond))[0]
        d = d[d.condition == CONDIZIONI[cond]].reset_index(drop=True)
        d["fid"] = d.fid + i * 1_000_000  # finestre distinte tra insiemi
        d["insieme"] = n
        parti.append(d)
        print(f"{n}: {len(d):,} righe", flush=True)
    return pd.concat(parti, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condizione", choices=list(CONDIZIONI), default="onset")
    ap.add_argument("--semi", type=int, default=6)
    ap.add_argument("--thread", type=int, default=8)
    ap.add_argument("--righe-salvate", action="store_true")
    ap.add_argument("--oof-da", default=None)
    a = ap.parse_args()
    cond, t0 = a.condizione, time.time()
    d = righe(cond, a.semi, a.righe_salvate)
    colonne = [c for c in d.columns if c not in META + ["insieme"]]
    X, y, ele, b = d[colonne].to_numpy(np.float32), d.y.to_numpy(), d.ele.to_numpy(), d.blocco.to_numpy()
    print(f"{cond}: {len(d):,} righe, {ele.sum():,} eleggibili, {len(colonne)} caratteristiche ({time.time() - t0:.0f} s)", flush=True)
    oof = {}
    if a.oof_da:
        cv = pd.read_parquet(a.oof_da, columns=["insieme", "l", "y", "passo"] + [f"p_{n}" for n in MOTORI])
        cv = cv[cv.insieme.isin(allenamento(cond, a.semi))]
        assert (cv.insieme.to_numpy() == d.insieme.to_numpy()).all() and (cv.l.to_numpy() == d.l.to_numpy()).all()
        assert (cv.y.to_numpy() == d.y.to_numpy()).all() and (cv.passo.to_numpy() == d.passo.to_numpy()).all()
        oof = {n: cv[f"p_{n}"].to_numpy(np.float32) for n in MOTORI}
        del cv
    for nome in MOTORI:
        fit, prevedi = motore(nome, a.thread)
        if nome not in oof:
            p = np.full(len(d), np.nan, np.float32)
            for k in range(3):
                tr = ele & (b != k)
                p[b == k] = prevedi(fit(X[tr], y[tr]), X[b == k])
            oof[nome] = p
        fit(X[ele], y[ele]).save_model(str(USCITA / file_modello(cond, nome)))
        print(f"{nome}: fuori fold e finale ({time.time() - t0:.0f} s)", flush=True)
    gid = pd.factorize(d.fid)[0]
    L = (d.groupby(gid).l.max() + 1).sort_index().to_numpy()
    ctx, nomi_ctx = contesto(oof["lgb"], gid, d.passo.to_numpy(), d.l.to_numpy(), L, onset=cond == "onset")
    del d
    Z = np.hstack([X, np.column_stack([oof[n] for n in MOTORI]), ctx])
    del X, ctx
    fit, _ = motore("lgb", a.thread)
    secondo = f"finale_stack_{cond}_secondo.txt"
    fit(Z[ele], y[ele]).save_model(str(USCITA / secondo))
    (USCITA / file_parametri(cond)).write_text(json.dumps(dict(
        lgb=dict(PARAMETRI, num_threads=a.thread), xgb=XGB, cat=CAT, alberi=ALBERI, soglia=SOGLIA[cond], condizione=cond,
        allenamento=allenamento(cond, a.semi), colonne=colonne, colonne_secondo=colonne + [f"p1_{n}" for n in MOTORI] + nomi_ctx,
        file={n: file_modello(cond, n) for n in MOTORI}, secondo=secondo,
        librerie={n: version(n) for n in ("lightgbm", "xgboost-cpu", "catboost")}), indent=1))
    print(f"secondo livello: {Z.shape[1]} caratteristiche ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
