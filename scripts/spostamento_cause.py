"""Che cosa distingue i mesi rigenerati dal train: AUC di ogni caratteristica da sola e confronto delle distribuzioni.

Stesse celle e fold di spostamento_split.py. Per ogni corridoio: ultimo mese del train contro
validation, e un mese del train contro il successivo come riferimento.

    uv run python scripts/spostamento_cause.py [--thread 16]
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from spostamento_split import celle, roc_auc
from tfb.dati import pannelli, tensori

warnings.filterwarnings("ignore", category=RuntimeWarning)


def auc_una(a: pd.DataFrame, b: pd.DataFrame, col: list[str], thread: int) -> float:
    x = pd.concat([a.assign(y=0), b.assign(y=1)], ignore_index=True)
    rng = np.random.default_rng(0)
    fold = np.zeros(len(x), int)
    for y in (0, 1):
        giorni = np.unique(x.giorno[x.y == y])
        assegna = dict(zip(rng.permutation(giorni), np.arange(len(giorni)) % 5))
        m = (x.y == y).to_numpy()
        fold[m] = x.giorno[m].map(assegna).to_numpy()
    par = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=500, num_threads=thread, verbose=-1, seed=0)
    p = np.zeros(len(x))
    for k in range(5):
        al, te = fold != k, fold == k
        m = lgb.train(par, lgb.Dataset(x.loc[al, col], x.y[al]), num_boost_round=100)
        p[te] = m.predict(x.loc[te, col], num_threads=thread)
    return roc_auc(x.y.to_numpy(), p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", type=int, default=16)
    a = ap.parse_args()
    singole = ["v", "q", "o", "r", "istante", "l"]
    coppie = [["v", "q"], ["q", "o"], ["v", "o"], ["q", "istante"], ["q", "l"], ["o", "l"], ["v", "l"]]
    righe, stat = [], []
    for p in pannelli():
        rng = np.random.default_rng(0)
        date = pd.to_datetime(pd.Series(tensori(p, "train").date))
        per = date.dt.to_period("M")
        mesi = sorted(per.unique())
        ultimo = celle(p, "train", np.flatnonzero((per == mesi[-1]).to_numpy()), rng)
        penultimo = celle(p, "train", np.flatnonzero((per == mesi[-2]).to_numpy()), rng)
        val = celle(p, "validation", np.arange(len(tensori(p, "validation").date)), rng)
        for nome, (x0, x1) in {"train→train": (penultimo, ultimo), "train→validation": (ultimo, val)}.items():
            for c in singole + coppie:
                col = c if isinstance(c, list) else [c]
                righe.append(dict(pannello=p, confronto=nome, caratteristiche="+".join(col), auc=auc_una(x0, x1, col, a.thread)))
        for nome, x in (("train penultimo", penultimo), ("train ultimo", ultimo), ("validation", val)):
            libero = x[x.r > 0.9]
            stat.append(dict(pannello=p, mese=nome, v_libero=libero.v.mean(), q_libero=libero.q.mean(), o_libero=libero.o.mean(),
                             o_su_q=(libero.o / libero.q.clip(lower=1)).median() * 1e3, quota_coda=(x.r <= 0.6).mean(),
                             q_medio=x.q.mean(), sd_v_libero=libero.v.std()))
        print(p, flush=True)
    r = pd.DataFrame(righe)
    pd.set_option("display.width", 200)
    print("\nAUC di ogni caratteristica da sola (media sui corridoi)")
    print(r.pivot_table(index="caratteristiche", columns="confronto", values="auc").round(3).sort_values("train→validation").to_string())
    s = pd.DataFrame(stat)
    print("\nDistribuzioni (media sui corridoi); celle in flusso libero: r > 0,9")
    print(s.groupby("mese")[["v_libero", "sd_v_libero", "q_libero", "o_libero", "o_su_q", "q_medio", "quota_coda"]].mean().round(4).to_string())
    s.to_csv("outputs/spostamento_cause.csv", index=False)


if __name__ == "__main__":
    main()
