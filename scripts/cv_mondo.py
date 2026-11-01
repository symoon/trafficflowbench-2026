"""Task 1: validazione a blocchi con un cambio di mondo simulato nel blocco valutato.

I mesi valutati sono mondi diversi dal train: la velocità in flusso libero di ogni tratto si
sposta di circa 1,2 km/h (deviazione standard tra tratti, fino a 3–5), solo in flusso libero
(README, «Ogni mese valutato è un mondo diverso»). La validazione a blocchi sta tutta nel mondo
del train e non lo vede. Qui, nel blocco valutato, ogni tratto riceve uno scarto δ ~ N(0; 1,2)
applicato per intero se v ≥ 0,95·v_f, nullo sotto 0,8·v_f e lineare in mezzo, sia allo strato
mascherato (ingressi) sia alla verità; le caratteristiche si ricalcolano sul mondo spostato.

Si confrontano interpolazione, LightGBM a 800 e a 3200 alberi (stesso modello, due troncamenti),
nel mondo del train e in MONDI mondi spostati.

Con --allenamento mondi o entrambi i modelli si allenano anche (o solo) sulle stesse celle in
mondi spostati (prepara_task1_mondi.py): è la randomizzazione del mondo.

Con --sd si valutano anche mondi con scarti di ampiezza diversa da quella usata in allenamento
(1,2), per vedere se il guadagno regge quando il cambio vero non è quello simulato.

    uv run python scripts/cv_mondo.py [--mondi 2] [--thread 16] [--allenamento train|mondi|entrambi] [--sd 1.2 2.4]
"""
import argparse
import dataclasses
import time
import zlib
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_task1 import CARTELLA, META, PARAMETRI
from tfb.caratteristiche import base, calcola
from tfb.dati import pannelli, tensori, tratti
from tfb.metrica import s_lwr_stima, s_state

warnings.filterwarnings("ignore", category=RuntimeWarning)
FRAZIONE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
ALBERI = (800, 3200)
SD_MONDO = 1.2


def sposta(t, vf: np.ndarray, delta: np.ndarray, rif: np.ndarray):
    """Copia del tensore con la velocità spostata di delta per tratto, pesata sul regime del flusso.

    `rif` è la velocità vera (D, 288, L): il peso dipende dallo stato vero della cella, così
    ingressi e verità si spostano allo stesso modo.
    """
    r = rif / vf[None, None, :]
    w = np.clip((r - 0.80) / 0.15, 0.0, 1.0)
    return dataclasses.replace(t, speed=(t.speed + delta[None, None, :] * w).astype(t.speed.dtype))


def misura(celle: pd.DataFrame, vero, v_hat: np.ndarray, q_hat: np.ndarray) -> tuple[float, float]:
    corsie = tratti(vero.pannello).lanes.to_numpy(float)
    q_tot = q_hat * corsie[celle.l.to_numpy()]
    return s_state(celle, vero, v_hat, q_tot)["S_state"], s_lwr_stima(celle, vero, v_hat, q_tot)["S_LWR"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mondi", type=int, default=2)
    ap.add_argument("--thread", type=int, default=16)
    ap.add_argument("--allenamento", choices=["train", "mondi", "entrambi"], default="train")
    ap.add_argument("--sd", type=float, nargs="+", default=[SD_MONDO])
    a = ap.parse_args()
    suffisso = "" if a.sd == [SD_MONDO] else "_sd" + "_".join(f"{x:g}" for x in a.sd)
    mondi = [0.0] + [sd for sd in a.sd for _ in range(a.mondi)]
    t0 = time.time()
    d = pd.concat([pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    colonne = [c for c in d.columns if c not in META]
    fonti = {"train": [d], "mondi": [], "entrambi": [d]}[a.allenamento]
    if a.allenamento != "train":
        fonti.append(pd.concat([pd.read_parquet(CARTELLA / "caratteristiche_mondi" / f"{p}.parquet") for p in pannelli()],
                               ignore_index=True).drop(columns="copia"))
    A = pd.concat(fonti, ignore_index=True)
    bv_a, bq_a = base(A, "v"), base(A, "q")
    righe = []
    for k in range(3):
        al = (A.blocco != k).to_numpy()
        modelli = {}
        for nome, y, b_ in (("v", A.y_v.to_numpy(), bv_a), ("q", A.y_q.to_numpy(), bq_a)):
            r = y - b_
            ok = al & np.isfinite(r)
            modelli[nome] = lgb.train(dict(PARAMETRI, num_threads=a.thread), lgb.Dataset(A.loc[ok, colonne], r[ok]),
                                      num_boost_round=max(ALBERI))
        print(f"fold {k}: modelli ({time.time() - t0:.0f} s)", flush=True)
        for p in pannelli():
            celle = d[(d.pannello == p) & (d.blocco == k) & d.completo].reset_index(drop=True)
            masc, vero = tensori(p, "train"), tensori(p, "train", "vero")
            vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
            rng = np.random.default_rng([k, zlib.crc32(p.encode())])
            for mondo, sd in enumerate(mondi):
                if mondo == 0:
                    m, v, X = masc, vero, celle
                else:
                    delta = rng.normal(0.0, sd, len(vf)).astype(np.float32)
                    rif = np.where(np.isfinite(vero.speed), vero.speed, np.nanmedian(vero.speed))
                    m, v = sposta(masc, vf, delta, rif), sposta(vero, vf, delta, rif)
                    X = calcola(m, celle, FRAZIONE)
                xv, xq = base(X, "v"), base(X, "q")
                for nome_b, b_ in (("v", xv), ("q", xq)):
                    med = np.nanmedian(b_)
                    b_[:] = np.where(np.isfinite(b_), b_, med)
                metodi = {"interpolazione": (xv, xq)}
                for n in ALBERI:
                    metodi[f"lgb_{n}"] = (xv + modelli["v"].predict(X[colonne], num_iteration=n, num_threads=a.thread),
                                          xq + modelli["q"].predict(X[colonne], num_iteration=n, num_threads=a.thread))
                for metodo, (vh, qh) in metodi.items():
                    ss, sl = misura(celle, v, vh, qh)
                    righe.append(dict(fold=k, pannello=p, mondo=mondo, sd=sd, spostato=mondo > 0, metodo=metodo, S_state=ss, S_LWR=sl))
            print(f"fold {k} {p} ({time.time() - t0:.0f} s)", flush=True)
        tab = pd.DataFrame(righe)
        tab.to_csv(CARTELLA / f"cv_mondo_{a.allenamento}{suffisso}.csv", index=False)
    tab = pd.DataFrame(righe)
    pd.set_option("display.width", 200)
    s = tab.groupby(["metodo", "spostato", "fold"])[["S_state", "S_LWR"]].mean().unstack("spostato")
    print("\nMedia sui corridoi: mondo del train (False) e mondi spostati (True)")
    print(s.round(4).to_string())
    m = tab.groupby(["metodo", "spostato"])[["S_state", "S_LWR"]].mean().unstack("spostato")
    m[("Δ", "S_state")] = m[("S_state", True)] - m[("S_state", False)]
    m[("Δ", "S_LWR")] = m[("S_LWR", True)] - m[("S_LWR", False)]
    print("\nSintesi sui fold\n", m.round(4).to_string())
    if len(a.sd) > 1:
        print("\nPer ampiezza dello scarto (sd = 0: mondo del train)")
        print(tab.groupby(["metodo", "sd"])[["S_state", "S_LWR"]].mean().unstack("sd").round(4).to_string())


if __name__ == "__main__":
    main()
