"""Finestre del Task 2 ricostruite sul train con la regola di selezione pubblicata.

Regola (config/synthetic_release_v1.json, pagina Evaluation, forum 742350 e 745700):
- storia da T-60 a T-5 (12 passi), orizzonte da T+5 a T+30 (6 passi);
- coda: velocità ≤ 0,6 × velocità libera;
- `queue_onset`: nessuna coda nella storia; `queue_ongoing`: almeno due osservazioni
  in coda su uno stesso tratto; in entrambe almeno una coda nell'orizzonte;
- copertura della storia ≥ 0,7; IoU della persistenza (stato a T-5) ≤ 0,9;
- origini scelte nell'ordine del tempo, distanti almeno 360 minuti da qualunque finestra già
  scelta (con la distanza per condizione si ritrovano 47 delle 80 finestre ufficiali del
  train, con la distanza unica 53: tutte le onset e metà delle ongoing).

Le etichette ufficiali vengono dallo stato sottostante, qui dalle osservazioni vere del
train: vicino alla soglia il rumore di misura sposta qualche cella.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dati import ISTANTI, tensori, tratti

STORIA, ORIZZONTE = 12, 6
SPAZIATURA = 72  # 360 minuti in passi da 5
SOGLIA = 0.60


def etichette(pannello: str) -> tuple[np.ndarray, np.ndarray]:
    """(coda, eleggibile) sul train vero, tensori (D, 288, L)."""
    v = tensori(pannello, "train", "vero")
    vf = tratti(pannello).free_speed_kmh.to_numpy(np.float32)
    ele = v.eleggibile & np.isfinite(v.speed)
    return ele & (v.speed <= SOGLIA * vf), ele


def iou(pred: np.ndarray, vero: np.ndarray, ele: np.ndarray) -> float:
    p, t = pred[ele], vero[ele]
    u = np.logical_or(p, t).sum()
    return 1.0 if u == 0 else float(np.logical_and(p, t).sum() / u)


def costruisci(pannello: str) -> pd.DataFrame:
    """Tutte le finestre del train che la regola selezionerebbe, scorrendo il tempo una volta sola."""
    coda, ele = etichette(pannello)
    D, T, L = coda.shape
    righe = []
    ultimo = -10**9
    for g in range(D):
        for t in range(STORIA, T - ORIZZONTE):
            assoluto = g * ISTANTI + t
            st = slice(t - STORIA, t)
            fu = slice(t + 1, t + 1 + ORIZZONTE)
            if not coda[g, fu].any():
                continue
            if ele[g, st].mean() < 0.7:
                continue
            per_tratto = coda[g, st].sum(axis=0)
            if per_tratto.sum() == 0:
                cond = "queue_onset"
            elif per_tratto.max() >= 2:
                cond = "queue_ongoing"
            else:
                continue
            if assoluto - ultimo < SPAZIATURA:
                continue
            pers = np.broadcast_to(coda[g, t - 1], (ORIZZONTE, L))
            if iou(pers, coda[g, fu], ele[g, fu]) > 0.9:
                continue
            ultimo = assoluto
            righe.append(dict(pannello=pannello, g=g, t=t, condition=cond,
                              celle_coda=int(coda[g, fu][ele[g, fu]].sum())))
    return pd.DataFrame(righe)


def candidati_ongoing(pannello: str) -> pd.DataFrame:
    """Tutte le origini che la regola ammette come queue_ongoing, senza distanza minima."""
    coda, ele = etichette(pannello)
    D, T, L = coda.shape
    righe = []
    for g in range(D):
        for t in range(STORIA, T - ORIZZONTE):
            st, fu = slice(t - STORIA, t), slice(t + 1, t + 1 + ORIZZONTE)
            if not coda[g, fu].any() or ele[g, st].mean() < 0.7:
                continue
            per_tratto = coda[g, st].sum(axis=0)
            if per_tratto.sum() == 0 or per_tratto.max() < 2:
                continue
            if iou(np.broadcast_to(coda[g, t - 1], (ORIZZONTE, L)), coda[g, fu], ele[g, fu]) > 0.9:
                continue
            righe.append((g, t))
    return pd.DataFrame(righe, columns=["g", "t"])


def candidati_onset(pannello: str, solo_a_30: bool = True) -> pd.DataFrame:
    """Origini queue_onset senza distanza minima: con `solo_a_30` la prima coda dell'orizzonte è a T+30,
    come nelle ufficiali; altrimenti in un passo qualunque dell'orizzonte."""
    coda, ele = etichette(pannello)
    T = coda.shape[1]
    qa = coda.any(axis=2)
    cq = np.concatenate([np.zeros((len(qa), 1), int), np.cumsum(qa, axis=1)], axis=1)
    ce = np.concatenate([np.zeros((len(qa), 1)), np.cumsum(ele.mean(axis=2), axis=1)], axis=1)
    t = np.arange(STORIA, T - ORIZZONTE)
    storia_vuota = cq[:, t] - cq[:, t - STORIA] == 0
    copertura = (ce[:, t] - ce[:, t - STORIA]) / STORIA >= 0.7
    if solo_a_30:
        coda_futura = (cq[:, t + ORIZZONTE] - cq[:, t + 1] == 0) & qa[:, t + ORIZZONTE]
    else:
        coda_futura = cq[:, t + ORIZZONTE + 1] - cq[:, t + 1] > 0
    g, i = np.nonzero(storia_vuota & copertura & coda_futura)
    return pd.DataFrame({"g": g, "t": t[i]})


def ongoing_casuali(pannello: str, seme: int = 0) -> pd.DataFrame:
    """Finestre queue_ongoing scelte a caso tra i candidati, distanti almeno 360 minuti tra loro.

    Le ongoing ufficiali del train non sono la prima origine ammessa: nello stesso giorno cadono
    in posizioni sparse tra i candidati (quantile mediano 0,58), come una scelta casuale.
    Prendere sempre la prima origine favorisce l'inizio degli episodi, dove la persistenza
    rende di più (0,815 contro 0,760 sulle ufficiali).
    """
    import zlib
    c = candidati_ongoing(pannello)
    rng = np.random.default_rng([seme, zlib.crc32(pannello.encode())])
    assoluto = (c.g * ISTANTI + c.t).to_numpy()
    scelti: list[int] = []
    for i in rng.permutation(len(c)):
        if all(abs(assoluto[i] - assoluto[j]) >= SPAZIATURA for j in scelti):
            scelti.append(i)
    s = c.iloc[sorted(scelti)].reset_index(drop=True)
    return s.assign(pannello=pannello, condition="queue_ongoing")[["pannello", "g", "t", "condition"]]
