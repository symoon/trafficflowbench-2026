"""Task 2: caratteristiche per tratto di una finestra e insiemi di previsione.

Ogni finestra si descrive con la sua storia (T-60 … T-5, 12 passi), la riga a T e i
dati statici della rete. Tutto ciò che entra ha timestamp ≤ T, più una frequenza
storica di coda per tratto e ora calcolata sul train, che precede validation e private.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dati import ISTANTI, tratti

SOGLIA = 0.60
VICINI = (-4, -3, -2, -1, 1, 2, 3, 4)  # negativi a monte, positivi a valle


def frequenza_coda(coda: np.ndarray, ele: np.ndarray, giorni: np.ndarray) -> np.ndarray:
    """(288, L): quota di giorni in coda per istante e tratto, mediata su ±15 minuti."""
    c = coda[giorni].sum(axis=0).astype(np.float64)
    n = ele[giorni].sum(axis=0).astype(np.float64)
    k = np.ones(7)
    cs = np.apply_along_axis(lambda x: np.convolve(x, k, mode="same"), 0, c)
    ns = np.apply_along_axis(lambda x: np.convolve(x, k, mode="same"), 0, n)
    return (cs / np.maximum(ns, 1.0)).astype(np.float32)


def _ultimo_valido(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni tratto l'ultimo valore non NaN lungo il tempo (asse 0) e quanti passi fa."""
    T = x.shape[0]
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(T)[:, None], -1).max(axis=0)
    val = np.where(idx >= 0, x[np.clip(idx, 0, T - 1), np.arange(x.shape[1])], np.nan)
    return val, np.where(idx >= 0, T - 1 - idx, np.nan)


def _distanza_coda(coda: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni tratto la distanza in km dal tratto in coda più vicino a valle e a monte (NaN se nessuno)."""
    L = len(pos)
    idx = np.flatnonzero(coda)
    valle = np.full(L, np.nan, np.float32)
    monte = np.full(L, np.nan, np.float32)
    if len(idx):
        j = np.searchsorted(idx, np.arange(L))
        ok = j < len(idx)
        valle[ok] = np.abs(pos[idx[j[ok]]] - pos[np.arange(L)[ok]])
        jm = np.searchsorted(idx, np.arange(L), side="right") - 1
        ok = jm >= 0
        monte[ok] = np.abs(pos[np.arange(L)[ok]] - pos[idx[jm[ok]]])
    return valle, monte


def caratteristiche(pannello: str, storia_v: np.ndarray, storia_q: np.ndarray, storia_o: np.ndarray,
                    T_v: np.ndarray, T_q: np.ndarray, T_o: np.ndarray, t_origine: int,
                    frequenza: np.ndarray) -> pd.DataFrame:
    """Una riga per tratto. storia_* è (12, L) con NaN dove non osservato; T_* è (L,)."""
    tr = tratti(pannello)
    vf = tr.free_speed_kmh.to_numpy(np.float32)
    corsie = tr.lanes.to_numpy(np.float32)
    cap = (tr.capacity_vph / tr.lanes).to_numpy(np.float32)
    pos = tr.pos_km.to_numpy(np.float32)
    L = len(vf)
    tutto_v = np.vstack([storia_v, T_v[None]])  # 13 passi, l'ultimo è T
    tutto_q = np.vstack([storia_q, T_q[None]]) / corsie
    tutto_o = np.vstack([storia_o, T_o[None]])
    r = tutto_v / vf
    f: dict[str, np.ndarray] = {}
    ult_r, eta = _ultimo_valido(r)
    f["r_ultimo"], f["eta_ultimo"] = ult_r, eta
    for k in (0, 1, 2, 3, 6, 12):  # passi indietro da T
        f[f"r_T-{k}"] = r[12 - k]
    f["r_min"] = np.nanmin(np.where(np.isfinite(r), r, np.inf), axis=0)
    f["r_min"][~np.isfinite(f["r_min"])] = np.nan
    f["r_media_15"] = np.nanmean(r[-4:], axis=0) if np.isfinite(r[-4:]).any() else np.full(L, np.nan)
    f["r_media_60"] = np.nanmean(r, axis=0)
    f["pendenza_r"] = f["r_media_15"] - np.nanmean(r[:6], axis=0)
    ult_q, _ = _ultimo_valido(tutto_q)
    ult_o, _ = _ultimo_valido(tutto_o)
    f["q_ultimo"], f["q_su_cap"], f["o_ultimo"] = ult_q, ult_q / cap, ult_o
    f["q_media_60"] = np.nanmean(tutto_q, axis=0)
    f["o_media_15"] = np.nanmean(tutto_o[-4:], axis=0)
    for k in VICINI:
        sh = np.full(L, np.nan, np.float32)
        if k > 0:
            sh[:-k] = ult_r[k:]
        else:
            sh[-k:] = ult_r[:k]
        f[f"r_vic{k:+d}"] = sh
    # geometria della coda all'ultimo stato noto e 15 e 30 minuti prima
    coda_ult = np.isfinite(ult_r) & (ult_r <= SOGLIA)
    f["in_coda"] = coda_ult.astype(np.float32)
    for nome, riga in (("ult", ult_r), ("T-3", r[9]), ("T-6", r[6])):
        cd = np.isfinite(riga) & (riga <= SOGLIA)
        valle, monte = _distanza_coda(cd, pos)
        f[f"km_coda_valle_{nome}"], f[f"km_coda_monte_{nome}"] = valle, monte
    f["passi_in_coda"] = np.nansum((r <= SOGLIA).astype(np.float32), axis=0)
    f["n_coda_corridoio"] = np.full(L, coda_ult.sum(), np.float32)
    f["frequenza"] = frequenza[min(t_origine + 6, ISTANTI - 1)]
    f["frequenza_T"] = frequenza[t_origine]
    f["vf"], f["cap_corsia"], f["corsie"] = vf, cap, corsie
    f["lunghezza"] = tr.length_km.to_numpy(np.float32)
    f["istante"] = np.full(L, t_origine, np.float32)
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def insieme(p: np.ndarray, soglia: float, almeno_uno: bool) -> np.ndarray:
    """Celle previste in coda: sopra soglia; se nessuna e almeno_uno, la più probabile."""
    s = p >= soglia
    if almeno_uno and not s.any() and len(p):
        s[np.argmax(p)] = True
    return s


def contesto(p: np.ndarray, gid: np.ndarray, passo: np.ndarray, l: np.ndarray, L: np.ndarray,
             onset: bool) -> tuple[np.ndarray, list[str]]:
    """Contesto di una probabilità per cella nella sua finestra, per il secondo livello dello stack.

    gid: finestra di ogni riga (0 … G−1), passo 1 … 6, l tratto (negativi a monte), L tratti per finestra.
    Tratti vicini, massimi a monte e a valle entro 6 tratti, massimo, rango e somma nel passo;
    per le ongoing anche i passi vicini e media e massimo sul tratto. Usa solo le probabilità
    della stessa finestra, quindi solo dati ≤ T.
    """
    gid, passo, l, L = (np.asarray(x, np.int64) for x in (gid, passo, l, L))
    off = np.concatenate([[0], np.cumsum(6 * L)])
    Lr = L[gid]
    pos = off[gid] + (passo - 1) * Lr + l
    D = np.full(off[-1], np.nan, np.float32)
    D[pos] = p

    def sposta(ok: np.ndarray, dove: np.ndarray) -> np.ndarray:
        return np.where(ok, D[np.where(ok, dove, pos)], np.nan).astype(np.float32)

    out = {}
    vic = {d: sposta((l + d >= 0) & (l + d < Lr), pos + d) for d in range(-6, 7) if d}
    for d in (-3, -2, -1, 1, 2, 3):
        out[f"p1_vic{d:+d}"] = vic[d]
    out["p1_max_monte6"] = np.fmax.reduce([vic[-d] for d in range(1, 7)])
    out["p1_max_valle6"] = np.fmax.reduce([vic[d] for d in range(1, 7)])
    s = pd.Series(p)
    g = s.groupby(gid * 6 + passo - 1)
    mx = g.transform("max").to_numpy()
    out["p1_max_passo"] = mx
    out["p1_rel"] = p / np.maximum(mx, 1e-6)
    out["p1_rango"] = g.rank(ascending=False, method="min").to_numpy()
    out["p1_somma_passo"] = g.transform("sum").to_numpy()
    if not onset:
        for dt in (-1, 1):
            out[f"p1_passo{dt:+d}"] = sposta((passo + dt >= 1) & (passo + dt <= 6), pos + dt * Lr)
        g2 = s.groupby(gid * int(L.max()) + l)
        out["p1_media_tratto"] = g2.transform("mean").to_numpy()
        out["p1_max_tratto"] = g2.transform("max").to_numpy()
    nomi = list(out)
    return np.column_stack([np.asarray(out[n], np.float32) for n in nomi]), nomi


NOTTE = (12, 48)  # istanti dall'1:00 alle 4:00, come caratteristiche.NOTTE


def scarto_e1(vs: np.ndarray, mesi: np.ndarray, g: int, t: int, livello: np.ndarray) -> np.ndarray:
    """(L,) E1 del Task 2: livello notturno del mese con le sole notti fino a T compreso, meno quello del train.

    vs: (D, 288, L) velocità osservate ed eleggibili dello strato mascherato dello split (NaN altrove);
    mesi: mese di ogni giorno. Entrano le notti dei giorni precedenti dello stesso mese e quella del
    giorno g fino all'istante t: nessun dato dopo T. 0 dove non c'è nessuna osservazione.
    """
    giorni = np.flatnonzero(mesi[:g] == mesi[g])
    parti = [vs[giorni, NOTTE[0]:NOTTE[1], :].reshape(-1, vs.shape[2])]
    if t >= NOTTE[0]:
        parti.append(vs[g, NOTTE[0]:min(NOTTE[1], t + 1), :])
    x = np.concatenate(parti)
    if not len(x):
        return np.zeros(vs.shape[2], np.float32)
    d = np.nanmedian(x, axis=0) - livello
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)


def togli_scarto(v: np.ndarray, d: np.ndarray, vf: np.ndarray) -> np.ndarray:
    """Velocità riportate nel mondo del train: − d con il peso del flusso libero (task1.peso_libero)."""
    return v - d * np.clip((v / vf - 0.80) / 0.15, 0.0, 1.0)
