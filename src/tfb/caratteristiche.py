"""Caratteristiche per il modello del Task 1, calcolate sui tensori (giorno, istante, tratto).

Tutto parte dall'evidenza del giorno: celle osservate ed eleggibili. Per ogni cella
bersaglio si guardano lo stesso tratto prima e dopo (asse del tempo), i tratti a
monte e a valle nello stesso istante (asse dello spazio) e la forma che i tratti
vicini disegnano attraverso lo stesso buco: se il vicino in quell'intervallo si
discosta dalla sua retta tra gli estremi del buco, lo fa probabilmente anche il
bersaglio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dati import Tensori, tratti
from .task1 import evidenza, interpola

NOTTE = (12, 48)  # istanti dall'1:00 alle 4:00: flusso libero quasi ovunque
MESE = ("v_lib_mese", "v_lib_scarto", "v_it_meno_lib")  # colonne di caratteristiche_mese
E5 = ("lr_v", "lr_q", "lr_v_meno_it", "lr_q_meno_it")  # colonne di caratteristiche_e5
RANGO_V, RANGO_Q = 4, 2  # scelti su celle osservate tenute da parte (e5_basso_rango.py rango)


def _precedente(oss: np.ndarray, asse: int) -> np.ndarray:
    """Indice dell'ultima cella osservata strettamente prima lungo l'asse, -1 se nessuna."""
    n = oss.shape[asse]
    forma = [1] * oss.ndim
    forma[asse] = n
    idx = np.where(oss, np.arange(n).reshape(forma), -1)
    acc = np.maximum.accumulate(idx, axis=asse)
    # spostamento di uno: per una cella osservata il "precedente" è quello prima di lei
    out = np.full_like(acc, -1)
    sl_dst = [slice(None)] * oss.ndim
    sl_src = [slice(None)] * oss.ndim
    sl_dst[asse], sl_src[asse] = slice(1, None), slice(None, -1)
    out[tuple(sl_dst)] = acc[tuple(sl_src)]
    return out


def _successivo(oss: np.ndarray, asse: int) -> np.ndarray:
    """Indice della prima cella osservata strettamente dopo lungo l'asse, n se nessuna."""
    n = oss.shape[asse]
    rov = np.flip(oss, axis=asse)
    p = _precedente(rov, asse)
    return np.flip(np.where(p >= 0, n - 1 - p, n), axis=asse)


def _prendi(x: np.ndarray, idx: np.ndarray, asse: int) -> np.ndarray:
    """x lungo l'asse agli indici idx (NaN fuori intervallo)."""
    n = x.shape[asse]
    ok = (idx >= 0) & (idx < n)
    v = np.take_along_axis(x, np.clip(idx, 0, n - 1), axis=asse)
    return np.where(ok, v, np.nan)


def _sposta(x: np.ndarray, k: int, asse: int) -> np.ndarray:
    """x spostato di k lungo l'asse: out[i] = x[i + k], NaN fuori dai bordi."""
    out = np.full_like(x, np.nan)
    n = x.shape[asse]
    if abs(k) >= n:
        return out
    dst = [slice(None)] * x.ndim
    src = [slice(None)] * x.ndim
    if k >= 0:
        dst[asse], src[asse] = slice(0, n - k), slice(k, n)
    else:
        dst[asse], src[asse] = slice(-k, n), slice(0, n + k)
    out[tuple(dst)] = x[tuple(src)]
    return out


def _residuo_buco(serie: np.ndarray, ia: np.ndarray, ib: np.ndarray, t: np.ndarray) -> np.ndarray:
    """serie(t) meno la retta tra serie(ia) e serie(ib), lungo l'asse del tempo."""
    a = _prendi(serie, ia, 1)
    b = _prendi(serie, ib, 1)
    w = np.where(ib > ia, (t - ia) / np.maximum(ib - ia, 1), 0.5)
    return serie - (a + (b - a) * w)


def livello_mese(t: Tensori) -> np.ndarray:
    """(D, L): velocità di notte di ogni tratto nel mese del giorno, mediana delle celle osservate.

    Ogni mese valutato è un mondo con un suo livello in flusso libero per tratto (README, «Ogni
    mese valutato è un mondo diverso»), costante nel mese più rumore bianco: qui lo si stima dal
    mese stesso, come si può fare nel Task 1 con tutte le osservazioni rilasciate dello split.
    """
    vs, _ = evidenza(t)
    notte = vs[:, NOTTE[0]:NOTTE[1], :]
    mesi = pd.to_datetime(pd.Series(t.date)).dt.to_period("M").to_numpy()
    out = np.full((len(t.date), vs.shape[2]), np.nan, np.float32)
    for m in np.unique(mesi):
        i = mesi == m
        out[i] = np.nanmedian(notte[i].reshape(-1, vs.shape[2]), axis=0)
    return out


def caratteristiche_mese(t: Tensori, celle: pd.DataFrame, v_it: np.ndarray) -> dict[str, np.ndarray]:
    """Livello del mese per le celle indicate; `v_it` è l'interpolazione nel tempo nelle stesse celle."""
    vf = tratti(t.pannello).free_speed_kmh.to_numpy(np.float32)
    li = celle.l.to_numpy()
    lv = livello_mese(t)[celle.g.to_numpy(), li]
    return dict(zip(MESE, (lv, lv - vf[li], v_it - lv)))


def basso_rango(Y: np.ndarray, W: np.ndarray, r: int, lam: float = 1.0, giri: int = 15) -> np.ndarray:
    """Completamento a rango r di Y (n × m) dai soli valori dove W: medie di colonna più A·Bᵀ, minimi quadrati alternati con cresta λ."""
    Wf = W.astype(np.float64)
    mu = np.where(W, Y, 0.0).sum(0) / np.maximum(Wf.sum(0), 1)
    R = np.where(W, Y - mu, 0.0)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    A, B = U[:, :r] * np.sqrt(S[:r]), Vt[:r].T * np.sqrt(S[:r])
    I = lam * np.eye(r)
    for _ in range(giri):
        A = np.linalg.solve(np.einsum("ij,jk,jl->ikl", Wf, B, B, optimize=True) + I, (R @ B)[..., None])[..., 0]
        B = np.linalg.solve(np.einsum("ij,ik,il->jkl", Wf, A, A, optimize=True) + I, (R.T @ A)[..., None])[..., 0]
    return mu + A @ B.T


def mesi(t: Tensori) -> np.ndarray:
    return pd.to_datetime(pd.Series([str(d) for d in t.date])).dt.to_period("M").to_numpy()


def completa_mese(t: Tensori, rango_v: int = RANGO_V, rango_q: int = RANGO_Q, tenere: np.ndarray | None = None,
                  soli=None) -> tuple[np.ndarray, np.ndarray]:
    """E5: (D, 288, L) stime a basso rango di velocità e flusso per corsia, mese per mese.

    Per ogni mese la matrice (giorno × istante) × tratto delle osservazioni eleggibili si completa a
    rango r: il mese intero stima il profilo di ogni tratto. `tenere` toglie celle dall'adattamento;
    `soli` limita ai mesi indicati.
    """
    vs, qs = evidenza(t)
    corsie = tratti(t.pannello).lanes.to_numpy(np.float32)
    qs = qs / corsie[None, None, :]
    if tenere is not None:
        vs, qs = np.where(tenere, np.nan, vs), np.where(tenere, np.nan, qs)
    D, T, L = vs.shape
    lv, lq = np.full(vs.shape, np.nan, np.float32), np.full(vs.shape, np.nan, np.float32)
    m = mesi(t)
    for mese in (np.unique(m) if soli is None else soli):
        g = np.flatnonzero(m == mese)
        for x, out, r in ((vs, lv, rango_v), (qs, lq, rango_q)):
            Y = x[g].reshape(-1, L)
            W = np.isfinite(Y)
            out[g] = basso_rango(np.nan_to_num(Y), W, r).reshape(len(g), T, L)
    return lv, lq


def caratteristiche_e5(t: Tensori, celle: pd.DataFrame, v_it: np.ndarray, q_it: np.ndarray) -> dict[str, np.ndarray]:
    """Stime a basso rango del mese per le celle indicate e la loro differenza dall'interpolazione nel tempo."""
    lv, lq = completa_mese(t)
    g, ti, li = celle.g.to_numpy(), celle.t.to_numpy(), celle.l.to_numpy()
    a, b = lv[g, ti, li], lq[g, ti, li]
    return dict(zip(E5, (a, b, a - v_it, b - q_it)))


def calcola(t: Tensori, celle: pd.DataFrame, frazione_regime: dict[str, float], mese: bool = False,
            e5: bool = False) -> pd.DataFrame:
    """Caratteristiche per le celle (g, t, l) indicate; una riga per cella, stesso ordine.

    Con `mese` aggiunge le caratteristiche relative al livello del mese (caratteristiche_mese),
    con `e5` le stime a basso rango del mese (caratteristiche_e5).
    """
    tr = tratti(t.pannello)
    corsie = tr.lanes.to_numpy(np.float32)
    vf = tr.free_speed_kmh.to_numpy(np.float32)
    cap = (tr.capacity_vph / tr.lanes).to_numpy(np.float32)
    lun = tr.length_km.to_numpy(np.float32)
    kjam = (tr.k_jam / tr.lanes).to_numpy(np.float32)
    pos = tr.pos_km.to_numpy(np.float32)

    vs, qs = evidenza(t)
    qs = qs / corsie  # flusso per corsia: confrontabile tra tratti e corridoi
    occ = np.where(np.isfinite(vs), t.occ, np.nan)
    oss = np.isfinite(vs)
    D, T, L = vs.shape
    tt = np.arange(T, dtype=np.float32)[None, :, None]

    g, ti, li = celle.g.to_numpy(), celle.t.to_numpy(), celle.l.to_numpy()
    f: dict[str, np.ndarray] = {}

    # Asse del tempo, stesso tratto
    ia, ib = _precedente(oss, 1), _successivo(oss, 1)
    f["dt_prima"] = (ti - ia[g, ti, li]).astype(np.float32)
    f["dt_dopo"] = (ib[g, ti, li] - ti).astype(np.float32)
    serie = {}
    for nome, x in (("v", vs), ("q", qs), ("o", occ)):
        it = interpola(x, 1)
        serie[nome] = it
        f[f"{nome}_prima"] = _prendi(x, ia, 1)[g, ti, li]
        f[f"{nome}_dopo"] = _prendi(x, ib, 1)[g, ti, li]
        f[f"{nome}_it"] = it[g, ti, li]

    # Asse dello spazio, stesso istante (monte = indice minore)
    sa, sb = _precedente(oss, 2), _successivo(oss, 2)
    km_monte = np.where(sa >= 0, pos[None, None, :] - pos[np.clip(sa, 0, L - 1)], np.nan)
    km_valle = np.where(sb < L, pos[np.clip(sb, 0, L - 1)] - pos[None, None, :], np.nan)
    f["km_monte"] = km_monte[g, ti, li].astype(np.float32)
    f["km_valle"] = km_valle[g, ti, li].astype(np.float32)
    for nome, x in (("v", vs), ("q", qs), ("o", occ)):
        f[f"{nome}_monte"] = _prendi(x, sa, 2)[g, ti, li]
        f[f"{nome}_valle"] = _prendi(x, sb, 2)[g, ti, li]
        f[f"{nome}_is"] = interpola(x, 2)[g, ti, li]

    # Forma dei vicini attraverso lo stesso buco temporale del bersaglio
    for nome in ("v", "q"):
        it = serie[nome]
        for k in (-2, -1, 1, 2):
            vic = _sposta(it, k, 2)
            r = _residuo_buco(vic, ia, ib, tt)
            f[f"{nome}_res_vic{k:+d}"] = r[g, ti, li]
        for dk in (-1, 1):
            for k in (-1, 1):
                f[f"{nome}_vic{k:+d}_t{dk:+d}"] = _sposta(_sposta(it, k, 2), dk, 1)[g, ti, li]

    # Quante osservazioni intorno: misura di quanto è "pieno" il contesto
    o = oss.astype(np.float32)
    dens = np.zeros_like(o)
    for k in range(-3, 4):
        dens += np.nan_to_num(_sposta(o, k, 1))
    f["oss_intorno"] = dens[g, ti, li]

    # Statici e calendario
    f["vf"] = vf[li]
    f["cap_corsia"] = cap[li]
    f["corsie"] = corsie[li]
    f["lunghezza"] = lun[li]
    f["kjam_corsia"] = kjam[li]
    f["istante"] = ti.astype(np.float32)
    giorni = pd.to_datetime(pd.Series(t.date)).dt.weekday.to_numpy()
    f["giorno_sett"] = giorni[g].astype(np.float32)
    f["quota_mascherata"] = celle.mask_regime.map(frazione_regime).to_numpy(np.float32)
    f["pos_relativa"] = (pos[li] / max(pos.max(), 1e-6)).astype(np.float32)
    if mese:
        f.update(caratteristiche_mese(t, celle, f["v_it"]))
    if e5:
        f.update(caratteristiche_e5(t, celle, f["v_it"], f["q_it"]))

    X = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    return X


def base(X: pd.DataFrame, nome: str) -> np.ndarray:
    """Valore di partenza per il residuo: tempo, poi spazio."""
    b = X[f"{nome}_it"].to_numpy()
    return np.where(np.isfinite(b), b, X[f"{nome}_is"].to_numpy())
