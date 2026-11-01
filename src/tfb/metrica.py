"""Punteggi locali sul train, sugli stessi conti dei valutatori ufficiali.

`s_state` riproduce score_task1.py. `s_lwr_stima` è solo una stima: il valutatore
del Task 3 usa flussi di confine che gli organizzatori non rilasciano e che, per
loro dichiarazione, rendono esatta la conservazione sulle osservazioni rilasciate.
Qui quel termine noto si sostituisce con la variazione vera di N = (q/v)·L, quindi
la stima vale 1 per la risposta perfetta (0,96 sul valutatore ufficiale).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dati import REGIMI, Tensori, tratti

DT_H = 5.0 / 60.0


def s_state(b: pd.DataFrame, vero: Tensori, v_hat: np.ndarray, q_hat: np.ndarray) -> dict:
    """S_state per regime e medio. `v_hat`, `q_hat` sono allineati alle righe di `b`."""
    corsie = tratti(vero.pannello).lanes.to_numpy(float)
    g, t, l = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()
    dv = v_hat - vero.speed[g, t, l]
    dq = (q_hat - vero.flow[g, t, l]) / corsie[l]
    out = {}
    for r in REGIMI:
        m = (b.mask_regime == r).to_numpy()
        rv, rq = np.sqrt(np.mean(dv[m] ** 2)), np.sqrt(np.mean(dq[m] ** 2))
        out[r] = dict(rmse_v=rv, rmse_q=rq, S=0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
    out["S_state"] = float(np.mean([out[r]["S"] for r in REGIMI]))
    return out


def s_lwr_stima(b: pd.DataFrame, vero: Tensori, v_hat: np.ndarray, q_hat: np.ndarray) -> dict:
    """Stima di S_LWR per regime: 1 - Σ|ΔN̂ - ΔN| / Σ|ΔN| sulle transizioni tra celle eleggibili."""
    lun = tratti(vero.pannello).length_km.to_numpy(float)
    N = vero.flow / np.maximum(vero.speed, 1.0) * lun
    Nh = N.copy()
    g, t, l = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()
    Nh[g, t, l] = q_hat / np.maximum(v_hat, 1.0) * lun[l]
    ele = vero.eleggibile & np.isfinite(N)
    ok = ele[:, 1:, :] & ele[:, :-1, :]
    dN, dNh = np.diff(N, axis=1), np.diff(Nh, axis=1)
    out = {}
    for r in REGIMI:
        giorni = vero_regime(b, vero) == r
        m = ok & giorni[:, None, None]
        num = np.abs(dNh - dN)[m].sum()
        den = np.abs(dN)[m].sum()
        out[r] = float(max(0.0, 1 - num / den))
    out["S_LWR"] = float(np.mean([out[r] for r in REGIMI]))
    return out


def vero_regime(b: pd.DataFrame, vero: Tensori) -> np.ndarray:
    """Regime di ogni giorno del tensore, ricavato dalle righe bersaglio."""
    reg = np.full(len(vero.date), "", dtype=object)
    per_giorno = b.groupby("g").mask_regime.first()
    reg[per_giorno.index.to_numpy()] = per_giorno.to_numpy()
    return reg


def parametri_fisica(pannello: str) -> pd.DataFrame:
    """Parametri per tratto (ordine di `tratti`) come network_parameters del valutatore del Task 3.

    lanes, free_speed e capacity da fd_parameters (media per tratto), con ripiego su links.csv;
    critical_density = capacity / free_speed; k_jam mancante = 100 per corsia, almeno 1,05·k_crit;
    length_km da links.csv.
    """
    from .dati import RILASCIO
    rete = RILASCIO / "corridors" / pannello / "network"
    n = pd.read_csv(rete / "links.csv"); n["link_id"] = n.link_id.astype(str); n = n.drop_duplicates("link_id")
    out = n[["link_id", "length_km", "free_speed_kmh", "capacity_vph", "lanes"]].copy()
    out["length_km"] = pd.to_numeric(out.length_km, errors="coerce").fillna(1.0)
    out["free_speed_kmh"] = pd.to_numeric(out.free_speed_kmh, errors="coerce").fillna(100.0)
    corsie = pd.to_numeric(out.lanes, errors="coerce")
    out["lanes"] = corsie.fillna(1.0).clip(lower=1.0)
    out["capacity_vph"] = pd.to_numeric(out.capacity_vph, errors="coerce").fillna(1800.0 * corsie.fillna(4.0))
    f = pd.read_csv(rete / "fd_parameters.csv"); f["link_id"] = f.link_id.astype(str)
    m = f.groupby("link_id", as_index=False)[["free_speed_kmh", "capacity_vph", "lanes", "k_jam"]].mean()
    out = out.merge(m, on="link_id", how="left", suffixes=("", "_fd"))
    for c in ("free_speed_kmh", "capacity_vph", "lanes"):
        out[c] = out[f"{c}_fd"].fillna(out[c])
    out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    out["k_jam"] = np.maximum(out.k_jam.fillna(100.0 * out.lanes.clip(lower=1.0)), out.critical_density * 1.05)
    return tratti(pannello)[["link_id"]].merge(out[["link_id", "length_km", "free_speed_kmh", "capacity_vph", "lanes",
                                                     "critical_density", "k_jam"]], on="link_id", how="left")


def s_fisica_stima(b: pd.DataFrame, vero: Tensori, v_hat: np.ndarray, q_hat: np.ndarray,
                   valida: np.ndarray | None = None) -> dict:
    """S_FD, S_LWR e S_physics per regime con l'universo del valutatore ufficiale (modalità A).

    Stato: celle eleggibili dei giorni del regime, con le risposte nelle celle bersaglio.
    S_FD come il valutatore su tutte quelle righe. S_LWR solo sui tratti osservabili (vicini di
    linea principale tutti presenti) e sulle transizioni con rampe valide; il termine noto dei
    flussi di confine, non rilasciato, si sostituisce con la variazione vera di N (stima).
    `valida` è validita_rampe dello split, da passare già tagliata se `vero` è una parte dei giorni.
    """
    from .dati import validita_rampe
    p = vero.pannello
    par = parametri_fisica(p)
    L = par.length_km.to_numpy(float)
    g, t, l = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()
    v, q = vero.speed.astype(float), vero.flow.astype(float)
    vh, qh = v.copy(), q.copy()
    vh[g, t, l], qh[g, t, l] = v_hat, q_hat
    ele = vero.eleggibile & np.isfinite(v) & np.isfinite(q)
    bersaglio = np.zeros(ele.shape, bool); bersaglio[g, t, l] = True
    N, Nh = q / np.maximum(v, 1.0) * L, qh / np.maximum(vh, 1.0) * L
    # tratti osservabili: ogni vicino di linea principale (in, out) ha righe nel regime
    from .dati import RILASCIO
    topo = pd.read_csv(RILASCIO / "corridors" / p / "network" / "lwr_mainline_topology.csv", dtype=str).set_index("link_id")
    ids = list(par.link_id)
    if valida is None:
        valida = validita_rampe(p, vero.split)
    reg = vero_regime(b, vero)
    out = {}
    for r in REGIMI:
        giorni = reg == r
        e = ele & giorni[:, None, None]
        presenti = set(np.asarray(ids)[e.any(axis=(0, 1))])
        oss = np.array([
            l_ in topo.index and all(x in presenti for c in ("incoming_link_ids", "outgoing_link_ids")
                                     for x in str(topo.loc[l_, c]).split(";") if x.strip() in topo.index)
            for l_ in ids])
        s_fd = _s_fd(par, vh, qh, e, bersaglio)
        m = e[:, 1:] & e[:, :-1] & oss[None, None, :] & valida[:, :-1]
        dN, dNh = np.diff(N, axis=1), np.diff(Nh, axis=1)
        s_lwr = max(0.0, 1 - min(1.0, np.abs(dNh - dN)[m].sum() / (np.abs(dN)[m].sum() + 1e-9)))
        out[r] = dict(S_FD=float(max(0.0, s_fd)), S_LWR=float(s_lwr), transizioni=int(m.sum()),
                      quota_tratti_osservabili=float(oss.mean()))
    for k in ("S_FD", "S_LWR"):
        out[k] = float(np.mean([out[r][k] for r in REGIMI]))
    out["S_physics"] = out["S_FD"] / 3 + 2 * out["S_LWR"] / 3
    return out


def _s_fd(par: pd.DataFrame, vh: np.ndarray, qh: np.ndarray, e: np.ndarray, bersaglio: np.ndarray) -> float:
    """S_FD del valutatore sulle righe `e`: scarto relativo del flusso dal diagramma triangolare."""
    corsie = par.lanes.to_numpy(float)
    vf, cap = par.free_speed_kmh.to_numpy(float), par.capacity_vph.to_numpy(float)
    kc, kj = par.critical_density.to_numpy(float), par.k_jam.to_numpy(float)
    k_l, q_l = qh / np.maximum(vh, 1.0) / corsie, qh / corsie
    q_fd = np.where(k_l <= kc / corsie, vf * k_l, (cap / corsie) / np.maximum(kj / corsie - kc / corsie, 1e-9) * np.maximum(kj / corsie - k_l, 0))
    if (qh[e & bersaglio] < 50).mean() > 0.20:
        return 0.0
    return float(max(0.0, 1 - np.abs(q_l - q_fd)[e].sum() / (np.abs(q_l)[e].sum() + 1e-9)))


def s_fd_valutato(b: pd.DataFrame, masc: Tensori, v_hat: np.ndarray, q_hat: np.ndarray) -> dict:
    """S_FD esatto in validation e private: non serve la verità.

    Le righe sono le celle eleggibili dei giorni del regime; nelle celle bersaglio valgono le
    risposte, altrove i valori rilasciati. È la stessa formula di s_fisica_stima.
    """
    par = parametri_fisica(masc.pannello)
    g, t, l = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()
    vh, qh = masc.speed.astype(float), masc.flow.astype(float)
    vh[g, t, l], qh[g, t, l] = v_hat, q_hat
    bersaglio = np.zeros(vh.shape, bool); bersaglio[g, t, l] = True
    ele = masc.eleggibile & np.isfinite(vh) & np.isfinite(qh)
    reg = vero_regime(b, masc)
    out = {r: _s_fd(par, vh, qh, ele & (reg == r)[:, None, None], bersaglio) for r in REGIMI}
    out["S_FD"] = float(np.mean([out[r] for r in REGIMI]))
    return out
