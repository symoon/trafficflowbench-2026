"""Task 1, E2: la zona di una finestra del Task 2 ricostruita tutta insieme da una rete convoluzionale piccola.

Allenamento e validazione in scripts/e2_sequenze.py; valutazione sui pseudo-mesi in scripts/cv_pseudomesi.py;
nell'invio (task1.ricostruisci_lgb_buchi con rete=True) la previsione della rete nella regione da T a T+24 si
media con quella del modello LightGBM sui buchi. Serve PyTorch (gruppo `gpu` di pyproject.toml), solo qui.

La zona va da T−12 a T+36 su tutto il corridoio. Canali per cella:
- velocità relativa al livello notturno del tratto nel mese, flusso per corsia e occupazione, solo dove
  osservati (zero altrove), e la maschera di osservazione;
- la maschera dei bersagli da prevedere (da T a T+24);
- il valore di partenza (task1.ricostruisci_v1, la base dei modelli LightGBM), in canali a parte;
- la distanza dall'ultima e dalla prossima osservazione sullo stesso tratto, l'istante del giorno, i dati
  statici del tratto, la maschera di validità della zona.
Uscita: residui di v/25 e di q per corsia/600 rispetto al valore di partenza.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dati import Tensori, tratti

PRIMA, DOPO = 12, 36
PASSI = PRIMA + DOPO + 1
REGIONE = (0, 24)  # bersagli che la rete prevede, rispetto all'origine T (come task1.regione_finestre)
CANALI = ["v_oss", "q_oss", "o_oss", "osservata", "bersaglio", "v_base", "q_base", "dt_prima", "dt_dopo",
          "ora_sin", "ora_cos", "vf", "corsie", "cap", "lunghezza", "pos", "valida"]


def zona(x: np.ndarray, g: int, t: int, riempi=0.0) -> np.ndarray:
    """x[g, t−12 : t+37, :] con riempimento fuori dal giorno."""
    T = x.shape[1]
    out = np.full((PASSI, x.shape[2]), riempi, dtype=x.dtype)
    a, b = max(0, t - PRIMA), min(T, t + DOPO + 1)
    out[a - (t - PRIMA):b - (t - PRIMA)] = x[g, a:b]
    return out


def ingressi(mb: Tensori, bersaglio: np.ndarray, w: pd.DataFrame, v1=None, q1=None) -> tuple[np.ndarray, np.ndarray]:
    """Ingressi X (n, canali, PASSI, L) e valori di partenza B (n, 2, PASSI, L) delle zone delle finestre `w` (g, t).

    `bersaglio` (D, 288, L) sono i bersagli dello strato; il livello del tratto è quello del mese dello strato.
    """
    from .caratteristiche import _precedente, _successivo, livello_mese
    from .task1 import evidenza, ricostruisci_v1
    tr = tratti(mb.pannello)
    corsie, vf = tr.lanes.to_numpy(np.float32), tr.free_speed_kmh.to_numpy(np.float32)
    cap = (tr.capacity_vph / tr.lanes).to_numpy(np.float32)
    pos = tr.pos_km.to_numpy(np.float32); pos = pos / max(pos.max(), 1e-6)
    statici = np.stack([vf / 100, corsie / 5, cap / 2500, tr.length_km.to_numpy(np.float32), pos])
    livello = livello_mese(mb)
    if v1 is None:
        v1, q1 = ricostruisci_v1(mb)
    vs, qs = evidenza(mb)
    occ = np.where(np.isfinite(vs), mb.occ, np.nan)
    oss = np.isfinite(vs)
    ia, ib = _precedente(oss, 1), _successivo(oss, 1)
    tt = np.arange(oss.shape[1])[None, :, None]
    dta = np.where(ia >= 0, tt - ia, 99).astype(np.float32)
    dtb = np.where(ib < oss.shape[1], ib - tt, 99).astype(np.float32)
    uno = np.ones(mb.speed.shape, np.float32)
    # bersagli da prevedere: da T a T+24, come la regione di v10; nella storia non ce ne sono
    reg = np.zeros((PASSI, len(vf)), np.float32); reg[PRIMA + REGIONE[0]:PRIMA + REGIONE[1] + 1] = 1
    Xs, B = [], []
    for g, t in zip(w.g, w.t):
        o = zona(oss.astype(np.float32), g, t)
        lv = livello[g][None, :]
        ore = (t - PRIMA + np.arange(PASSI)) / 288 * 2 * np.pi
        x = np.stack([
            np.nan_to_num((zona(vs, g, t, np.nan) - lv) / 25), np.nan_to_num(zona(qs, g, t, np.nan) / corsie / 600),
            np.nan_to_num(zona(occ, g, t, np.nan)), o, zona(bersaglio.astype(np.float32), g, t) * reg,
            (zona(v1, g, t, np.nan) - lv) / 25, zona(q1, g, t, np.nan) / corsie / 600,
            np.minimum(zona(dta, g, t, 99), 36) / 18, np.minimum(zona(dtb, g, t, 99), 36) / 18,
            np.broadcast_to(np.sin(ore)[:, None], o.shape), np.broadcast_to(np.cos(ore)[:, None], o.shape),
            *[np.broadcast_to(s_[None, :], o.shape) for s_ in statici], zona(uno, g, t)])
        Xs.append(np.nan_to_num(x).astype(np.float16))
        B.append(np.stack([zona(v1, g, t, np.nan), zona(q1, g, t, np.nan)]).astype(np.float32))
    return np.stack(Xs), np.stack(B)


def costruisci_rete(c_in: int = len(CANALI), c: int = 64):
    import torch.nn as nn

    class Blocco(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.a = nn.Conv2d(c, c, 3, padding=d, dilation=d)
            self.b = nn.Conv2d(c, c, 3, padding=1)
            self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
            self.f = nn.GELU()

        def forward(self, x):
            return x + self.b(self.f(self.n2(self.a(self.f(self.n1(x))))))

    class Rete(nn.Module):
        def __init__(self):
            super().__init__()
            self.dentro = nn.Conv2d(c_in, c, 1)
            # dilatazioni (tempo, spazio): il campo copre tutta la zona nel tempo e ±20 tratti
            self.blocchi = nn.Sequential(*[Blocco(d) for d in ((1, 1), (2, 1), (4, 2), (8, 1), (2, 4), (4, 1), (1, 2), (1, 1))])
            self.fuori = nn.Conv2d(c, 2, 1)

        def forward(self, x):
            return self.fuori(self.blocchi(self.dentro(x)))

    return Rete()


def carica_rete(percorso):
    import torch
    r = costruisci_rete()
    r.load_state_dict(torch.load(percorso, map_location="cpu"))
    return r.eval()


def prevedi_regione(rete, st: Tensori, bersaglio: np.ndarray, w: pd.DataFrame, v1: np.ndarray, q1: np.ndarray,
                    g: np.ndarray, ti: np.ndarray, li: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Velocità e flusso per corsia della rete nelle celle (g, ti, li) da T a T+24 di una finestra di `w`; NaN altrove."""
    import torch
    corsie = tratti(st.pannello).lanes.to_numpy(np.float32)
    vh, qh = np.full(len(g), np.nan, np.float32), np.full(len(g), np.nan, np.float32)
    if not len(w):
        return vh, qh
    finestra = np.full(st.speed.shape[:2], -1)
    for j, (gg, tt) in enumerate(zip(w.g, w.t)):
        finestra[gg, max(0, tt + REGIONE[0]):tt + REGIONE[1] + 1] = j
    jf = finestra[g, ti]
    X, B = ingressi(st, bersaglio, w, v1, q1)
    with torch.no_grad():
        u = np.concatenate([rete(torch.tensor(X[i:i + 4].astype(np.float32))).numpy() for i in range(0, len(X), 4)])
    sel = jf >= 0
    j, s, l = jf[sel], ti[sel] - w.t.to_numpy()[jf[sel]] + PRIMA, li[sel]
    vh[sel] = B[j, 0, s, l] + 25 * u[j, 0, s, l]
    qh[sel] = B[j, 1, s, l] / corsie[l] + 600 * u[j, 1, s, l]
    return vh, qh
