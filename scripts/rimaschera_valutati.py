"""Task 1: confronto dei modelli nei mesi valutati, rimascherando celle osservate.

I mesi valutati sono mondi diversi dal train e né la validazione a blocchi né il cambio
di mondo simulato (cv_mondo.py) hanno previsto la classifica di v7 e v9. Qui, in
validation e private, si oscura a caso una frazione delle celle osservate ed eleggibili
e si ricostruiscono come i bersagli veri, con la stessa catena di task1.ricostruisci_lgb:
la verità è il valore osservato. Le maschere ufficiali sono estratte a caso per cella,
quindi le celle oscurate qui somigliano ai bersagli; il contesto è solo un po' più rado.

Si usano solo gli ingressi rilasciati, nessuna etichetta, e solo per confrontare modelli
già congelati in outputs/invii/<versione>/modelli: qui non si allena nulla. S_LWR si
stima sulle sole transizioni tra celle con valore osservato, quindi è schiacciato verso 1:
contano le differenze tra metodi, non il livello.

Con --vicino K si oscurano solo celle entro K istanti, sullo stesso tratto, da un buco che non è
bersaglio (orizzonte e margine delle finestre del Task 2): nel train questi buchi non ci sono,
e stanno attorno alle code.

Con --e1 ogni versione si misura anche con E1 (task1.equivariante): lo scarto del livello notturno
del mese da quello del train si toglie dagli ingressi e si restituisce alle velocità previste.

    uv run python scripts/rimaschera_valutati.py [--frazione 0.1] [--semi 2] [--versioni v6 v7 v9] [--vicino 6] [--e1]
"""
import argparse
import dataclasses
import json
import time
import warnings
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.caratteristiche import base, calcola
from tfb.dati import RADICE_PROGETTO, bersagli_task1, pannelli, tensori, tratti
from tfb.invio import SPLIT_VALUTATI
from tfb.metrica import s_lwr_stima, s_state
from tfb.task1 import peso_libero, ricostruisci_v1, riporta, scarto_livello

warnings.filterwarnings("ignore", category=RuntimeWarning)
FRAZIONE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
INVII = RADICE_PROGETTO / "outputs" / "invii"
USCITA = RADICE_PROGETTO / "outputs" / "task1" / "rimaschera_valutati.csv"


def carica(versione: str):
    c = INVII / versione / "modelli" / "task1"
    colonne = json.loads((c / "finale_parametri.json").read_text())["colonne"]
    return {n: lgb.Booster(model_file=str(c / f"finale_{n}.txt")) for n in ("v", "q")}, colonne


def vicine_ai_buchi(m, k: int) -> np.ndarray:
    """(D, 288, L) bool: entro k istanti, sullo stesso tratto, da una cella eleggibile vuota che non è bersaglio."""
    b = bersagli_task1(m.pannello, m.split, m)
    bersaglio = np.zeros(m.speed.shape, bool); bersaglio[b.g, b.t, b.l] = True
    buco = m.eleggibile & ~np.isfinite(m.speed) & ~bersaglio
    vicino = buco.copy()
    for d in range(1, k + 1):
        vicino[:, d:] |= buco[:, :-d]
        vicino[:, :-d] |= buco[:, d:]
    return vicino


def oscura(m, frazione: float, rng, ammesse=None):
    """Copia dello strato con una frazione delle celle osservate ed eleggibili oscurata come le ufficiali."""
    ok = m.eleggibile & np.isfinite(m.speed) & np.isfinite(m.flow)
    cand = np.flatnonzero((ok if ammesse is None else ok & ammesse).ravel())
    scelte = np.sort(rng.choice(cand, size=int(frazione * len(cand)), replace=False))
    g, t, l = np.unravel_index(scelte, m.speed.shape)
    celle = pd.DataFrame(dict(g=g, t=t, l=l, mask_regime=m.regime[g]))
    canali = {}
    for nome in ("speed", "flow", "occ", "dens_occ"):
        x = getattr(m, nome).copy()
        x[g, t, l] = np.nan
        canali[nome] = x
    return dataclasses.replace(m, **canali), celle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frazione", type=float, default=0.1)
    ap.add_argument("--semi", type=int, default=2)
    ap.add_argument("--versioni", nargs="+", default=["v6", "v7", "v9"])
    ap.add_argument("--thread", type=int, default=12)
    ap.add_argument("--vicino", type=int, default=0)
    ap.add_argument("--e1", action="store_true")
    a = ap.parse_args()
    uscita = USCITA if not a.vicino else USCITA.with_name(f"rimaschera_valutati_vicino{a.vicino}.csv")
    if a.e1:
        uscita = uscita.with_name(uscita.stem + "_e1.csv")
    if a.versioni != ["v6", "v7", "v9"]:  # un file per confronto, senza sovrascrivere i precedenti
        uscita = uscita.with_name(uscita.stem + "_" + "_".join(a.versioni) + ".csv")
    modelli = {v: carica(v) for v in a.versioni}
    t0 = time.time()
    righe = []
    for p in pannelli():
        corsie = tratti(p).lanes.to_numpy(np.float32)
        vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
        for s in SPLIT_VALUTATI:
            m = tensori(p, s)
            ammesse = vicine_ai_buchi(m, a.vicino) if a.vicino else None
            for seme in range(a.semi):
                rng = np.random.default_rng([seme, zlib.crc32(p.encode()), zlib.crc32(s.encode())])
                m2, celle = oscura(m, a.frazione, rng, ammesse)
                if celle.empty:
                    continue
                g, ti, li = celle.g.to_numpy(), celle.t.to_numpy(), celle.l.to_numpy()
                d = scarto_livello(m2)
                metodi = {}
                for e1, strato in ((False, m2), (True, riporta(m2, d)))[:2 if a.e1 else 1]:
                    coda = "_e1" if e1 else ""
                    X = calcola(strato, celle, FRAZIONE, mese=True, e5=True)  # colonne in più: ogni modello prende le sue
                    v1, q1 = ricostruisci_v1(strato)
                    bv, bq = base(X, "v"), base(X, "q")
                    # come task1.ricostruisci_lgb: dove manca anche lo spazio resta la ricostruzione v1
                    bv = np.where(np.isfinite(bv), bv, v1[g, ti, li])
                    bq = np.where(np.isfinite(bq), bq, q1[g, ti, li] / corsie[li])
                    uscite = {"interpolazione": (bv, bq)}
                    for v, (mod, colonne) in modelli.items():
                        uscite[v] = (bv + mod["v"].predict(X[colonne], num_threads=a.thread),
                                     bq + mod["q"].predict(X[colonne], num_threads=a.thread))
                    for nome, (vh, qh) in uscite.items():
                        vh = np.clip(vh, 3.0, None)
                        if e1:  # come task1.equivariante
                            vh = vh + d[g, li] * peso_libero(vh, vf[li])
                        metodi[nome + coda] = (vh, qh)
                for metodo, (vh, qh) in metodi.items():
                    vh, qt = np.clip(vh, 3.0, None), np.clip(qh * corsie[li], 50.0, None)
                    ss, sl = s_state(celle, m, vh, qt), s_lwr_stima(celle, m, vh, qt)
                    righe.append(dict(pannello=p, split=s, seme=seme, metodo=metodo, celle=len(celle),
                                      S_state=ss["S_state"], S_LWR=sl["S_LWR"],
                                      **{f"rmse_v_{r}": ss[r]["rmse_v"] for r in FRAZIONE},
                                      **{f"rmse_q_{r}": ss[r]["rmse_q"] for r in FRAZIONE}))
            print(f"{p} {s}: {len(celle):,} celle per seme ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(righe).to_csv(uscita, index=False)
    tab = pd.DataFrame(righe)
    pd.set_option("display.width", 200)
    media = tab.groupby(["metodo", "split"])[["S_state", "S_LWR"]].mean().unstack("split")
    print("\nMedia su corridoi e semi\n", media.round(4).to_string())
    rif = tab[tab.metodo == a.versioni[0]].set_index(["pannello", "split", "seme"])
    print(f"\nDifferenze rispetto a {a.versioni[0]} (meglio: corridoi·semi con differenza > 0)")
    for v in [m for m in tab.metodo.unique() if m != a.versioni[0]]:
        d = tab[tab.metodo == v].set_index(["pannello", "split", "seme"])[["S_state", "S_LWR"]] - rif[["S_state", "S_LWR"]]
        for s in SPLIT_VALUTATI:
            x = d.xs(s, level="split")
            print(f"{v} {s}: ΔS_state {x.S_state.mean():+.4f} (meglio {int((x.S_state > 0).sum())}/{len(x)}), "
                  f"ΔS_LWR {x.S_LWR.mean():+.4f} (meglio {int((x.S_LWR > 0).sum())}/{len(x)})")


if __name__ == "__main__":
    main()
