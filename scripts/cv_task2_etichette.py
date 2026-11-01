"""Task 2: i confronti tra metodi cambiano se le etichette approssimano lo stato sottostante?

La verità ufficiale del Task 2 viene dallo stato sottostante, non dalle osservazioni rumorose
(SCORING_SPEC.md): il rumore non fa scattare l'indicatore attorno alla soglia. Le etichette della
nostra CV sono osservate. Qui le si sostituisce con la velocità osservata mediata nel tempo su una
finestra centrata di 3 o 5 passi (il rumore è bianco, circa 1,2 km/h; lo stato cambia in decine di
minuti), coda = media ≤ 0,6·v_f, e si ricalcola l'IoU per finestra degli stessi metodi.

    uv run python scripts/cv_task2_etichette.py --condizione ongoing
"""
import argparse
import pickle

import numpy as np
import pandas as pd

from cv_task2 import PANNELLI, iou_finestra
from cv_task2_decoder import almeno_uno, percorso
from cv_task2_stack import CONDIZIONI, SOGLIA, USCITA, VALUTA, boot, metodi, per_fold
from tfb.dati import tensori, tratti
from tfb.finestre import ORIZZONTE


def lisciate(p: str, ampiezza: int) -> tuple[np.ndarray, np.ndarray]:
    """(D, 288, L) coda dalla media centrata della velocità osservata ed eleggibile; eleggibilità invariata."""
    v = tensori(p, "train", "vero")
    ok = v.eleggibile & np.isfinite(v.speed)
    x = np.where(ok, v.speed, 0.0)
    n = ok.astype(float)
    k = ampiezza // 2
    somma, conta = x.copy(), n.copy()
    for d in range(1, k + 1):
        somma[:, d:] += x[:, :-d]; conta[:, d:] += n[:, :-d]
        somma[:, :-d] += x[:, d:]; conta[:, :-d] += n[:, d:]
    vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
    m = somma / np.maximum(conta, 1)
    return ok & (m <= 0.6 * vf[None, None, :]), ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condizione", default="ongoing")
    ap.add_argument("--file", default=None, help="probabilità (default stack_<condizione>.parquet)")
    a = ap.parse_args()
    cond = a.condizione
    f = a.file or str(USCITA / f"stack_{cond}.parquet")
    colonne = ["insieme", "fid", "passo", "l"]
    pr = pd.read_parquet(f)
    pr = pr[pr.insieme.isin(VALUTA[cond])]
    tutti = metodi(pr)
    etich = {w: {p: lisciate(p, w) for p in PANNELLI} for w in (3, 5)}
    ris, quota = [], []
    for n in VALUTA[cond]:
        fin = pickle.loads(percorso(n, "finestre").read_bytes())
        r = (pr.insieme == n).to_numpy()
        x = pr.loc[r, colonne]
        fid = x.fid.to_numpy()
        tagli = np.flatnonzero(np.diff(fid)) + 1
        passo, l = x.passo.to_numpy().astype(int) - 1, x.l.to_numpy()
        pm = {k: v[r] for k, v in tutti.items()}
        for a0, z in zip(np.r_[0, tagli], np.r_[tagli, len(fid)]):
            f0 = fin[int(fid[a0])]
            cand = np.zeros(f0["vero"].shape, bool)
            cand[ORIZZONTE - 1 if cond == "onset" else slice(None)] = True
            fu = slice(f0["t"] + 1, f0["t"] + 1 + ORIZZONTE)
            versioni = {"osservate": f0}
            for w in (3, 5):
                coda, ele = etich[w][f0["pannello"]]
                versioni[f"lisce{w}"] = dict(f0, vero=coda[f0["g"], fu], ele=ele[f0["g"], fu])
            e = f0["ele"]
            quota.append((n, int((f0["vero"] & e).sum()), int((versioni["lisce5"]["vero"] != f0["vero"])[e].sum())))
            for nome, v in pm.items():
                P = np.zeros(f0["vero"].shape)
                P[passo[a0:z], l[a0:z]] = v[a0:z]
                s = almeno_uno((P >= SOGLIA[cond]) & cand, np.where(cand, P, -1))
                for et, ff in versioni.items():
                    ris.append((n, f0["pannello"], f0["g"], f0["t"], f0["blocco"], nome, et, iou_finestra(s, ff)))
    q = pd.DataFrame(quota, columns=["insieme", "celle_in_coda", "celle_cambiate"])
    print(f"{cond}: celle la cui etichetta cambia con la media su 5 passi, per cella in coda osservata: "
          f"{q.celle_cambiate.sum() / q.celle_in_coda.sum():.3f}")
    r = pd.DataFrame(ris, columns=["insieme", "pannello", "g", "t", "fold", "metodo", "etichette", "iou"])
    r.to_parquet(USCITA / f"etichette_{cond}.parquet", index=False)
    r["valutazione"] = np.where(r.insieme.str.startswith("casuali"), "casuali", r.insieme)
    pd.set_option("display.width", 220)
    for val in dict.fromkeys(["casuali" if cond == "ongoing" else "attuali", "attuali", "ufficiali"]):
        for et in ("osservate", "lisce3", "lisce5"):
            y = r[(r.valutazione == val) & (r.etichette == et)]
            if y.empty:
                continue
            t = per_fold(y)
            d = t.sub(t.loc["lgb"], axis=1)
            righe = []
            for m in t.index:
                riga = {"media": t.loc[m, "media"], "min_fold": t.loc[m, "min_fold"], "Δ media": d.loc[m, "media"], "Δ min_fold": d.loc[m, "min_fold"]}
                if m != "lgb" and val != "ufficiali":
                    riga["boot"], riga["boot_2.5"], riga["boot_97.5"] = boot(y, m)
                righe.append(pd.Series(riga, name=m))
            print(f"\n{cond}, {val}, etichette {et}: IoU e Δ su lgb\n" + pd.DataFrame(righe).round(4).to_string())


if __name__ == "__main__":
    main()
