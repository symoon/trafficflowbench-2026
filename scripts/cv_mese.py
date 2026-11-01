"""Task 1: caratteristiche relative al livello del mese, con e senza randomizzazione del mondo.

Ogni mese valutato è un mondo con un suo livello di velocità in flusso libero per tratto
(circa 1,2 km/h dal train, fino a 3–5), e LightGBM, che ha visto un solo mondo, ne perde
0,0047 di S_state (cv_mondo.py). Qui si aggiunge il livello del mese stimato dallo strato
mascherato dello stesso mese (caratteristiche.caratteristiche_mese) e si confrontano quattro
modelli a 800 alberi, a blocchi, sui giorni completi:
- base o con il livello del mese;
- allenati sul solo train o anche sulle stesse celle in mondi spostati (prepara_task1_mondi.py).
Si valutano nel mondo del train e in mondi spostati come in cv_mondo.py, con gli stessi semi:
il modello base allenato sul train deve ritrovare lgb_800 di cv_mondo_train.csv.

Il controllo è in parte circolare (i mondi valutati vengono dallo stesso modello di spostamento):
quello che decide è rimaschera_valutati.py sui mesi veri.

    uv run python scripts/cv_mese.py prepara
    uv run python scripts/cv_mese.py cv [--thread 16] [--mondi 2] [--sd 1.2 2.4]
"""
import argparse
import time
import warnings
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_mondo import FRAZIONE, SD_MONDO, misura, sposta
from cv_task1 import CARTELLA, META, PARAMETRI
from tfb.caratteristiche import MESE, base, calcola, caratteristiche_mese, livello_mese
from tfb.dati import pannelli, tensori, tratti

warnings.filterwarnings("ignore", category=RuntimeWarning)
USCITA = CARTELLA / "caratteristiche_mese"
ALBERI = 800


def prepara() -> None:
    """Colonne del mese per le righe di caratteristiche/ e caratteristiche_mondi/, nello stesso ordine."""
    USCITA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for p in pannelli():
        masc, vero = tensori(p, "train"), tensori(p, "train", "vero")
        c = pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet", columns=["g", "t", "l", "v_it"])
        pd.DataFrame(caratteristiche_mese(masc, c, c.v_it.to_numpy())).to_parquet(USCITA / f"{p}.parquet", index=False)
        # mondi: stessi semi e stesso spostamento di prepara_task1_mondi.py
        mo = pd.read_parquet(CARTELLA / "caratteristiche_mondi" / f"{p}.parquet", columns=["g", "t", "l", "v_it", "blocco", "copia"])
        # le righe dei mondi sono le stesse celle, blocco per blocco: se no i file vengono da preparazioni diverse
        for copia in mo.copia.unique():
            assert np.array_equal(mo.loc[mo.copia == copia, ["g", "t", "l"]].to_numpy(), c[["g", "t", "l"]].to_numpy()), p
        vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
        rif = np.where(np.isfinite(vero.speed), vero.speed, np.nanmedian(vero.speed))
        lv0 = np.nanmedian(livello_mese(masc), axis=0)  # livello di ogni tratto nel mondo del train
        pezzi, controllo = [], []
        for copia in sorted(mo.copia.unique()):
            for k in range(3):
                rng = np.random.default_rng([copia, k, zlib.crc32(p.encode())])
                delta = rng.normal(0.0, SD_MONDO, len(vf)).astype(np.float32)
                m = sposta(masc, vf, delta, rif)
                i = np.flatnonzero(((mo.copia == copia) & (mo.blocco == k)).to_numpy())
                x = pd.DataFrame(caratteristiche_mese(m, mo.iloc[i], mo.v_it.to_numpy()[i]), index=i)
                pezzi.append(x)
                # il livello del mese deve seguire lo scarto del mondo, tratto per tratto
                per_tratto = x.assign(l=mo.l.to_numpy()[i]).groupby("l").v_lib_mese.median()
                controllo.append(np.corrcoef(per_tratto.to_numpy() - lv0[per_tratto.index], delta[per_tratto.index])[0, 1])
        pd.concat(pezzi).sort_index().to_parquet(USCITA / f"{p}__mondi.parquet", index=False)
        print(f"{p}: {len(c):,} + {len(mo):,} righe; correlazione livello del mese / scarto del mondo "
              f"{np.mean(controllo):.3f} ({time.time() - t0:.0f} s)", flush=True)


def carica() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.concat([pd.concat([pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet"),
                              pd.read_parquet(USCITA / f"{p}.parquet")], axis=1) for p in pannelli()], ignore_index=True)
    mo = pd.concat([pd.concat([pd.read_parquet(CARTELLA / "caratteristiche_mondi" / f"{p}.parquet").drop(columns="copia"),
                               pd.read_parquet(USCITA / f"{p}__mondi.parquet")], axis=1) for p in pannelli()], ignore_index=True)
    return d, mo


def cv(thread: int, n_mondi: int, sd: list[float]) -> None:
    t0 = time.time()
    d, mo = carica()
    tutte = [c for c in d.columns if c not in META]
    insiemi = {"base": [c for c in tutte if c not in MESE], "mese": tutte}
    usa = tutte + ["y_v", "y_q", "blocco"]
    allenamenti = {"train": d[usa], "entrambi": pd.concat([d[usa], mo[usa]], ignore_index=True)}
    del mo
    mondi = [0.0] + [s for s in sd for _ in range(n_mondi)]
    righe = []
    for k in range(3):
        modelli = {}
        for nome_a, A in allenamenti.items():
            al = (A.blocco != k).to_numpy()
            for nome, y in (("v", A.y_v.to_numpy()), ("q", A.y_q.to_numpy())):
                r = y - base(A, nome)
                ok = al & np.isfinite(r)
                for nome_c, col in insiemi.items():
                    modelli[(nome_a, nome_c, nome)] = lgb.train(dict(PARAMETRI, num_threads=thread),
                                                                lgb.Dataset(A.loc[ok, col], r[ok]), num_boost_round=ALBERI)
            print(f"fold {k}: modelli {nome_a} ({time.time() - t0:.0f} s)", flush=True)
        for p in pannelli():
            celle = d[(d.pannello == p) & (d.blocco == k) & d.completo].reset_index(drop=True)
            masc, vero = tensori(p, "train"), tensori(p, "train", "vero")
            vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
            rng = np.random.default_rng([k, zlib.crc32(p.encode())])  # come cv_mondo.py
            for mondo, s in enumerate(mondi):
                if mondo == 0:
                    v, X = vero, celle
                else:
                    delta = rng.normal(0.0, s, len(vf)).astype(np.float32)
                    rif = np.where(np.isfinite(vero.speed), vero.speed, np.nanmedian(vero.speed))
                    m, v = sposta(masc, vf, delta, rif), sposta(vero, vf, delta, rif)
                    X = calcola(m, celle, FRAZIONE, mese=True)
                xv, xq = base(X, "v"), base(X, "q")
                for b_ in (xv, xq):
                    b_[:] = np.where(np.isfinite(b_), b_, np.nanmedian(b_))
                metodi = {"interpolazione": (xv, xq)}
                for (nome_a, nome_c, nome), mod in modelli.items():
                    if nome == "v":
                        col = insiemi[nome_c]
                        metodi[f"{nome_a}_{nome_c}"] = (xv + mod.predict(X[col], num_threads=thread),
                                                        xq + modelli[(nome_a, nome_c, "q")].predict(X[col], num_threads=thread))
                for metodo, (vh, qh) in metodi.items():
                    ss, sl = misura(celle, v, vh, qh)
                    righe.append(dict(fold=k, pannello=p, mondo=mondo, sd=s, metodo=metodo, S_state=ss, S_LWR=sl))
            print(f"fold {k} {p} ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(righe).to_csv(CARTELLA / "cv_mese.csv", index=False)
    sintesi(pd.DataFrame(righe))


def sintesi(tab: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\nMedia su corridoi e fold, per ampiezza dello scarto (sd = 0: mondo del train)")
    print(tab.groupby(["metodo", "sd"])[["S_state", "S_LWR"]].mean().unstack("sd").round(4).to_string())
    # confronti appaiati: media sui mondi della stessa ampiezza, poi per corridoio e fold
    x = tab.groupby(["sd", "fold", "pannello", "metodo"])[["S_state", "S_LWR"]].mean().unstack("metodo")
    for rif in ("train_base", "entrambi_base"):
        print(f"\nRispetto a {rif} (meglio: corridoi·fold con differenza > 0)")
        for m in ("train_mese", "entrambi_base", "entrambi_mese"):
            if m == rif:
                continue
            for s in sorted(tab.sd.unique()):
                dd = x.xs(s, level="sd")
                ds, dl = dd["S_state"][m] - dd["S_state"][rif], dd["S_LWR"][m] - dd["S_LWR"][rif]
                print(f"{m:>14} sd {s:g}: ΔS_state {ds.mean():+.5f} (meglio {int((ds > 0).sum())}/{len(ds)}), "
                      f"ΔS_LWR {dl.mean():+.5f} (meglio {int((dl > 0).sum())}/{len(dl)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["prepara", "cv"])
    ap.add_argument("--thread", type=int, default=16)
    ap.add_argument("--mondi", type=int, default=2)
    ap.add_argument("--sd", type=float, nargs="+", default=[SD_MONDO, 2 * SD_MONDO])
    a = ap.parse_args()
    {"prepara": prepara, "cv": lambda: cv(a.thread, a.mondi, a.sd)}[a.fase]()
