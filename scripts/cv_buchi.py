"""Task 1: i buchi del Task 2 riprodotti nel train, per misurarli e per allenare il modello a gestirli.

Nei mesi valutati, per ogni finestra del Task 2, gli organizzatori oscurano su tutto il corridoio
le celle eleggibili non bersaglio dall'inizio dell'orizzonte fino a un'ora dopo la sua fine
(T+5 … T+90 minuti, 18 istanti): lo dice MASK_SPEC.md e i buchi di validation si ricostruiscono
tutti così dalle finestre. Le finestre partono all'inizio delle code. Nel train questi buchi non
ci sono, quindi la validazione a blocchi non li vede, mentre il valutatore del Task 3 vi mette la
verità e confronta le nostre ricostruzioni dei bersagli che ci cadono dentro.

Qui, per ogni corridoio e blocco, si scelgono fino a FINESTRE_PER_BLOCCO finestre del train (regola
pubblicata, finestre.costruisci e ongoing_casuali, una per giorno), si oscura il corridoio nello
stesso modo e si calcolano le caratteristiche dei bersagli da T a T+24 sullo strato con i buchi
e su quello normale. Come nei mesi valutati, la storia della finestra (T-60 … T-5) è tutta
osservata e non ha bersagli: il Task 2 la ripubblica, quindi la maschera la salta. Poi, a blocchi:
- LightGBM allenato come oggi (righe normali) e allenato anche sulle righe con i buchi;
- valutati sulle finestre del blocco tenuto fuori, sullo strato con i buchi;
- S_state e la parte di S_LWR nella regione della finestra, con la verità nelle celle oscurate,
  come il valutatore.

    uv run python scripts/cv_buchi.py prepara
    uv run python scripts/cv_buchi.py cv [--thread 16]
    uv run python scripts/cv_buchi.py completi [--thread 8]
    uv run python scripts/cv_buchi.py copie --copie 3
    uv run python scripts/cv_buchi.py cv --copie 3 [--thread 16]

La fase `completi` controlla che allenare anche sulle righe con i buchi non peggiori il resto:
800 alberi, valutazione sui giorni completi della validazione a blocchi (S_state e S_LWR stimati).

Le fasi con --copie misurano se al modello sui buchi servono più esempi. La copia 0 è quella di
`prepara`, con al più FINESTRE_PER_BLOCCO finestre per blocco, e resta l'unica su cui si valuta.
Le copie 1 … N prendono un'altra finestra a caso in ogni giorno che ne ha una e servono solo ad
allenare: si confrontano i modelli allenati con la copia 0, con le copie 0–1 e con le copie 0–N.
"""
import argparse
import dataclasses
import time
import warnings
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_task1 import PARAMETRI
from tfb.caratteristiche import base, calcola
from tfb.dati import RADICE_PROGETTO, RILASCIO, bersagli_task1, pannelli, tensori, tratti, validita_rampe
from tfb.finestre import costruisci, ongoing_casuali
from tfb.metrica import parametri_fisica, s_lwr_stima, s_state

warnings.filterwarnings("ignore", category=RuntimeWarning)
CARTELLA = RADICE_PROGETTO / "outputs" / "task1"
USCITA = CARTELLA / "caratteristiche_buchi"
FRAZIONE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
CONFINI = (0, 92, 183, 273)
FINESTRE_PER_BLOCCO = 25
BUCO = (1, 18)        # istanti oscurati rispetto all'origine T, estremi compresi
STORIA = 12           # istanti prima di T ripubblicati dal Task 2: osservati, senza bersagli
REGIONE = (0, 24)     # bersagli di cui si ricalcolano le caratteristiche
ALBERI = (800, 3200)
META = ["y_v", "y_q", "g", "t", "l", "blocco", "completo", "mask_regime", "timestamp", "station_id", "link_id",
        "pannello", "finestra", "rel", "in_buco", "strato"]


def finestre(p: str, copia: int = 0, usate: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una finestra per giorno, scelta a caso tra quelle della regola e non già in `usate`.

    La copia 0 ne tiene al più FINESTRE_PER_BLOCCO per blocco; le altre una per ogni giorno che ne ha.
    """
    u = pd.concat([costruisci(p)[["g", "t"]], ongoing_casuali(p)[["g", "t"]]]).drop_duplicates()
    u = u[u.t + REGIONE[1] < 288].reset_index(drop=True)
    if usate is not None:
        u = u.merge(usate[["g", "t"]], how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    rng = np.random.default_rng([zlib.crc32(p.encode()), 7 + copia])
    u = u.iloc[rng.permutation(len(u))].drop_duplicates("g")
    u["blocco"] = np.searchsorted(CONFINI, u.g.to_numpy(), side="right") - 1
    if copia == 0:
        u = u.groupby("blocco", group_keys=False).head(FINESTRE_PER_BLOCCO)
    return u.sort_values("g").reset_index(drop=True)


def oscura(m, vero, w: pd.DataFrame, bersaglio: np.ndarray):
    """Strato come nei mesi valutati attorno alle finestre `w`.

    Storia (T-12 … T-1) scoperta con i valori veri; dall'inizio dell'orizzonte a un'ora dopo
    la sua fine, tutto il corridoio oscurato nelle celle non bersaglio.
    """
    buco, storia = np.zeros(m.speed.shape, bool), np.zeros(m.speed.shape, bool)
    for g, t in zip(w.g, w.t):
        buco[g, t + BUCO[0]:t + BUCO[1] + 1, :] = True
        storia[g, t - STORIA:t, :] = True
    buco &= m.eleggibile & ~bersaglio
    storia &= m.eleggibile
    canali = {}
    for nome in ("speed", "flow", "occ", "dens_occ"):
        x = getattr(m, nome).copy()
        x[storia] = getattr(vero, nome)[storia]
        x[buco] = np.nan
        canali[nome] = x
    return dataclasses.replace(m, **canali)


def righe(p: str, m, vero, b: pd.DataFrame, w: pd.DataFrame, normale: bool) -> pd.DataFrame:
    """Caratteristiche dei bersagli da T a T+24 delle finestre `w`, sullo strato con i buchi e, se richiesto, su quello normale."""
    bersaglio = np.zeros(m.speed.shape, bool); bersaglio[b.g, b.t, b.l] = True
    mb = oscura(m, vero, w, bersaglio)
    indice = pd.DataFrame({"g": b.g, "t": b.t, "riga": np.arange(len(b))})
    pezzi = []
    for i, (g, t) in enumerate(zip(w.g, w.t)):
        r = indice[(indice.g == g) & indice.t.between(t + REGIONE[0], t + REGIONE[1])]
        c = b.iloc[r.riga.to_numpy()].reset_index(drop=True)
        pezzi.append(c.assign(finestra=i, rel=c.t - t, in_buco=c.t.between(t + BUCO[0], t + BUCO[1])))
    celle = pd.concat(pezzi, ignore_index=True)
    celle["blocco"] = np.searchsorted(CONFINI, celle.g.to_numpy(), side="right") - 1
    corsie = tratti(p).lanes.to_numpy(np.float32)
    g, t, l = celle.g.to_numpy(), celle.t.to_numpy(), celle.l.to_numpy()
    uscite = []
    for strato, mm in (("buchi", mb), ("normale", m))[:2 if normale else 1]:
        X = calcola(mm, celle, FRAZIONE)
        X["y_v"] = vero.speed[g, t, l]
        X["y_q"] = vero.flow[g, t, l] / corsie[l]
        for col in ("g", "t", "l", "blocco", "mask_regime", "timestamp", "station_id", "link_id", "finestra", "rel", "in_buco"):
            X[col] = celle[col].to_numpy()
        X["completo"], X["pannello"], X["strato"] = False, p, strato
        uscite.append(X)
    return pd.concat(uscite, ignore_index=True)


def prepara() -> None:
    USCITA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for p in pannelli():
        m, vero = tensori(p, "train"), tensori(p, "train", "vero")
        b = bersagli_task1(p, "train", m)
        w = finestre(p)
        X = righe(p, m, vero, b, w, normale=True)
        X.to_parquet(USCITA / f"{p}.parquet", index=False)
        w.assign(pannello=p).to_csv(USCITA / f"{p}__finestre.csv", index=False)
        buchi = X[X.strato == "buchi"]
        print(f"{p}: {len(w)} finestre, {len(buchi):,} celle, di cui {int(buchi.in_buco.sum()):,} nei buchi ({time.time() - t0:.0f} s)", flush=True)


def copie(n: int) -> None:
    """Copie 1 … n dei buchi, solo strato con i buchi: righe in più per allenare."""
    t0 = time.time()
    for p in pannelli():
        m, vero = tensori(p, "train"), tensori(p, "train", "vero")
        b = bersagli_task1(p, "train", m)
        usate = pd.read_csv(USCITA / f"{p}__finestre.csv")
        for c in range(1, n + 1):
            w = finestre(p, c, usate)
            usate = pd.concat([usate, w], ignore_index=True)
            X = righe(p, m, vero, b, w, normale=False)
            X.to_parquet(USCITA / f"{p}__copia{c}.parquet", index=False)
            w.assign(pannello=p).to_csv(USCITA / f"{p}__finestre_copia{c}.csv", index=False)
            print(f"{p} copia {c}: {len(w)} finestre, {len(X):,} celle ({time.time() - t0:.0f} s)", flush=True)


def transizioni(p: str, vero) -> np.ndarray:
    """(D, 287, L) bool: transizioni nell'universo del valutatore del Task 3 (modalità A)."""
    par = parametri_fisica(p)
    topo = pd.read_csv(RILASCIO / "corridors" / p / "network" / "lwr_mainline_topology.csv", dtype=str).set_index("link_id")
    ids = list(par.link_id)
    ele = vero.eleggibile & np.isfinite(vero.speed) & np.isfinite(vero.flow)
    presenti = set(np.asarray(ids)[ele.any(axis=(0, 1))])
    oss = np.array([l_ in topo.index and all(x in presenti for c in ("incoming_link_ids", "outgoing_link_ids")
                                             for x in str(topo.loc[l_, c]).split(";") if x.strip() in topo.index)
                    for l_ in ids])
    return ele[:, 1:] & ele[:, :-1] & oss[None, None, :] & validita_rampe(p, "train")[:, :-1]


def misura(p, vero, ok_tr, w, X, vh, qh) -> dict:
    """S_state nei buchi e nella regione; residuo di conservazione nella regione delle finestre."""
    corsie = tratti(p).lanes.to_numpy(float)
    L = tratti(p).length_km.to_numpy(float)
    g, t, l = X.g.to_numpy(), X.t.to_numpy(), X.l.to_numpy()
    dv = vh - vero.speed[g, t, l]
    dq = (qh - vero.flow[g, t, l]) / corsie[l]
    ib = X.in_buco.to_numpy()
    out = dict(rmse_v_buco=float(np.sqrt(np.nanmean(dv[ib] ** 2))), rmse_q_buco=float(np.sqrt(np.nanmean(dq[ib] ** 2))),
               rmse_v_regione=float(np.sqrt(np.nanmean(dv ** 2))), rmse_q_regione=float(np.sqrt(np.nanmean(dq ** 2))),
               celle_buco=int(ib.sum()))
    N = vero.flow / np.maximum(vero.speed, 1.0) * L
    Nh = N.copy()
    Nh[g, t, l] = qh / np.maximum(vh, 1.0) * L[l]
    regione = np.zeros(ok_tr.shape, bool)
    for gg, tt in zip(w.g, w.t):
        regione[gg, max(0, tt + REGIONE[0] - 1):tt + REGIONE[1] + 1, :] = True
    m = ok_tr & regione
    out["residuo"] = float(np.abs(np.diff(Nh, axis=1) - np.diff(N, axis=1))[m].sum())
    out["dN_regione"] = float(np.abs(np.diff(N, axis=1))[m].sum())
    return out


def cv(thread: int, n_copie: int = 0) -> None:
    t0 = time.time()
    d = pd.concat([pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    colonne = [c for c in d.columns if c not in META]
    usa = colonne + ["y_v", "y_q", "blocco"]  # quanto serve per allenare
    bu = pd.concat([pd.read_parquet(USCITA / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    fin = {p: pd.read_csv(USCITA / f"{p}__finestre.csv") for p in pannelli()}
    righe_b = bu[bu.strato == "buchi"]
    allenamenti = {"std": d[usa], "buchi": pd.concat([d[usa], righe_b[usa]], ignore_index=True)}
    del d
    alberi, uscita = ALBERI, CARTELLA / "cv_buchi.csv"
    if n_copie:
        extra = [pd.concat([pd.read_parquet(USCITA / f"{p}__copia{c}.parquet", columns=usa) for p in pannelli()], ignore_index=True)
                 for c in range(1, n_copie + 1)]
        for n in sorted({1, n_copie}):
            allenamenti[f"buchi+{n}"] = pd.concat([allenamenti["buchi"], *extra[:n]], ignore_index=True)
        del extra
        alberi, uscita = ALBERI[:1], CARTELLA / "cv_buchi_copie.csv"
        print("righe di allenamento: " + ", ".join(f"{k} {len(A):,}" for k, A in allenamenti.items()), flush=True)
    tab = []
    for k in range(3):
        modelli = {}
        for nome_a, A in allenamenti.items():
            al = (A.blocco != k).to_numpy()
            for nome, y in (("v", A.y_v.to_numpy()), ("q", A.y_q.to_numpy())):
                r = y - base(A, nome)
                ok = al & np.isfinite(r)
                modelli[(nome_a, nome)] = lgb.train(dict(PARAMETRI, num_threads=thread), lgb.Dataset(A.loc[ok, colonne], r[ok]),
                                                    num_boost_round=max(alberi))
            print(f"fold {k}: modelli {nome_a} ({time.time() - t0:.0f} s)", flush=True)
        for p in pannelli():
            vero = tensori(p, "train", "vero")
            ok_tr = transizioni(p, vero)
            corsie = tratti(p).lanes.to_numpy(np.float32)
            w = fin[p][fin[p].blocco == k]
            N = vero.flow / np.maximum(vero.speed, 1.0) * tratti(p).length_km.to_numpy(np.float32)
            nel_blocco = np.zeros(ok_tr.shape[0], bool); nel_blocco[CONFINI[k]:CONFINI[k + 1]] = True
            dN_blocco = float(np.abs(np.diff(N, axis=1))[ok_tr & nel_blocco[:, None, None]].sum())
            giorni_blocco = CONFINI[k + 1] - CONFINI[k]
            for strato in ("buchi", "normale"):
                X = bu[(bu.pannello == p) & (bu.blocco == k) & (bu.strato == strato)].reset_index(drop=True)
                if X.empty:
                    continue
                xv, xq = base(X, "v"), base(X, "q")
                for b_ in (xv, xq):
                    b_[:] = np.where(np.isfinite(b_), b_, np.nanmedian(b_))
                metodi = {"interpolazione": (xv, xq)}
                for nome_a in allenamenti:
                    if strato == "normale" and nome_a != "std":
                        continue
                    for n in alberi:
                        metodi[f"lgb_{nome_a}_{n}"] = (
                            xv + modelli[(nome_a, "v")].predict(X[colonne], num_iteration=n, num_threads=thread),
                            xq + modelli[(nome_a, "q")].predict(X[colonne], num_iteration=n, num_threads=thread))
                for metodo, (vh, qh) in metodi.items():
                    vh = np.clip(vh, 3.0, None)
                    qh = np.clip(qh * corsie[X.l.to_numpy()], 50.0, None)
                    tab.append(dict(fold=k, pannello=p, strato=strato, metodo=metodo, finestre=len(w),
                                    dN_blocco=dN_blocco, giorni_blocco=giorni_blocco,
                                    **misura(p, vero, ok_tr, w, X, vh, qh)))
            print(f"fold {k} {p} ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(tab).to_csv(uscita, index=False)
    t = pd.DataFrame(tab)
    pd.set_option("display.width", 220)
    s = t.groupby(["strato", "metodo"]).agg(rmse_v_buco=("rmse_v_buco", "mean"), rmse_q_buco=("rmse_q_buco", "mean"),
                                           rmse_v_regione=("rmse_v_regione", "mean"), rmse_q_regione=("rmse_q_regione", "mean"),
                                           residuo=("residuo", "sum"), dN=("dN_regione", "sum"))
    s["E_regione"] = s.residuo / s.dN
    print("\nMedie su corridoi e fold (E_regione = residuo di conservazione / Σ|ΔN| nella regione delle finestre)")
    print(s.round(4).to_string())
    if n_copie:
        confronta(t[t.strato == "buchi"], "lgb_buchi_800")


def confronta(t: pd.DataFrame, rif: str) -> None:
    """Confronti appaiati per corridoio e fold con il metodo `rif`.

    S_LWR del mese: il residuo nella regione di 10 finestre sul Σ|ΔN| di un mese di 30 giorni; tra due
    metodi conta la differenza, perché fuori dalla regione le previsioni sono le stesse.
    """
    t = t.assign(perdita_mese=t.residuo / t.finestre * 10 / (t.dN_blocco / t.giorni_blocco * 30))
    x = t.pivot_table(index=["fold", "pannello"], columns="metodo", values=["perdita_mese", "rmse_q_buco", "rmse_v_buco"])
    print(f"\nRispetto a {rif} (positivo = meglio; meglio in quanti corridoi·fold)")
    for m in x["perdita_mese"].columns.drop(rif):
        dl = x["perdita_mese"][rif] - x["perdita_mese"][m]
        dq = x["rmse_q_buco"][rif] - x["rmse_q_buco"][m]
        dv = x["rmse_v_buco"][rif] - x["rmse_v_buco"][m]
        print(f"{m:>22}: ΔS_LWR mese {dl.mean():+.4f} (min {dl.min():+.4f}, {int((dl > 0).sum())}/{len(dl)}), "
              f"ΔRMSE q buco {dq.mean():+.2f} ({int((dq > 0).sum())}/{len(dq)}), ΔRMSE v buco {dv.mean():+.3f} ({int((dv > 0).sum())}/{len(dv)})")



def completi(thread: int) -> None:
    t0 = time.time()
    d = pd.concat([pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    meta_std = [c for c in d.columns if c in META]
    colonne = [c for c in d.columns if c not in META]
    bu = pd.concat([pd.read_parquet(USCITA / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    righe_b = bu[bu.strato == "buchi"]
    allenamenti = {"std": d[meta_std + colonne], "buchi": pd.concat([d[meta_std + colonne], righe_b[meta_std + colonne]], ignore_index=True)}
    tab = []
    for k in range(3):
        modelli = {}
        for nome_a, A in allenamenti.items():
            al = (A.blocco != k).to_numpy()
            for nome, y in (("v", A.y_v.to_numpy()), ("q", A.y_q.to_numpy())):
                r = y - base(A, nome)
                ok = al & np.isfinite(r)
                modelli[(nome_a, nome)] = lgb.train(dict(PARAMETRI, num_threads=thread), lgb.Dataset(A.loc[ok, colonne], r[ok]),
                                                    num_boost_round=ALBERI[0])
        print(f"fold {k}: modelli ({time.time() - t0:.0f} s)", flush=True)
        for p in pannelli():
            vero = tensori(p, "train", "vero")
            corsie = tratti(p).lanes.to_numpy(np.float32)
            X = d[(d.pannello == p) & (d.blocco == k) & d.completo].reset_index(drop=True)
            xv, xq = base(X, "v"), base(X, "q")
            for b_ in (xv, xq):
                b_[:] = np.where(np.isfinite(b_), b_, np.nanmedian(b_))
            for nome_a in allenamenti:
                vh = np.clip(xv + modelli[(nome_a, "v")].predict(X[colonne], num_threads=thread), 3.0, None)
                qh = np.clip((xq + modelli[(nome_a, "q")].predict(X[colonne], num_threads=thread)) * corsie[X.l.to_numpy()], 50.0, None)
                tab.append(dict(fold=k, pannello=p, metodo=f"lgb_{nome_a}_800", S_state=s_state(X, vero, vh, qh)["S_state"],
                                S_LWR=s_lwr_stima(X, vero, vh, qh)["S_LWR"]))
        pd.DataFrame(tab).to_csv(CARTELLA / "cv_buchi_completi.csv", index=False)
    t = pd.DataFrame(tab)
    print("\nGiorni completi senza buchi, medie su corridoi e fold")
    print(t.groupby("metodo")[["S_state", "S_LWR"]].mean().round(4).to_string())
    x = t.pivot_table(index=["fold", "pannello"], columns="metodo", values=["S_state", "S_LWR"])
    for c in ("S_state", "S_LWR"):
        dd = x[c]["lgb_buchi_800"] - x[c]["lgb_std_800"]
        print(f"Δ{c} buchi − std: {dd.mean():+.5f} (meglio {int((dd > 0).sum())}/{len(dd)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["prepara", "cv", "completi", "copie"])
    ap.add_argument("--thread", type=int, default=16)
    ap.add_argument("--copie", type=int, default=0)
    a = ap.parse_args()
    {"prepara": prepara, "cv": lambda: cv(a.thread, a.copie), "completi": lambda: completi(a.thread),
     "copie": lambda: copie(a.copie)}[a.fase]()
