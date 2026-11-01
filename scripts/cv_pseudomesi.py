"""Task 1: valutazione su pseudo-mesi completi del train, con la procedura dell'invio.

La misura di cv_buchi.py guarda solo la regione delle finestre e mette la verità nei bersagli fuori
dalla regione (REVISIONE_AVVERSARIALE_v2 §2.1). Qui ogni mese del train diventa un mese valutato:
- 10 finestre del Task 2 per corridoio, come in window_index.csv degli split valutati: 5 onset, le
  prime ammesse in ordine di tempo; 5 ongoing scorrendo i giorni in ordine e, nel giorno, a caso tra
  i candidati; tutte distanti almeno 360 minuti. Nei mesi valutati cadono nei primi giorni (mediana
  3–4), come queste. D12_I405_N e D12_I405_S non hanno il Task 2 e restano senza buchi;
- storia protetta e corridoio oscurato da T+1 a T+18 (cv_buchi.oscura); i bersagli del train che
  cadono nella storia (T−12 … T−1) si tolgono, perché nei mesi valutati lì non ci sono bersagli;
- tutti i bersagli del mese ricostruiti, ognuno con il ramo che gli assegna il metodo: un ramo fuori
  dalla regione da T a T+24 delle finestre e uno dentro, come task1.ricostruisci_lgb_buchi;
- sul mese intero S_state, S_FD e S_LWR (s_fisica_stima, universo del valutatore) e la parte del
  totale che dipende da v e q: 0,35·S_state + 0,05·S_FD + 0,10·S_LWR.
Il mese è valutato con i modelli allenati sui due blocchi che non lo contengono.

Rami (modelli a 800 alberi, PARAMETRI di cv_task1):
    std     righe di caratteristiche/ (v6)
    buchi   std + righe con i buchi della copia 0 (v10, cv_buchi.py prepara)
    buchi3  buchi + copie 1–3 (cv_buchi.py copie --copie 3)
    mondi   std + le stesse celle in mondi spostati (v9, prepara_task1_mondi.py)
    mese    mondi con le caratteristiche del livello del mese (cv_mese.py prepara)
Metodi: nome=fuori/dentro, con /e1 in coda per E1 (REVISIONE_AVVERSARIALE_v2 §5): prima di
calcolare le caratteristiche lo scarto di livello del mese di ogni tratto, δ = livello notturno del
mese − livello del train, si toglie dalle velocità osservate con il peso del flusso libero di
cv_mondo.sposta, e si restituisce alle velocità previste. Il ramo `interp` è l'interpolazione.
I rami `rete0` e `rete1` sono le reti di e2_sequenze.py (λ = 0 e 1) dello stesso fold, solo dentro la
regione; `media0` e `media1` la media della rete e di buchi3 (E2 e E3, REVISIONE_AVVERSARIALE_v2 §6–7).

XGBoost accanto a LightGBM: `modelli --motore xgb` allena gli stessi rami e fold con XGBoost (PARAMETRI_XGB,
alberi a profondità fissa invece che a foglie, gli stessi 800 alberi) e salva in oof_{ramo}.parquet i residui
previsti fuori dal fold da entrambi. Nei metodi `ramo@x` è XGBoost, `ramo@NN` la media dei residui con peso
NN% a LightGBM, `media0@NN` la media della rete e di buchi3@NN.

Con --mondo SD ogni pseudo-mese è anche spostato in un mondo con δ ~ N(0; SD) per tratto (ingressi e
verità, cv_mondo.sposta), un mondo per corridoio e mese.

    uv run python scripts/cv_pseudomesi.py modelli --rami std buchi buchi3 [--motore xgb] [--thread 20]
    uv run python scripts/cv_pseudomesi.py valuta --metodi v8=std/std v10=std/buchi ... [--mondo 1.2] [--rif v10]
    uv run python scripts/cv_pseudomesi.py valuta --metodi v14=mese/media0/e1 x50=mese@50/media0@50/e1 --rif v14
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

from cv_buchi import REGIONE, STORIA, oscura
from cv_mondo import sposta
from cv_task1 import CARTELLA, META, PARAMETRI
from tfb.caratteristiche import MESE, base, calcola
from tfb.dati import ISTANTI, RILASCIO, bersagli_task1, pannelli, tensori, tratti, validita_rampe
from tfb.finestre import SPAZIATURA, candidati_ongoing, costruisci
from tfb.metrica import s_fisica_stima, s_state
from tfb.task1 import peso_libero, ricostruisci_v1, riporta, scarto_livello

warnings.filterwarnings("ignore", category=RuntimeWarning)
FRAZIONE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
CONFINI = (0, 92, 183, 273)
ALBERI = 800
USCITA = CARTELLA / "pseudomesi"
MODELLI = USCITA / "modelli"
RAMI = {
    "std": dict(fonti=["train"], mese=False),
    "buchi": dict(fonti=["train", "buchi0"], mese=False),
    "buchi3": dict(fonti=["train", "buchi0", "buchi1", "buchi2", "buchi3"], mese=False),
    "mondi": dict(fonti=["train", "mondi"], mese=False),
    "mese": dict(fonti=["train", "mondi"], mese=True),
}
PESI = dict(S_state=0.35, S_FD=0.05, S_LWR=0.10)
# configurazione standard, non regolata: la media guadagna da errori diversi da quelli di LightGBM
PARAMETRI_XGB = dict(objective="reg:squarederror", eta=0.05, max_depth=8, min_child_weight=200, subsample=0.8,
                     colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", seed=0)
PESI_LGB = (100, 75, 50, 25, 0)


def leggi(fonte: str, mese: bool) -> pd.DataFrame:
    def uno(p: str) -> pd.DataFrame:
        if fonte == "train":
            x = pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet")
            extra = CARTELLA / "caratteristiche_mese" / f"{p}.parquet"
        elif fonte == "mondi":
            x = pd.read_parquet(CARTELLA / "caratteristiche_mondi" / f"{p}.parquet").drop(columns="copia")
            extra = CARTELLA / "caratteristiche_mese" / f"{p}__mondi.parquet"
        else:
            if mese:
                raise SystemExit("le righe con i buchi non hanno le caratteristiche del mese")
            c = int(fonte[-1])
            nome = f"{p}.parquet" if c == 0 else f"{p}__copia{c}.parquet"
            x = pd.read_parquet(CARTELLA / "caratteristiche_buchi" / nome)
            return x[x.strato == "buchi"] if c == 0 else x
        return pd.concat([x, pd.read_parquet(extra)], axis=1) if mese else x
    return pd.concat([uno(p) for p in pannelli()], ignore_index=True)


def modelli(rami: list[str], thread: int, motore: str) -> None:
    MODELLI.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ramo in rami:
        spec = RAMI[ramo]
        pezzi = [leggi(f, spec["mese"]) for f in spec["fonti"]]
        colonne = [c for c in pezzi[0].columns if c not in META]
        usa = colonne + ["y_v", "y_q", "blocco"]
        fonte = np.concatenate([np.full(len(x), i, np.int8) for i, x in enumerate(pezzi)])
        A = pd.concat([x[usa] for x in pezzi], ignore_index=True)
        del pezzi
        oof = []
        for k in range(3):
            al = (A.blocco != k).to_numpy()
            for nome, y in (("v", A.y_v.to_numpy()), ("q", A.y_q.to_numpy())):
                r = y - base(A, nome)
                ok = al & np.isfinite(r)
                if motore == "lgb":
                    m = lgb.train(dict(PARAMETRI, num_threads=thread), lgb.Dataset(A.loc[ok, colonne], r[ok]), num_boost_round=ALBERI)
                    m.save_model(str(MODELLI / f"{ramo}_f{k}_{nome}.txt"))
                    continue
                import xgboost as xgb
                d = xgb.QuantileDMatrix(A.loc[ok, colonne], r[ok], nthread=thread)
                m = xgb.train(dict(PARAMETRI_XGB, nthread=thread), d, num_boost_round=ALBERI)
                del d
                m.save_model(str(MODELLI / f"{ramo}_xgb_f{k}_{nome}.ubj"))
                # residui fuori dal fold: XGBoost e il LightGBM dello stesso ramo e fold
                ho = ~al & np.isfinite(r)
                Xh = A.loc[ho, colonne]
                l_ = lgb.Booster(model_file=str(MODELLI / f"{ramo}_f{k}_{nome}.txt"))
                oof.append(pd.DataFrame(dict(nome=nome, fonte=fonte[ho], blocco=np.int8(k), r=r[ho].astype(np.float32),
                                             lgb=l_.predict(Xh, num_threads=thread).astype(np.float32),
                                             xgb=m.inplace_predict(Xh).astype(np.float32))))
            print(f"{ramo} {motore} fold {k}: {int(al.sum()):,} righe ({time.time() - t0:.0f} s)", flush=True)
        if motore == "lgb":
            (MODELLI / f"{ramo}.json").write_text(json.dumps(dict(spec, colonne=colonne, alberi=ALBERI)))
            continue
        assert json.loads((MODELLI / f"{ramo}.json").read_text())["colonne"] == colonne
        (MODELLI / f"{ramo}_xgb.json").write_text(json.dumps(dict(spec, colonne=colonne, alberi=ALBERI, parametri=PARAMETRI_XGB)))
        o = pd.concat(oof, ignore_index=True)
        o.to_parquet(USCITA / f"oof_{ramo}.parquet", index=False)
        sintesi_oof(o, spec["fonti"], ramo)


def sintesi_oof(o: pd.DataFrame, fonti: list[str], ramo: str) -> None:
    """RMSE dei residui fuori dal fold per peso di LightGBM nella media; per fold, la media 50/50 meno LightGBM."""
    print(f"\n{ramo}: RMSE fuori dal fold (v in km/h, q per corsia in veicoli/h) per peso di LightGBM")
    for (nome, f), x in o.groupby(["nome", "fonte"]):
        e = lambda p, y: float(np.sqrt(np.mean((y.r - (p / 100 * y.lgb + (1 - p / 100) * y.xgb)) ** 2)))
        tutti = "  ".join(f"{p}%: {e(p, x):.3f}" for p in PESI_LGB)
        fold = " ".join(f"{e(50, y) - e(100, y):+.3f}" for _, y in x.groupby("blocco"))
        print(f"  {nome} {fonti[f]:>7} ({len(x):,}): {tutti} | 50% − 100% per fold: {fold}")


def finestre_mese(p: str, date: np.ndarray) -> pd.DataFrame:
    """10 finestre per mese (5 onset, 5 ongoing) come negli split valutati; vuoto senza Task 2."""
    if not (RILASCIO / "task2" / p).exists():
        return pd.DataFrame(columns=["g", "t", "condition"])
    c = costruisci(p)
    onset = c[c.condition == "queue_onset"]
    cand = candidati_ongoing(p)
    mesi = pd.to_datetime(pd.Series(date)).dt.to_period("M").to_numpy()
    rng = np.random.default_rng([zlib.crc32(p.encode()), 11])
    out = []
    for m in pd.unique(mesi):
        giorni = np.flatnonzero(mesi == m)
        scelte = onset[onset.g.isin(giorni)].sort_values(["g", "t"]).head(5)
        righe = [dict(g=g, t=t, condition="queue_onset") for g, t in zip(scelte.g, scelte.t)]
        assoluti = [g * ISTANTI + t for g, t in zip(scelte.g, scelte.t)]
        n = 0
        for g in giorni:
            x = cand[cand.g == g].t.to_numpy()
            for t in x[rng.permutation(len(x))]:
                a = g * ISTANTI + t
                if all(abs(a - b) >= SPAZIATURA for b in assoluti):
                    righe.append(dict(g=g, t=int(t), condition="queue_ongoing"))
                    assoluti.append(a)
                    n += 1
                    if n == 5:
                        break
            if n == 5:
                break
        out += righe
    return pd.DataFrame(out)


def taglia(t, giorni: np.ndarray):
    """Tensori dei soli giorni indicati (contigui): i conti del Task 1 e del Task 3 stanno nel giorno."""
    s = slice(giorni[0], giorni[-1] + 1)
    campi = {f.name: getattr(t, f.name) for f in dataclasses.fields(t)}
    return dataclasses.replace(t, **{k: v[s] for k, v in campi.items()
                                     if k != "link_id" and isinstance(v, np.ndarray) and len(v) == len(t.date)})


def analizza(metodi: list[str]) -> dict:
    out = {}
    for m in metodi:
        nome, spec = m.split("=")
        parti = spec.split("/")
        out[nome] = dict(fuori=parti[0], dentro=parti[1], e1="e1" in parti[2:])
    return out


def valuta(metodi: list[str], mondo: float, rif: str, thread: int, solo: list[str] | None) -> None:
    t0 = time.time()
    spec = analizza(metodi)
    usati = {r for s in spec.values() for r in (s["fuori"], s["dentro"])}
    lam_reti = sorted({r.split("@")[0][-1] for r in usati if r.startswith(("rete", "media"))})
    usati |= {"buchi3" + r[len("media0"):] for r in usati if r.startswith("media")}
    alberi = {r for r in usati if r != "interp" and not r.startswith(("rete", "media"))}
    rami = sorted({r.split("@")[0] for r in alberi if not r.endswith("@x")})
    rami_x = sorted({r.split("@")[0] for r in alberi if "@" in r})
    reti = {}
    if lam_reti:
        import torch

        from e2_sequenze import CARTELLA as SEQ
        from tfb.rete import carica_rete, prevedi_regione
        torch.set_num_threads(thread)
        for lam in lam_reti:
            for k in range(3):
                reti[(lam, k)] = carica_rete(SEQ / f"rete_f{k}_lam{lam}.pt")
    caricati = {}
    for r in rami:
        colonne = json.loads((MODELLI / f"{r}.json").read_text())["colonne"]
        for k in range(3):
            caricati[(r, k)] = ({n: lgb.Booster(model_file=str(MODELLI / f"{r}_f{k}_{n}.txt")) for n in ("v", "q")}, colonne)
    for r in rami_x:
        import xgboost as xgb
        colonne = json.loads((MODELLI / f"{r}_xgb.json").read_text())["colonne"]
        for k in range(3):
            mod = {n: xgb.Booster(model_file=str(MODELLI / f"{r}_xgb_f{k}_{n}.ubj")) for n in ("v", "q")}
            for b_ in mod.values():
                b_.set_param({"nthread": thread})
            caricati[(f"{r}@x", k)] = (mod, colonne)
    serve_mese = any(set(MESE) & set(c) for _, c in caricati.values())
    suffisso = "" if not mondo else f"_mondo{mondo:g}"
    uscita = USCITA / f"valuta{suffisso}.csv"
    righe = []
    for p in solo or pannelli():
        m, vero = tensori(p, "train"), tensori(p, "train", "vero")
        tr = tratti(p)
        vf, corsie = tr.free_speed_kmh.to_numpy(np.float32), tr.lanes.to_numpy(np.float32)
        b = bersagli_task1(p, "train", m)
        bersaglio = np.zeros(m.speed.shape, bool); bersaglio[b.g, b.t, b.l] = True
        w = finestre_mese(p, m.date)
        mb = oscura(m, vero, w, bersaglio) if len(w) else m
        regione = np.zeros(m.speed.shape, bool)
        storia = np.zeros(m.speed.shape[:2], bool)
        for g, t in zip(w.g, w.t):
            regione[g, t + REGIONE[0]:t + REGIONE[1] + 1, :] = True
            storia[g, t - STORIA:t] = True
        # nei mesi valutati la storia delle finestre è tutta osservata e non ha bersagli
        b = b[~storia[b.g, b.t]].reset_index(drop=True)
        valida = validita_rampe(p, "train")
        mesi = pd.to_datetime(pd.Series(m.date)).dt.to_period("M").to_numpy()
        for j, mm in enumerate(pd.unique(mesi)):
            giorni = np.flatnonzero(mesi == mm)
            k = int(np.searchsorted(CONFINI, giorni[0], side="right") - 1)
            strato, verita = taglia(mb, giorni), taglia(vero, giorni)
            if mondo:
                rng = np.random.default_rng([zlib.crc32(p.encode()), j, 23])
                delta = rng.normal(0.0, mondo, len(vf)).astype(np.float32)
                rif_v = np.where(np.isfinite(verita.speed), verita.speed, np.nanmedian(verita.speed))
                strato, verita = sposta(strato, vf, delta, rif_v), sposta(verita, vf, delta, rif_v)
            bm = b[b.g.isin(giorni)].reset_index(drop=True)
            bm["g"] -= giorni[0]
            g, ti, li = bm.g.to_numpy(), bm.t.to_numpy(), bm.l.to_numpy()
            dentro = regione[giorni][g, ti, li]
            w_m = w[np.isin(w.g, giorni)].assign(g=lambda x: x.g - giorni[0]).reset_index(drop=True)
            finestra = np.full(strato.speed.shape[:2], -1)
            for j_, (gg, tt_) in enumerate(zip(w_m.g, w_m.t)):
                finestra[gg, tt_ + REGIONE[0]:tt_ + REGIONE[1] + 1] = j_
            jf = finestra[g, ti]
            assert np.array_equal(jf >= 0, dentro)
            bers_m = np.zeros(strato.speed.shape, bool); bers_m[g, ti, li] = True
            # E1 come task1.equivariante: lo strato riportato nel mondo del train
            d_hat = scarto_livello(strato)
            riportato = riporta(strato, d_hat)
            previsioni = {}

            def residui(ramo: str, e1: bool):
                """Residui previsti da un ramo ad alberi: LightGBM, XGBoost (@x) o la loro media (@NN)."""
                chiave = ("residui", ramo, e1)
                if chiave in previsioni:
                    return previsioni[chiave]
                X = previsioni[("X", e1)][0]
                if "@" in ramo and not ramo.endswith("@x"):
                    r0, p = ramo.split("@")[0], int(ramo.split("@")[1]) / 100
                    a_, b_ = residui(r0, e1), residui(f"{r0}@x", e1)
                    previsioni[chiave] = (p * a_[0] + (1 - p) * b_[0], p * a_[1] + (1 - p) * b_[1])
                else:
                    mod, colonne = caricati[(ramo, k)]
                    if ramo.endswith("@x"):
                        previsioni[chiave] = tuple(mod[n].inplace_predict(X[colonne]) for n in ("v", "q"))
                    else:
                        previsioni[chiave] = tuple(mod[n].predict(X[colonne], num_threads=thread) for n in ("v", "q"))
                return previsioni[chiave]

            def prevedi(ramo: str, e1: bool):
                chiave = (ramo, e1)
                if chiave in previsioni:
                    return previsioni[chiave]
                st = riportato if e1 else strato
                if ("X", e1) not in previsioni:
                    v1, q1 = ricostruisci_v1(st)
                    X = calcola(st, bm, FRAZIONE, mese=serve_mese)
                    bv, bq = base(X, "v"), base(X, "q")
                    previsioni[("X", e1)] = (X, np.where(np.isfinite(bv), bv, v1[g, ti, li]),
                                             np.where(np.isfinite(bq), bq, q1[g, ti, li] / corsie[li]), v1, q1)
                X, bv, bq, v1, q1 = previsioni[("X", e1)]
                if ramo.startswith("media"):  # media0, media1, media0@50: la rete e buchi3 (con lo stesso suffisso)
                    a_, b_ = prevedi(f"rete{ramo[5]}", e1), prevedi("buchi3" + ramo[6:], e1)
                    previsioni[chiave] = ((a_[0] + b_[0]) / 2, (a_[1] + b_[1]) / 2)
                    return previsioni[chiave]
                if ramo == "interp":
                    vh, qh = bv, bq
                elif ramo.startswith("rete"):  # come task1.ricostruisci_lgb_buchi con rete=True
                    vh, qh = prevedi_regione(reti[(ramo[-1], k)], st, bers_m, w_m, v1, q1, g, ti, li)
                else:
                    rv, rq = residui(ramo, e1)
                    vh, qh = bv + rv, bq + rq
                vh = np.clip(vh, 3.0, None)
                if e1:
                    vh = vh + d_hat[g, li] * peso_libero(vh, vf[li])
                previsioni[chiave] = (vh, np.clip(qh * corsie[li], 50.0, None))
                return previsioni[chiave]

            for nome, s in spec.items():
                vf_, qf_ = prevedi(s["fuori"], s["e1"])
                vd_, qd_ = prevedi(s["dentro"], s["e1"])
                vh, qh = np.where(dentro, vd_, vf_), np.where(dentro, qd_, qf_)
                ss = s_state(bm, verita, vh, qh)
                sf = s_fisica_stima(bm, verita, vh, qh, valida=valida[giorni])
                r = dict(pannello=p, mese=str(mm), blocco=k, metodo=nome, finestre=int(np.isin(w.g, giorni).sum()),
                         quota_regione=float(dentro.mean()), S_state=ss["S_state"], S_FD=sf["S_FD"], S_LWR=sf["S_LWR"],
                         d_hat_sd=float(np.std(d_hat[0])))
                r["parte"] = sum(PESI[c] * r[c] for c in PESI)
                righe.append(r)
            print(f"{p} {mm}: {len(bm):,} bersagli, {dentro.mean():.1%} nella regione ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(righe).to_csv(uscita, index=False)
    sintesi(pd.DataFrame(righe), rif)


def sintesi(t: pd.DataFrame, rif: str) -> None:
    pd.set_option("display.width", 220)
    print("\nMedie sui corridoi·mesi")
    print(t.groupby("metodo")[["S_state", "S_FD", "S_LWR", "parte"]].mean().round(5).to_string())
    x = t.pivot_table(index=["pannello", "mese"], columns="metodo", values=["S_state", "S_FD", "S_LWR", "parte"])
    rng = np.random.default_rng(0)
    print(f"\nRispetto a {rif}: differenza media, intervallo bootstrap al 90% sui corridoi·mesi, quanti meglio / peggio")
    for m in x["parte"].columns.drop(rif):
        cella = []
        for c in ("S_state", "S_FD", "S_LWR", "parte"):
            d = (x[c][m] - x[c][rif]).to_numpy()
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
            cella.append(f"{c} {d.mean():+.5f} [{np.quantile(bs, .05):+.5f}, {np.quantile(bs, .95):+.5f}] "
                         f"{int((d > 1e-9).sum())}/{int((d < -1e-9).sum())}")
        print(f"{m:>10}: " + " | ".join(cella))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["modelli", "valuta"])
    ap.add_argument("--rami", nargs="+", default=["std", "buchi", "buchi3"])
    ap.add_argument("--motore", choices=["lgb", "xgb"], default="lgb")
    ap.add_argument("--metodi", nargs="+", default=["interp=interp/interp", "v8=std/std", "v10=std/buchi", "v10c=std/buchi3"])
    ap.add_argument("--mondo", type=float, default=0.0)
    ap.add_argument("--rif", default="v10")
    ap.add_argument("--thread", type=int, default=20)
    ap.add_argument("--pannelli", nargs="+")
    a = ap.parse_args()
    if a.fase == "modelli":
        modelli(a.rami, a.thread, a.motore)
    else:
        valuta(a.metodi, a.mondo, a.rif, a.thread, a.pannelli)
