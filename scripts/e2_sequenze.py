"""Task 1, E2/E3 (REVISIONE_AVVERSARIALE_v2 §6–7): la zona di ogni finestra del Task 2 ricostruita tutta insieme da una rete piccola.

Il modello dei buchi di v10 corregge cella per cella. Qui una rete convoluzionale piccola vede insieme,
su tutto il corridoio, la zona di una finestra da T−12 a T+36: storia protetta, 90 minuti oscurati,
uscita. Poi ricostruisce i bersagli da T a T+24, cioè quelli che in v10 vanno al modello dei buchi.
Ingressi per cella (canali):
- velocità relativa al livello notturno del tratto nel mese, flusso per corsia e occupazione, solo
  dove osservati (zero altrove), e la maschera di osservazione;
- la maschera dei bersagli;
- il valore di partenza (task1.ricostruisci_v1, come la base dei modelli LightGBM), in canali a parte:
  un riempimento non deve sembrare un'osservazione;
- la distanza dall'ultima e dalla prossima osservazione sullo stesso tratto, l'istante del giorno,
  i dati statici del tratto, la maschera di validità della zona.
Uscita: residui di v/25 e q per corsia/600 rispetto al valore di partenza.

E3: alla perdita sui bersagli (pesi di S_state: 0,54 sulla velocità, 0,46 sul flusso) si aggiunge
λ · Σ|e_N(t+1) − e_N(t)| / Σ|ΔN| sulle transizioni tra celle eleggibili, con e_N = L·q̂/v̂ − N vero nei
bersagli e 0 altrove, perché il valutatore mette la verità nelle celle oscurate non bersaglio. Si
penalizzano le variazioni dell'errore di densità, non le variazioni di N.

Stesse finestre e stessi fold di cv_buchi.py: si allena sulle copie 0–3 dei blocchi non valutati e si
valuta sulla copia 0 del blocco tenuto fuori. Il confronto con il modello LightGBM sui buchi (ramo
buchi3 di cv_pseudomesi.py, stesso fold) si fa sulle stesse celle, con RMSE nei buchi e residuo di
conservazione nella regione nell'universo del valutatore (cv_buchi.transizioni).

    uv run python scripts/e2_sequenze.py prepara                                # CPU
    uv run --group gpu python scripts/e2_sequenze.py cv --lam 0 1 [--epoche 30]   # GPU
    uv run --group gpu python scripts/e2_sequenze.py finale --lam 0 --dispositivo cpu   # rete dell'invio, tutte le zone
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from cv_buchi import BUCO, USCITA as BUCHI, oscura, transizioni
from tfb.caratteristiche import base
from tfb.dati import RADICE_PROGETTO, bersagli_task1, pannelli, tensori, tratti
from tfb.rete import CANALI, PRIMA, PASSI, REGIONE, costruisci_rete, ingressi, zona

warnings.filterwarnings("ignore", category=RuntimeWarning)
CARTELLA = RADICE_PROGETTO / "outputs" / "task1" / "sequenze"
COPIE = 4
CONFINI = (0, 92, 183, 273)


def prepara() -> None:
    CARTELLA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for p in pannelli():
        m, vero = tensori(p, "train"), tensori(p, "train", "vero")
        tr = tratti(p)
        corsie = tr.lanes.to_numpy(np.float32)
        Lkm = tr.length_km.to_numpy(np.float32)
        b = bersagli_task1(p, "train", m)
        bersaglio = np.zeros(m.speed.shape, bool); bersaglio[b.g, b.t, b.l] = True
        valida = (vero.eleggibile & np.isfinite(vero.speed) & np.isfinite(vero.flow)).astype(np.float32)
        Xs, Ys, B, V, meta = [], [], [], [], []
        for c in range(COPIE):
            nome = "__finestre.csv" if c == 0 else f"__finestre_copia{c}.csv"
            w = pd.read_csv(BUCHI / f"{p}{nome}")
            x, b_ = ingressi(oscura(m, vero, w, bersaglio), bersaglio, w)
            Xs += list(x); B += list(b_)
            for g, t in zip(w.g, w.t):
                Ys.append(np.stack([zona(vero.speed, g, t, np.nan), zona(vero.flow, g, t, np.nan)]).astype(np.float32))
                V.append(zona(valida, g, t))
                meta.append(dict(copia=c, g=g, t=t, blocco=int(np.searchsorted(CONFINI, g, side="right") - 1)))
        np.savez(CARTELLA / f"{p}.npz", X=np.stack(Xs), Y=np.stack(Ys), B=np.stack(B), V=np.stack(V).astype(bool),
                 corsie=corsie, Lkm=Lkm)
        pd.DataFrame(meta).to_csv(CARTELLA / f"{p}.csv", index=False)
        print(f"{p}: {len(meta)} zone, {len(corsie)} tratti ({time.time() - t0:.0f} s)", flush=True)
    (CARTELLA / "canali.json").write_text(json.dumps(CANALI))


# --- allenamento (GPU o CPU) ---

def carica_corridoio(p: str):
    z = np.load(CARTELLA / f"{p}.npz")
    return {k: z[k] for k in z.files}, pd.read_csv(CARTELLA / f"{p}.csv")


def obiettivi(D: dict):
    """Residui normalizzati e pesi dei bersagli da prevedere (canale bersaglio)."""
    corsie = D["corsie"][None, None, :]
    X, Y, B = D["X"], D["Y"], D["B"]
    da = X[:, CANALI.index("bersaglio")].astype(bool) & np.isfinite(Y[:, 0]) & np.isfinite(Y[:, 1])
    rv = np.nan_to_num((Y[:, 0] - B[:, 0]) / 25)
    rq = np.nan_to_num((Y[:, 1] - B[:, 1]) / corsie / 600)
    return np.stack([rv, rq]).transpose(1, 0, 2, 3).astype(np.float32), da


def perdita(uscita, R, peso, extra, lam: float):
    """Perdita di S_state sui bersagli, più λ volte il termine sulle variazioni dell'errore di N (E3)."""
    import torch
    m = peso
    n = m.sum().clamp(min=1)
    ls = (0.54 * ((uscita[:, 0] - R[:, 0]) ** 2 * m).sum() + 0.46 * ((uscita[:, 1] - R[:, 1]) ** 2 * m).sum()) / n
    if lam == 0:
        return ls
    Bv, Bq, Yv, Yq, V, corsie, Lkm = extra
    vh = (Bv + 25 * uscita[:, 0]).clamp(min=3.0)
    qh = ((Bq / corsie + 600 * uscita[:, 1]) * corsie).clamp(min=50.0)
    N = Yq / Yv.clamp(min=1.0) * Lkm
    eN = torch.where(m > 0, qh / vh * Lkm - N, torch.zeros_like(N))
    tr = V[:, 1:] * V[:, :-1]
    dN = (torch.abs(N[:, 1:] - N[:, :-1]) * tr).sum().clamp(min=1e-3)
    return ls + lam * (torch.abs(eN[:, 1:] - eN[:, :-1]) * tr).sum() / dN


def carica_tutti() -> dict:
    dati = {p: carica_corridoio(p) for p in pannelli()}
    for p, (D, meta) in dati.items():
        D["R"], D["peso"] = obiettivi(D)
    return dati


def allena(dati: dict, al: list, lam: float, epoche: int, dev, seme: int, rng, etichetta: str,
           larghezza: int = 64, lotto: int = 48):
    """Allena una rete sulle zone `al` (coppie corridoio, indice) con ritagli di `larghezza` tratti."""
    import torch
    t0 = time.time()
    cuda = dev.type == "cuda"
    torch.manual_seed(seme)
    rete = costruisci_rete(len(CANALI)).to(dev)
    opt = torch.optim.AdamW(rete.parameters(), lr=2e-3, weight_decay=1e-4)
    passi = epoche * (len(al) // lotto)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=passi)
    rete.train()
    for e in range(epoche):
        ordine = rng.permutation(len(al))
        tot = 0.0
        for s in range(len(al) // lotto):
            pezzi = []
            for j in ordine[s * lotto:(s + 1) * lotto]:
                p, i = al[j]
                D = dati[p][0]
                L = D["X"].shape[-1]
                a = int(rng.integers(0, max(1, L - larghezza + 1)))
                sl = slice(a, a + larghezza)
                pezzi.append((D["X"][i, :, :, sl], D["R"][i, :, :, sl], D["peso"][i, :, sl],
                              D["B"][i, :, :, sl], D["Y"][i, :, :, sl], D["V"][i, :, sl], D["corsie"][sl], D["Lkm"][sl]))
            w_ = max(x[0].shape[-1] for x in pezzi)
            def pad(x, w_=w_):
                return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, w_ - x.shape[-1])])
            X = torch.tensor(np.stack([pad(x[0]).astype(np.float32) for x in pezzi]), device=dev)
            R = torch.tensor(np.stack([pad(x[1]) for x in pezzi]), device=dev)
            P = torch.tensor(np.stack([pad(x[2].astype(np.float32)) for x in pezzi]), device=dev)
            extra = None
            if lam:
                Bv = torch.tensor(np.nan_to_num(np.stack([pad(x[3][0]) for x in pezzi]), nan=1.0), device=dev)
                Bq = torch.tensor(np.nan_to_num(np.stack([pad(x[3][1]) for x in pezzi]), nan=1.0), device=dev)
                Yv = torch.tensor(np.nan_to_num(np.stack([pad(x[4][0]) for x in pezzi]), nan=1.0), device=dev)
                Yq = torch.tensor(np.nan_to_num(np.stack([pad(x[4][1]) for x in pezzi])), device=dev)
                V = torch.tensor(np.stack([pad(x[5].astype(np.float32)) for x in pezzi]), device=dev)
                co = torch.tensor(np.stack([np.pad(x[6], (0, w_ - len(x[6])), constant_values=1) for x in pezzi]), device=dev)[:, None, :]
                lk = torch.tensor(np.stack([np.pad(x[7], (0, w_ - len(x[7]))) for x in pezzi]), device=dev)[:, None, :]
                extra = (Bv, Bq, Yv, Yq, V, co, lk)
            # sulla GPU in bfloat16, sulla CPU in precisione piena
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                u = rete(X)
            l_ = perdita(u.float(), R, P, extra, lam)
            opt.zero_grad(); l_.backward(); opt.step(); sched.step()
            tot += float(l_.detach())
        print(f"{etichetta} λ={lam:g} epoca {e}: perdita {tot / max(1, len(al) // lotto):.5f} ({time.time() - t0:.0f} s)", flush=True)
    return rete.eval()


def cv(lams: list[float], epoche: int) -> None:
    import torch

    dev = torch.device("cuda")
    dati = carica_tutti()
    rng = np.random.default_rng(0)
    righe = []
    MOD = RADICE_PROGETTO / "outputs" / "task1" / "pseudomesi" / "modelli"
    col_buchi = json.loads((MOD / "buchi3.json").read_text())["colonne"]
    for k in range(3):
        # indici di allenamento: tutte le copie dei blocchi non valutati
        al = [(p, i) for p, (D, meta) in dati.items() for i in np.flatnonzero(meta.blocco.to_numpy() != k)]
        for lam in lams:
            rete = allena(dati, al, lam, epoche, dev, k, rng, f"fold {k}")
            torch.save(rete.state_dict(), CARTELLA / f"rete_f{k}_lam{lam:g}.pt")
            # valutazione sulla copia 0 del blocco tenuto fuori, sulle stesse celle del modello LightGBM
            for p, (D, meta) in dati.items():
                te = np.flatnonzero((meta.blocco.to_numpy() == k) & (meta.copia.to_numpy() == 0))
                if not len(te):
                    continue
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    u = np.concatenate([rete(torch.tensor(D["X"][te[i:i + 16]].astype(np.float32), device=dev)).float().cpu().numpy()
                                        for i in range(0, len(te), 16)])
                righe += valuta_corridoio(p, D, meta, te, u, k, f"rete_lam{lam:g}", MOD, col_buchi)
            pd.DataFrame(righe).to_csv(CARTELLA / "cv.csv", index=False)
    sintesi(pd.DataFrame(righe))


def finale(lam: float, epoche: int, dispositivo: str, thread: int) -> None:
    """La rete dell'invio: stessa ricetta di cv, su tutte le zone dei tre blocchi."""
    import torch
    torch.set_num_threads(thread)
    dati = carica_tutti()
    al = [(p, i) for p, (D, meta) in dati.items() for i in range(len(meta))]
    rete = allena(dati, al, lam, epoche, torch.device(dispositivo), 0, np.random.default_rng(0), "finale")
    uscita = RADICE_PROGETTO / "outputs" / "task1" / f"rete_buchi_lam{lam:g}.pt"
    torch.save(rete.state_dict(), uscita)
    print(f"salvata {uscita} ({len(al)} zone)", flush=True)


_lgb_cache: dict = {}


def valuta_corridoio(p, D, meta, te, u, k, nome, MOD, col_buchi) -> list[dict]:
    """Misure nella zona delle finestre per rete, LightGBM sui buchi (buchi3) e loro media."""
    import lightgbm as lgb
    corsie, Lkm = D["corsie"], D["Lkm"]
    if (p, k) not in _lgb_cache:
        vero = tensori(p, "train", "vero")
        X = pd.read_parquet(BUCHI / f"{p}.parquet")
        X = X[(X.strato == "buchi") & (X.blocco == k)].reset_index(drop=True)
        mod = {n: lgb.Booster(model_file=str(MOD / f"buchi3_f{k}_{n}.txt")) for n in ("v", "q")}
        bv, bq = base(X, "v"), base(X, "q")
        for b_ in (bv, bq):
            b_[:] = np.where(np.isfinite(b_), b_, np.nanmedian(b_))
        X["lgb_v"] = np.clip(bv + mod["v"].predict(X[col_buchi], num_threads=8), 3.0, None)
        X["lgb_q"] = np.clip((bq + mod["q"].predict(X[col_buchi], num_threads=8)) * corsie[X.l.to_numpy()], 50.0, None)
        _lgb_cache[(p, k)] = (X[["g", "t", "l", "lgb_v", "lgb_q"]], transizioni(p, vero))
    lg, ok_tr = _lgb_cache[(p, k)]
    out = []
    per = {}
    for j, i in enumerate(te):
        g, t = int(meta.g[i]), int(meta.t[i])
        da = D["peso"][i]  # bersagli da T a T+24 con verità
        tt, ll = np.nonzero(da)
        Yv, Yq = D["Y"][i, 0][da], D["Y"][i, 1][da]
        rv = np.clip(D["B"][i, 0][da] + 25 * u[j, 0][da], 3.0, None)
        rq = np.clip((D["B"][i, 1][da] / corsie[ll] + 600 * u[j, 1][da]) * corsie[ll], 50.0, None)
        x = pd.DataFrame(dict(g=g, t=t - PRIMA + tt, l=ll)).merge(lg, on=["g", "t", "l"], how="left")
        if x.lgb_v.isna().any():
            raise ValueError(f"{p} finestra {g},{t}: celle senza previsione LightGBM")
        metodi = {nome: (rv, rq), "lgb_buchi3": (x.lgb_v.to_numpy(), x.lgb_q.to_numpy())}
        metodi[f"media_{nome}"] = ((rv + metodi["lgb_buchi3"][0]) / 2, (rq + metodi["lgb_buchi3"][1]) / 2)
        rel = tt - PRIMA
        buco = (rel >= BUCO[0]) & (rel <= BUCO[1])
        N = D["Y"][i, 1] / np.maximum(D["Y"][i, 0], 1.0) * Lkm[None, :]
        # transizioni della regione (da T−1 a T+24) nell'universo del valutatore
        a0 = t + REGIONE[0] - 1
        trz = np.zeros((PASSI - 1, len(corsie)), bool)
        for s in range(PASSI - 1):
            ta = t - PRIMA + s
            if a0 <= ta <= t + REGIONE[1] and 0 <= ta < ok_tr.shape[1]:  # come cv_buchi.misura
                trz[s] = ok_tr[g, ta]
        for mtd, (vh, qh) in metodi.items():
            Nh = N.copy()
            Nh[tt, ll] = qh / np.maximum(vh, 1.0) * Lkm[ll]
            e = np.nan_to_num(Nh - N)
            r = per.setdefault(mtd, dict(sv=0.0, sq=0.0, n=0, svb=0.0, sqb=0.0, nb=0, res=0.0))
            dv, dq = vh - Yv, (qh - Yq) / corsie[ll]
            r["sv"] += float((dv ** 2).sum()); r["sq"] += float((dq ** 2).sum()); r["n"] += len(dv)
            r["svb"] += float((dv[buco] ** 2).sum()); r["sqb"] += float((dq[buco] ** 2).sum()); r["nb"] += int(buco.sum())
            r["res"] += float(np.abs(np.diff(e, axis=0))[trz].sum())
    for mtd, r in per.items():
        out.append(dict(fold=k, pannello=p, metodo=mtd, finestre=len(te), rmse_v_buco=np.sqrt(r["svb"] / max(r["nb"], 1)),
                        rmse_q_buco=np.sqrt(r["sqb"] / max(r["nb"], 1)), rmse_v_regione=np.sqrt(r["sv"] / max(r["n"], 1)),
                        rmse_q_regione=np.sqrt(r["sq"] / max(r["n"], 1)), residuo=r["res"]))
    return out


def sintesi(t: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    t = t.drop_duplicates(["fold", "pannello", "metodo"])
    print("\nMedie su corridoi e fold")
    print(t.groupby("metodo")[["rmse_v_buco", "rmse_q_buco", "rmse_v_regione", "rmse_q_regione", "residuo"]].agg(
        {"rmse_v_buco": "mean", "rmse_q_buco": "mean", "rmse_v_regione": "mean", "rmse_q_regione": "mean", "residuo": "sum"}).round(4).to_string())
    x = t.pivot_table(index=["fold", "pannello"], columns="metodo", values=["rmse_q_buco", "rmse_v_buco", "residuo"])
    print("\nRispetto a lgb_buchi3 (positivo = meglio; meglio in quanti corridoi·fold)")
    for m in x["residuo"].columns.drop("lgb_buchi3"):
        dr = (x["residuo"]["lgb_buchi3"] - x["residuo"][m]) / x["residuo"]["lgb_buchi3"]
        dq = x["rmse_q_buco"]["lgb_buchi3"] - x["rmse_q_buco"][m]
        dv = x["rmse_v_buco"]["lgb_buchi3"] - x["rmse_v_buco"][m]
        print(f"{m:>22}: residuo {dr.mean():+.2%} ({int((dr > 0).sum())}/{len(dr)}), ΔRMSE q buco {dq.mean():+.2f} "
              f"({int((dq > 0).sum())}/{len(dq)}), ΔRMSE v buco {dv.mean():+.3f} ({int((dv > 0).sum())}/{len(dv)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["prepara", "cv", "finale"])
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--epoche", type=int, default=30)
    ap.add_argument("--dispositivo", default="cuda")
    ap.add_argument("--thread", type=int, default=16)
    a = ap.parse_args()
    if a.fase == "prepara":
        prepara()
    elif a.fase == "cv":
        cv(a.lam, a.epoche)
    else:
        finale(a.lam[0], a.epoche, a.dispositivo, a.thread)
