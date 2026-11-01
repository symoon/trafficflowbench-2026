"""Confronta un invio rigenerato con quello di riferimento (lo stesso file caricato su Kaggle).

La premiazione chiede che il codice riproduca l'invio scelto entro l'1% del punteggio privato.
Per ogni compito si misura quanto i due file differiscono e, per il Task 1, si dà un limite superiore
allo scarto di S_state sul mese privato: per la disuguaglianza triangolare
|RMSE(nuovo, vero) − RMSE(rif, vero)| ≤ RMSE(nuovo − rif), quindi in ogni corridoio e regime
|ΔS| ≤ 0,54·d_v/25 + 0,46·d_q/600, con d_q sul flusso per corsia; sul totale pesa 0,35.

    uv run python scripts/confronta_invio.py submissions/v19.csv riferimento/v19.csv
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

from tfb.dati import tratti
from tfb.invio import chiave


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nuovo")
    ap.add_argument("riferimento")
    a = ap.parse_args()
    hn, hr = sha(a.nuovo), sha(a.riferimento)
    print(f"sha256 nuovo:       {hn}\nsha256 riferimento: {hr}")
    if hn == hr:
        print("ESITO: file identici byte per byte")
        return
    tipi = {"submission_id": np.int64, "task": str, "speed_kmh": float, "flow_vph": float, "queue_pred": np.int8, "path_flow": float}
    n, r = pd.read_csv(a.nuovo, dtype=tipi), pd.read_csv(a.riferimento, dtype=tipi)
    if len(n) != len(r) or not (n.submission_id.to_numpy() == r.submission_id.to_numpy()).all():
        raise SystemExit("ESITO: righe o ID diversi, i file non sono confrontabili")
    k = chiave()
    assert (k.submission_id.to_numpy() == n.submission_id.to_numpy()).all()

    # Task 1
    s = (n.task == "state").to_numpy()
    st = k.loc[s, ["split", "panel", "mask_regime", "link_id"]].astype(str).reset_index(drop=True)
    corsie = pd.concat([tratti(p)[["link_id"]].assign(panel=p, corsie=tratti(p).lanes.to_numpy(float)) for p in st.panel.unique()])
    st = st.merge(corsie, on=["panel", "link_id"], how="left", validate="many_to_one")
    st["dv"] = n.speed_kmh.to_numpy()[s] - r.speed_kmh.to_numpy()[s]
    st["dq"] = (n.flow_vph.to_numpy()[s] - r.flow_vph.to_numpy()[s]) / st.corsie.to_numpy()
    g = st.groupby(["split", "panel", "mask_regime"]).agg(d_v=("dv", lambda x: np.sqrt(np.mean(x ** 2))),
                                                          d_q=("dq", lambda x: np.sqrt(np.mean(x ** 2))))
    g["limite_S"] = 0.54 * g.d_v / 25 + 0.46 * g.d_q / 600
    print(f"\nTask 1: {len(st):,} celle; |Δv| massimo {np.abs(st.dv).max():.4f} km/h, |Δq| per corsia massimo {np.abs(st.dq).max():.3f} veic/h")
    print(f"        celle diverse (oltre 0,001): v {(np.abs(st.dv) > 1e-3).sum():,}, q {(np.abs(st.dq) > 1e-3).sum():,}")
    print("        RMSE della differenza per split:", g.groupby("split")[["d_v", "d_q"]].max().round(5).to_dict("index"))
    lim1 = 0.35 * g.loc["private"].groupby("panel").limite_S.mean().mean()

    # Task 2
    s = (n.task == "queue").to_numpy()
    qn, qr = n.queue_pred.to_numpy()[s], r.queue_pred.to_numpy()[s]
    w = k.loc[s, "window_id"].astype(str).to_numpy()
    diverse = qn != qr
    finestre = pd.DataFrame({"w": w, "n": qn, "r": qr}).groupby("w").apply(
        lambda x: 1.0 if not (x.n | x.r).any() else float((x.n & x.r).sum() / (x.n | x.r).sum()), include_groups=False)
    print(f"\nTask 2: {s.sum():,} celle, {diverse.sum():,} diverse in {(finestre < 1).sum()} finestre su {len(finestre)}; "
          f"IoU medio tra i due invii {finestre.mean():.5f}")
    # 1 - IoU è una distanza (Jaccard): |IoU(A,T) - IoU(B,T)| <= 1 - IoU(A,B) per qualunque verità T.
    # Aggregazione come la metrica: media per pannello, poi le due direzioni, poi i corridoi.
    fw = k.loc[s, ["window_id", "split", "panel"]].astype(str).drop_duplicates("window_id").set_index("window_id")
    fw["d"] = 1 - finestre.reindex(fw.index).to_numpy()
    fw["corridoio"] = fw.panel.str.rsplit("_", n=1).str[0]
    lim2 = (fw.groupby(["split", "corridoio", "panel"]).d.mean().groupby(["split", "corridoio"]).mean()
            .groupby("split").mean())
    print("        limite superiore dello scarto di S_queue per split (distanza di Jaccard):", lim2.round(5).to_dict())

    # Task 4
    s = (n.task == "odme").to_numpy()
    fn, fr = n.path_flow.to_numpy()[s], r.path_flow.to_numpy()[s]
    rel = np.abs(fn - fr).sum() / max(np.abs(fr).sum(), 1e-9)
    print(f"\nTask 4: {s.sum():,} percorsi, |Δ| massimo {np.abs(fn - fr).max():.4f}, scarto L1 relativo {rel:.2e}")

    print(f"\nLimite superiore dello scarto sul totale privato dovuto a S_state: {lim1:.6f}")
    print(f"Limite superiore dello scarto sul totale privato dovuto a S_queue: {0.30 * lim2['private']:.6f}")
    print(f"Somma (Task 4 {'identico' if rel == 0 else f'scarto L1 {rel:.1e}'}): {lim1 + 0.30 * lim2['private']:.6f} "
          "(soglia della premiazione: 1% del punteggio, ~0,0086)")
    print("La parte fisica (0,15) usa le stesse v e q: con scarti di questo ordine cambia dello stesso ordine.")


if __name__ == "__main__":
    main()
