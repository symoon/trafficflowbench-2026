"""Raccoglie i modelli allenati da riproduci.sh nella cartella della versione, con i nomi che legge l'invio.

genera_invio.py usa outputs/invii/<versione>/modelli/ se esiste: qui la si riempie con i modelli appena
allenati in outputs/task1 e outputs/task2. v18 e v19 differiscono solo nel modello principale del Task 1
(v19 con le caratteristiche E5).

    uv run python scripts/assembla_modelli.py v19
"""
import argparse
import shutil

from tfb.dati import RADICE_PROGETTO

T1, T2 = RADICE_PROGETTO / "outputs" / "task1", RADICE_PROGETTO / "outputs" / "task2"
PRINCIPALE = {"v18": "finale_mese", "v19": "finale_e5"}
# nome nella cartella della versione: file allenato
COMUNI = {
    "task1/finale_buchi_v.txt": T1 / "finale_buchi3_v.txt",
    "task1/finale_buchi_q.txt": T1 / "finale_buchi3_q.txt",
    "task1/finale_buchi_parametri.json": T1 / "finale_buchi3_parametri.json",
    "task1/rete_buchi.pt": T1 / "rete_buchi_lam0.pt",
    "task2/finale_parametri.json": T2 / "finale_parametri.json",
    "task2/finale_queue_onset.txt": T2 / "finale_queue_onset.txt",
    "task2/finale_queue_ongoing.txt": T2 / "finale_queue_ongoing.txt",
    "task2/finale_stack_ongoing_lgb.txt": T2 / "finale_stack_ongoing_lgb.txt",
    "task2/finale_stack_ongoing_xgb.json": T2 / "finale_stack_ongoing_xgb.json",
    "task2/finale_stack_ongoing_cat.cbm": T2 / "finale_stack_ongoing_cat.cbm",
    "task2/finale_stack_ongoing_secondo.txt": T2 / "finale_stack_ongoing_secondo.txt",
    "task2/finale_stack_ongoing_parametri.json": T2 / "finale_stack_ongoing_parametri.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("versione", choices=sorted(PRINCIPALE))
    a = ap.parse_args()
    nome = PRINCIPALE[a.versione]
    file = dict(COMUNI)
    for x in ("v.txt", "q.txt", "parametri.json"):
        file[f"task1/finale_{x}"] = T1 / f"{nome}_{x}"
    mancanti = [str(s) for s in file.values() if not s.exists()]
    if mancanti:
        raise SystemExit("modelli mancanti:\n  " + "\n  ".join(mancanti))
    cartella = RADICE_PROGETTO / "outputs" / "invii" / a.versione / "modelli"
    if cartella.exists():
        shutil.rmtree(cartella)
    for dest, src in sorted(file.items()):
        (cartella / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, cartella / dest)
    print(f"{len(file)} modelli in {cartella}", flush=True)


if __name__ == "__main__":
    main()
