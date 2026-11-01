#!/usr/bin/env bash
# Single entry point: rebuilds a final submission from the released data (data/kaggle_public),
# training included, and writes submissions/<version>.csv and .zip.
#
#   scripts/riproduci.sh [v19|v18]                       # full run: features, training, inference (default v19)
#   scripts/riproduci.sh v19 --frozen-models DIR         # inference only, with the models shipped in DIR
#
# v19 is the selected final submission (public LB 0.86106); v18 (0.85998) is the second selected one and
# differs only in the main Task 1 model (no E5 features). Steps already completed are skipped (marker files in
# outputs/riproduzione/); delete that folder to start over. Each step logs wall time and peak memory to
# outputs/riproduzione/<step>.log. CPU only: no GPU is needed.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=${1:-v19}
case "$VERSION" in v18|v19) ;; *) echo "version must be v18 or v19" >&2; exit 2 ;; esac
FROZEN=""
if [ "${2:-}" = "--frozen-models" ]; then FROZEN=${3:?missing folder after --frozen-models}; fi

LOG=outputs/riproduzione
mkdir -p "$LOG"
if [ ! -d data/kaggle_public/corridors ]; then
    echo "data/kaggle_public not found: run scripts/scarica_dati.sh first (Kaggle API credentials needed)" >&2; exit 1
fi

# Python 3.12 and pinned dependencies (uv.lock); torch for the E2 network, xgboost and catboost for the Task 2 stack
uv sync --frozen --group gpu --group xgb --group catboost
PY=.venv/bin/python

step() {  # step <name> <command...>: runs once, logs time and peak memory
    local name=$1; shift
    if [ -f "$LOG/$name.done" ]; then echo "[skip] $name"; return; fi
    echo "[$(date '+%F %T')] $name: $*"
    /usr/bin/time -v -o "$LOG/$name.time" "$@" 2>&1 | tee "$LOG/$name.log"
    grep -E "Elapsed \(wall clock\)|Maximum resident" "$LOG/$name.time" | sed 's/^\s*/    /'
    touch "$LOG/$name.done"
}

if [ -n "$FROZEN" ]; then
    rm -rf "outputs/invii/$VERSION/modelli"
    mkdir -p "outputs/invii/$VERSION"
    cp -r "$FROZEN" "outputs/invii/$VERSION/modelli"
else
    # Task 2 training windows (published selection rule applied to the train month)
    step finestre_train          $PY scripts/finestre_train.py
    # Task 1 features: train cells, the same cells in shifted worlds, month level, low-rank month completion (E5)
    step prepara_task1           $PY scripts/prepara_task1.py
    step prepara_task1_mondi     $PY scripts/prepara_task1_mondi.py --copie 1
    step mese_prepara            $PY scripts/cv_mese.py prepara
    # Task 1 model for the cells around the Task 2 windows: the organizers' Task 2 holes reproduced in train
    step buchi_prepara           $PY scripts/cv_buchi.py prepara
    step buchi_copie             $PY scripts/cv_buchi.py copie --copie 3
    step finale_buchi            $PY scripts/allena_task1_finale.py --alberi 800 --thread 16 --buchi --copie 3 --nome finale_buchi3
    # E2: small convolutional network over the whole window zone (CPU)
    step e2_prepara              $PY scripts/e2_sequenze.py prepara
    step e2_finale               $PY scripts/e2_sequenze.py finale --lam 0 --epoche 30 --dispositivo cpu --thread 16
    # main Task 1 model
    if [ "$VERSION" = v19 ]; then
        step e5_prepara          $PY scripts/e5_basso_rango.py prepara --rango-v 4 --rango-q 2
        step e5_prepara_mondi    $PY scripts/e5_basso_rango.py prepara_mondi
        step finale_e5           $PY scripts/allena_task1_finale.py --alberi 800 --thread 16 --mondi --mese --e5 --nome finale_e5
    else
        step finale_mese         $PY scripts/allena_task1_finale.py --alberi 800 --thread 16 --mondi --mese --nome finale_mese
    fi
    # Task 2: LightGBM onset and ongoing models, then the three-engine stack for ongoing
    step task2_finale            $PY scripts/allena_task2_finale.py --thread 16 --semi 6
    step task2_stack_ongoing     $PY scripts/allena_task2_stack.py --condizione ongoing --semi 6 --thread 15
    $PY scripts/assembla_modelli.py "$VERSION"
fi

# inference on validation and private, submission file, manifest with hashes
rm -f "$LOG/genera_$VERSION.done"
step "genera_$VERSION"          $PY scripts/genera_invio.py "$VERSION"
echo "Submission: submissions/$VERSION.csv (zip: submissions/$VERSION.zip)"
sha256sum "submissions/$VERSION.csv"
if [ -f "riferimento/$VERSION.sha256" ]; then
    echo "Reference (file uploaded to Kaggle): $(cat "riferimento/$VERSION.sha256")"
fi
