#!/usr/bin/env bash
# Costruisce il pacchetto finale per gli organizzatori in outputs/pacchetto/:
# - TrafficFlowBench_SimoneZappatini/: codice del commit corrente (git archive), README in inglese come README.md,
#   diario italiano in docs/, report PDF in report/ (è l'albero del repository GitHub);
# - frozen_models.zip: modelli congelati di v18 e v19 con i manifesti (allegato della release);
# - TrafficFlowBench_SimoneZappatini.zip: le due cose insieme (allegato della release).
#
#   scripts/pacchetto.sh
set -euo pipefail
cd "$(dirname "$0")/.."
NOME=TrafficFlowBench_SimoneZappatini
OUT=outputs/pacchetto
if grep -q "{{" README_EN.md; then echo "README_EN.md ha ancora segnaposto" >&2; exit 1; fi
test -f docs/report/report.pdf || { echo "manca docs/report/report.pdf" >&2; exit 1; }
git diff --quiet HEAD -- . || { echo "modifiche non committate: il pacchetto viene da HEAD" >&2; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT/$NOME"

git archive HEAD | tar -x -C "$OUT/$NOME"
(
    cd "$OUT/$NOME"
    mv README.md docs/lab_notebook_it.md
    mv README_EN.md README.md
    mkdir -p report
    mv docs/report/report.pdf "report/${NOME}_report.pdf"
    mv docs/report/report.html docs/report/figure.py docs/report/fig_*.svg report/
    rmdir docs/report
    rm -rf consegna   # strumenti di consegna, non fanno parte del metodo
    # copie delle pagine Kaggle e dei post del forum: testi di altri, non si ripubblicano
    rm -rf docs/kaggle docs/forum
)
mkdir -p "$OUT/modelli/frozen_models"
for v in v18 v19; do
    mkdir -p "$OUT/modelli/frozen_models/$v"
    cp -r "outputs/invii/$v/modelli/." "$OUT/modelli/frozen_models/$v/"
    cp "outputs/invii/$v/manifesto.json" "$OUT/modelli/frozen_models/$v/manifest.json"
done
(cd "$OUT/modelli" && zip -qr ../frozen_models.zip frozen_models)
cp -r "$OUT/modelli/frozen_models" "$OUT/$NOME/"
(cd "$OUT" && zip -qr "$NOME.zip" "$NOME")
rm -rf "$OUT/$NOME/frozen_models" "$OUT/modelli"
ls -la "$OUT"
sha256sum "$OUT"/*.zip
