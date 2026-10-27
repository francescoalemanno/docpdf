#!/usr/bin/env bash
# Scarica i modelli dell'OCR (progetto ocrs-models) se mancano. Vengono inclusi nell'eseguibile
# al momento della compilazione, ma non stanno nel repository: la loro licenza è quella del
# progetto d'origine (https://github.com/robertknight/ocrs-models).
set -euo pipefail
DIR=$(dirname -- "$(readlink -f -- "$0")")
URL=https://ocrs-models.s3-accelerate.amazonaws.com
declare -A SHA256=(
    [text-detection.onnx]=a917b23dbd9524b465df7e922641b3ff2981623df4ded5a0234004ef2fee7cfe
    [text-recognition.onnx]=86c145c2edb96c8caed5b1ebb8f44d706408922211451c309c625157dd6061c5
)
for name in "${!SHA256[@]}"; do
    file="$DIR/$name"
    if [[ -f $file ]] && echo "${SHA256[$name]}  $file" | sha256sum -c --status; then
        continue
    fi
    echo "Scaricamento del modello OCR $name…"
    curl -fsSL -o "$file.part" "$URL/$name"
    if ! echo "${SHA256[$name]}  $file.part" | sha256sum -c --status; then
        rm -f -- "$file.part"
        echo "Il file scaricato $name non corrisponde a quello atteso." >&2
        exit 1
    fi
    mv -- "$file.part" "$file"
done
