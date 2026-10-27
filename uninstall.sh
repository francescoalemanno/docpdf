#!/usr/bin/env bash
# Rimuove DocPDF installato da install.sh (il codice in questa cartella resta).
set -euo pipefail
ROOT=$(dirname -- "$(readlink -f -- "$0")")
DATA_HOME=${XDG_DATA_HOME:-"$HOME/.local/share"}

# Vero se $1 è stato creato da DocPDF (questa versione o quella in Python).
ours() {
    [[ -L $1 ]] && [[ $(readlink -- "$1") == *docpdf* ]] && return 0
    [[ -f $1 ]] && grep -aq 'DocPDF\|docpdf' -- "$1"
}

for target in "$HOME/.local/bin/docpdf" "$DATA_HOME/nemo/actions/docpdf.nemo_action" \
              "$DATA_HOME/nemo/actions/docpdf-bin" "$DATA_HOME/applications/docpdf.desktop"; do
    if ours "$target"; then
        rm -- "$target"
        printf 'Rimosso: %s\n' "$target"
    elif [[ -e $target || -L $target ]]; then
        printf 'Non rimosso (non è stato creato da DocPDF): %s\n' "$target"
    fi
done
# impostazioni della finestra (ed eventuale vecchio ambiente Python)
for dir in "$DATA_HOME/docpdf"; do
    if [[ -d $dir ]]; then
        rm -rf -- "$dir"
        printf 'Rimosso: %s\n' "$dir"
    fi
done
printf 'Disinstallazione completata. I file in %s sono rimasti intatti.\n' "$ROOT"
