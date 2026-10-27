#!/usr/bin/env bash
set -euo pipefail
ROOT=$(dirname -- "$(readlink -f -- "$0")")
DATA_HOME=${XDG_DATA_HOME:-"$HOME/.local/share"}
APP_DATA="$DATA_HOME/docpdf"
DESKTOP="$DATA_HOME/applications/docpdf.desktop"

# Vero se $1 è un link simbolico che punta dentro questa cartella.
points_into_root() {
    local link
    [[ -L $1 ]] && link=$(readlink -- "$1") && [[ $link == "$ROOT/"* ]]
}

for target in "$HOME/.local/bin/docpdf" "$DATA_HOME/nemo/actions/docpdf-bin" \
              "$DATA_HOME/nemo/actions/docpdf.nemo_action"; do
    if points_into_root "$target"; then
        rm -- "$target"
        printf 'Rimosso: %s\n' "$target"
    elif [[ -e $target || -L $target ]]; then
        printf 'Non rimosso (non è un link installato da questo installer): %s\n' "$target"
    fi
done
if [[ -f $DESKTOP ]] && grep -q -- "$ROOT/bin/docpdf" "$DESKTOP"; then
    rm -- "$DESKTOP"
    printf 'Rimosso: %s\n' "$DESKTOP"
fi
if [[ -d $APP_DATA ]]; then
    rm -rf -- "$APP_DATA"
    printf 'Rimosso l'"'"'ambiente Python: %s\n' "$APP_DATA"
fi
printf 'Disinstallazione completata. I file in %s sono rimasti intatti.\n' "$ROOT"
