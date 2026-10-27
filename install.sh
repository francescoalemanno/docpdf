#!/usr/bin/env bash
set -euo pipefail
ROOT=$(dirname -- "$(readlink -f -- "$0")")
DATA_HOME=${XDG_DATA_HOME:-"$HOME/.local/share"}
APP_DATA="$DATA_HOME/docpdf"
VENV="$APP_DATA/venv"
BIN_LINK="$HOME/.local/bin/docpdf"
ACTION_DEST="$DATA_HOME/nemo/actions"
# L'azione usa Exec=<docpdf-bin/docpdf %F>: Nemo cerca l'eseguibile relativo
# alla cartella delle azioni, quindi lì serve un collegamento a bin/.
HELPER_LINK="$ACTION_DEST/docpdf-bin"
ACTION_LINK="$ACTION_DEST/docpdf.nemo_action"
DESKTOP="$DATA_HOME/applications/docpdf.desktop"

# Vero se $1 è un link simbolico che punta dentro questa cartella.
points_into_root() {
    local link
    [[ -L $1 ]] && link=$(readlink -- "$1") && [[ $link == "$ROOT/"* ]]
}

# Vero se $1 è un file creato da una versione di DocPDF installata a mano.
legacy_docpdf_file() {
    [[ -f $1 && ! -L $1 ]] && grep -q 'docpdf' -- "$1"
}

check_target() {
    if [[ -e $1 || -L $1 ]] && ! points_into_root "$1" && ! legacy_docpdf_file "$1"; then
        printf 'Installazione interrotta: esiste già un elemento non gestito da questo installer:\n  %s\n' "$1" >&2
        exit 1
    fi
}

command -v python3 >/dev/null || { echo "Serve python3." >&2; exit 1; }
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || { echo "Serve Python 3.10 o più recente." >&2; exit 1; }
for target in "$BIN_LINK" "$HELPER_LINK" "$ACTION_LINK"; do
    check_target "$target"
done

# Ambiente Python dedicato. Usa i pacchetti di sistema già presenti
# (per esempio PyQt6) e installa solo quelli mancanti.
if [[ ! -x $VENV/bin/python ]]; then
    echo "Creazione dell'ambiente Python in $VENV…"
    python3 -m venv --system-site-packages "$VENV"
fi
echo "Installazione delle dipendenze Python…"
"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check -r "$ROOT/requirements.txt"

# File della vecchia installazione manuale, ora sostituiti da questa cartella
rm -f -- "$APP_DATA/core.py" "$APP_DATA/gui.py"
rm -rf -- "$APP_DATA/__pycache__"

chmod +x -- "$ROOT/bin/docpdf"
mkdir -p -- "$(dirname -- "$BIN_LINK")" "$ACTION_DEST" "$(dirname -- "$DESKTOP")"
rm -f -- "$BIN_LINK" "$ACTION_LINK"
ln -sfn -- "$ROOT/bin/docpdf" "$BIN_LINK"
ln -sfn -- "$ROOT/bin" "$HELPER_LINK"
ln -sfn -- "$ROOT/actions/docpdf.nemo_action" "$ACTION_LINK"

cat > "$DESKTOP" <<DESKTOP
[Desktop Entry]
Type=Application
Name=DocPDF
Comment=Foto di documenti in un'unica pagina PDF
Exec=$ROOT/bin/docpdf %F
Icon=x-office-document
Terminal=false
Categories=Office;Graphics;
MimeType=image/jpeg;image/png;image/webp;image/heic;image/tiff;
DESKTOP

if ! command -v tesseract >/dev/null; then
    echo "Attenzione: tesseract non è installato. Senza OCR i documenti non vengono girati" \
         "nel verso giusto automaticamente (su Arch: sudo pacman -S tesseract tesseract-data-ita)." >&2
elif ! tesseract --list-langs 2>/dev/null | grep -qx 'ita'; then
    echo "Consiglio: installa la lingua italiana di tesseract (su Arch: sudo pacman -S tesseract-data-ita)." >&2
fi

echo "DocPDF installato. Avvialo con 'docpdf' oppure dal menu di Nemo (clic destro sulle foto)."
echo "Se l'azione non compare in Nemo, riavvialo con: nemo -q"
