#!/usr/bin/env bash
# Compila DocPDF e lo installa per l'utente corrente.
set -euo pipefail
ROOT=$(dirname -- "$(readlink -f -- "$0")")
DATA_HOME=${XDG_DATA_HOME:-"$HOME/.local/share"}
BIN="$HOME/.local/bin/docpdf"
ACTIONS="$DATA_HOME/nemo/actions"
ACTION="$ACTIONS/docpdf.nemo_action"
DESKTOP="$DATA_HOME/applications/docpdf.desktop"

# Vero se $1 è stato creato da DocPDF (questa versione o quella in Python).
ours() {
    [[ -L $1 ]] && [[ $(readlink -- "$1") == *docpdf* ]] && return 0
    [[ -f $1 ]] && grep -aq 'DocPDF\|docpdf' -- "$1"
}

for target in "$BIN" "$ACTION" "$DESKTOP"; do
    if [[ -e $target || -L $target ]] && ! ours "$target"; then
        printf 'Installazione interrotta: esiste già un elemento non creato da DocPDF:\n  %s\n' "$target" >&2
        exit 1
    fi
done

command -v cargo >/dev/null || {
    echo "Serve Rust per compilare (su Arch: sudo pacman -S rust, altrimenti https://rustup.rs)." >&2
    exit 1
}
echo "Compilazione (la prima volta ci vuole qualche minuto)…"
cargo build --release --locked --manifest-path "$ROOT/Cargo.toml"

# Residui della versione in Python: ambiente virtuale e collegamenti
[[ -L $ACTIONS/docpdf-bin ]] && rm -- "$ACTIONS/docpdf-bin"
# (nella stessa cartella la finestra salva le sue impostazioni: si toglie solo venv)
if [[ -d $DATA_HOME/docpdf/venv ]]; then
    rm -rf -- "$DATA_HOME/docpdf/venv"
    echo "Rimosso il vecchio ambiente Python in $DATA_HOME/docpdf/venv"
fi

rm -f -- "$BIN" "$ACTION"
install -Dm755 -- "$ROOT/target/release/docpdf" "$BIN"
mkdir -p -- "$ACTIONS" "$(dirname -- "$DESKTOP")"

cat > "$ACTION" <<ACTION
[Nemo Action]
Name=Unisci documenti in un PDF
Comment=Raddrizza e ritaglia le foto dei documenti e le mette in un'unica pagina PDF (DocPDF)
Exec=$BIN %F
Icon-Name=x-office-document
Selection=notnone
Extensions=jpg;jpeg;png;webp;heic;heif;tif;tiff;bmp;dir;
ACTION

cat > "$DESKTOP" <<DESKTOP
[Desktop Entry]
Type=Application
Name=DocPDF
Comment=Foto di documenti in un'unica pagina PDF
Exec=$BIN %F
Icon=x-office-document
Terminal=false
Categories=Office;Graphics;
MimeType=image/jpeg;image/png;image/webp;image/heic;image/tiff;
StartupWMClass=docpdf
DESKTOP

echo "DocPDF installato in $BIN. Avvialo con 'docpdf' oppure dal menu di Nemo (clic destro sulle foto)."
echo "Se l'azione non compare in Nemo, riavvialo con: nemo -q"
