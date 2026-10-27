# DocPDF

Trasforma le foto dei documenti (carta d'identità, tessera sanitaria, patente, fogli) in **un'unica pagina PDF A4**: ogni documento viene trovato nella foto, ritagliato dallo sfondo, raddrizzato e girato nel verso giusto.

- **Rilevamento automatico** del documento, anche fotografato storto o in prospettiva; se nella stessa foto ci sono più documenti (per esempio tessera sanitaria e carta d'identità), vengono separati.
- **Orientamento automatico** con l'OCR (Tesseract): i documenti capovolti o girati di 90° vengono rimessi dritti.
- **Fronte e retro affiancati**: con due foto che contengono lo stesso numero di documenti (una dei fronti, una dei retri), ogni fronte viene messo accanto al suo retro; il fronte va per primo (lo riconosce dal testo: «Repubblica Italiana», «Carta di identità»… contro la zona `<<<<` o «Tessera europea di assicurazione malattia» del retro).
- **Proporzioni esatte**: le tessere (formato ID-1, 85,6 × 54 mm) escono con le proporzioni reali e gli angoli arrotondati puliti; si può scegliere anche passaporto, A4, A5 o formato libero.
- **Pulizia dell'immagine** (facoltativa per ogni documento): bilanciamento del bianco, contrasto e nitidezza.
- **Impaginazione**: «Adatta alla pagina» (predefinita, ingrandisce i documenti fino a riempire il foglio), dimensione reale, 150% o 200%; facoltativamente in bianco e nero.

## Uso

In Nemo seleziona le foto (oppure la cartella che le contiene), clic destro → **Unisci documenti in un PDF**. Si apre una finestra con l'anteprima della pagina:

| Comando | Tasto |
|---|---|
| Correggi il ritaglio (trascina i 4 angoli; rotella = zoom) | `E` o doppio clic sulla miniatura |
| Ruota a destra / a sinistra | `R` / `Shift+R` |
| Capovolgi | `F` |
| Duplica (per ritagliare a mano un altro documento della stessa foto) | `D` |
| Togli dalla pagina | `Canc` |
| Aggiungi foto (oppure trascinale nella finestra) | `Ctrl+O` |
| Salva il PDF | `Ctrl+S` |

L'ordine si cambia trascinando le miniature. I documenti con ritaglio incerto sono evidenziati in arancione. Il PDF viene salvato nella cartella della prima foto con il nome `<cartella>_documenti.pdf` (modificabile); se esiste già viene chiesta conferma prima di sovrascriverlo.

Si può usare anche da terminale:

```bash
docpdf foto1.jpg foto2.jpg            # apre la finestra
docpdf cartella/                      # tutte le immagini della cartella
docpdf --auto cartella/ -o uscita.pdf # senza finestra
```

## Installazione

```bash
git clone https://github.com/francescoalemanno/docpdf.git ~/docpdf
cd ~/docpdf
./install.sh
```

L'installer crea un ambiente Python dedicato in `~/.local/share/docpdf/venv` (usa i pacchetti di sistema già presenti, come PyQt6, e scarica solo quelli mancanti: OpenCV, numpy…), poi collega:

- `~/.local/bin/docpdf` → `bin/docpdf`
- l'azione di Nemo in `~/.local/share/nemo/actions` (con il collegamento `docpdf-bin` → `bin/`, perché l'azione usa `Exec=<docpdf-bin/docpdf %F>`)
- la voce **DocPDF** nel menu delle applicazioni

La cartella clonata va lasciata al suo posto, perché i collegamenti puntano ai suoi file. Se l'azione non compare subito, riavvia Nemo con `nemo -q`.

### Dipendenze di sistema

- Python ≥ 3.10
- `tesseract` con la lingua italiana, per l'orientamento automatico (su Arch: `sudo pacman -S tesseract tesseract-data-ita`; `tesseract-data-eng` è facoltativo)
- `imagemagick` (facoltativo) per aprire foto HEIC

## Aggiornamento

```bash
cd ~/docpdf
git pull
./install.sh
```

## Disinstallazione

```bash
cd ~/docpdf
./uninstall.sh
```

Rimuove i collegamenti, la voce di menu e l'ambiente Python in `~/.local/share/docpdf`; i file nella cartella clonata restano al loro posto.

## Struttura

- `docpdf/core.py` — rilevamento dei documenti (OpenCV), raddrizzamento, pulizia, orientamento con Tesseract, impaginazione e scrittura del PDF (ReportLab)
- `docpdf/gui.py` — la finestra (PyQt6) e la modalità `--auto`
- `bin/docpdf` — avvio con l'ambiente Python dell'installer
- `actions/docpdf.nemo_action` — l'azione per Nemo

## Privacy

Tutto avviene in locale: nessuna immagine viene inviata altrove. Il `.gitignore` esclude immagini e PDF, perché nel repository non finiscano mai foto di documenti reali.
