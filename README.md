# DocPDF

Trasforma le foto dei documenti (carta d'identità, tessera sanitaria, patente, fogli) in **un'unica pagina PDF A4**: ogni documento viene trovato nella foto, ritagliato dallo sfondo, raddrizzato e girato nel verso giusto.

- **Rilevamento automatico** del documento, anche fotografato storto o in prospettiva; se nella stessa foto ci sono più documenti (per esempio tessera sanitaria e carta d'identità), vengono separati.
- **Orientamento automatico** con l'OCR integrato: i documenti capovolti o girati di 90° vengono rimessi dritti; le foto senza testo restano come sono.
- **Fronte e retro affiancati**: con due foto che contengono lo stesso numero di documenti (una dei fronti, una dei retri), ogni fronte viene messo accanto al suo retro; il fronte va per primo (lo riconosce dal testo: «Repubblica Italiana», «Carta di identità»… contro la zona `<<<<` o «Tessera europea di assicurazione malattia» del retro).
- **Proporzioni esatte**: le tessere (formato ID-1, 85,6 × 54 mm) escono con le proporzioni reali e gli angoli arrotondati puliti; si può scegliere anche passaporto, A4, A5 o formato libero.
- **Pulizia dell'immagine** (facoltativa per ogni documento): bilanciamento del bianco, contrasto e nitidezza.
- **Impaginazione**: «Adatta alla pagina» (predefinita, ingrandisce i documenti fino a riempire il foglio), dimensione reale, 150% o 200%; facoltativamente in bianco e nero.

## Uso

In Nemo seleziona le foto (oppure la cartella che le contiene), clic destro → **Unisci documenti in un PDF**. Si apre una finestra con l'anteprima della pagina:

| Comando | Tasto |
|---|---|
| Correggi il ritaglio (trascina i 4 angoli; rotella = zoom; `Invio` = OK, `Esc` = annulla) | `E` o doppio clic sulla miniatura |
| Ruota a destra / a sinistra | `R` / `Shift+R` |
| Capovolgi | `F` |
| Duplica (per ritagliare a mano un altro documento della stessa foto) | `D` |
| Togli dalla pagina | `Canc` |
| Documento precedente / successivo | frecce |
| Aggiungi foto (oppure trascinale nella finestra) | `Ctrl+O` |
| Salva il PDF | `Ctrl+S` |

L'ordine si cambia trascinando le miniature. I documenti con ritaglio incerto sono evidenziati in arancione. Il PDF viene salvato nella cartella della prima foto con il nome `<cartella>_documenti.pdf` (modificabile); se esiste già viene chiesta conferma prima di sovrascriverlo.

Si può usare anche da terminale:

```bash
docpdf foto1.jpg foto2.jpg            # apre la finestra
docpdf cartella/                      # tutte le immagini della cartella
docpdf --auto cartella/ -o uscita.pdf # senza finestra (-v: formato, rotazione e fronte/retro trovati)
```

## Installazione

```bash
git clone https://github.com/francescoalemanno/docpdf.git ~/docpdf
cd ~/docpdf
./install.sh
```

L'installer compila il programma (serve Rust: su Arch `sudo pacman -S rust`) e installa:

- l'eseguibile `~/.local/bin/docpdf`
- l'azione di Nemo `~/.local/share/nemo/actions/docpdf.nemo_action`
- la voce **DocPDF** nel menu delle applicazioni

Se trova la vecchia versione in Python, ne rimuove l'ambiente virtuale. Se l'azione non compare subito, riavvia Nemo con `nemo -q`.

Il risultato è **un unico file eseguibile** (circa 30 MB, con dentro i modelli dell'OCR): non servono Python, Tesseract né altre librerie oltre a quelle di base del sistema. Si può anche copiare `target/release/docpdf` su un altro PC Linux e usarlo così com'è. Unica dipendenza facoltativa: `imagemagick`, per aprire le foto HEIC.

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

Rimuove l'eseguibile, l'azione di Nemo, la voce di menu e le impostazioni della finestra (`~/.local/share/docpdf`); i file nella cartella clonata restano al loro posto.

## Struttura

Scritto in Rust; librerie principali: [egui/eframe](https://github.com/emilk/egui) per la finestra, [image](https://github.com/image-rs/image) e [imageproc](https://github.com/image-rs/imageproc) per le immagini, [ocrs](https://github.com/robertknight/ocrs) (con [rten](https://github.com/robertknight/rten)) per l'OCR, [pdf-writer](https://github.com/typst/pdf-writer) per il PDF.

- `src/detect.rs` — rilevamento dei documenti: bordi (Canny) e soglie (Otsu) su luminosità e colore, contorni, quadrilateri; ogni lato viene poi agganciato al bordo vero più vicino e i candidati sono valutati per nettezza dei bordi, forma e formato
- `src/geom.rs` — geometria dei quadrilateri
- `src/imgops.rs` — caricamento (con orientamento EXIF), raddrizzamento prospettico, pulizia (bianco, livelli, CLAHE, nitidezza), angoli arrotondati
- `src/orient.rs` — orientamento con l'OCR: legge l'inizio delle righe di testo più lunghe dritte e capovolte e sceglie il verso letto con più sicurezza; riconosce fronte e retro dalle scritte
- `src/item.rs` — un documento trovato in una foto
- `src/layout.rs` — abbinamento fronte/retro e impaginazione sull'A4
- `src/pdf.rs` — scrittura del PDF (immagini JPEG a 300 dpi)
- `src/gui.rs` — la finestra
- `models/` — i modelli dell'OCR di [Ocrs](https://github.com/robertknight/ocrs) (Robert Knight), inclusi nell'eseguibile al momento della compilazione

## Licenza

Il codice di DocPDF è sotto licenza MIT (`LICENSE`). I modelli dell'OCR in `models/` sono di Robert Knight e sono distribuiti con licenza [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), senza modifiche: origine e testo della licenza in [`models/README.md`](models/README.md) e [`models/LICENSE`](models/LICENSE). Anche l'eseguibile compilato li contiene, quindi se lo distribuisci includi queste informazioni.

## Privacy

Tutto avviene in locale: nessuna immagine viene inviata altrove. Il `.gitignore` esclude immagini e PDF, perché nel repository non finiscano mai foto di documenti reali.
