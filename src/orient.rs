//! Orientamento automatico e lettura del testo con l'OCR integrato (ocrs).
//!
//! Per capire il verso si confronta la sicurezza del modello di riconoscimento sulle stesse
//! righe di testo lette dritte e capovolte: il rilevamento delle parole si fa una volta sola.

use std::sync::OnceLock;

use image::RgbImage;
use ocrs::{ImageSource, OcrEngine, OcrEngineParams, OcrInput};
use rten::Model;
use rten_tensor::{NdTensor, NdTensorView};
use rten_tensor::prelude::*;

use crate::imgops::rotate;

static DETECTION_MODEL: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/text-detection.onnx"));
static RECOGNITION_MODEL: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/text-recognition.onnx"));
/// Alfabeto del modello di riconoscimento (l'etichetta 0 è il "vuoto" del CTC).
const ALPHABET: &str =
    " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~EABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
/// Lato lungo dell'immagine data all'OCR.
pub const OCR_SIZE: u32 = 1000;
/// Righe di testo lette al massimo (le più lunghe).
const MAX_LINES: usize = 8;
/// Si legge solo l'inizio di ogni riga (circa 15 caratteri): basta per il verso e per le scritte
/// di fronte e retro, e il costo del riconoscimento è proporzionale alla lunghezza letta.
const LINE_PX: usize = 512;
/// Per girare il documento servono almeno tanti caratteri letti con sicurezza, e un distacco
/// netto dal secondo verso migliore.
const MIN_SCORE: f32 = 4.0;
const MIN_MARGIN: f32 = 1.3;

struct Ocr {
    engine: OcrEngine,
    /// Seconda copia del modello di riconoscimento, per leggerne le probabilità.
    rec: Model,
    alphabet: Vec<char>,
}

fn ocr() -> Option<&'static Ocr> {
    static OCR: OnceLock<Option<Ocr>> = OnceLock::new();
    OCR.get_or_init(|| {
        let load = || -> anyhow::Result<Ocr> {
            let engine = OcrEngine::new(OcrEngineParams {
                detection_model: Some(Model::load_static_slice(DETECTION_MODEL)?),
                recognition_model: Some(Model::load_static_slice(RECOGNITION_MODEL)?),
                ..Default::default()
            })?;
            let rec = Model::load_static_slice(RECOGNITION_MODEL)?;
            Ok(Ocr { engine, rec, alphabet: ALPHABET.chars().collect() })
        };
        load().map_err(|e| eprintln!("OCR non disponibile: {e}")).ok()
    })
    .as_ref()
}

/// Carica i modelli. Va chiamata fuori dai thread di rayon prima di elaborare le foto: se il
/// caricamento partisse dentro il pool, i thread in attesa del lucchetto bloccherebbero quelli
/// che servono al caricamento stesso.
pub fn init() {
    debug_assert!(rayon::current_thread_index().is_none());
    ocr();
}

/// Testo letto e punteggio di sicurezza di un orientamento.
#[derive(Default)]
struct Reading {
    score: f32,
    text: String,
}

impl Ocr {
    /// Decodifica CTC "greedy": somma delle sicurezze dei caratteri riconosciuti e testo.
    fn decode(&self, logp: NdTensorView<f32, 2>, out: &mut Reading) {
        let mut prev = 0;
        let mut line = String::new();
        for t in 0..logp.size(0) {
            let row = logp.slice(t);
            let (mut best, mut best_lp) = (0, f32::MIN);
            for c in 0..row.size(0) {
                if row[c] > best_lp {
                    best_lp = row[c];
                    best = c;
                }
            }
            if best != 0 && best != prev {
                let p = best_lp.exp();
                if p > 0.5 {
                    out.score += p * p;
                }
                if let Some(&ch) = self.alphabet.get(best - 1) {
                    line.push(ch);
                }
            }
            prev = best;
        }
        if !line.trim().is_empty() {
            out.text.push_str(line.trim());
            out.text.push('\n');
        }
    }

    /// Righe di testo più lunghe, pronte per il riconoscimento.
    fn lines(&self, img: &RgbImage) -> anyhow::Result<Vec<NdTensor<f32, 2>>> {
        let src = ImageSource::from_bytes(img.as_raw(), img.dimensions())?;
        let input: OcrInput = self.engine.prepare_input(src)?;
        let words = self.engine.detect_words(&input)?;
        let mut lines = self.engine.find_text_lines(&input, &words);
        let width = |l: &Vec<_>| l.iter().map(|w: &rten_imageproc::RotatedRect| w.width()).sum::<f32>();
        lines.sort_by(|a, b| width(b).total_cmp(&width(a)));
        let mut out = Vec::new();
        for l in lines.iter().filter(|l| !l.is_empty()).take(MAX_LINES) {
            let li = self.engine.prepare_recognition_input(&input, l)?;
            if li.size(1) >= 8 {
                out.push(li);
            }
        }
        out.sort_by_key(|li| std::cmp::Reverse(li.size(1).min(LINE_PX)));
        Ok(out)
    }

    /// Legge l'inizio delle righe in ciascuno dei versi richiesti (false = dritte, true = capovolte
    /// di 180°). Capovolta, una riga comincia dall'altra estremità.
    fn read(&self, lines: &[NdTensor<f32, 2>], ways: &[bool]) -> anyhow::Result<Vec<Reading>> {
        let mut res: Vec<Reading> = ways.iter().map(|_| Reading::default()).collect();
        let k = ways.len();
        // righe di larghezza simile in un unico lotto (riempite di nero a destra, come fa ocrs)
        let mut rest = lines;
        let crop = |li: &NdTensor<f32, 2>| li.size(1).min(LINE_PX);
        while let Some(first) = rest.first() {
            let (h, w) = (first.size(0), crop(first));
            let n = rest.iter().take_while(|li| crop(li) * 4 >= w * 3).count();
            let (group, tail) = rest.split_at(n);
            rest = tail;
            let mut batch = NdTensor::<f32, 4>::full([k * n, 1, h, w], -0.5);
            for (i, li) in group.iter().enumerate() {
                let (lw, cw) = (li.size(1), crop(li));
                for (j, &flip) in ways.iter().enumerate() {
                    for y in 0..h {
                        for tx in 0..cw {
                            let v = if flip { li[[h - 1 - y, lw - 1 - tx]] } else { li[[y, tx]] };
                            batch[[k * i + j, 0, y, tx]] = v;
                        }
                    }
                }
            }
            let out: NdTensor<f32, 3> = self.rec.run_one(batch.into(), None)?.try_into()?;
            // uscita [sequenza, lotto, classi]
            for b in 0..k * n {
                let mut seq = out.slice((.., b, ..)).to_tensor();
                seq.make_contiguous();
                self.decode(seq.view(), &mut res[b % k]);
            }
        }
        Ok(res)
    }
}

/// Rotazione (gradi, multipli di 90 in senso orario) che rende il testo leggibile, e il testo letto.
/// `landscape` forza l'orientamento del formato; con None si provano anche i 90°.
pub fn auto_rotation(img: &RgbImage, landscape: Option<bool>) -> (u32, String) {
    let (w, h) = img.dimensions();
    let fallback = if landscape.is_some_and(|l| (w >= h) != l) { 90 } else { 0 };
    let Some(ocr) = ocr() else { return (fallback, String::new()) };
    let bases = match landscape {
        Some(_) => vec![fallback],
        None => vec![0, 90],
    };
    // si leggono le righe dritte e capovolte: vince il verso letto con più sicurezza,
    // ma solo se stacca bene gli altri (senza testo vero sono tutti rumore)
    let mut found: Vec<(Reading, u32)> = Vec::new();
    for base in bases {
        let Ok(lines) = ocr.lines(&rotate(img, base)) else { continue };
        let Ok(readings) = ocr.read(&lines, &[false, true]) else { continue };
        for (k, r) in readings.into_iter().enumerate() {
            found.push((r, (base + 180 * k as u32) % 360));
        }
    }
    found.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
    let second = found.get(1).map_or(0.0, |f| f.0.score);
    match found.into_iter().next() {
        Some((r, deg)) if r.score >= MIN_SCORE && r.score >= MIN_MARGIN * second => (deg, r.text),
        _ => (fallback, String::new()),
    }
}

/// Tipo di documento riconosciuto dalle scritte: serve ad abbinare fronte e retro giusti.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    IdCard,
    HealthCard,
    DrivingLicense,
}

const FRONT: i8 = -1;
const BACK: i8 = 1;

/// Scritte tipiche: (testo, documento se è specifica di uno solo, lato).
const HINTS: &[(&str, Option<Kind>, i8)] = &[
    ("REPUBBLICA ITALIANA", None, FRONT),
    ("CARTA DI IDENTIT", Some(Kind::IdCard), FRONT),
    ("IDENTITY CARD", Some(Kind::IdCard), FRONT),
    ("COGNOME / SURNAME", Some(Kind::IdCard), FRONT),
    ("<<", Some(Kind::IdCard), BACK),
    ("ESTREMI ATTO", Some(Kind::IdCard), BACK),
    ("COGNOME E NOME DEL", Some(Kind::IdCard), BACK),
    ("INDIRIZZO DI RESIDENZA", Some(Kind::IdCard), BACK),
    ("TESSERA SANITARIA", Some(Kind::HealthCard), FRONT),
    ("TESSERA EUROPEA", Some(Kind::HealthCard), BACK),
    ("EUROPEAN HEALTH", Some(Kind::HealthCard), BACK),
    ("PATENTE DI GUIDA", Some(Kind::DrivingLicense), FRONT),
];

const HINT_CHARS: usize = 12;

/// Solo lettere maiuscole, cifre e '<': l'OCR spesso perde spazi e punteggiatura.
fn normalize(s: &str) -> Vec<char> {
    s.to_uppercase().chars().filter(|c| c.is_ascii_alphanumeric() || *c == '<').collect()
}

/// `pat` compare in `text` con al massimo `max_err` errori (inserimenti, cancellazioni, sostituzioni)?
fn fuzzy_contains(text: &[char], pat: &[char], max_err: usize) -> bool {
    // distanza di modifica con inizio libero nel testo (algoritmo di Sellers)
    let mut col: Vec<usize> = (0..=pat.len()).collect();
    for &t in text {
        let mut prev_diag = col[0];
        col[0] = 0;
        for i in 1..=pat.len() {
            let cur = col[i];
            col[i] = (prev_diag + (pat[i - 1] != t) as usize).min(col[i - 1] + 1).min(cur + 1);
            prev_diag = cur;
        }
        if col[pat.len()] <= max_err {
            return true;
        }
    }
    false
}

/// Lato (-1 = fronte, +1 = retro, 0 = non si sa) e tipo di documento, dalle scritte lette.
pub fn guess_doc(text: &str) -> (i8, Option<Kind>) {
    let t = normalize(text);
    let found: Vec<_> = HINTS
        .iter()
        .filter(|(h, ..)| {
            let mut p = normalize(h);
            p.truncate(HINT_CHARS);
            fuzzy_contains(&t, &p, p.len() / 6)
        })
        .collect();
    let votes = |side: i8| found.iter().filter(|h| h.2 == side).count();
    let (front, back) = (votes(FRONT), votes(BACK));
    let side = (back > front) as i8 - (front > back) as i8;
    // il tipo con più scritte trovate; in caso di parità non si sa
    let mut kinds: Vec<(usize, Kind)> = Vec::new();
    for k in found.iter().filter_map(|h| h.1) {
        match kinds.iter_mut().find(|c| c.1 == k) {
            Some(c) => c.0 += 1,
            None => kinds.push((1, k)),
        }
    }
    kinds.sort_by_key(|c| std::cmp::Reverse(c.0));
    let kind = match kinds[..] {
        [(n, k), (m, _), ..] if n > m => Some(k),
        [(_, k)] => Some(k),
        _ => None,
    };
    (side, kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_tolerate_ocr_errors() {
        use Kind::*;
        assert_eq!(guess_doc("TESSERA FUROPEA DI ASSICURAZIONE MALATTIA"), (1, Some(HealthCard)));
        assert_eq!(guess_doc("REPUBBLICA ITALANA\nCARTA DIIDENTITA /IDENTITY CARD"), (-1, Some(IdCard)));
        assert_eq!(guess_doc("REPUBBLICA ITALIANA\nTESSERA SANITARIA"), (-1, Some(HealthCard)));
        assert_eq!(guess_doc("C<ITACA38543NU2<<<<<<<<"), (1, Some(IdCard)));
        assert_eq!(guess_doc("12/08/2028 LIDIA"), (0, None));
    }
}
