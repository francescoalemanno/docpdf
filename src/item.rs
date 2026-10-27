//! Un documento trovato in una foto, con ritaglio, formato, rotazione e pulizia.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use image::RgbImage;
use rayon::prelude::*;

use crate::detect::detect_documents;
use crate::geom::{Quad, order_corners, quad_sides};
use crate::imgops::{self, Fmt, guess_format};
use crate::orient::{Kind, OCR_SIZE, auto_rotation, guess_doc};

#[derive(Clone)]
pub struct Item {
    pub path: Arc<PathBuf>,
    pub original: Arc<RgbImage>,
    pub quad: Quad,
    pub fmt: Fmt,
    pub rotation: u32,
    pub enhance: bool,
    /// Sicurezza del ritaglio automatico (0..1).
    pub confidence: f32,
    /// -1 = fronte, +1 = retro, 0 = non si sa.
    pub side: i8,
    /// Tipo di documento, se riconosciuto dalle scritte.
    pub kind: Option<Kind>,
    /// Documento raddrizzato e pulito a piena risoluzione, prima della rotazione.
    cache: Option<(CacheKey, Arc<RgbImage>)>,
}

type CacheKey = (Quad, Fmt, bool);

/// Sotto questa sicurezza il ritaglio va controllato.
pub const UNCERTAIN: f32 = 0.45;

impl Item {
    /// Tutti i documenti trovati nella foto.
    pub fn from_path(path: &Path) -> Result<Vec<Item>> {
        let img = Arc::new(imgops::load_image(path)?);
        let path = Arc::new(path.to_path_buf());
        Ok(detect_documents(&img)
            .into_par_iter()
            .map(|(quad, confidence)| {
                let mut it = Item {
                    path: path.clone(),
                    original: img.clone(),
                    quad,
                    fmt: guess_format(&quad),
                    rotation: 0,
                    enhance: true,
                    confidence,
                    side: 0,
                    kind: None,
                    cache: None,
                };
                it.auto_orient();
                it
            })
            .collect())
    }

    pub fn auto_orient(&mut self) {
        let flat = imgops::warp_small(&self.original, &self.quad, self.fmt, OCR_SIZE);
        let (rotation, text) = auto_rotation(&flat, self.fmt.landscape());
        self.rotation = rotation;
        (self.side, self.kind) = guess_doc(&text);
    }

    pub fn is_landscape_crop(&self) -> bool {
        let (w, h) = quad_sides(&order_corners(&self.quad));
        w >= h
    }

    fn finish(&self, mut img: RgbImage) -> RgbImage {
        if self.enhance {
            img = imgops::enhance(&img);
        }
        if self.fmt == Fmt::Card {
            imgops::round_corners(&mut img, 3.18 / 85.6);
        }
        img
    }

    /// Anteprima veloce (lato lungo al massimo `max_side`), non ancora ruotata.
    pub fn preview(&self, max_side: u32) -> RgbImage {
        self.finish(imgops::warp_small(&self.original, &self.quad, self.fmt, max_side))
    }

    /// Documento raddrizzato e pulito a piena risoluzione, non ancora ruotato.
    pub fn processed(&mut self) -> Arc<RgbImage> {
        let key = (self.quad, self.fmt, self.enhance);
        if let Some((k, img)) = &self.cache
            && *k == key
        {
            return img.clone();
        }
        let out = Arc::new(self.finish(imgops::warp(&self.original, &self.quad, self.fmt)));
        self.cache = Some((key, out.clone()));
        out
    }

    pub fn result(&mut self) -> RgbImage {
        imgops::rotate(&self.processed(), self.rotation)
    }

    /// Dimensione in pixel dopo la rotazione, senza calcolare l'immagine.
    fn result_dims(&self) -> (u32, u32) {
        let (w, h) = imgops::warp_size(&self.quad, self.fmt);
        if self.rotation.is_multiple_of(180) { (w, h) } else { (h, w) }
    }

    /// Dimensione fisica (larghezza, altezza) in mm dopo la rotazione.
    pub fn size_mm(&self) -> (f32, f32) {
        let (w, h) = self.result_dims();
        let (long_mm, short_mm) = match self.fmt.size_mm() {
            Some(s) => s,
            None => (180.0, 180.0 * w.min(h) as f32 / w.max(h) as f32),
        };
        if w >= h { (long_mm, short_mm) } else { (short_mm, long_mm) }
    }
}
