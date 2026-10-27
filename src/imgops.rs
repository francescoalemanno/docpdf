//! Caricamento, raddrizzamento, pulizia e rotazione delle immagini.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader, Rgb, RgbImage, imageops};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};
use rayon::prelude::*;

use crate::geom::{Quad, order_corners, quad_sides};

/// Formati noti (lato lungo x lato corto, mm).
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Fmt {
    Card,
    Passport,
    A4,
    A5,
    Free,
}

impl Fmt {
    pub const ALL: [Fmt; 5] = [Fmt::Card, Fmt::Passport, Fmt::A4, Fmt::A5, Fmt::Free];

    pub fn label(self) -> &'static str {
        match self {
            Fmt::Card => "Tessera / Carta d'identità",
            Fmt::Passport => "Pagina passaporto",
            Fmt::A4 => "Foglio A4",
            Fmt::A5 => "Foglio A5",
            Fmt::Free => "Libero",
        }
    }

    pub fn size_mm(self) -> Option<(f32, f32)> {
        match self {
            Fmt::Card => Some((85.6, 54.0)),
            Fmt::Passport => Some((125.0, 88.0)),
            Fmt::A4 => Some((297.0, 210.0)),
            Fmt::A5 => Some((210.0, 148.0)),
            Fmt::Free => None,
        }
    }

    /// Orientamento forzato dal formato: tessere orizzontali, fogli verticali.
    pub fn landscape(self) -> Option<bool> {
        match self {
            Fmt::Card => Some(true),
            Fmt::A4 | Fmt::A5 => Some(false),
            _ => None,
        }
    }
}

pub const CARD_RATIO: f32 = 85.6 / 54.0;

pub fn guess_format(q: &Quad) -> Fmt {
    let (w, h) = quad_sides(q);
    let r = w.max(h) / w.min(h).max(1e-6);
    if (r - CARD_RATIO).abs() < 0.12 {
        Fmt::Card
    } else if (r - 297.0 / 210.0).abs() < 0.06 {
        Fmt::A4
    } else {
        Fmt::Free
    }
}

// ---------------------------------------------------------------- caricamento

/// Immagine RGB con l'orientamento EXIF già applicato.
pub fn load_image(path: &Path) -> Result<RgbImage> {
    let native = || -> Result<DynamicImage> {
        let mut dec = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
        let orientation = dec.orientation()?;
        let mut img = DynamicImage::from_decoder(dec)?;
        img.apply_orientation(orientation);
        Ok(img)
    };
    let img = match native() {
        Ok(img) => img,
        // HEIC e formati esotici: si passa da ImageMagick, se c'è
        Err(_) => {
            let mut arg = path.as_os_str().to_owned();
            arg.push("[0]");
            let out = Command::new("magick")
                .arg(arg)
                .args(["-auto-orient", "png:-"])
                .output()
                .context("formato non supportato (per HEIC serve ImageMagick)")?;
            anyhow::ensure!(out.status.success(), "ImageMagick non riesce a leggere il file");
            image::load_from_memory(&out.stdout)?
        }
    };
    Ok(img.into_rgb8())
}

/// Rimpicciolisce (veloce, media dei pixel) in modo che il lato lungo sia al massimo `max_side`.
pub fn shrink(img: &RgbImage, max_side: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let s = max_side as f32 / w.max(h) as f32;
    if s >= 1.0 {
        return img.clone();
    }
    imageops::thumbnail(img, ((w as f32 * s).round() as u32).max(1), ((h as f32 * s).round() as u32).max(1))
}


// ---------------------------------------------------------------- raddrizzamento

/// Dimensioni in pixel del documento raddrizzato. Per i formati noti forza le proporzioni esatte.
pub fn warp_size(q: &Quad, fmt: Fmt) -> (u32, u32) {
    let (w, h) = quad_sides(&order_corners(q));
    let (ow, oh) = match fmt.size_mm() {
        Some((long_mm, short_mm)) => {
            let long_px = w.max(h).clamp(600.0, 4000.0);
            if w >= h {
                (long_px, long_px * short_mm / long_mm)
            } else {
                (long_px * short_mm / long_mm, long_px)
            }
        }
        None => (w, h),
    };
    ((ow.round() as u32).max(2), (oh.round() as u32).max(2))
}

/// Raddrizza la prospettiva del documento delimitato da `q`.
pub fn warp(img: &RgbImage, q: &Quad, fmt: Fmt) -> RgbImage {
    warp_to(img, q, warp_size(q, fmt))
}

/// Come `warp`, ma rimpicciolito (lato lungo al massimo `max_side`) e più veloce: per l'OCR e le anteprime.
pub fn warp_small(img: &RgbImage, q: &Quad, fmt: Fmt, max_side: u32) -> RgbImage {
    let (w, h) = warp_size(q, fmt);
    let s = (max_side as f32 / w.max(h) as f32).min(1.0);
    let size = (((w as f32 * s).round() as u32).max(2), ((h as f32 * s).round() as u32).max(2));
    warp_with(img, q, size, Interpolation::Bilinear)
}

pub fn warp_to(img: &RgbImage, q: &Quad, size: (u32, u32)) -> RgbImage {
    warp_with(img, q, size, Interpolation::Bicubic)
}

fn warp_with(img: &RgbImage, q: &Quad, (ow, oh): (u32, u32), interp: Interpolation) -> RgbImage {
    let q = order_corners(q);
    let (fw, fh) = ((ow - 1) as f32, (oh - 1) as f32);
    let dst = [(0.0, 0.0), (fw, 0.0), (fw, fh), (0.0, fh)];
    let src = q.map(|p| (p[0], p[1]));
    let mut out = RgbImage::new(ow, oh);
    if let Some(proj) = Projection::from_control_points(src, dst) {
        warp_into(img, proj, interp, Border::Replicate, &mut out);
    }
    out
}

pub fn rotate(img: &RgbImage, deg: u32) -> RgbImage {
    match deg % 360 {
        90 => imageops::rotate90(img),
        180 => imageops::rotate180(img),
        270 => imageops::rotate270(img),
        _ => img.clone(),
    }
}

// ---------------------------------------------------------------- pulizia

fn percentile_of_hist(hist: &[u64], total: u64, p: f32) -> usize {
    let target = (total as f64 * p as f64 / 100.0).ceil() as u64;
    let mut acc = 0;
    for (i, &c) in hist.iter().enumerate() {
        acc += c;
        if acc >= target.max(1) {
            return i;
        }
    }
    hist.len() - 1
}

/// Bilanciamento del bianco, contrasto e nitidezza, senza snaturare i colori.
pub fn enhance(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let px = img.as_raw();
    let n = (w * h) as u64;

    // bianco: i pixel più chiari diventano neutri
    let mut lum_hist = [0u64; 256];
    for p in px.as_chunks::<3>().0 {
        lum_hist[(p[0] as usize + p[1] as usize + p[2] as usize) / 3] += 1;
    }
    let thr = percentile_of_hist(&lum_hist, n, 92.0);
    let (mut sum, mut cnt) = ([0.0f64; 3], 0u64);
    for p in px.as_chunks::<3>().0 {
        if (p[0] as usize + p[1] as usize + p[2] as usize) / 3 >= thr {
            for c in 0..3 {
                sum[c] += p[c] as f64;
            }
            cnt += 1;
        }
    }
    let mut gain = [1.0f32; 3];
    if cnt > 50 {
        let r = sum.map(|s| (s / cnt as f64) as f32);
        let mean = (r[0] + r[1] + r[2]) / 3.0;
        gain = r.map(|v| (mean / v.max(1.0)).clamp(0.8, 1.25));
    }

    // stiramento dei livelli
    let mut val_hist = vec![0u64; 320];
    for p in px.as_chunks::<3>().0 {
        for c in 0..3 {
            val_hist[((p[c] as f32 * gain[c]) as usize).min(319)] += 1;
        }
    }
    let lo = percentile_of_hist(&val_hist, n * 3, 0.7) as f32;
    let hi = percentile_of_hist(&val_hist, n * 3, 99.3) as f32;
    let (off, mul) = if hi - lo > 30.0 { (lo, 255.0 / (hi - lo)) } else { (0.0, 1.0) };
    let mut f: Vec<f32> = px
        .par_chunks_exact(3)
        .flat_map_iter(|p| (0..3).map(move |c| ((p[c] as f32 * gain[c] - off) * mul).clamp(0.0, 255.0)))
        .collect();

    // contrasto locale leggero sulla luminanza
    let luma: Vec<u8> = f
        .as_chunks::<3>().0.iter()
        .map(|p| (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).round().clamp(0.0, 255.0) as u8)
        .collect();
    let eq = clahe(&luma, w as usize, h as usize, 1.3, 8);
    f.par_chunks_exact_mut(3).zip(luma.par_iter().zip(eq.par_iter())).for_each(|(p, (&a, &b))| {
        let d = b as f32 - a as f32;
        for v in p.iter_mut() {
            *v = (*v + d).clamp(0.0, 255.0);
        }
    });

    // nitidezza (unsharp mask)
    let base = RgbImage::from_raw(w, h, f.iter().map(|v| v.round() as u8).collect()).unwrap();
    let blur = imageproc::filter::gaussian_blur_f32(&base, 1.6);
    let out: Vec<u8> = f
        .par_iter()
        .zip(blur.as_raw().par_iter())
        .map(|(&a, &b)| (1.4 * a - 0.4 * b as f32).round().clamp(0.0, 255.0) as u8)
        .collect();
    RgbImage::from_raw(w, h, out).unwrap()
}

/// CLAHE come in OpenCV: istogrammi per riquadro con limite di contrasto, interpolati bilinearmente.
fn clahe(src: &[u8], w: usize, h: usize, clip_limit: f32, tiles: usize) -> Vec<u8> {
    let tw = w.div_ceil(tiles).max(1);
    let th = h.div_ceil(tiles).max(1);
    let (nx, ny) = (w.div_ceil(tw), h.div_ceil(th));
    let maps: Vec<[u8; 256]> = (0..nx * ny)
        .into_par_iter()
        .map(|t| {
            let (tx, ty) = (t % nx, t / nx);
            let (x0, y0) = (tx * tw, ty * th);
            let (x1, y1) = ((x0 + tw).min(w), (y0 + th).min(h));
            let mut hist = [0u32; 256];
            for y in y0..y1 {
                for &v in &src[y * w + x0..y * w + x1] {
                    hist[v as usize] += 1;
                }
            }
            let area = ((x1 - x0) * (y1 - y0)) as u32;
            let limit = ((clip_limit * area as f32 / 256.0) as u32).max(1);
            let mut excess = 0;
            for c in hist.iter_mut() {
                if *c > limit {
                    excess += *c - limit;
                    *c = limit;
                }
            }
            let (add, rest) = (excess / 256, (excess % 256) as usize);
            for (i, c) in hist.iter_mut().enumerate() {
                *c += add + (i < rest) as u32;
            }
            let mut map = [0u8; 256];
            let mut acc = 0u32;
            let scale = 255.0 / area.max(1) as f32;
            for (i, c) in hist.iter().enumerate() {
                acc += c;
                map[i] = (acc as f32 * scale).round().min(255.0) as u8;
            }
            map
        })
        .collect();
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = (y as f32 + 0.5) / th as f32 - 0.5;
        let ty0 = fy.floor().clamp(0.0, (ny - 1) as f32) as usize;
        let ty1 = (ty0 + 1).min(ny - 1);
        let wy = (fy - ty0 as f32).clamp(0.0, 1.0);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = (x as f32 + 0.5) / tw as f32 - 0.5;
            let tx0 = fx.floor().clamp(0.0, (nx - 1) as f32) as usize;
            let tx1 = (tx0 + 1).min(nx - 1);
            let wx = (fx - tx0 as f32).clamp(0.0, 1.0);
            let v = src[y * w + x] as usize;
            let m = |tx: usize, ty: usize| maps[ty * nx + tx][v] as f32;
            let top = m(tx0, ty0) * (1.0 - wx) + m(tx1, ty0) * wx;
            let bot = m(tx0, ty1) * (1.0 - wx) + m(tx1, ty1) * wx;
            *o = (top * (1.0 - wy) + bot * wy).round() as u8;
        }
    });
    out
}

/// Imbianca gli angoli fuori dal bordo arrotondato della tessera.
pub fn round_corners(img: &mut RgbImage, radius_frac: f32) {
    let (w, h) = img.dimensions();
    let r = (radius_frac * w.max(h) as f32).round();
    if r < 2.0 {
        return;
    }
    let ri = r as u32;
    let centers = [
        (r, r, 0, 0),
        (w as f32 - r - 1.0, r, w - ri - 1, 0),
        (r, h as f32 - r - 1.0, 0, h - ri - 1),
        (w as f32 - r - 1.0, h as f32 - r - 1.0, w - ri - 1, h - ri - 1),
    ];
    for (cx, cy, x0, y0) in centers {
        for y in y0..(y0 + ri + 1).min(h) {
            for x in x0..(x0 + ri + 1).min(w) {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                // solo il quarto esterno dell'angolo
                if dx * (x0 as f32 - cx).signum() < 0.0 || dy * (y0 as f32 - cy).signum() < 0.0 {
                    continue;
                }
                let a = (r + 0.5 - dx.hypot(dy)).clamp(0.0, 1.0);
                if a < 1.0 {
                    let p = img.get_pixel_mut(x, y);
                    *p = Rgb(p.0.map(|v| (v as f32 * a + 255.0 * (1.0 - a)).round() as u8));
                }
            }
        }
    }
}
