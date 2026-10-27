//! Rilevamento dei documenti nella foto (anche più d'uno).

use image::{GrayImage, Luma, RgbImage};
use imageproc::contours::find_contours;
use imageproc::contrast::otsu_level;
use imageproc::distance_transform::Norm;
use imageproc::edges::canny;
use imageproc::filter::{gaussian_blur_f32, median_filter};
use imageproc::geometry::{approximate_polygon_dp, arc_length, convex_hull, min_area_rect};
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use imageproc::morphology::{close, dilate, open};
use imageproc::point::Point;
use rayon::prelude::*;

use crate::geom::*;
use crate::imgops::{CARD_RATIO, shrink};

const WORK_SIZE: u32 = 640;

/// Canali in cui cercare i bordi: grigio, a e b di Lab, saturazione HSV.
fn channels(img: &RgbImage) -> [GrayImage; 4] {
    let (w, h) = img.dimensions();
    let mut out = [(); 4].map(|_| GrayImage::new(w, h));
    for (x, y, p) in img.enumerate_pixels() {
        let [r, g, b] = p.0;
        let (gray, a, bb) = gray_lab_ab(r, g, b);
        let mx = r.max(g).max(b) as f32;
        let mn = r.min(g).min(b) as f32;
        let s = if mx > 0.0 { 255.0 * (mx - mn) / mx } else { 0.0 };
        out[0].put_pixel(x, y, Luma([gray]));
        out[1].put_pixel(x, y, Luma([a]));
        out[2].put_pixel(x, y, Luma([bb]));
        out[3].put_pixel(x, y, Luma([s.round() as u8]));
    }
    out
}

/// Grigio e componenti a, b di Lab con la codifica a 8 bit di OpenCV (+128).
fn gray_lab_ab(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let lin = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (rl, gl, bl) = (lin(r), lin(g), lin(b));
    let x = (0.412453 * rl + 0.357580 * gl + 0.180423 * bl) / 0.950456;
    let y = 0.212671 * rl + 0.715160 * gl + 0.072169 * bl;
    let z = (0.019334 * rl + 0.119193 * gl + 0.950227 * bl) / 1.088754;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    let a = 500.0 * (fx - fy) + 128.0;
    let bb = 200.0 * (fy - fz) + 128.0;
    let gray = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
    let c = |v: f32| v.round().clamp(0.0, 255.0) as u8;
    (c(gray), c(a), c(bb))
}

fn percentile(v: &[f32], p: f32) -> f32 {
    let mut s = v.to_vec();
    let k = ((s.len() - 1) as f32 * p / 100.0) as usize;
    *s.select_nth_unstable_by(k, f32::total_cmp).1
}

/// Gradiente (Sobel) di un canale sfocato: componenti x, y.
fn gradient(ch: &GrayImage) -> (Vec<f32>, Vec<f32>) {
    let b = gaussian_blur_f32(ch, 1.1);
    let gx = horizontal_sobel(&b).as_raw().iter().map(|&v| v as f32).collect();
    let gy = vertical_sobel(&b).as_raw().iter().map(|&v| v as f32).collect();
    (gx, gy)
}

/// Gradiente combinato dei canali: in ogni pixel quello del canale più netto, con ogni canale
/// normalizzato sui propri bordi (così un bordo solo di colore conta quanto uno di luminosità).
struct GradMap {
    w: usize,
    h: usize,
    gx: Vec<f32>,
    gy: Vec<f32>,
}

impl GradMap {
    fn new(w: usize, h: usize, grads: &[(Vec<f32>, Vec<f32>)]) -> GradMap {
        let scales: Vec<f32> = grads
            .iter()
            .map(|(gx, gy)| {
                let mag: Vec<f32> = gx.iter().zip(gy).map(|(x, y)| x.hypot(*y)).collect();
                // pavimento: un canale quasi senza bordi non deve amplificare il rumore
                1.0 / percentile(&mag, 99.0).max(60.0)
            })
            .collect();
        let (mut gx, mut gy) = (vec![0.0f32; w * h], vec![0.0f32; w * h]);
        for i in 0..w * h {
            let mut best = 0.0f32;
            for ((cx, cy), s) in grads.iter().zip(&scales) {
                let (x, y) = (cx[i] * s, cy[i] * s);
                let m = x.hypot(y);
                if m > best {
                    best = m;
                    // i bordi molto netti (testo) non devono pesare più di un bordo pulito
                    let k = if m > 1.0 { 1.0 / m } else { 1.0 };
                    (gx[i], gy[i]) = (x * k, y * k);
                }
            }
        }
        GradMap { w, h, gx, gy }
    }

    /// Gradiente nella direzione `n` vicino al punto (massimo su ±1 pixel lungo `n`).
    fn along(&self, p: Pt, n: Pt) -> f32 {
        let mut m = 0.0f32;
        for d in [-1.0, 0.0, 1.0] {
            let (x, y) = ((p[0] + n[0] * d).round(), (p[1] + n[1] * d).round());
            if x < 0.0 || y < 0.0 || x as usize >= self.w || y as usize >= self.h {
                continue;
            }
            let i = y as usize * self.w + x as usize;
            m = m.max((self.gx[i] * n[0] + self.gy[i] * n[1]).abs());
        }
        m
    }

    /// Gradiente medio lungo la parte centrale del segmento a-b.
    fn side_strength(&self, a: Pt, b: Pt) -> f32 {
        const N: usize = 32;
        let l = dist(a, b).max(1e-3);
        let n = [-(b[1] - a[1]) / l, (b[0] - a[0]) / l];
        (0..N)
            .map(|k| {
                let t = 0.15 + 0.7 * k as f32 / (N - 1) as f32;
                self.along([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], n)
            })
            .sum::<f32>()
            / N as f32
    }

    /// Nettezza del bordo lungo il perimetro: media dei lati, con il lato più debole che conta
    /// quanto la media (un lato che taglia il documento invece di seguirne il bordo è debole).
    fn edge_strength(&self, q: &Quad) -> f32 {
        let sides = (0..4).map(|i| self.side_strength(q[i], q[(i + 1) % 4]));
        let (sum, min) = sides.fold((0.0f32, f32::MAX), |(s, m), v| (s + v, m.min(v)));
        (sum / 4.0 + min) / 2.0
    }

    /// Sposta ogni lato (i due estremi lungo la normale, indipendentemente) sul bordo più netto
    /// nelle vicinanze, poi ricalcola gli spigoli: i contorni delle maschere sono approssimativi.
    fn snap(&self, q: &Quad) -> Quad {
        let (w, h) = quad_sides(q);
        let reach = (0.06 * w.min(h)).max(4.0);
        let c = centroid(q);
        let mut sides = [([0.0f32; 2], [0.0f32; 2]); 4];
        for i in 0..4 {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            let l = dist(a, b).max(1.0);
            let mut n = [-(b[1] - a[1]) / l, (b[0] - a[0]) / l];
            let mid = [(a[0] + b[0]) / 2.0 - c[0], (a[1] + b[1]) / 2.0 - c[1]];
            if n[0] * mid[0] + n[1] * mid[1] < 0.0 {
                n = [-n[0], -n[1]];
            }
            let mv = |p: Pt, d: f32| [p[0] + n[0] * d, p[1] + n[1] * d];
            let score = |da: f32, db: f32| self.side_strength(mv(a, da), mv(b, db));
            // prima a passi larghi, poi di un pixel attorno al migliore
            let step = (reach / 8.0).max(1.0);
            let mut best = (score(0.0, 0.0), 0.0, 0.0);
            let k = (reach / step).ceil() as i32;
            for i in -k..=k {
                for j in -k..=k {
                    let (da, db) = (i as f32 * step, j as f32 * step);
                    let s = score(da, db);
                    if s > best.0 {
                        best = (s, da, db);
                    }
                }
            }
            // fra i bordi netti quasi quanto il migliore vince il più esterno: dentro il documento
            // ci sono scritte e righe a contrasto più alto del bordo, fuori di solito no
            let strong = 0.7 * best.0;
            for i in -k..=k {
                for j in -k..=k {
                    let (da, db) = (i as f32 * step, j as f32 * step);
                    if da + db > best.1 + best.2 && score(da, db) >= strong {
                        best = (score(da, db), da, db);
                    }
                }
            }
            let (_, da0, db0) = best;
            let f = step.ceil() as i32;
            for i in -f..=f {
                for j in -f..=f {
                    let (da, db) = (da0 + i as f32, db0 + j as f32);
                    let s = score(da, db);
                    if s > best.0 {
                        best = (s, da, db);
                    }
                }
            }
            sides[i] = (mv(a, best.1), mv(b, best.2));
        }
        let mut out = *q;
        for i in 0..4 {
            let (p, q2) = (sides[(i + 3) % 4], sides[i]);
            match intersect_segments_lines(p, q2) {
                Some(x) => out[i] = x,
                None => return *q,
            }
        }
        if (0..4).any(|i| dist(out[i], q[i]) > 0.2 * w.min(h)) {
            return *q;
        }
        out
    }
}

/// Intersezione delle rette passanti per due segmenti.
fn intersect_segments_lines((a1, b1): (Pt, Pt), (a2, b2): (Pt, Pt)) -> Option<Pt> {
    let (d1, d2) = (sub(b1, a1), sub(b2, a2));
    let det = d1[0] * d2[1] - d1[1] * d2[0];
    if det.abs() < 1e-6 {
        return None;
    }
    let r = sub(a2, a1);
    let t = (r[0] * d2[1] - r[1] * d2[0]) / det;
    Some([a1[0] + t * d1[0], a1[1] + t * d1[1]])
}

/// Douglas-Peucker su un poligono chiuso. Si parte da due punti opposti (il più lontano dal
/// baricentro e il più lontano da questo), che sono certamente vertici: imageproc invece tiene
/// sempre il primo punto, che può cadere a metà di un lato.
fn approximate_closed(poly: &[Point<i32>], eps: f64) -> Vec<Point<i32>> {
    let d2 = |a: Point<i32>, b: Point<i32>| ((a.x - b.x) as i64).pow(2) + ((a.y - b.y) as i64).pow(2);
    let n = poly.len();
    let (cx, cy) = poly.iter().fold((0i64, 0i64), |(x, y), p| (x + p.x as i64, y + p.y as i64));
    let c = Point::new((cx / n as i64) as i32, (cy / n as i64) as i32);
    let a = (0..n).max_by_key(|&i| d2(poly[i], c)).unwrap();
    let b = (0..n).max_by_key(|&i| d2(poly[i], poly[a])).unwrap();
    let (a, b) = (a.min(b), a.max(b));
    let first: Vec<_> = poly[a..=b].to_vec();
    let second: Vec<_> = poly[b..].iter().chain(&poly[..=a]).copied().collect();
    let mut out = approximate_polygon_dp(&first, eps, false);
    out.pop();
    let mut rest = approximate_polygon_dp(&second, eps, false);
    rest.pop();
    out.append(&mut rest);
    out
}

/// Frazione del perimetro del quadrilatero su cui la maschera è accesa (entro 2 pixel).
fn outline_coverage(mask: &GrayImage, q: &Quad) -> f32 {
    const N: usize = 25;
    let (w, h) = (mask.width() as i32, mask.height() as i32);
    let on = |p: Pt| {
        (-2..=2).any(|dy| {
            (-2..=2).any(|dx| {
                let (x, y) = (p[0].round() as i32 + dx, p[1].round() as i32 + dy);
                x >= 0 && y >= 0 && x < w && y < h && mask.get_pixel(x as u32, y as u32).0[0] > 0
            })
        })
    };
    let hits: usize = (0..4)
        .map(|i| {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            (0..N)
                .filter(|&k| {
                    let t = (k as f32 + 0.5) / N as f32;
                    on([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
                })
                .count()
        })
        .sum();
    hits as f32 / (4 * N) as f32
}

/// Quadrilatero formato dai quattro lati più lunghi di un poligono, prolungati: toglie gli
/// spigoli tagliati di sbieco.
fn longest_sides_quad(poly: &[Pt]) -> Option<Quad> {
    let n = poly.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| dist(poly[b], poly[(b + 1) % n]).total_cmp(&dist(poly[a], poly[(a + 1) % n])));
    idx.truncate(4);
    idx.sort_unstable();
    let side = |k: usize| (poly[idx[k]], poly[(idx[k] + 1) % n]);
    let mut q = [[0.0; 2]; 4];
    for (k, corner) in q.iter_mut().enumerate() {
        *corner = intersect_segments_lines(side((k + 3) % 4), side(k))?;
    }
    is_convex(&q).then_some(q)
}

type Contour = (f32, f32, Vec<Pt>, Vec<Point<i32>>);

/// Quadrilateri candidati (con il "riempimento") dalle regioni di una maschera. Il riempimento
/// è la parte del quadrilatero occupata dalla regione, oppure, per i contorni aperti (un bordo
/// con un buco, che non racchiude nulla), la parte del perimetro coperta dal contorno.
fn contour_candidates(mask: &GrayImage, min_area: f32) -> Vec<(Quad, f32)> {
    // (area dell'inviluppo, area racchiusa, contorno, inviluppo)
    let mut cnts: Vec<Contour> = find_contours::<i32>(mask)
        .into_iter()
        .filter(|c| c.points.len() >= 4)
        .filter_map(|c| {
            let pts: Vec<Pt> = c.points.iter().map(|p| [p.x as f32, p.y as f32]).collect();
            let hull = convex_hull(c.points);
            let hull_pts: Vec<Pt> = hull.iter().map(|p| [p.x as f32, p.y as f32]).collect();
            let hull_area = polygon_area(&hull_pts);
            (hull.len() >= 3 && hull_area >= min_area).then(|| (hull_area, polygon_area(&pts), pts, hull))
        })
        .collect();
    cnts.sort_by(|a, b| b.0.total_cmp(&a.0));
    cnts.truncate(8);
    let mut out = Vec::new();
    for (_, area, pts, hull) in cnts {
        let peri = arc_length(&hull, true);
        let (mut quad, mut cut) = (None, None);
        for eps in [0.02, 0.03, 0.04, 0.06] {
            let ap: Vec<Pt> = approximate_closed(&hull, eps * peri).iter().map(|p| [p.x as f32, p.y as f32]).collect();
            if ap.len() == 4 {
                quad = Some([ap[0], ap[1], ap[2], ap[3]]);
                break;
            }
            if cut.is_none() && (5..=6).contains(&ap.len()) {
                cut = longest_sides_quad(&ap);
            }
        }
        let quad = quad.unwrap_or_else(|| min_area_rect(&hull).map(|p| [p.x as f32, p.y as f32]));
        let refined = refine_quad_with_lines(&pts, &order_corners(&quad));
        // con uno spigolo smussato (bordo interrotto proprio lì) si prova anche la versione
        // con i quattro lati più lunghi prolungati
        for q in std::iter::once(refined).chain(cut.map(|q| order_corners(&q))) {
            let qa = polygon_area(&q);
            let fill = if qa > 0.0 { (area / qa).min(1.0) } else { 0.0 };
            out.push((q, fill.max(outline_coverage(mask, &q))));
        }
    }
    out
}

fn scored_candidates(cands: &[(Quad, f32)], g: &GradMap) -> Vec<(f32, Quad)> {
    let (w, h) = (g.w as f32, g.h as f32);
    let mut out: Vec<(f32, Quad)> = cands
        .par_iter()
        .filter_map(|&(q, fill)| {
            if !is_convex(&q) {
                return None;
            }
            let q = g.snap(&q);
            if !is_convex(&q) {
                return None;
            }
            let frac = polygon_area(&q) / (w * h);
            if !(0.04..=0.985).contains(&frac) {
                return None;
            }
            let (sw, sh) = quad_sides(&q);
            let ratio = sw.max(sh) / sw.min(sh).max(1e-6);
            if ratio > 3.0 {
                return None;
            }
            // angoli vicini a 90° (tolleranza per la prospettiva)
            let max_cos = (0..4)
                .map(|i| {
                    let (v1, v2) = (sub(q[(i + 3) % 4], q[i]), sub(q[(i + 1) % 4], q[i]));
                    let dot = v1[0] * v2[0] + v1[1] * v2[1];
                    (dot / (v1[0].hypot(v1[1]) * v2[0].hypot(v2[1]) + 1e-9)).abs()
                })
                .fold(0.0f32, f32::max);
            if max_cos > 0.45 {
                return None;
            }
            let edge = g.edge_strength(&q).min(1.0);
            let fmt_bonus = [CARD_RATIO, 297.0 / 210.0, 125.0 / 88.0]
                .iter()
                .map(|r| (-((ratio - r) / 0.1).powi(2)).exp())
                .fold(0.0f32, f32::max);
            // non a ridosso del bordo foto (tipico dei falsi positivi che inquadrano tutto)
            let margin = q
                .iter()
                .map(|p| p[0].min(p[1]).min(w - p[0]).min(h - p[1]))
                .fold(f32::MAX, f32::min);
            let border_pen = if margin < 3.0 { 0.15 } else { 0.0 };
            let score = 0.45 * edge + 0.2 * fill.min(1.0) + 0.15 * (1.0 - max_cos) + 0.12 * fmt_bonus
                + 0.08 * (frac / 0.25).min(1.0)
                - border_pen;
            Some((score, q))
        })
        .collect();
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out
}

/// Frazione del quadrilatero più piccolo coperta dall'altro (stimata su una griglia).
fn overlap(q1: &Quad, q2: &Quad) -> f32 {
    let bbox = |q: &Quad| {
        let xs = q.map(|p| p[0]);
        let ys = q.map(|p| p[1]);
        (
            xs.iter().copied().fold(f32::MAX, f32::min),
            ys.iter().copied().fold(f32::MAX, f32::min),
            xs.iter().copied().fold(f32::MIN, f32::max),
            ys.iter().copied().fold(f32::MIN, f32::max),
        )
    };
    let (a, b) = (bbox(q1), bbox(q2));
    let (x0, y0, x1, y1) = (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3));
    let step = ((x1 - x0).max(y1 - y0) / 150.0).max(1.0);
    let (mut n1, mut n2, mut both) = (0u32, 0u32, 0u32);
    let mut y = y0;
    while y <= y1 {
        let mut x = x0;
        while x <= x1 {
            let (i1, i2) = (inside_convex(q1, [x, y]), inside_convex(q2, [x, y]));
            n1 += i1 as u32;
            n2 += i2 as u32;
            both += (i1 && i2) as u32;
            x += step;
        }
        y += step;
    }
    both as f32 / n1.min(n2).max(1) as f32
}

/// Sceglie più documenti non sovrapposti (es. due tessere nella stessa foto).
fn pick_documents(scored: &[(f32, Quad)]) -> Vec<(f32, Quad)> {
    let Some(&(best, _)) = scored.first() else { return vec![] };
    let mut chosen: Vec<(f32, Quad)> = Vec::new();
    for &(score, q) in scored {
        if score < (0.6 * best).max(0.42) {
            break;
        }
        if chosen.iter().any(|(_, c)| overlap(&q, c) > 0.15) {
            continue;
        }
        chosen.push((score, q));
        if chosen.len() >= 6 {
            break;
        }
    }
    chosen
}

/// Trova i documenti nella foto, ordinati dall'alto in basso e da sinistra a destra.
/// Restituisce [(4 spigoli in coordinate originali, punteggio 0..1)].
pub fn detect_documents(img: &RgbImage) -> Vec<(Quad, f32)> {
    let (big_w, big_h) = img.dimensions();
    let small = shrink(img, WORK_SIZE);
    let scale = small.width() as f32 / big_w as f32;
    let (w, h) = small.dimensions();
    let min_area = 0.04 * (w * h) as f32;
    let chans = channels(&small);

    // gradiente anche sul colore: i documenti spesso si staccano dallo sfondo per tinta
    let grads: Vec<(Vec<f32>, Vec<f32>)> = chans.par_iter().map(gradient).collect();
    let gmap = GradMap::new(w as usize, h as usize, &grads);

    let masks: Vec<GrayImage> = chans
        .par_iter()
        .zip(&grads)
        .flat_map_iter(|(ch, grad)| {
            // soglie di Canny relative ai bordi presenti nel canale
            let mag: Vec<f32> = grad.0.iter().zip(&grad.1).map(|(x, y)| x.hypot(*y)).collect();
            let g90 = percentile(&mag, 90.0);
            let blur = median_filter(ch, 2, 2);
            let mut out = Vec::new();
            for f in [1.0, 0.5, 2.0] {
                let hi = (g90 * f).max(12.0);
                let e = canny(&blur, 0.4 * hi, hi);
                let e = dilate(&e, Norm::LInf, 2);
                out.push(close(&e, Norm::LInf, 4));
            }
            // soglia di Otsu: separa documento chiaro/scuro da sfondo uniforme
            let b = gaussian_blur_f32(ch, 1.4);
            let t = otsu_level(&b);
            let th = GrayImage::from_fn(w, h, |x, y| Luma([if b.get_pixel(x, y).0[0] > t { 255 } else { 0 }]));
            let inv = GrayImage::from_fn(w, h, |x, y| Luma([255 - th.get_pixel(x, y).0[0]]));
            out.push(open(&th, Norm::LInf, 4));
            out.push(open(&inv, Norm::LInf, 4));
            out.into_iter()
        })
        .collect();
    let cands: Vec<(Quad, f32)> = masks.par_iter().flat_map_iter(|m| contour_candidates(m, min_area)).collect();

    let scored = scored_candidates(&cands, &gmap);
    let mut chosen = pick_documents(&scored);
    if chosen.is_empty() {
        return vec![(full_quad(big_w, big_h), 0.0)];
    }
    // ordine di lettura: per righe (dall'alto), poi da sinistra
    let key = |q: &Quad| {
        let c = centroid(q);
        ((c[1] / (h as f32 * 0.15)).round() as i32, c[0])
    };
    chosen.sort_by(|a, b| {
        let (ka, kb) = (key(&a.1), key(&b.1));
        ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1))
    });
    chosen.into_iter().map(|(sc, q)| (scale_quad(&q, 1.0 / scale), sc)).collect()
}

/// Un solo documento: il migliore, oppure quello più vicino a `near`.
pub fn detect_document(img: &RgbImage, near: Option<&Quad>) -> (Quad, f32) {
    let found = detect_documents(img);
    match near {
        Some(n) if found.len() > 1 => {
            let c = centroid(n);
            *found.iter().min_by(|a, b| dist(centroid(&a.0), c).total_cmp(&dist(centroid(&b.0), c))).unwrap()
        }
        _ => *found.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap(),
    }
}
