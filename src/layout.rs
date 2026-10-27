//! Impaginazione dei documenti su un foglio A4.

pub const A4_W: f32 = 210.0;
pub const A4_H: f32 = 297.0;
const MARGIN: f32 = 12.0;
const GAP: f32 = 8.0;

/// Mette in fila i documenti trovati in più foto.
/// Caso tipico: foto dei fronti + foto dei retri con lo stesso numero di documenti
/// -> fronte e retro di ciascun documento uno accanto all'altro.
/// `side` dice se un documento è il fronte (-1) o il retro (+1).
pub fn order_groups<T>(groups: Vec<Vec<T>>, side: impl Fn(&T) -> i8) -> Vec<T> {
    let groups: Vec<Vec<T>> = groups.into_iter().filter(|g| !g.is_empty()).collect();
    if groups.len() == 2 && groups[0].len() == groups[1].len() {
        // coppie fronte/retro, il fronte per primo
        let mut it = groups.into_iter();
        let (a, b) = (it.next().unwrap(), it.next().unwrap());
        return a
            .into_iter()
            .zip(b)
            .flat_map(|(x, y)| if side(&y) < side(&x) { [y, x] } else { [x, y] })
            .collect();
    }
    groups.into_iter().flatten().collect()
}

/// Riga di documenti: (indice, larghezza, altezza) in mm.
type Row = Vec<(usize, f32, f32)>;

/// Disposizione a righe; restituisce le righe, l'altezza totale e la larghezza massima.
fn pack(sizes: &[(f32, f32)], scale: f32, avail_w: f32) -> (Vec<Row>, f32, f32) {
    let mut rows: Vec<Row> = Vec::new();
    let (mut row, mut row_w): (Row, f32) = (Vec::new(), 0.0);
    for (i, &(w, h)) in sizes.iter().enumerate() {
        let (w, h) = (w * scale, h * scale);
        if !row.is_empty() && row_w + GAP + w > avail_w {
            rows.push(std::mem::take(&mut row));
            row_w = 0.0;
        }
        row_w += if row.is_empty() { 0.0 } else { GAP } + w;
        row.push((i, w, h));
    }
    if !row.is_empty() {
        rows.push(row);
    }
    let row_h = |r: &Row| r.iter().map(|t| t.2).fold(0.0, f32::max);
    let row_w = |r: &Row| r.iter().map(|t| t.1).sum::<f32>() + GAP * (r.len() - 1) as f32;
    let total_h = rows.iter().map(row_h).sum::<f32>() + GAP * (rows.len().saturating_sub(1)) as f32;
    let max_w = rows.iter().map(row_w).fold(0.0, f32::max);
    (rows, total_h, max_w)
}

/// Posizioni (x, y, w, h) in mm su A4 verticale, origine in alto a sinistra.
/// `max_scale` = None: ingrandisce fino a riempire il foglio.
pub fn layout(sizes: &[(f32, f32)], max_scale: Option<f32>) -> Vec<(f32, f32, f32, f32)> {
    if sizes.is_empty() {
        return vec![];
    }
    let (aw, ah) = (A4_W - 2.0 * MARGIN, A4_H - 2.0 * MARGIN);
    let (mut lo, mut hi) = (0.05f32, max_scale.unwrap_or(20.0));
    let (mut rows, mut th, mw) = pack(sizes, hi, aw);
    if th > ah || mw > aw {
        for _ in 0..40 {
            let mid = (lo + hi) / 2.0;
            let (_, t, m) = pack(sizes, mid, aw);
            if t <= ah && m <= aw {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (rows, th, _) = pack(sizes, lo, aw);
    }
    // centrato verticalmente (leggermente verso l'alto se c'è molto spazio)
    let free = ah - th;
    let mut y = MARGIN + if free < 60.0 { free / 2.0 } else { (free / 2.0).min(25.0) };
    let mut pos = vec![(0.0, 0.0, 0.0, 0.0); sizes.len()];
    for r in &rows {
        let rh = r.iter().map(|t| t.2).fold(0.0, f32::max);
        let rw = r.iter().map(|t| t.1).sum::<f32>() + GAP * (r.len() - 1) as f32;
        let mut x = MARGIN + (aw - rw) / 2.0;
        for &(i, w, h) in r {
            pos[i] = (x, y + (rh - h) / 2.0, w, h);
            x += w + GAP;
        }
        y += rh + GAP;
    }
    pos
}
