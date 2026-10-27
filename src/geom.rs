//! Geometria dei quadrilateri: ordinamento degli spigoli, lati, rette, aree.

pub type Pt = [f32; 2];
/// 4 spigoli: alto-sx, alto-dx, basso-dx, basso-sx.
pub type Quad = [Pt; 4];

pub fn sub(a: Pt, b: Pt) -> Pt {
    [a[0] - b[0], a[1] - b[1]]
}

pub fn dist(a: Pt, b: Pt) -> f32 {
    let d = sub(a, b);
    d[0].hypot(d[1])
}

pub fn centroid(pts: &[Pt]) -> Pt {
    let n = pts.len().max(1) as f32;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
    [sx / n, sy / n]
}

/// Ordina 4 punti: alto-sx, alto-dx, basso-dx, basso-sx.
pub fn order_corners(q: &Quad) -> Quad {
    let c = centroid(q);
    let mut pts = *q;
    // angolo crescente in coordinate immagine (y verso il basso) = senso orario
    pts.sort_by(|a, b| {
        let aa = (a[1] - c[1]).atan2(a[0] - c[0]);
        let ab = (b[1] - c[1]).atan2(b[0] - c[0]);
        aa.total_cmp(&ab)
    });
    let start = (0..4)
        .min_by(|&i, &j| (pts[i][0] + pts[i][1]).total_cmp(&(pts[j][0] + pts[j][1])))
        .unwrap();
    pts.rotate_left(start);
    pts
}

/// (larghezza, altezza) medie del quadrilatero ordinato.
pub fn quad_sides(q: &Quad) -> (f32, f32) {
    let w = (dist(q[1], q[0]) + dist(q[2], q[3])) / 2.0;
    let h = (dist(q[3], q[0]) + dist(q[2], q[1])) / 2.0;
    (w, h)
}

/// Area (sempre positiva) di un poligono.
pub fn polygon_area(pts: &[Pt]) -> f32 {
    let n = pts.len();
    let mut s = 0.0f64;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        s += a[0] as f64 * b[1] as f64 - b[0] as f64 * a[1] as f64;
    }
    (s / 2.0).abs() as f32
}

pub fn is_convex(q: &Quad) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let (a, b, c) = (q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() < 1e-6 {
            return false;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
}

/// Il punto è dentro il quadrilatero convesso?
pub fn inside_convex(q: &Quad, p: Pt) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        if cross != 0.0 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Retta (direzione, punto) stimata con pesi di Huber, come cv2.fitLine(DIST_HUBER).
fn fit_line_huber(pts: &[Pt]) -> (Pt, Pt) {
    let mut w = vec![1.0f32; pts.len()];
    let (mut dir, mut p0) = ([1.0, 0.0], [0.0, 0.0]);
    for _ in 0..8 {
        let sw: f32 = w.iter().sum();
        let mx = pts.iter().zip(&w).map(|(p, w)| p[0] * w).sum::<f32>() / sw;
        let my = pts.iter().zip(&w).map(|(p, w)| p[1] * w).sum::<f32>() / sw;
        let (mut sxx, mut syy, mut sxy) = (0.0f32, 0.0f32, 0.0f32);
        for (p, w) in pts.iter().zip(&w) {
            let (dx, dy) = (p[0] - mx, p[1] - my);
            sxx += w * dx * dx;
            syy += w * dy * dy;
            sxy += w * dx * dy;
        }
        let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        dir = [th.cos(), th.sin()];
        p0 = [mx, my];
        let n = [-dir[1], dir[0]];
        for (p, w) in pts.iter().zip(w.iter_mut()) {
            let r = ((p[0] - mx) * n[0] + (p[1] - my) * n[1]).abs();
            *w = if r <= 1.345 { 1.0 } else { 1.345 / r };
        }
    }
    (dir, p0)
}

fn line_intersection(l1: (Pt, Pt), l2: (Pt, Pt)) -> Option<Pt> {
    let ((v1, p1), (v2, p2)) = (l1, l2);
    let det = v1[0] * -v2[1] - (-v2[0]) * v1[1];
    if det.abs() < 1e-9 {
        return None;
    }
    let (bx, by) = (p2[0] - p1[0], p2[1] - p1[1]);
    let t = (bx * -v2[1] - (-v2[0]) * by) / det;
    Some([p1[0] + t * v1[0], p1[1] + t * v1[1]])
}

/// Ricostruisce gli spigoli intersecando rette stimate sui lati del contorno.
/// Così gli angoli arrotondati delle tessere non "tagliano" il documento.
pub fn refine_quad_with_lines(contour: &[Pt], quad: &Quad) -> Quad {
    let mut lines = Vec::with_capacity(4);
    for i in 0..4 {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        let l = dist(a, b);
        if l < 1.0 {
            return *quad;
        }
        let u = [(b[0] - a[0]) / l, (b[1] - a[1]) / l];
        let n = [-u[1], u[0]];
        // solo la parte centrale del lato (esclude gli angoli arrotondati)
        let tol = (0.03 * l).max(4.0);
        let sel: Vec<Pt> = contour
            .iter()
            .copied()
            .filter(|p| {
                let r = sub(*p, a);
                let t = r[0] * u[0] + r[1] * u[1];
                let d = (r[0] * n[0] + r[1] * n[1]).abs();
                t > 0.12 * l && t < 0.88 * l && d < tol
            })
            .collect();
        lines.push(if sel.len() < 5 { (u, a) } else { fit_line_huber(&sel) });
    }
    let mut out = [[0.0; 2]; 4];
    for i in 0..4 {
        match line_intersection(lines[(i + 3) % 4], lines[i]) {
            Some(p) => out[i] = p,
            None => return *quad,
        }
    }
    // se il raffinamento ha spostato troppo gli spigoli, si tiene il quadrilatero di partenza
    let (w, h) = quad_sides(quad);
    if (0..4).any(|i| dist(out[i], quad[i]) > 0.15 * w.min(h)) {
        return *quad;
    }
    out
}

pub fn full_quad(w: u32, h: u32) -> Quad {
    let (w, h) = ((w - 1) as f32, (h - 1) as f32);
    [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]
}

pub fn scale_quad(q: &Quad, s: f32) -> Quad {
    q.map(|p| [p[0] * s, p[1] * s])
}
