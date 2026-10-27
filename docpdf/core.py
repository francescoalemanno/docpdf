"""Motore di DocPDF: rilevamento del documento, raddrizzamento, orientamento, impaginazione PDF."""
from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

# Formati noti (lato lungo x lato corto, mm)
FORMATS = {
    "card": ("Tessera / Carta d'identità", (85.6, 54.0)),
    "passport": ("Pagina passaporto", (125.0, 88.0)),
    "a4": ("Foglio A4", (297.0, 210.0)),
    "a5": ("Foglio A5", (210.0, 148.0)),
    "free": ("Libero", None),
}
CARD_RATIO = 85.6 / 54.0
A4_W, A4_H = 210.0, 297.0
DPI = 300


# ---------------------------------------------------------------- caricamento

def load_image(path: str) -> np.ndarray:
    """Restituisce un'immagine BGR, con l'orientamento EXIF già applicato."""
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        # HEIC e formati esotici: si passa da ImageMagick
        out = subprocess.run(["magick", path + "[0]", "-auto-orient", "png:-"],
                             capture_output=True, check=True).stdout
        im = Image.open(io.BytesIO(out)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- geometria

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordina 4 punti: alto-sx, alto-dx, basso-dx, basso-sx."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # senso orario partendo da sinistra-alto circa
    start = np.argmin(pts.sum(axis=1))
    return np.roll(pts, -start, axis=0)


def quad_sides(q: np.ndarray):
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return w, h


def _line_intersection(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    a = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t = np.linalg.solve(a, np.array([x2 - x1, y2 - y1], dtype=np.float64))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1])


def refine_quad_with_lines(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Ricostruisce gli spigoli intersecando rette stimate sui lati del contorno.
    Così gli angoli arrotondati delle tessere non 'tagliano' il documento."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1:
            return quad
        u = ab / L
        n = np.array([-u[1], u[0]])
        rel = pts - a
        t = rel @ u
        d = np.abs(rel @ n)
        # solo la parte centrale del lato (esclude gli angoli arrotondati)
        sel = (t > 0.12 * L) & (t < 0.88 * L) & (d < max(4.0, 0.03 * L))
        if sel.sum() < 5:
            lines.append((u[0], u[1], a[0], a[1]))
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts[sel], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((vx, vy, x0, y0))
    out = []
    for i in range(4):
        p = _line_intersection(lines[i - 1], lines[i])
        if p is None:
            return quad
        out.append(p)
    out = np.array(out, dtype=np.float32)
    # se il raffinamento ha spostato troppo gli spigoli, si tiene il quadrilatero di partenza
    w, h = quad_sides(quad)
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.15 * min(w, h):
        return quad
    return out


# ---------------------------------------------------------------- rilevamento

def _edge_strength(gradmag: np.ndarray, q: np.ndarray) -> float:
    """Media del gradiente lungo il perimetro (parte centrale dei lati)."""
    vals = []
    H, W = gradmag.shape
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        for t in np.linspace(0.15, 0.85, 40):
            x, y = a + (b - a) * t
            xi, yi = int(round(x)), int(round(y))
            if 1 <= xi < W - 1 and 1 <= yi < H - 1:
                vals.append(gradmag[yi - 1:yi + 2, xi - 1:xi + 2].max())
            else:
                vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def _contour_candidates(mask: np.ndarray, min_area: float):
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(c)
        if area < min_area:
            break
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.02, 0.03, 0.04, 0.06):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(4, 2).astype(np.float32)
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        quad = order_corners(quad)
        quad = refine_quad_with_lines(hull, quad)
        # riempimento: quanto il contorno somiglia al quadrilatero
        qa = cv2.contourArea(quad.reshape(-1, 1, 2))
        fill = area / qa if qa > 0 else 0
        out.append((quad, fill))
    return out


def _scored_candidates(cands, gnorm, w, h):
    out = []
    for q, fill in cands:
        if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.int32)):
            continue
        area = cv2.contourArea(q.reshape(-1, 1, 2))
        frac = area / (w * h)
        if frac < 0.04 or frac > 0.985:
            continue
        sw, sh = quad_sides(q)
        ratio = max(sw, sh) / max(1e-6, min(sw, sh))
        if ratio > 3.0:
            continue
        # angoli vicini a 90° (tolleranza per la prospettiva)
        angs = []
        for i in range(4):
            v1, v2 = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
            cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angs.append(abs(cosv))
        if max(angs) > 0.45:
            continue
        edge = min(1.0, _edge_strength(gnorm, q))
        rect_score = 1.0 - max(angs)
        fmt_bonus = max(np.exp(-((ratio - r) / 0.06) ** 2) for r in (CARD_RATIO, 297 / 210, 125 / 88))
        # non a ridosso del bordo foto (tipico dei falsi positivi che inquadrano tutto)
        margin = min(q[:, 0].min(), q[:, 1].min(), w - q[:, 0].max(), h - q[:, 1].max())
        border_pen = 0.15 if margin < 3 else 0.0
        score = (0.45 * edge + 0.2 * min(fill, 1.0) + 0.15 * rect_score
                 + 0.12 * fmt_bonus + 0.08 * min(1.0, frac / 0.25) - border_pen)
        out.append((score, q))
    out.sort(key=lambda t: -t[0])
    return out


def _best_candidate(cands, gnorm, w, h):
    sc = _scored_candidates(cands, gnorm, w, h)
    return (sc[0][1], sc[0][0]) if sc else (None, -1.0)


def _overlap(q1, q2, w, h) -> float:
    """Frazione del quadrilatero più piccolo coperta dall'altro."""
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m1, [q1.astype(np.int32)], 1)
    cv2.fillPoly(m2, [q2.astype(np.int32)], 1)
    inter = np.count_nonzero(m1 & m2)
    return inter / max(1, min(np.count_nonzero(m1), np.count_nonzero(m2)))


def _pick_documents(scored, w, h):
    """Sceglie più documenti non sovrapposti (es. due tessere nella stessa foto)."""
    if not scored:
        return []
    best = scored[0][0]
    chosen = []
    for score, q in scored:
        if score < max(0.42, 0.72 * best):
            break
        if any(_overlap(q, c, w, h) > 0.15 for _, c in chosen):
            continue
        chosen.append((score, q))
        if len(chosen) >= 6:
            break
    return chosen


def detect_documents(img: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Trova i documenti nella foto (anche più d'uno), ordinati dall'alto in basso e da sinistra a destra.
    Restituisce [(4 spigoli in coordinate originali, punteggio 0..1)]."""
    H, W = img.shape[:2]
    scale = 900.0 / max(H, W)
    small = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    img_area = h * w
    min_area = 0.04 * img_area

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    channels = [gray, lab[:, :, 1], lab[:, :, 2], hsv[:, :, 1]]

    gx = cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F, 0, 1)
    gradmag = cv2.magnitude(gx, gy)
    # gradiente anche sul colore: i documenti spesso si staccano dallo sfondo per tinta
    for ch in channels[1:]:
        b = cv2.GaussianBlur(ch, (5, 5), 0)
        gradmag = np.maximum(gradmag, 1.5 * cv2.magnitude(cv2.Sobel(b, cv2.CV_32F, 1, 0),
                                                          cv2.Sobel(b, cv2.CV_32F, 0, 1)))
    gnorm = gradmag / (np.percentile(gradmag, 99.5) + 1e-6)

    cands = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for ch in channels:
        blur = cv2.bilateralFilter(ch, 9, 50, 50)
        med = np.median(blur)
        for lo, hi in ((0.33, 1.0), (0.15, 0.6), (0.6, 1.6)):
            e = cv2.Canny(blur, max(5, int(lo * med)), max(15, int(hi * med)))
            e = cv2.dilate(e, kernel, iterations=1)
            e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel, iterations=2)
            # i contorni chiusi delimitano regioni: riempiamo e cerchiamo le più grandi
            cands += _contour_candidates(e, min_area)
        # soglia di Otsu: separa documento chiaro/scuro da sfondo uniforme
        _, th = cv2.threshold(cv2.GaussianBlur(ch, (7, 7), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for m in (th, 255 - th):
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=2)
            cands += _contour_candidates(m, min_area)

    best, best_score = _best_candidate(cands, gnorm, w, h)
    if best_score < 0.5:
        # GrabCut inizializzato sulla zona centrale: utile quando i bordi sono deboli
        try:
            mask = np.zeros((h, w), np.uint8)
            rect = (int(w * 0.04), int(h * 0.04), int(w * 0.92), int(h * 0.92))
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(small, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
            fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=3)
            cands += _contour_candidates(fg, min_area)
        except cv2.error:
            pass

    chosen = _pick_documents(_scored_candidates(cands, gnorm, w, h), w, h)
    if not chosen:
        return [(full_quad(img), 0.0)]
    # ordine di lettura: per righe (dall'alto), poi da sinistra
    chosen.sort(key=lambda t: (round(t[1][:, 1].mean() / (h * 0.15)), t[1][:, 0].mean()))
    return [((q / scale).astype(np.float32), float(sc)) for sc, q in chosen]


def detect_document(img: np.ndarray, near: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Un solo documento: il migliore, oppure quello più vicino a `near`."""
    found = detect_documents(img)
    if near is None or len(found) == 1:
        return max(found, key=lambda t: t[1])
    c = np.asarray(near).reshape(4, 2).mean(axis=0)
    return min(found, key=lambda t: np.linalg.norm(t[0].mean(axis=0) - c))


def full_quad(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    return np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)


def guess_format(quad: np.ndarray) -> str:
    w, h = quad_sides(quad)
    r = max(w, h) / max(1e-6, min(w, h))
    if abs(r - CARD_RATIO) < 0.12:
        return "card"
    if abs(r - 297 / 210) < 0.06:
        return "a4"
    return "free"


def warp(img: np.ndarray, quad: np.ndarray, fmt: str) -> np.ndarray:
    """Raddrizza la prospettiva. Per i formati noti forza le proporzioni esatte."""
    q = order_corners(quad)
    w, h = quad_sides(q)
    size = FORMATS[fmt][1]
    if size:
        long_mm, short_mm = size
        long_px = max(w, h)
        long_px = min(max(long_px, 600), 4000)
        if w >= h:
            ow, oh = long_px, long_px * short_mm / long_mm
        else:
            ow, oh = long_px * short_mm / long_mm, long_px
    else:
        ow, oh = w, h
    ow, oh = int(round(ow)), int(round(oh))
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- pulizia

def enhance(img: np.ndarray) -> np.ndarray:
    """Bilanciamento del bianco, contrasto e nitidezza, senza snaturare i colori."""
    f = img.astype(np.float32)
    # bianco: i pixel più chiari diventano neutri
    lum = f.mean(axis=2)
    thr = np.percentile(lum, 92)
    bright = f[lum >= thr]
    if len(bright) > 50:
        ref = bright.mean(axis=0)
        gain = ref.mean() / np.maximum(ref, 1)
        gain = np.clip(gain, 0.8, 1.25)
        f *= gain
    # stiramento dei livelli
    lo = np.percentile(f, 0.7)
    hi = np.percentile(f, 99.3)
    if hi - lo > 30:
        f = (f - lo) * (255.0 / (hi - lo))
    f = np.clip(f, 0, 255).astype(np.uint8)
    # contrasto locale leggero sulla luminanza
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    f = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # nitidezza (unsharp mask)
    blur = cv2.GaussianBlur(f, (0, 0), 1.6)
    return cv2.addWeighted(f, 1.4, blur, -0.4, 0)


def round_corners(img: np.ndarray, radius_frac: float) -> np.ndarray:
    """Imbianca gli angoli fuori dal bordo arrotondato della tessera."""
    h, w = img.shape[:2]
    r = int(round(radius_frac * max(w, h)))
    if r < 2:
        return img
    mask = np.full((h, w), 255, np.uint8)
    for cx, cy, x0, y0 in ((r, r, 0, 0), (w - r - 1, r, w - r - 1, 0),
                           (r, h - r - 1, 0, h - r - 1), (w - r - 1, h - r - 1, w - r - 1, h - r - 1)):
        mask[y0:y0 + r + 1, x0:x0 + r + 1] = 0
        cv2.circle(mask, (cx, cy), r, 255, -1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[:, :, None] / 255.0
    return (img * mask + 255 * (1 - mask)).astype(np.uint8)


# ---------------------------------------------------------------- orientamento

_LANGS = None


def _tesseract_langs() -> str:
    """Lingue OCR disponibili tra italiano e inglese (es. "ita+eng")."""
    global _LANGS
    if _LANGS is None:
        try:
            out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
            have = {l.strip() for l in out.splitlines()[1:]}
        except OSError:
            have = set()
        _LANGS = "+".join(l for l in ("ita", "eng") if l in have) or "osd"
    return _LANGS


def _ocr(img: np.ndarray) -> tuple[float, str]:
    """(somma delle confidenze delle parole riconosciute da Tesseract, testo letto)."""
    h, w = img.shape[:2]
    s = 1600.0 / max(h, w)
    g = cv2.cvtColor(cv2.resize(img, (int(w * s), int(h * s))), cv2.COLOR_BGR2GRAY)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        cv2.imwrite(tf.name, g)
        name = tf.name
    try:
        out = subprocess.run(["tesseract", name, "-", "-l", _tesseract_langs(), "--psm", "11", "tsv"],
                             capture_output=True, text=True, timeout=60,
                             env={**os.environ, "OMP_THREAD_LIMIT": "1"}).stdout
    except Exception:
        return 0.0, ""
    finally:
        os.unlink(name)
    score, words = 0.0, []
    for line in out.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            conf = float(parts[10])
        except ValueError:
            continue
        word = parts[11].strip()
        if conf > 30:
            words.append(word)
        if conf > 50 and len(re.sub(r"[^A-Za-z0-9]", "", word)) >= 3:
            score += conf * min(len(word), 12)
    return score, " ".join(words)


BACK_HINTS = ("<<", "EUROPEA DI ASSICURAZIONE", "EUROPEAN HEALTH", "ESTREMI ATTO", "PADRE E DELLA MADRE",
              "INDIRIZZO DI RESIDENZA")
FRONT_HINTS = ("REPUBBLICA ITALIANA", "CARTA DI IDENTIT", "IDENTITY CARD", "TESSERA SANITARIA",
               "PATENTE DI GUIDA", "COGNOME / SURNAME", "SURNAME")


def guess_side(text: str) -> int:
    """-1 = fronte, +1 = retro, 0 = non si sa."""
    t = re.sub(r"\s+", " ", text.upper())
    back = sum(h in t for h in BACK_HINTS)
    front = sum(h in t for h in FRONT_HINTS)
    return (back > front) - (front > back)


def auto_rotation(img: np.ndarray, landscape: bool | None) -> tuple[int, str]:
    """Rotazione (gradi, multipli di 90 in senso orario) che rende il testo leggibile, e il testo letto.
    landscape=True/False forza l'orientamento del formato; None prova tutte e quattro."""
    h, w = img.shape[:2]
    if landscape is None:
        opts = [0, 90, 180, 270]
    else:
        base = 0 if (w >= h) == landscape else 90
        opts = [base, base + 180]
    with ThreadPoolExecutor(len(opts)) as ex:
        res = list(ex.map(lambda r: _ocr(rotate(img, r)), opts))
    best = int(np.argmax([sc for sc, _ in res]))
    if res[best][0] <= 0:
        return opts[0], ""
    return opts[best] % 360, res[best][1]


def rotate(img: np.ndarray, deg: int) -> np.ndarray:
    deg %= 360
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ---------------------------------------------------------------- elemento

@dataclass
class Item:
    path: str
    original: np.ndarray
    quad: np.ndarray
    fmt: str = "free"
    rotation: int = 0
    enhance: bool = True
    confidence: float = 0.0
    side: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_path(cls, path: str) -> list["Item"]:
        """Tutti i documenti trovati nella foto."""
        img = load_image(path)
        items = []
        for quad, conf in detect_documents(img):
            it = cls(path=path, original=img, quad=quad, confidence=conf)
            it.fmt = guess_format(quad)
            it.auto_orient()
            items.append(it)
        return items

    def auto_orient(self):
        flat = warp(self.original, self.quad, self.fmt)
        landscape = True if self.fmt == "card" else (False if self.fmt in ("a4", "a5") else None)
        self.rotation, text = auto_rotation(flat, landscape)
        self.side = guess_side(text)

    def copy(self) -> "Item":
        return Item(path=self.path, original=self.original, quad=self.quad.copy(), fmt=self.fmt,
                    rotation=self.rotation, enhance=self.enhance, confidence=self.confidence,
                    side=self.side)

    def result(self) -> np.ndarray:
        key = (self.quad.tobytes(), self.fmt, self.rotation, self.enhance)
        if self._cache.get("key") != key:
            out = warp(self.original, self.quad, self.fmt)
            if self.enhance:
                out = enhance(out)
            if self.fmt == "card":
                out = round_corners(out, 3.18 / 85.6)
            self._cache = {"key": key, "img": rotate(out, self.rotation)}
        return self._cache["img"]

    def size_mm(self) -> tuple[float, float]:
        """Dimensione fisica (larghezza, altezza) dopo la rotazione."""
        img = self.result()
        h, w = img.shape[:2]
        size = FORMATS[self.fmt][1]
        if size:
            long_mm, short_mm = size
        else:
            long_mm = 180.0
            short_mm = long_mm * min(w, h) / max(w, h)
        return (long_mm, short_mm) if w >= h else (short_mm, long_mm)


# ---------------------------------------------------------------- impaginazione

def order_groups(groups: list[list[Item]]) -> list[Item]:
    """Mette in fila i documenti trovati in più foto.
    Caso tipico: foto dei fronti + foto dei retri con lo stesso numero di documenti
    -> fronte e retro di ciascun documento uno accanto all'altro."""
    groups = [g for g in groups if g]
    if len(groups) == 2 and len(groups[0]) == len(groups[1]):
        # coppie fronte/retro, il fronte per primo
        return [it for pair in zip(*groups) for it in sorted(pair, key=lambda i: i.side)]
    return [it for g in groups for it in g]


def _pack(sizes, scale, avail_w, gap):
    """Disposizione a righe; restituisce lista di righe [(indice, w, h)] e altezza totale."""
    rows, row, row_w = [], [], 0.0
    for i, (w, h) in enumerate(sizes):
        w, h = w * scale, h * scale
        if row and row_w + gap + w > avail_w:
            rows.append(row)
            row, row_w = [], 0.0
        row.append((i, w, h))
        row_w += (gap if len(row) > 1 else 0) + w
    if row:
        rows.append(row)
    total_h = sum(max(h for _, _, h in r) for r in rows) + gap * (len(rows) - 1)
    max_w = max(sum(w for _, w, _ in r) + gap * (len(r) - 1) for r in rows)
    return rows, total_h, max_w


def layout(sizes, max_scale, margin=12.0, gap=8.0):
    """Posizioni (x, y, w, h) in mm su A4 verticale, origine in alto a sinistra."""
    if not sizes:
        return []
    aw, ah = A4_W - 2 * margin, A4_H - 2 * margin
    lo, hi = 0.05, (max_scale if max_scale else 20.0)
    rows, th, mw = _pack(sizes, hi, aw, gap)
    if th > ah or mw > aw:
        for _ in range(40):
            mid = (lo + hi) / 2
            r, t, m = _pack(sizes, mid, aw, gap)
            if t <= ah and m <= aw:
                lo = mid
            else:
                hi = mid
        rows, th, mw = _pack(sizes, lo, aw, gap)
    # centrato verticalmente (leggermente verso l'alto se c'è molto spazio)
    free = ah - th
    y = margin + (free / 2 if free < 60 else min(free / 2, 25.0))
    pos = [None] * len(sizes)
    for r in rows:
        rh = max(h for _, _, h in r)
        rw = sum(w for _, w, _ in r) + gap * (len(r) - 1)
        x = margin + (aw - rw) / 2
        for i, w, h in r:
            pos[i] = (x, y + (rh - h) / 2, w, h)
            x += w + gap
        y += rh + gap
    return pos


def make_pdf(items: list[Item], out_path: str, max_scale: float | None, grayscale: bool = False):
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    sizes = [it.size_mm() for it in items]
    pos = layout(sizes, max_scale)
    c = canvas.Canvas(out_path, pagesize=(A4_W * mm, A4_H * mm))
    c.setTitle(os.path.splitext(os.path.basename(out_path))[0])
    for it, (x, y, w, h) in zip(items, pos):
        img = it.result()
        tw, th = int(w / 25.4 * DPI), int(h / 25.4 * DPI)
        if img.shape[1] > tw:
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if grayscale:
            pil = pil.convert("L")
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=88, optimize=True)
        buf.seek(0)
        c.drawImage(ImageReader(buf), x * mm, (A4_H - y - h) * mm, w * mm, h * mm)
    c.showPage()
    c.save()
