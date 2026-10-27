"""Finestra di DocPDF e modalità da riga di comando."""
from __future__ import annotations

import os
import subprocess
import sys
import traceback

import cv2
import numpy as np
from PyQt6.QtCore import (QObject, QRectF, QRunnable, QSettings, QSize, Qt,
                          QThreadPool, pyqtSignal)
from PyQt6.QtGui import (QBrush, QColor, QIcon, QImage, QKeySequence, QPainter,
                         QPen, QPixmap, QPolygonF, QShortcut)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
                             QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
                             QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                             QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from . import core

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp")
SCALES = [("Dimensione reale (100%)", 1.0), ("Grande (150%)", 1.5),
          ("Molto grande (200%)", 2.0), ("Adatta alla pagina", None)]


def to_qimage(img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def expand_paths(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            out += sorted(os.path.join(a, f) for f in os.listdir(a) if f.lower().endswith(IMG_EXT))
        elif a.lower().endswith(IMG_EXT) and os.path.isfile(a):
            out.append(a)
    return out


# ---------------------------------------------------------------- elaborazione in background

class _Signals(QObject):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class LoadJob(QRunnable):
    def __init__(self, idx, path):
        super().__init__()
        self.idx, self.path, self.signals = idx, path, _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.idx, core.Item.from_path(self.path))
        except Exception:
            self.signals.failed.emit(self.idx, traceback.format_exc(limit=1))


# ---------------------------------------------------------------- editor degli spigoli

class Handle(QGraphicsEllipseItem):
    def __init__(self, r, on_move):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.on_move = on_move
        self.setBrush(QBrush(QColor(255, 140, 0, 110)))
        self.setPen(QPen(QColor(255, 140, 0), max(2, r / 6)))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(10)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            self.on_move()
        return super().itemChange(change, value)


class ZoomView(QGraphicsView):
    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)


class CropDialog(QDialog):
    """Trascina i 4 cerchi sugli spigoli del documento. Rotella = zoom."""

    def __init__(self, item: core.Item, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ritaglio: trascina i 4 angoli sugli spigoli del documento")
        self.item = item
        self.resize(1100, 800)
        img = item.original
        self.scene = QGraphicsScene(self)
        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(to_qimage(img))))
        self.poly = QGraphicsPolygonItem()
        self.poly.setPen(QPen(QColor(0, 170, 255), max(2, max(img.shape[:2]) / 400)))
        self.poly.setBrush(QBrush(QColor(0, 170, 255, 40)))
        self.scene.addItem(self.poly)
        r = max(img.shape[:2]) / 70
        self.handles = [Handle(r, self._update) for _ in range(4)]
        for h in self.handles:
            self.scene.addItem(h)
        self.view = ZoomView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.preview = QLabel()
        self.preview.setFixedWidth(300)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        b_auto = QPushButton("Rileva di nuovo")
        b_auto.clicked.connect(lambda: self._set(core.detect_document(img, near=self.quad())[0]))
        b_full = QPushButton("Tutta la foto")
        b_full.clicked.connect(lambda: self._set(core.full_quad(img)))
        b_ok = QPushButton("OK")
        b_ok.setDefault(True)
        b_ok.clicked.connect(self.accept)
        b_no = QPushButton("Annulla")
        b_no.clicked.connect(self.reject)

        side = QVBoxLayout()
        side.addWidget(QLabel("Anteprima:"))
        side.addWidget(self.preview)
        side.addStretch()
        side.addWidget(QLabel("Rotella del mouse: zoom\nTrascina lo sfondo: sposta"))
        side.addWidget(b_auto)
        side.addWidget(b_full)
        side.addWidget(b_no)
        side.addWidget(b_ok)
        lay = QHBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(side)
        self._set(item.quad)

    def showEvent(self, e):
        super().showEvent(e)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _set(self, quad):
        for h, (x, y) in zip(self.handles, core.order_corners(quad)):
            h.setPos(float(x), float(y))
        self._update()

    def quad(self) -> np.ndarray:
        return core.order_corners(np.array([[h.pos().x(), h.pos().y()] for h in self.handles], np.float32))

    def _update(self):
        if not hasattr(self, "preview"):
            return
        self.poly.setPolygon(QPolygonF([h.pos() for h in self.handles]))
        q = self.quad()
        if cv2.contourArea(q.reshape(-1, 1, 2)) < 100:
            return
        out = core.warp(self.item.original, q, core.guess_format(q))
        pm = QPixmap.fromImage(to_qimage(out))
        self.preview.setPixmap(pm.scaled(290, 400, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))


# ---------------------------------------------------------------- anteprima pagina

class PagePreview(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.items, self.pos, self.selected = [], [], -1
        self.setMinimumSize(380, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, items, pos, selected):
        self.items, self.pos, self.selected = items, pos, selected
        self.update()

    def _page_rect(self):
        W, H = self.width() - 20, self.height() - 20
        s = min(W / core.A4_W, H / core.A4_H)
        pw, ph = core.A4_W * s, core.A4_H * s
        return QRectF((self.width() - pw) / 2, (self.height() - ph) / 2, pw, ph), s

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(90, 90, 96))
        page, s = self._page_rect()
        p.fillRect(page.translated(3, 3), QColor(0, 0, 0, 80))
        p.fillRect(page, Qt.GlobalColor.white)
        if not self.items:
            p.setPen(QColor(120, 120, 120))
            p.drawText(page, Qt.AlignmentFlag.AlignCenter,
                       "Trascina qui le foto dei documenti\n(oppure premi «Aggiungi foto»)")
        for i, (it, (x, y, w, h)) in enumerate(zip(self.items, self.pos)):
            r = QRectF(page.x() + x * s, page.y() + y * s, w * s, h * s)
            p.drawImage(r, it.thumb)
            if i == self.selected:
                p.setPen(QPen(QColor(0, 140, 255), 3))
                p.drawRect(r.adjusted(-3, -3, 3, 3))

    def mousePressEvent(self, e):
        page, s = self._page_rect()
        for i, (x, y, w, h) in enumerate(self.pos):
            if QRectF(page.x() + x * s, page.y() + y * s, w * s, h * s).contains(e.position()):
                self.clicked.emit(i)
                return


# ---------------------------------------------------------------- finestra principale

class Main(QWidget):
    def __init__(self, paths):
        super().__init__()
        self.setWindowTitle("DocPDF – documenti in un'unica pagina")
        self.setAcceptDrops(True)
        self.resize(1150, 860)
        self.settings = QSettings("docpdf", "docpdf")
        self.items: list[core.Item] = []
        self.pending = 0
        self.out_dir = None
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max(2, min(4, os.cpu_count() or 2)))

        # lista a sinistra
        self.list = QListWidget()
        self.list.setIconSize(QSize(200, 140))
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.setMinimumWidth(240)
        self.list.setSpacing(4)
        self.list.model().rowsMoved.connect(self._reordered)
        self.list.currentRowChanged.connect(self._selection_changed)
        self.list.itemDoubleClicked.connect(lambda _: self.edit_crop())

        def tb(text, tip, slot, key=None):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip + (f"  [{key}]" if key else ""))
            b.clicked.connect(slot)
            b.setMinimumHeight(34)
            if key:
                QShortcut(QKeySequence(key), self, slot)
            return b

        self.b_left = tb("⟲", "Ruota a sinistra", lambda: self.rotate(-90), "Shift+R")
        self.b_right = tb("⟳", "Ruota a destra", lambda: self.rotate(90), "R")
        self.b_flip = tb("↻ 180°", "Capovolgi", lambda: self.rotate(180), "F")
        self.b_crop = tb("✂ Ritaglio", "Correggi gli angoli (doppio clic sulla miniatura)",
                         self.edit_crop, "E")
        self.b_dup = tb("⧉", "Duplica: per aggiungere a mano un altro documento presente nella stessa foto",
                        self.duplicate, "D")
        self.b_del = tb("🗑", "Togli dalla pagina", self.remove, "Delete")
        self.fmt = QComboBox()
        for k, (label, _) in core.FORMATS.items():
            self.fmt.addItem(label, k)
        self.fmt.setToolTip("Formato reale del documento (serve per le proporzioni e la dimensione di stampa)")
        self.fmt.activated.connect(self._fmt_changed)
        self.enh = QCheckBox("Migliora immagine")
        self.enh.setToolTip("Bilancia il bianco, aumenta contrasto e nitidezza")
        self.enh.toggled.connect(self._enh_changed)

        item_bar = QHBoxLayout()
        for w in (self.b_left, self.b_right, self.b_flip, self.b_crop, self.b_dup, self.b_del):
            item_bar.addWidget(w)
        item_bar2 = QHBoxLayout()
        item_bar2.addWidget(self.fmt, 1)
        item_bar2.addWidget(self.enh)

        b_add = QPushButton("＋ Aggiungi foto…")
        b_add.clicked.connect(self.add_dialog)
        QShortcut(QKeySequence("Ctrl+O"), self, self.add_dialog)

        left = QVBoxLayout()
        left.addWidget(b_add)
        left.addWidget(self.list, 1)
        left.addWidget(QLabel("Documento selezionato:"))
        left.addLayout(item_bar)
        left.addLayout(item_bar2)

        # anteprima
        self.preview = PagePreview()
        self.preview.clicked.connect(self.list.setCurrentRow)

        # barra in basso
        self.scale = QComboBox()
        for label, v in SCALES:
            self.scale.addItem(label, v)
        self.scale.setCurrentIndex(int(self.settings.value("scale", 3)))
        self.scale.currentIndexChanged.connect(self._global_changed)
        self.gray = QCheckBox("Bianco e nero")
        self.gray.setChecked(self.settings.value("gray", "false") == "true")
        self.gray.toggled.connect(self._global_changed)
        self.name = QLineEdit()
        self.name.setPlaceholderText("nome del file PDF")
        self.open_after = QCheckBox("Apri dopo il salvataggio")
        self.open_after.setChecked(self.settings.value("open_after", "true") == "true")
        self.b_save = QPushButton("💾  Salva PDF")
        self.b_save.setMinimumHeight(40)
        self.b_save.setStyleSheet("font-weight: bold; font-size: 14px; padding: 0 18px;")
        self.b_save.clicked.connect(self.save)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save)
        self.status = QLabel()

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Dimensione:"))
        bottom.addWidget(self.scale)
        bottom.addWidget(self.gray)
        bottom.addSpacing(20)
        bottom.addWidget(QLabel("File:"))
        bottom.addWidget(self.name, 1)
        bottom.addWidget(QLabel(".pdf"))
        bottom.addWidget(self.open_after)
        bottom.addWidget(self.b_save)

        top = QHBoxLayout()
        top.addLayout(left)
        top.addWidget(self.preview, 1)
        root = QVBoxLayout(self)
        root.addLayout(top, 1)
        root.addLayout(bottom)
        root.addWidget(self.status)

        self._update_controls()
        if paths:
            self.add_paths(paths)

    # ------------------------------------------------------------ caricamento
    def add_paths(self, paths):
        paths = expand_paths(paths)
        if not paths:
            return
        if self.out_dir is None:
            self.out_dir = os.path.dirname(os.path.abspath(paths[0]))
            if not self.name.text():
                self.name.setText(self._default_name())
        # ogni aggiunta è un "lotto": i documenti si mettono in pagina quando il lotto è completo,
        # così si può abbinare fronte e retro tra foto diverse
        batch = {"groups": [None] * len(paths), "left": len(paths)}
        for k, p in enumerate(paths):
            job = LoadJob(k, p)
            job.signals.done.connect(lambda i, its, b=batch: self._loaded(b, i, its))
            job.signals.failed.connect(lambda i, msg, b=batch: self._failed(b, i, msg))
            self.pending += 1
            self.pool.start(job)
        self._set_status()

    def _default_name(self):
        folder = os.path.basename(self.out_dir.rstrip("/")) or "documenti"
        return f"{folder}_documenti"

    def _loaded(self, batch, idx, items):
        for it in items:
            it.thumb = to_qimage(it.result())
        batch["groups"][idx] = items
        self._batch_step(batch)

    def _failed(self, batch, idx, msg):
        batch["groups"][idx] = []
        self._batch_step(batch)
        QMessageBox.warning(self, "Errore", f"Impossibile elaborare un'immagine:\n{msg}")

    def _batch_step(self, batch):
        self.pending -= 1
        batch["left"] -= 1
        if batch["left"] == 0:
            first_new = len(self.items)
            self.items += core.order_groups(batch["groups"])
            self._rebuild_list(first_new if first_new < len(self.items) else None)
        self._set_status()

    def _rebuild_list(self, select=None):
        row = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        for it in self.items:
            li = QListWidgetItem(QIcon(QPixmap.fromImage(it.thumb)), "")
            li.setData(Qt.ItemDataRole.UserRole, id(it))
            li.setToolTip(it.path)
            if it.confidence < 0.45:
                li.setToolTip("Rilevamento incerto: controlla il ritaglio\n" + it.path)
                li.setForeground(QColor(220, 120, 0))
            self.list.addItem(li)
        self.list.blockSignals(False)
        self.list.setCurrentRow(min(max(row, 0), len(self.items) - 1))
        self.refresh()

    def _set_status(self, extra=""):
        if self.pending:
            self.status.setText(f"⏳ Elaborazione di {self.pending} foto in corso…")
        else:
            uncertain = sum(1 for it in self.items if it.confidence < 0.45)
            msg = f"{len(self.items)} documenti sulla pagina."
            if uncertain:
                msg += f"  ⚠ {uncertain} con ritaglio incerto (in arancione): controllali."
            self.status.setText(extra or msg)

    def add_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Scegli le foto", self.out_dir or os.path.expanduser("~"),
            "Immagini (" + " ".join("*" + e for e in IMG_EXT) + ")")
        self.add_paths(files)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls()])

    # ------------------------------------------------------------ azioni sul documento
    def current(self):
        r = self.list.currentRow()
        return self.items[r] if 0 <= r < len(self.items) else None

    def _changed_current(self):
        it = self.current()
        if not it:
            return
        it.thumb = to_qimage(it.result())
        self.list.currentItem().setIcon(QIcon(QPixmap.fromImage(it.thumb)))
        self.refresh()

    def rotate(self, deg):
        it = self.current()
        if it:
            it.rotation = (it.rotation + deg) % 360
            self._changed_current()

    def edit_crop(self):
        it = self.current()
        if not it:
            return
        dlg = CropDialog(it, self)
        if dlg.exec():
            q = dlg.quad()
            old_landscape = (lambda w, h: w >= h)(*core.quad_sides(core.order_corners(it.quad)))
            it.quad = q
            it.fmt = core.guess_format(q)
            it.confidence = 1.0
            # se cambia l'orientamento del ritaglio si ricalcola la rotazione
            if (lambda w, h: w >= h)(*core.quad_sides(q)) != old_landscape:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                it.auto_orient()
                QApplication.restoreOverrideCursor()
            self.list.currentItem().setForeground(self.list.palette().text().color())
            self.list.currentItem().setToolTip("")
            self._changed_current()
            self._update_controls()
            self._set_status()

    def duplicate(self):
        it = self.current()
        if not it:
            return
        r = self.list.currentRow() + 1
        new = it.copy()
        new.thumb = it.thumb
        self.items.insert(r, new)
        self._rebuild_list(r)
        self.edit_crop()

    def remove(self):
        r = self.list.currentRow()
        if 0 <= r < len(self.items):
            del self.items[r]
            self.list.takeItem(r)
            self.refresh()
            self._set_status()

    def _fmt_changed(self):
        it = self.current()
        if it:
            it.fmt = self.fmt.currentData()
            self._changed_current()

    def _enh_changed(self, on):
        it = self.current()
        if it and it.enhance != on:
            it.enhance = on
            self._changed_current()

    def _reordered(self):
        ids = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        by_id = {id(it): it for it in self.items}
        self.items = [by_id[i] for i in ids]
        self.refresh()

    def _selection_changed(self, _):
        self._update_controls()
        self.refresh()

    def _update_controls(self):
        it = self.current()
        for w in (self.b_left, self.b_right, self.b_flip, self.b_crop, self.b_dup, self.b_del, self.fmt, self.enh):
            w.setEnabled(it is not None)
        if it:
            self.fmt.setCurrentIndex(self.fmt.findData(it.fmt))
            self.enh.blockSignals(True)
            self.enh.setChecked(it.enhance)
            self.enh.blockSignals(False)
        self.b_save.setEnabled(bool(self.items))

    def _global_changed(self):
        self.settings.setValue("scale", self.scale.currentIndex())
        self.settings.setValue("gray", "true" if self.gray.isChecked() else "false")
        self.refresh()

    def refresh(self):
        pos = core.layout([it.size_mm() for it in self.items], self.scale.currentData())
        self.preview.set_data(self.items, pos, self.list.currentRow())
        self._update_controls()

    # ------------------------------------------------------------ salvataggio
    def save(self):
        if not self.items:
            return
        if self.pending:
            QMessageBox.information(self, "Attendi", "Alcune foto sono ancora in elaborazione.")
            return
        name = self.name.text().strip() or self._default_name()
        if name.lower().endswith(".pdf"):
            name = name[:-4]
        out = os.path.join(self.out_dir, name + ".pdf")
        if os.path.exists(out):
            ans = QMessageBox.question(
                self, "File esistente", f"«{os.path.basename(out)}» esiste già. Sovrascriverlo?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            core.make_pdf(self.items, out, self.scale.currentData(), self.gray.isChecked())
        except Exception as ex:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Errore", f"Salvataggio non riuscito:\n{ex}")
            return
        QApplication.restoreOverrideCursor()
        self.settings.setValue("open_after", "true" if self.open_after.isChecked() else "false")
        self._set_status(f"✅ Salvato: {out}")
        if self.open_after.isChecked():
            subprocess.Popen(["xdg-open", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    args = sys.argv[1:]
    if "--auto" in args:
        # modalità senza finestra: docpdf --auto foto1 foto2 ... [-o uscita.pdf]
        args.remove("--auto")
        out = None
        if "-o" in args:
            i = args.index("-o")
            out = args[i + 1]
            del args[i:i + 2]
        paths = expand_paths(args)
        items = core.order_groups([core.Item.from_path(p) for p in paths])
        if not out:
            d = os.path.dirname(os.path.abspath(paths[0]))
            out = os.path.join(d, os.path.basename(d) + "_documenti.pdf")
        core.make_pdf(items, out, None)
        print(out)
        return
    app = QApplication(sys.argv)
    app.setApplicationName("DocPDF")
    w = Main(args)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
