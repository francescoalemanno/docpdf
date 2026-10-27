//! Finestra di DocPDF.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

use anyhow::anyhow;
use eframe::egui::{
    self, Align, Color32, ColorImage, CornerRadius, CursorIcon, Frame, Id, Key, Layout, Modifiers, Pos2, Rect,
    Sense, Shape, Stroke, StrokeKind, TextureHandle, TextureOptions, Vec2, pos2, vec2,
};
use image::RgbImage;
use rayon::prelude::*;

use crate::detect::detect_document;
use crate::geom::{Quad, full_quad, is_convex, order_corners, polygon_area};
use crate::imgops::{self, Fmt, guess_format};
use crate::item::{Item, UNCERTAIN};
use crate::layout::{A4_H, A4_W, layout, order_groups};

const SCALES: [(&str, Option<f32>); 4] = [
    ("Dimensione reale (100%)", Some(1.0)),
    ("Grande (150%)", Some(1.5)),
    ("Molto grande (200%)", Some(2.0)),
    ("Adatta alla pagina", None),
];
/// Lato lungo delle anteprime dei documenti, in pixel.
const THUMB: u32 = 640;
const ORANGE: Color32 = Color32::from_rgb(230, 130, 0);
const BLUE: Color32 = Color32::from_rgb(0, 140, 255);

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct Settings {
    scale: usize,
    gray: bool,
    open_after: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { scale: 3, gray: false, open_after: true }
    }
}

/// Un documento in pagina, con la sua anteprima.
struct Doc {
    item: Item,
    /// Anteprima non ancora ruotata.
    preview: RgbImage,
    tex: Option<TextureHandle>,
    gray_tex: Option<TextureHandle>,
}

impl Doc {
    fn new(item: Item, preview: RgbImage) -> Doc {
        Doc { item, preview, tex: None, gray_tex: None }
    }

    fn refresh_preview(&mut self) {
        self.preview = self.item.preview(THUMB);
        self.invalidate();
    }

    fn invalidate(&mut self) {
        self.tex = None;
        self.gray_tex = None;
    }

    fn texture(&mut self, ctx: &egui::Context, gray: bool) -> &TextureHandle {
        let slot = if gray { &mut self.gray_tex } else { &mut self.tex };
        slot.get_or_insert_with(|| {
            let mut img = imgops::rotate(&self.preview, self.item.rotation);
            if gray {
                img = image::DynamicImage::ImageLuma8(image::imageops::grayscale(&img)).into_rgb8();
            }
            ctx.load_texture("doc", to_color_image(&img), TextureOptions::LINEAR)
        })
    }
}

fn to_color_image(img: &RgbImage) -> ColorImage {
    ColorImage::from_rgb([img.width() as usize, img.height() as usize], img.as_raw())
}

enum Msg {
    Loaded { batch: u64, idx: usize, result: Result<Vec<(Item, RgbImage)>, String> },
    Saved(Result<PathBuf, String>),
}

/// Foto aggiunte insieme: vanno in pagina quando sono tutte pronte, così si possono abbinare
/// fronte e retro tra foto diverse.
struct Batch {
    id: u64,
    groups: Vec<Option<Vec<(Item, RgbImage)>>>,
    left: usize,
}

enum Action {
    Select(usize),
    Rotate(u32),
    Crop,
    Duplicate,
    Remove,
    SetFormat(Fmt),
    SetEnhance(bool),
    Move { from: usize, to: usize },
    Add,
    Save,
}

/// Editor degli spigoli: si trascinano i 4 cerchi sugli spigoli del documento.
struct Crop {
    idx: usize,
    /// Documento appena duplicato: se si annulla, si toglie.
    is_new: bool,
    photo: TextureHandle,
    photo_size: Vec2,
    quad: Quad,
    zoom: f32,
    pan: Vec2,
    dragging: Option<usize>,
    preview: Option<(Quad, TextureHandle)>,
}

pub struct App {
    docs: Vec<Doc>,
    selected: Option<usize>,
    batches: Vec<Batch>,
    next_batch: u64,
    pending: usize,
    saving: bool,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    out_dir: Option<PathBuf>,
    name: String,
    settings: Settings,
    message: Option<String>,
    error: Option<String>,
    confirm_overwrite: Option<PathBuf>,
    crop: Option<Crop>,
}

pub fn run(paths: Vec<PathBuf>) -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("DocPDF – documenti in un'unica pagina")
            .with_app_id("docpdf")
            .with_inner_size([1150.0, 860.0])
            .with_min_inner_size([760.0, 560.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("DocPDF", options, Box::new(move |cc| Ok(Box::new(App::new(cc, paths)))))
        .map_err(|e| anyhow!("impossibile aprire la finestra: {e}"))
}

impl App {
    fn new(cc: &eframe::CreationContext, paths: Vec<PathBuf>) -> App {
        let settings = cc.storage.and_then(|s| eframe::get_value(s, "impostazioni")).unwrap_or_default();
        let (tx, rx) = channel();
        let mut app = App {
            docs: Vec::new(),
            selected: None,
            batches: Vec::new(),
            next_batch: 0,
            pending: 0,
            saving: false,
            tx,
            rx,
            out_dir: None,
            name: String::new(),
            settings,
            message: None,
            error: None,
            confirm_overwrite: None,
            crop: None,
        };
        app.add_paths(paths, &cc.egui_ctx);
        app
    }

    // ------------------------------------------------------------ caricamento

    fn add_paths(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        let paths = crate::expand_paths(&paths);
        if paths.is_empty() {
            return;
        }
        if self.out_dir.is_none() {
            let dir = crate::photo_dir(&paths[0]);
            if self.name.is_empty() {
                self.name = crate::default_name(&dir);
            }
            self.out_dir = Some(dir);
        }
        let id = self.next_batch;
        self.next_batch += 1;
        self.batches.push(Batch { id, groups: paths.iter().map(|_| None).collect(), left: paths.len() });
        self.pending += paths.len();
        self.message = None;
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        std::thread::spawn(move || {
            crate::orient::init();
            paths.par_iter().enumerate().for_each(|(idx, p)| {
                let result = Item::from_path(p)
                    .map(|items| {
                        items
                            .into_par_iter()
                            .map(|it| {
                                let pv = it.preview(THUMB);
                                (it, pv)
                            })
                            .collect()
                    })
                    .map_err(|e| format!("{}: {e:#}", p.display()));
                let _ = tx.send(Msg::Loaded { batch: id, idx, result });
                ctx.request_repaint();
            });
        });
    }

    fn add_dialog(&mut self, ctx: &egui::Context) {
        let mut dlg = rfd::FileDialog::new().set_title("Scegli le foto").add_filter("Immagini", crate::IMG_EXT);
        if let Some(d) = self.out_dir.clone().or_else(std::env::home_dir) {
            dlg = dlg.set_directory(d);
        }
        if let Some(files) = dlg.pick_files() {
            self.add_paths(files, ctx);
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Loaded { batch, idx, result } => {
                    let Some(bi) = self.batches.iter().position(|b| b.id == batch) else { continue };
                    let b = &mut self.batches[bi];
                    b.groups[idx] = Some(result.unwrap_or_else(|e| {
                        self.error = Some(format!("Impossibile elaborare un'immagine:\n{e}"));
                        Vec::new()
                    }));
                    b.left -= 1;
                    self.pending -= 1;
                    if b.left == 0 {
                        let b = self.batches.remove(bi);
                        let groups = b.groups.into_iter().map(Option::unwrap_or_default).collect();
                        let first_new = self.docs.len();
                        self.docs.extend(
                            order_groups(groups, |(it, _): &(Item, RgbImage)| (it.side, it.kind))
                                .into_iter()
                                .map(|(it, pv)| Doc::new(it, pv)),
                        );
                        if first_new < self.docs.len() {
                            self.selected = Some(first_new);
                        }
                    }
                }
                Msg::Saved(result) => {
                    self.saving = false;
                    match result {
                        Ok(out) => {
                            self.message = Some(format!("✔ Salvato: {}", out.display()));
                            if self.settings.open_after {
                                let _ = std::process::Command::new("xdg-open").arg(&out).spawn();
                            }
                        }
                        Err(e) => self.error = Some(format!("Salvataggio non riuscito:\n{e}")),
                    }
                }
            }
            ctx.request_repaint();
        }
    }

    // ------------------------------------------------------------ azioni

    fn apply(&mut self, action: Action, ctx: &egui::Context) {
        match action {
            Action::Select(i) => self.selected = Some(i),
            Action::Add => self.add_dialog(ctx),
            Action::Save => self.save(false, ctx),
            Action::Move { from, to } => {
                let doc = self.docs.remove(from);
                let to = if from < to { to - 1 } else { to };
                self.docs.insert(to, doc);
                self.selected = Some(to);
            }
            _ => {
                let Some(i) = self.selected.filter(|&i| i < self.docs.len()) else { return };
                let doc = &mut self.docs[i];
                match action {
                    Action::Rotate(d) => {
                        doc.item.rotation = (doc.item.rotation + d) % 360;
                        doc.invalidate();
                    }
                    Action::SetFormat(f) => {
                        doc.item.fmt = f;
                        doc.refresh_preview();
                    }
                    Action::SetEnhance(on) => {
                        doc.item.enhance = on;
                        doc.refresh_preview();
                    }
                    Action::Crop => self.open_crop(i, false, ctx),
                    Action::Duplicate => {
                        let copy = Doc::new(doc.item.clone(), doc.preview.clone());
                        self.docs.insert(i + 1, copy);
                        self.selected = Some(i + 1);
                        self.open_crop(i + 1, true, ctx);
                    }
                    Action::Remove => {
                        self.docs.remove(i);
                        self.selected = if self.docs.is_empty() { None } else { Some(i.min(self.docs.len() - 1)) };
                    }
                    _ => {}
                }
            }
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) -> Vec<Action> {
        let mut out = Vec::new();
        ctx.input_mut(|i| {
            if i.consume_key(Modifiers::COMMAND, Key::O) {
                out.push(Action::Add);
            }
            if i.consume_key(Modifiers::COMMAND, Key::S) {
                out.push(Action::Save);
            }
        });
        if ctx.egui_wants_keyboard_input() {
            return out;
        }
        let n = self.docs.len();
        ctx.input_mut(|i| {
            // prima Shift+R: senza modificatori R corrisponderebbe anche a Shift+R
            if i.consume_key(Modifiers::SHIFT, Key::R) {
                out.push(Action::Rotate(270));
            }
            for (key, action) in [
                (Key::R, Action::Rotate(90)),
                (Key::F, Action::Rotate(180)),
                (Key::E, Action::Crop),
                (Key::D, Action::Duplicate),
                (Key::Delete, Action::Remove),
            ] {
                if i.consume_key(Modifiers::NONE, key) {
                    out.push(action);
                }
            }
            if n > 0 {
                let cur = self.selected.unwrap_or(0);
                if i.consume_key(Modifiers::NONE, Key::ArrowDown) || i.consume_key(Modifiers::NONE, Key::ArrowRight) {
                    out.push(Action::Select((cur + 1).min(n - 1)));
                }
                if i.consume_key(Modifiers::NONE, Key::ArrowUp) || i.consume_key(Modifiers::NONE, Key::ArrowLeft) {
                    out.push(Action::Select(cur.saturating_sub(1)));
                }
            }
        });
        out
    }

    fn pdf_path(&self) -> Option<PathBuf> {
        let dir = self.out_dir.clone()?;
        let mut name = self.name.trim().to_string();
        if name.to_lowercase().ends_with(".pdf") {
            name.truncate(name.len() - 4);
        }
        if name.is_empty() {
            name = crate::default_name(&dir);
        }
        Some(dir.join(format!("{name}.pdf")))
    }

    fn save(&mut self, overwrite: bool, ctx: &egui::Context) {
        if self.docs.is_empty() || self.saving {
            return;
        }
        if self.pending > 0 {
            self.error = Some("Alcune foto sono ancora in elaborazione.".into());
            return;
        }
        let Some(out) = self.pdf_path() else { return };
        if out.exists() && !overwrite {
            self.confirm_overwrite = Some(out);
            return;
        }
        let mut items: Vec<Item> = self.docs.iter().map(|d| d.item.clone()).collect();
        let (scale, gray) = (SCALES[self.settings.scale].1, self.settings.gray);
        let (tx, ctx) = (self.tx.clone(), ctx.clone());
        self.saving = true;
        self.message = None;
        std::thread::spawn(move || {
            let res = crate::pdf::make_pdf(&mut items, &out, scale, gray).map(|_| out).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::Saved(res));
            ctx.request_repaint();
        });
    }

    fn status(&self) -> String {
        if self.saving {
            return "Salvataggio del PDF…".into();
        }
        if self.pending > 0 {
            return format!("Elaborazione di {} foto in corso…", self.pending);
        }
        if let Some(m) = &self.message {
            return m.clone();
        }
        if self.docs.is_empty() {
            return "Aggiungi le foto dei documenti.".into();
        }
        let uncertain = self.docs.iter().filter(|d| d.item.confidence < UNCERTAIN).count();
        let mut s = format!("{} documenti sulla pagina.", self.docs.len());
        if uncertain > 0 {
            s += &format!("   ⚠ {uncertain} con ritaglio incerto (in arancione): controllali.");
        }
        s
    }

    // ------------------------------------------------------------ finestra principale

    fn main_ui(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let ctx = ui.ctx().clone();
        let gray = self.settings.gray;
        for d in &mut self.docs {
            d.texture(&ctx, false);
            d.texture(&ctx, gray);
        }
        egui::Panel::bottom("barra").show(ui, |ui| self.bottom_bar(ui, actions));
        egui::Panel::left("documenti").resizable(true).default_size(260.0).min_size(200.0).show(ui, |ui| {
            ui.add_space(6.0);
            let add = egui::Button::new("➕  Aggiungi foto…").min_size(vec2(ui.available_width(), 32.0));
            if ui.add(add).on_hover_text("Oppure trascina le foto nella finestra  [Ctrl+O]").clicked() {
                actions.push(Action::Add);
            }
            egui::Panel::bottom("comandi").show(ui, |ui| self.item_controls(ui, actions));
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| self.thumb_list(ui, actions));
        });
        egui::CentralPanel::default()
            .frame(Frame::new().fill(Color32::from_rgb(90, 90, 96)))
            .show(ui, |ui| self.page_preview(ui, actions));
    }

    fn thumb_list(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        ui.add_space(4.0);
        let width = ui.available_width();
        for (i, doc) in self.docs.iter().enumerate() {
            let tex = doc.tex.as_ref().unwrap();
            let size = fit(tex.size_vec2(), vec2(width - 16.0, 170.0));
            let (rect, resp) = ui.allocate_exact_size(vec2(width, size.y + 12.0), Sense::click_and_drag());
            let img = Rect::from_center_size(rect.center(), size);
            let p = ui.painter();
            if self.selected == Some(i) {
                p.rect_filled(rect, CornerRadius::same(6), ui.visuals().selection.bg_fill);
            } else if resp.hovered() {
                p.rect_filled(rect, CornerRadius::same(6), ui.visuals().widgets.hovered.weak_bg_fill);
            }
            p.image(tex.id(), img, full_uv(), Color32::WHITE);
            let uncertain = doc.item.confidence < UNCERTAIN;
            if uncertain {
                p.rect_stroke(img, 0.0, Stroke::new(3.0, ORANGE), StrokeKind::Outside);
            }
            // riordino trascinando
            if resp.drag_started() {
                resp.dnd_set_drag_payload(i);
            }
            if resp.dragged() {
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            }
            let below = ui.input(|inp| inp.pointer.interact_pos()).is_some_and(|p| p.y > rect.center().y);
            if resp.dnd_hover_payload::<usize>().is_some() {
                let y = if below { rect.bottom() } else { rect.top() };
                p.line_segment([pos2(rect.left(), y), pos2(rect.right(), y)], Stroke::new(3.0, BLUE));
            }
            if let Some(from) = resp.dnd_release_payload::<usize>() {
                actions.push(Action::Move { from: *from, to: i + below as usize });
            }
            if resp.clicked() {
                actions.push(Action::Select(i));
            }
            if resp.double_clicked() {
                actions.push(Action::Crop);
            }
            let name = doc.item.path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            resp.on_hover_text(if uncertain { format!("Rilevamento incerto: controlla il ritaglio\n{name}") } else { name });
        }
    }

    fn item_controls(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        ui.add_space(6.0);
        ui.label("Documento selezionato:");
        let doc = self.selected.and_then(|i| self.docs.get(i));
        ui.add_enabled_ui(doc.is_some(), |ui| {
            ui.horizontal_wrapped(|ui| {
                let mut b = |text: &str, tip: &str, a: Action| {
                    if ui.add(egui::Button::new(text).min_size(vec2(34.0, 30.0))).on_hover_text(tip).clicked() {
                        actions.push(a);
                    }
                };
                b("↺", "Ruota a sinistra  [Shift+R]", Action::Rotate(270));
                b("↻", "Ruota a destra  [R]", Action::Rotate(90));
                b("180°", "Capovolgi  [F]", Action::Rotate(180));
                b("✂ Ritaglio", "Correggi gli angoli (o doppio clic sulla miniatura)  [E]", Action::Crop);
                b("📋", "Duplica: per ritagliare a mano un altro documento della stessa foto  [D]", Action::Duplicate);
                b("🗑", "Togli dalla pagina  [Canc]", Action::Remove);
            });
            let (mut fmt, mut enh) = doc.map_or((Fmt::Free, true), |d| (d.item.fmt, d.item.enhance));
            let before = fmt;
            egui::ComboBox::from_id_salt("formato")
                .selected_text(fmt.label())
                .width(ui.available_width() - 8.0)
                .show_ui(ui, |ui| {
                    for f in Fmt::ALL {
                        ui.selectable_value(&mut fmt, f, f.label());
                    }
                })
                .response
                .on_hover_text("Formato reale del documento (proporzioni e dimensione di stampa)");
            if fmt != before {
                actions.push(Action::SetFormat(fmt));
            }
            if ui
                .checkbox(&mut enh, "Migliora immagine")
                .on_hover_text("Bilancia il bianco, aumenta contrasto e nitidezza")
                .changed()
            {
                actions.push(Action::SetEnhance(enh));
            }
        });
        ui.add_space(6.0);
    }

    fn bottom_bar(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Dimensione:");
            egui::ComboBox::from_id_salt("scala").selected_text(SCALES[self.settings.scale].0).show_ui(ui, |ui| {
                for (i, (label, _)) in SCALES.iter().enumerate() {
                    ui.selectable_value(&mut self.settings.scale, i, *label);
                }
            });
            ui.checkbox(&mut self.settings.gray, "Bianco e nero");
            ui.separator();
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let save = egui::Button::new(egui::RichText::new("💾  Salva PDF").strong().size(15.0))
                    .min_size(vec2(130.0, 36.0));
                if ui.add_enabled(!self.docs.is_empty() && !self.saving, save).on_hover_text("[Ctrl+S]").clicked() {
                    actions.push(Action::Save);
                }
                ui.checkbox(&mut self.settings.open_after, "Apri dopo il salvataggio");
                ui.label(".pdf");
                ui.add(
                    egui::TextEdit::singleline(&mut self.name)
                        .hint_text("nome del file PDF")
                        .desired_width(ui.available_width() - 40.0),
                );
                ui.label("File:");
            });
        });
        ui.horizontal(|ui| {
            if self.pending > 0 || self.saving {
                ui.spinner();
            }
            ui.label(self.status());
        });
        ui.add_space(4.0);
    }

    fn page_preview(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let area = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(area, Sense::click());
        let inner = area.shrink(16.0);
        let s = (inner.width() / A4_W).min(inner.height() / A4_H);
        let page = Rect::from_center_size(inner.center(), vec2(A4_W * s, A4_H * s));
        let p = ui.painter_at(area);
        p.rect_filled(page.translate(vec2(3.0, 3.0)), 0.0, Color32::from_black_alpha(80));
        p.rect_filled(page, 0.0, Color32::WHITE);
        if self.docs.is_empty() {
            p.text(
                page.center(),
                egui::Align2::CENTER_CENTER,
                "Trascina qui le foto dei documenti\n(oppure premi «Aggiungi foto»)",
                egui::FontId::proportional(16.0),
                Color32::from_gray(120),
            );
        }
        let sizes: Vec<_> = self.docs.iter().map(|d| d.item.size_mm()).collect();
        let pos = layout(&sizes, SCALES[self.settings.scale].1);
        let rects: Vec<Rect> =
            pos.iter().map(|&(x, y, w, h)| Rect::from_min_size(page.min + vec2(x * s, y * s), vec2(w * s, h * s))).collect();
        for (i, (doc, r)) in self.docs.iter().zip(&rects).enumerate() {
            let tex = if self.settings.gray { &doc.gray_tex } else { &doc.tex };
            p.image(tex.as_ref().unwrap().id(), *r, full_uv(), Color32::WHITE);
            if doc.item.confidence < UNCERTAIN {
                p.rect_stroke(*r, 0.0, Stroke::new(2.0, ORANGE), StrokeKind::Outside);
            }
            if self.selected == Some(i) {
                p.rect_stroke(r.expand(3.0), 0.0, Stroke::new(3.0, BLUE), StrokeKind::Outside);
            }
        }
        if let Some(ptr) = resp.interact_pointer_pos()
            && let Some(i) = rects.iter().position(|r| r.contains(ptr))
        {
            if resp.clicked() {
                actions.push(Action::Select(i));
            }
            if resp.double_clicked() {
                actions.push(Action::Crop);
            }
        }
    }

    // ------------------------------------------------------------ ritaglio

    fn open_crop(&mut self, idx: usize, is_new: bool, ctx: &egui::Context) {
        let it = &self.docs[idx].item;
        let photo = imgops::shrink(&it.original, 2048);
        self.crop = Some(Crop {
            idx,
            is_new,
            photo: ctx.load_texture("foto", to_color_image(&photo), TextureOptions::LINEAR),
            photo_size: vec2(it.original.width() as f32, it.original.height() as f32),
            quad: it.quad,
            zoom: 1.0,
            pan: Vec2::ZERO,
            dragging: None,
            preview: None,
        });
    }

    fn crop_ui(&mut self, ui: &mut egui::Ui) {
        let Some(c) = self.crop.as_mut() else { return };
        let original = self.docs[c.idx].item.original.clone();
        let mut done: Option<bool> = None;
        let ctx = ui.ctx().clone();

        egui::Panel::right("ritaglio_lato").exact_size(320.0).show(ui, |ui| {
            ui.add_space(8.0);
            ui.heading("Ritaglio");
            ui.label("Trascina i 4 cerchi sugli spigoli del documento.");
            ui.add_space(8.0);
            // anteprima del documento raddrizzato
            if polygon_area(&c.quad) > 100.0 && c.preview.as_ref().is_none_or(|(q, _)| *q != c.quad) {
                let img = imgops::warp_small(&original, &c.quad, guess_format(&c.quad), 600);
                c.preview = Some((c.quad, ctx.load_texture("anteprima", to_color_image(&img), TextureOptions::LINEAR)));
            }
            if let Some((_, tex)) = &c.preview {
                let size = fit(tex.size_vec2(), vec2(ui.available_width(), 380.0));
                ui.vertical_centered(|ui| ui.image((tex.id(), size)));
            }
            ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
                ui.add_space(8.0);
                let full = vec2(ui.available_width(), 32.0);
                if ui.add(egui::Button::new(egui::RichText::new("OK").strong()).min_size(full)).clicked() {
                    done = Some(true);
                }
                if ui.add(egui::Button::new("Annulla").min_size(full)).clicked() {
                    done = Some(false);
                }
                if ui.add(egui::Button::new("Tutta la foto").min_size(full)).clicked() {
                    c.quad = full_quad(original.width(), original.height());
                }
                if ui
                    .add(egui::Button::new("Rileva di nuovo").min_size(full))
                    .on_hover_text("Cerca di nuovo il documento più vicino al ritaglio attuale")
                    .clicked()
                {
                    c.quad = detect_document(&original, Some(&c.quad)).0;
                }
                ui.add_space(8.0);
                ui.weak("Rotella del mouse: zoom\nTrascina lo sfondo: sposta\nInvio: OK   Esc: annulla");
            });
        });

        egui::CentralPanel::default().frame(Frame::new().fill(Color32::from_gray(40))).show(ui, |ui| {
            let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
            let rect = resp.rect;
            let base = (rect.width() / c.photo_size.x).min(rect.height() / c.photo_size.y) * 0.95;
            let view = |c: &Crop| {
                let scale = base * c.zoom;
                (rect.center() - c.photo_size * scale / 2.0 + c.pan, scale)
            };
            // zoom attorno al puntatore
            if let Some(ptr) = resp.hover_pos() {
                let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                let f = pinch * (scroll / 300.0).exp();
                if (f - 1.0).abs() > 1e-4 {
                    let (o, s) = view(c);
                    let at = (ptr - o) / s;
                    c.zoom = (c.zoom * f).clamp(0.5, 25.0);
                    let (o2, s2) = view(c);
                    c.pan += ptr - (o2 + at * s2);
                }
            }
            let (origin, scale) = view(c);
            let to_screen = |p: [f32; 2]| origin + vec2(p[0], p[1]) * scale;
            let near_handle = |ptr: Pos2, q: &Quad| {
                (0..4)
                    .map(|k| (k, to_screen(q[k]).distance(ptr)))
                    .filter(|(_, d)| *d < 22.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(k, _)| k)
            };
            if resp.drag_started() {
                c.dragging = resp.interact_pointer_pos().and_then(|p| near_handle(p, &c.quad));
            }
            if resp.dragged() {
                let d = resp.drag_delta();
                match c.dragging {
                    Some(k) => {
                        let (w, h) = (c.photo_size.x - 1.0, c.photo_size.y - 1.0);
                        c.quad[k] = [(c.quad[k][0] + d.x / scale).clamp(0.0, w), (c.quad[k][1] + d.y / scale).clamp(0.0, h)];
                    }
                    None => c.pan += d,
                }
            }
            if resp.drag_stopped() {
                c.dragging = None;
            }
            if let Some(p) = resp.hover_pos() {
                ui.ctx().set_cursor_icon(if c.dragging.is_some() || near_handle(p, &c.quad).is_some() {
                    CursorIcon::Move
                } else if resp.dragged() {
                    CursorIcon::Grabbing
                } else {
                    CursorIcon::Grab
                });
            }

            let (origin, scale) = view(c);
            let to_screen = |p: [f32; 2]| origin + vec2(p[0], p[1]) * scale;
            painter.image(c.photo.id(), Rect::from_min_size(origin, c.photo_size * scale), full_uv(), Color32::WHITE);
            let pts: Vec<Pos2> = c.quad.iter().map(|&p| to_screen(p)).collect();
            let stroke = Stroke::new(2.0, Color32::from_rgb(0, 170, 255));
            if is_convex(&c.quad) {
                painter.add(Shape::convex_polygon(pts.clone(), Color32::from_rgba_unmultiplied(0, 170, 255, 40), stroke));
            } else {
                painter.add(Shape::closed_line(pts.clone(), stroke));
            }
            for p in pts {
                painter.circle(p, 11.0, Color32::from_rgba_unmultiplied(255, 140, 0, 110), Stroke::new(2.5, ORANGE));
            }
        });

        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
            done = Some(false);
        }
        if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
            done = Some(true);
        }
        match done {
            Some(true) => {
                let c = self.crop.take().unwrap();
                let doc = &mut self.docs[c.idx];
                let was_landscape = doc.item.is_landscape_crop();
                doc.item.quad = order_corners(&c.quad);
                doc.item.fmt = guess_format(&doc.item.quad);
                doc.item.confidence = 1.0;
                // se cambia l'orientamento del ritaglio (o è un documento nuovo) si ricalcola la rotazione
                if doc.item.is_landscape_crop() != was_landscape || c.is_new {
                    doc.item.auto_orient();
                }
                doc.refresh_preview();
                self.selected = Some(c.idx);
            }
            Some(false) => {
                let c = self.crop.take().unwrap();
                if c.is_new {
                    self.docs.remove(c.idx);
                    self.selected = (!self.docs.is_empty()).then_some(c.idx.saturating_sub(1));
                }
            }
            None => {}
        }
    }

    // ------------------------------------------------------------ finestre di dialogo

    fn dialogs(&mut self, ctx: &egui::Context) {
        if let Some(out) = self.confirm_overwrite.clone() {
            let mut answer = None;
            egui::Modal::new(Id::unique("sovrascrivi")).show(ctx, |ui| {
                ui.set_width(380.0);
                ui.heading("File esistente");
                let name = out.file_name().unwrap_or_default().to_string_lossy();
                ui.label(format!("«{name}» esiste già. Sovrascriverlo?"));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Sovrascrivi").clicked() {
                        answer = Some(true);
                    }
                    if ui.button("Annulla").clicked() {
                        answer = Some(false);
                    }
                });
            });
            if let Some(yes) = answer {
                self.confirm_overwrite = None;
                if yes {
                    self.save(true, ctx);
                }
            }
        }
        if let Some(msg) = self.error.clone() {
            let close = egui::Modal::new(Id::unique("errore"))
                .show(ctx, |ui| {
                    ui.set_width(420.0);
                    ui.heading("Attenzione");
                    ui.label(msg);
                    ui.add_space(8.0);
                    ui.button("OK").clicked()
                })
                .inner;
            if close {
                self.error = None;
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect());
        if !dropped.is_empty() {
            self.add_paths(dropped, &ctx);
        }
        if self.crop.is_some() {
            self.crop_ui(ui);
        } else {
            let mut actions = if self.confirm_overwrite.is_none() && self.error.is_none() {
                self.shortcuts(&ctx)
            } else {
                Vec::new()
            };
            self.main_ui(ui, &mut actions);
            for a in actions {
                self.apply(a, &ctx);
            }
        }
        self.dialogs(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "impostazioni", &self.settings);
    }
}

fn full_uv() -> Rect {
    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0))
}

/// Dimensione di `size` rimpicciolita (mantenendo le proporzioni) per stare in `max`.
fn fit(size: Vec2, max: Vec2) -> Vec2 {
    size * (max.x / size.x).min(max.y / size.y)
}
