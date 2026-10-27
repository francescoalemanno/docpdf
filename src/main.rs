//! DocPDF: foto di documenti in un'unica pagina PDF A4.

mod detect;
mod geom;
mod gui;
mod imgops;
mod item;
mod layout;
mod orient;
mod pdf;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use item::Item;

pub const IMG_EXT: &[&str] = &["jpg", "jpeg", "png", "webp", "heic", "heif", "tif", "tiff", "bmp"];

const USAGE: &str = "\
Uso:
  docpdf [foto o cartelle…]                         apre la finestra
  docpdf --auto [foto o cartelle…] [-o uscita.pdf]  senza finestra (-v: dettagli)
";

pub fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMG_EXT.contains(&e.to_lowercase().as_str()))
}

/// Foto indicate, con le cartelle espanse nelle immagini che contengono.
pub fn expand_paths(args: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for a in args {
        if a.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(a)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && is_image(p))
                .collect();
            files.sort();
            out.extend(files);
        } else if a.is_file() && is_image(a) {
            out.push(a.clone());
        }
    }
    out
}

/// Nome predefinito del PDF: <cartella>_documenti.
pub fn default_name(dir: &Path) -> String {
    let folder = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    format!("{}_documenti", if folder.is_empty() { "documenti" } else { &folder })
}

/// Cartella (assoluta) che contiene la foto.
pub fn photo_dir(p: &Path) -> PathBuf {
    std::path::absolute(p).ok().and_then(|a| a.parent().map(Path::to_path_buf)).unwrap_or_else(|| ".".into())
}

fn run_auto(args: Vec<String>) -> Result<()> {
    let (mut out, mut verbose, mut inputs) = (None, false, Vec::new());
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" => out = Some(PathBuf::from(it.next().context("manca il nome dopo -o")?)),
            "-v" => verbose = true,
            _ => inputs.push(PathBuf::from(a)),
        }
    }
    let paths = expand_paths(&inputs);
    if paths.is_empty() {
        bail!("nessuna immagine trovata\n\n{USAGE}");
    }
    let t0 = Instant::now();
    orient::init();
    let groups: Vec<Vec<Item>> = paths
        .par_iter()
        .map(|p| Item::from_path(p).with_context(|| p.display().to_string()))
        .collect::<Result<_>>()?;
    if verbose {
        for it in groups.iter().flatten() {
            eprintln!(
                "{}: {:?}, rotazione {}°, lato {}, sicurezza {:.2}",
                it.path.display(),
                it.fmt,
                it.rotation,
                it.side,
                it.confidence
            );
        }
        eprintln!("elaborazione: {:.2?}", t0.elapsed());
    }
    let mut items = layout::order_groups(groups, |it| it.side);
    let out = out.unwrap_or_else(|| {
        let d = photo_dir(&paths[0]);
        d.join(format!("{}.pdf", default_name(&d)))
    });
    pdf::make_pdf(&mut items, &out, None, false)?;
    if verbose {
        eprintln!("totale: {:.2?}", t0.elapsed());
    }
    println!("{}", out.display());
    Ok(())
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return;
    }
    let result = if let Some(i) = args.iter().position(|a| a == "--auto") {
        args.remove(i);
        run_auto(args)
    } else {
        gui::run(expand_paths(&args.iter().map(PathBuf::from).collect::<Vec<_>>()))
    };
    if let Err(e) = result {
        eprintln!("docpdf: {e:#}");
        std::process::exit(1);
    }
}
