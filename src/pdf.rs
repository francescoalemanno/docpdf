//! Scrittura del PDF: una pagina A4 con i documenti come immagini JPEG.

use std::path::Path;

use anyhow::Result;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use pdf_writer::{Content, Filter, Finish, Name, Pdf, Rect, Ref, TextStr};
use rayon::prelude::*;

use crate::item::Item;
use crate::layout::{A4_H, A4_W, layout};

const DPI: f32 = 300.0;
const PT_PER_MM: f32 = 72.0 / 25.4;

pub fn make_pdf(items: &mut [Item], out: &Path, max_scale: Option<f32>, grayscale: bool) -> Result<()> {
    let sizes: Vec<_> = items.iter().map(Item::size_mm).collect();
    let pos = layout(&sizes, max_scale);

    // immagini JPEG alla risoluzione di stampa, in parallelo
    let jpegs: Vec<(Vec<u8>, u32, u32)> = items
        .par_iter_mut()
        .zip(&pos)
        .map(|(it, &(_, _, w, _))| -> Result<_> {
            let mut img = it.result();
            let tw = (w / 25.4 * DPI).round() as u32;
            if img.width() > tw && tw > 0 {
                let th = ((img.height() as f32 * tw as f32 / img.width() as f32).round() as u32).max(1);
                img = imageops::resize(&img, tw, th, FilterType::Triangle);
            }
            let mut buf = Vec::new();
            let mut enc = JpegEncoder::new_with_quality(&mut buf, 88);
            if grayscale {
                enc.encode_image(&imageops::grayscale(&img))?;
            } else {
                enc.encode_image(&img)?;
            }
            Ok((buf, img.width(), img.height()))
        })
        .collect::<Result<_>>()?;

    let catalog = Ref::new(1);
    let tree = Ref::new(2);
    let page_id = Ref::new(3);
    let content_id = Ref::new(4);
    let info = Ref::new(5);
    let image_ref = |i: usize| Ref::new(6 + i as i32);

    let mut pdf = Pdf::new();
    pdf.catalog(catalog).pages(tree);
    pdf.pages(tree).kids([page_id]).count(1);
    let names: Vec<String> = (0..jpegs.len()).map(|i| format!("Im{i}")).collect();
    {
        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, A4_W * PT_PER_MM, A4_H * PT_PER_MM));
        page.parent(tree);
        page.contents(content_id);
        let mut res = page.resources();
        let mut xo = res.x_objects();
        for (i, n) in names.iter().enumerate() {
            xo.pair(Name(n.as_bytes()), image_ref(i));
        }
        xo.finish();
        res.finish();
        page.finish();
    }

    let mut content = Content::new();
    for (i, &(x, y, w, h)) in pos.iter().enumerate() {
        content.save_state();
        content.transform([w * PT_PER_MM, 0.0, 0.0, h * PT_PER_MM, x * PT_PER_MM, (A4_H - y - h) * PT_PER_MM]);
        content.x_object(Name(names[i].as_bytes()));
        content.restore_state();
    }
    pdf.stream(content_id, &content.finish());

    for (i, (data, w, h)) in jpegs.iter().enumerate() {
        let mut img = pdf.image_xobject(image_ref(i), data);
        img.filter(Filter::DctDecode);
        img.width(*w as i32);
        img.height(*h as i32);
        if grayscale {
            img.color_space().device_gray();
        } else {
            img.color_space().device_rgb();
        }
        img.bits_per_component(8);
        img.finish();
    }

    let title = out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    pdf.document_info(info).title(TextStr(&title)).producer(TextStr("DocPDF"));

    std::fs::write(out, pdf.finish())?;
    Ok(())
}
