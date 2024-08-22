mod io;
mod myrand;
mod nn;
mod raster;
mod tokenizer;
mod utils;

use harfbuzz_wasm::{debug, Font, Glyph, GlyphBuffer};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub unsafe fn shape(
    _shape_plan: u32,
    font_ref: u32,
    buf_ref: u32,
    _features: u32,
    _num_features: u32,
) -> i32 {
    utils::set_panic_hook();
    let font = Font::from_ref(font_ref);
    let mut buffer = GlyphBuffer::from_ref(buf_ref);
    let glyphs = &mut buffer.glyphs;
    let text = glyphs
        .iter()
        .map(|g| char::from_u32_unchecked(g.codepoint))
        .collect::<String>();
    if text.as_bytes()[0] != b'#' || text.len() == 1 {
        for g in glyphs {
            g.codepoint = font.get_glyph(g.codepoint, 0);
            g.x_advance = font.get_glyph_h_advance(g.codepoint);
        }
        return 1;
    }
    let scale = font.get_scale().0 / font.get_face().get_upem() as i32;
    debug(&format!("{:?} {}", &text[1..], scale));

    let res = nn::Generator::new().generate_for(&text.as_bytes()[1..]);
    let mut x_max = 0;
    let mut res = raster::rasterize_font(res, 20 * scale, 750 * scale, 0, 50 * scale)
        .into_iter()
        .map(|[x, y]| {
            x_max = x_max.max(x);
            Glyph {
                codepoint: font.get_glyph(0xF0001, 0),
                cluster: 0,
                x_advance: 0,
                y_advance: 0,
                x_offset: x,
                y_offset: y,
                flags: 0,
            }
        })
        .collect::<Vec<_>>();
    res.last_mut().unwrap().x_advance = x_max + 400 * scale;
    buffer.glyphs = res;
    1
}
