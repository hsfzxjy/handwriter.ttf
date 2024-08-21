use core::f64;

type T = f64;

pub struct Rasterizer<F> {
    step: T,
    plot: F,
}

impl<F> Rasterizer<F>
where
    F: FnMut(f64, f64),
{
    pub fn new(step: T, plot: F) -> Self {
        Self { step, plot }
    }
    fn draw_low(&mut self, x0: T, y0: T, x1: T, y1: T) {
        let dx = x1 - x0;
        let mut dy = y1 - y0;
        let mut yi = 1.;
        if dy < 0. {
            yi = -1.;
            dy = -dy;
        }
        let mut d = 2. * dy - dx;
        let mut y = y0;
        let mut x = x0;
        while x < x1 {
            (self.plot)(x, y);
            if d > 0. {
                y += yi;
                d += 2. * (dy - dx);
            } else {
                d += 2. * dy;
            }
            x += self.step;
        }
    }
    fn draw_high(&mut self, x0: T, y0: T, x1: T, y1: T) {
        let mut dx = x1 - x0;
        let dy = y1 - y0;
        let mut xi = 1.;
        if dx < 0. {
            xi = -1.;
            dx = -dx
        }
        let mut d = 2. * dx - dy;
        let mut x = x0;
        let mut y = y0;
        while y < y1 {
            (self.plot)(x, y);
            if d > 0. {
                x += xi;
                d += 2. * (dx - dy);
            } else {
                d += 2. * dx;
            }
            y += self.step;
        }
    }
    pub fn draw_line(&mut self, x0: T, y0: T, x1: T, y1: T) {
        if (y1 - y0).abs() < (x1 - x0).abs() {
            if x0 > x1 {
                self.draw_low(x1, y1, x0, y0);
            } else {
                self.draw_low(x0, y0, x1, y1)
            }
        } else {
            if y0 > y1 {
                self.draw_high(x1, y1, x0, y0);
            } else {
                self.draw_high(x0, y0, x1, y1)
            }
        }
    }
}

pub fn rasterize_font(
    points: Vec<[f64; 3]>,
    left: i32,
    top: i32,
    bottom: i32,
    pixsize: i32,
) -> Vec<[i32; 2]> {
    let (pt_left, pt_top, pt_bottom) = {
        let mut left = f64::INFINITY;
        let mut top = f64::INFINITY;
        let mut bottom = f64::NEG_INFINITY;
        for pt in &points {
            left = left.min(pt[0]);
            top = top.min(pt[1]);
            bottom = bottom.max(pt[1]);
        }
        (left, top, bottom)
    };
    let nv_pixel = (top - bottom) / pixsize;
    let scale = (top - bottom) as f64 / (pt_bottom - pt_top);

    let mut results = Vec::with_capacity(1024);
    let mut rast = Rasterizer::new((pt_bottom - pt_top) / nv_pixel as f64, |x, y| {
        let x = ((x - pt_left) * scale).round() as i32 + left;
        let y = top - ((y - pt_top) * scale).round() as i32;
        results.push([x, y])
    });
    let mut points = points.into_iter();
    let mut prev = points.next().unwrap();
    for pt in points {
        if prev[2] > 0.5 {
            prev = pt;
            continue;
        }
        rast.draw_line(prev[0], prev[1], pt[0], pt[1]);
        prev = pt;
    }
    results
}
