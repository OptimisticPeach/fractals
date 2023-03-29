use num_complex::Complex64;
use pixels::{Pixels, SurfaceTexture};
use rayon::prelude::*;

use winit::{
    dpi::LogicalSize,
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use winit_input_helper::WinitInputHelper;

const PX_X: usize = 1920;
const PX_Y: usize = 1000;

struct Settings {
    bottom_left: (f64, f64),
    width: f64,
    pixels_x: usize,
    pixels_y: usize,
    iterations: usize,
    modulus_bounds: f64,
    constant: Constant,
    scale: f64,
    col_pow: usize,
    show_axes: bool,
    done_frames: usize,
}

impl Settings {
    fn pos_x(&self, x: usize) -> f64 {
        (x as f64 / self.pixels_x as f64 + (0.5 / self.pixels_x as f64)) * self.width
            + self.bottom_left.0
    }

    fn pos_y(&self, y: usize) -> f64 {
        ((1.0 - (y as f64 / self.pixels_y as f64)) + (0.5 / self.pixels_y as f64)) * (self.width * self.scale)
            + self.bottom_left.1
    }

    fn constant(&self, position: (f64, f64)) -> (f64, f64) {
        match self.constant {
            Constant::Position => position,
            Constant::Const(x) => x,
        }
    }
}

#[derive(Debug)]
enum Constant {
    Position,
    Const((f64, f64)),
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            bottom_left: (-1.0, -1.0),
            width: 2.0,
            pixels_x: PX_X,
            pixels_y: PX_Y,
            iterations: 101,
            modulus_bounds: 10.0,
            constant: Constant::Const((0.4, 0.2)),
            scale: PX_Y as f64 / PX_X as f64,
            col_pow: 2,
            show_axes: true,
            done_frames: 0,
        }
    }
}

fn compute_next((a1, a2): (f64, f64), (c1, c2): (f64, f64)) -> (f64, f64) {
    // let a1 = a1.abs();
    // let a2 = a2.abs();
    let z = Complex64::new(a1, a2);
    let c = Complex64::new(c1, c2);

    let result = z * z * z * z * z + z;
    (result.re, result.im)
}

fn size((a1, a2): (f64, f64)) -> f64 {
    a1 * a1 + a2 * a2
}

fn num_iterations(
    mut point: (f64, f64),
    constant: (f64, f64),
    max_iterations: usize,
    mod_bounds: f64,
) -> (usize, (f64, f64)) {
    for i in 0..max_iterations {
        point = compute_next(point, constant);
        let size = size(point);
        if size > mod_bounds * mod_bounds {
            return (i, (0.0, 0.0));
        }
    }
    return (max_iterations, point);
}

fn map(left: f64, right: f64, val: f64) -> f64 {
    ((val - left) / (right - left)).clamp(0.0, 1.0)
}

const COLOURS: [u8; 9] = [0, 15, 30, 64, 255, 202, 12, 0, 0];
const COLOURS_F: [f64; 9] = {
    let mut result = [0.0; 9];
    let mut i = 0;
    while i < 9 {
        result[i] = COLOURS[i] as f64 / 255.0;
        i += 1;
    }
    result
};

fn make_colour<const O: usize>(map_1: f64, map_2: f64, f_v: f64) -> f64 {
    COLOURS_F[O] * (1.0 - map_1) + COLOURS_F[3 + O] * map_1 * (1.0 - map_2) + f_v * map_2
}

fn hsv(x: f64, k: f64) -> f64 {
    (((x + k).rem_euclid(6.0) - 3.0).abs() - 1.0).clamp(0.0, 1.0)
}

fn gradient(iterations: f64, max_iterations: usize, pow: usize, result: (f64, f64)) -> (f64, f64, f64) {
    let p = (iterations / max_iterations as f64).powi(pow as _);

    let angle = 6.0 * result.1.atan2(result.0) / (2.0 * std::f64::consts::PI);

    // let r = (std::f64::consts::FRAC_PI_3 * 2.0 - angle.abs() % (std::f64::consts::PI * 2.0)).max(0.0) / (std::f64::consts::FRAC_PI_3 * 2.0);
    // let g = (std::f64::consts::FRAC_PI_3 * 2.0 - (angle - std::f64::consts::FRAC_PI_3 * 2.0).abs() % (std::f64::consts::PI * 2.0)).max(0.0) / std::f64::consts::FRAC_PI_3 * 2.0;
    // let b = (std::f64::consts::FRAC_PI_3 * 2.0 - (angle - std::f64::consts::FRAC_PI_3 * 4.0).abs() % (std::f64::consts::PI * 2.0)).max(0.0) / std::f64::consts::FRAC_PI_3 * 2.0;
    // let r = angle.cos().max(0.0);
    // let g = (angle - std::f64::consts::FRAC_PI_3 * 2.0).cos().max(0.0);
    // let b = (angle - std::f64::consts::FRAC_PI_3 * 4.0).cos().max(0.0);
    let (r, g, b) = (hsv(angle, 0.0), hsv(angle, 2.0), hsv(angle, 4.0));
    // let (r, g, b) = (0.0, 0.0, 0.0);

    let map_1 = map(0.0, 0.9, p);
    let map_2 = map(0.9, 1.0, p);

    (make_colour::<0>(map_1, map_2, r), make_colour::<1>(map_1, map_2, g), make_colour::<2>(map_1, map_2, b))
}

fn draw(settings: &Settings, buffer: &mut [u8], cont_buffer: &mut [f64], acc_buffer: &mut [(f64, f64)]) {
    buffer
        .par_chunks_mut(settings.pixels_x * 4)
        .zip(cont_buffer.par_chunks_mut(settings.pixels_x))
        .zip(acc_buffer.par_chunks_mut(settings.pixels_x))
        .enumerate()
        .for_each(|(y, ((chunk_screen, chunk_cont), chunk_acc))| {
            let y = settings.pos_y(y);
            // let y = -y;
            for (x, (screen_result, (cont_result, acc_result))) in chunk_screen
                .chunks_exact_mut(4)
                .zip(chunk_cont.iter_mut().zip(chunk_acc.iter_mut())).enumerate() {
                let x = settings.pos_x(x);

                if settings.show_axes && (y.abs() < 0.01 || x.abs() < 0.01) {
                    screen_result.copy_from_slice(&[255, 255, 255, 255]);
                    continue
                }

                let r_x = (fastrand::f64() * 2.0 - 1.0) * (0.5 / settings.pixels_x as f64) * settings.width;
                let r_y = (fastrand::f64() * 2.0 - 1.0) * (0.5 / settings.pixels_y as f64) * settings.width * settings.scale;

                let (value, final_values) = num_iterations(
                    (x + r_x, y + r_y),
                    settings.constant((x + r_x, y + r_y)),
                    settings.iterations,
                    settings.modulus_bounds,
                );

                *cont_result += value as f64;
                acc_result.0 += final_values.0;
                acc_result.1 += final_values.1;

                let screen_value = *cont_result / settings.done_frames as f64;

                let value = gradient(screen_value, settings.iterations, settings.col_pow, *acc_result);
                let value_r = (value.0 * 255.0) as u8;
                let value_g = (value.1 * 255.0) as u8;
                let value_b = (value.2 * 255.0) as u8;
                let color = [value_r, value_g, value_b, 255];
                screen_result.copy_from_slice(&color);
            }
        });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();

    let window = {
        let size = LogicalSize::new(PX_X as f64, PX_X as f64);
        let scaled_size = LogicalSize::new(PX_Y as f64, PX_Y as f64);
        WindowBuilder::new()
            .with_title("Fractals")
            .with_inner_size(scaled_size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(PX_X as _, PX_Y as _, surface_texture)?
    };

    let mut settings = Settings::default();

    let mut paused = false;

    let mut window_size = (PX_X as f64, PX_Y as f64);

    let mut cont_buffer = vec![0.0; PX_X * PX_Y];
    let mut acc_buffer = vec![(0.0, 0.0); PX_X * PX_Y];

    event_loop.run(move |event, _, control_flow| {
        // The one and only event that winit_input_helper doesn't have for us...
        if let Event::RedrawRequested(_) = event {
            settings.done_frames += 1;
            draw(&settings, pixels.get_frame_mut(), &mut cont_buffer, &mut acc_buffer);
            if let Err(err) = pixels.render() {
                eprintln!("pixels.render: {:?}", err);
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        let mut should_reset = false;

        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape)
                || input.close_requested()
                || input.destroyed()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
            if input.key_pressed(VirtualKeyCode::P) {
                paused = !paused;
            }
            if input.key_pressed(VirtualKeyCode::C) {
                if let Constant::Position = &settings.constant {
                    settings.constant = Constant::Const((0.0, 0.0));
                } else {
                    settings.constant = Constant::Position;
                }
                should_reset = true;
            }

            if input.key_held(VirtualKeyCode::A) {
                settings.bottom_left.0 -= settings.width * 0.005;
                should_reset = true;
            }

            if input.key_held(VirtualKeyCode::D) {
                settings.bottom_left.0 += settings.width * 0.005;
                should_reset = true;
            }

            if input.key_held(VirtualKeyCode::S) {
                settings.bottom_left.1 -= settings.width * 0.005;
                should_reset = true;
            }

            if input.key_held(VirtualKeyCode::W) {
                settings.bottom_left.1 += settings.width * 0.005;
                should_reset = true;
            }

            if input.key_held(VirtualKeyCode::LControl) {
                let new_width = settings.width * 0.99;
                let diff_x = settings.width - new_width;
                settings.bottom_left.0 += diff_x / 2.0;

                let diff_y = settings.scale * diff_x;
                settings.bottom_left.1 += diff_y / 2.0;

                settings.width = new_width;
                should_reset = true;
            }

            if input.key_held(VirtualKeyCode::Space) {
                let new_width = settings.width * 1.01;
                let diff_x = settings.width - new_width;
                settings.bottom_left.0 += diff_x / 2.0;

                let diff_y = settings.scale * diff_x;
                settings.bottom_left.1 += diff_y / 2.0;

                settings.width = new_width;
                should_reset = true;
            }

            if input.key_pressed(VirtualKeyCode::Up) {
                settings.iterations += 10;
                should_reset = true;
            }
            if input.key_pressed(VirtualKeyCode::Down) {
                if settings.iterations >= 10 {
                    settings.iterations -= 10;
                }
                should_reset = true;
            }

            if input.key_pressed(VirtualKeyCode::I) {
                settings.col_pow += 1;
            }
            if input.key_pressed(VirtualKeyCode::K) {
                if settings.col_pow > 1 {
                    settings.col_pow -= 1;
                }
            }

            if input.key_pressed(VirtualKeyCode::O) {
                settings.modulus_bounds += 0.1;
                should_reset = true;
            }
            if input.key_pressed(VirtualKeyCode::L) {
                if settings.modulus_bounds > 0.1 {
                    settings.modulus_bounds -= 0.1;
                }
                should_reset = true;
            }

            if input.key_pressed(VirtualKeyCode::Z) {
                settings.show_axes = !settings.show_axes;
                should_reset = true;
            }

            if input.key_pressed(VirtualKeyCode::X) {
                println!(
                    "x in [{}, {}]\ny in [{}, {}]\ncenter at  ({} + {}i)\nconstant {:?}",
                    settings.bottom_left.0,
                    settings.bottom_left.0 + settings.width,
                    settings.bottom_left.1,
                    settings.bottom_left.1 + settings.width * settings.scale,
                    settings.bottom_left.0 + settings.width / 2.0,
                    settings.bottom_left.1 + settings.width * settings.scale / 2.0,
                    settings.constant
                );
            }

            input.mouse().map(|(mx, my)| {
                let mx = mx as f64;
                let my = my as f64;

                if !paused {
                    let px = 2.0 * mx / window_size.0 - 1.0;
                    let py = 2.0 * my / window_size.1 - 1.0;

                    if let Constant::Const(x) = &mut settings.constant {
                        *x = (px, py);
                        should_reset = true;
                    }
                }
            });

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    eprintln!("pixels.resize_surface: {:?}", err);
                    *control_flow = ControlFlow::Exit;
                    return;
                }

                window_size.0 = size.width as f64;
                window_size.1 = size.height as f64;
            }

            window.request_redraw();
        }
        if should_reset {
            println!("Resetting from {}", settings.done_frames);
            cont_buffer.iter_mut().for_each(|x| *x = 0.0);
            acc_buffer.iter_mut().for_each(|x| *x = (0.0, 0.0));
            settings.done_frames = 0;
        }
    });
}
