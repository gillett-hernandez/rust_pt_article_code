use minifb::Window;

mod math;

fn u32_to_rgb(v: u32) -> [u8; 3] {
    [
        ((v >> 16) % 256) as u8,
        ((v >> 8) % 256) as u8,
        (v % 256) as u8,
        // ((v >> 24) % 256) as u8,
    ]
}

fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

fn main() {
    let height = 256;
    let width = 256;

    // packed rgba layout
    let mut film = vec![rgb_to_u32(0, 0, 0); width * height];

    // do stuff to film
    for y in 0..height {
        for x in 0..width {
            let uv = (x as f32 / width as f32, y as f32 / height as f32);

            let v = (((uv.0 - 0.5) * (uv.1 - 0.5).powi(2) * 201.0).sin() + 1.0) / 2.0;
            
            film[y * width + x] =
                rgb_to_u32((255.0 * v) as u8, (255.0 * v) as u8, (255.0 * v) as u8);
        }
    }

    // view film using minifb
    if true {
        let mut window = Window::new(
            "debug display",
            width,
            height,
            minifb::WindowOptions {
                scale: minifb::Scale::X1,
                ..minifb::WindowOptions::default()
            },
        )
        .expect("failed to create window");
        window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

        while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
            window.update_with_buffer(&film, width, height).unwrap();
        }
    }

    // output image
    let mut rgbimage = image::RgbImage::new(width as u32, height as u32);
    for (x, y, pixel) in rgbimage.enumerate_pixels_mut() {
        let packed = film
            .get((y * width as u32 + x) as usize)
            .cloned()
            .unwrap_or(0u32);

        *pixel = image::Rgb(u32_to_rgb(packed));
    }
    rgbimage
        .save("./output/beauty.png")
        .expect("image failed to save");
}
