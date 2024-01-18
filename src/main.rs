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
    let film = vec![rgb_to_u32(128, 128, 128); width * height];

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
