use minifb::Window;

mod camera;
mod math;
mod primitive;

use camera::Camera;
use math::{random_on_unit_sphere, Point3, Ray, Vec3};
use primitive::{Intersection, RayIntersection, Sphere};

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

    let camera = Camera::new(
        Point3::new(3.0, 3.0, 3.0),
        Point3::ORIGIN,
        Vec3::Z,
        5.0,
        1.0,
    );

    let subject_sphere = Sphere {
        origin: Point3::ORIGIN,
        radius: 1.0,
    };

    let light_sphere = Sphere {
        origin: Point3::new(0.0, 0.0, 3.0),
        radius: 1.0,
    };

    // do stuff to film
    for y in 0..height {
        for x in 0..width {
            let uv = (x as f32 / width as f32, y as f32 / height as f32);

            let r = camera.get_ray(uv);

            let color = if let Some(Intersection { point, normal }) = subject_sphere.intersects(r) {
                // intersected sphere, now shoot another ray and see if that intersects the light

                let new_direction = normal + random_on_unit_sphere();
                let new_r = Ray::new(point, new_direction);
                if light_sphere.intersects(new_r).is_some() {
                    rgb_to_u32(255u8, 255u8, 255u8)
                } else {
                    0u32
                }
            } else {
                0u32
            };
            film[y * width + x] = color;
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
