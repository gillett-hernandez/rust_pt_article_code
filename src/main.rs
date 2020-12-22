#[macro_use]
extern crate packed_simd;

use std::f32::INFINITY;

use packed_simd::f32x4;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub mod camera;
pub mod film;
pub mod material;
pub mod math;
pub mod primitive;
pub mod random;
pub mod tonemap;

use camera::ProjectiveCamera;
use film::Film;
use material::{ConstDiffuseEmitter, ConstLambertian, Material, MaterialEnum};
use math::*;
use primitive::{IntersectionData, Primitive, Sphere};
use tonemap::{sRGB, Tonemapper};

pub fn output_film(filename: Option<&String>, film: &Film<XYZColor>) {
    let filename_str = filename.cloned().unwrap_or(String::from("output"));
    let exr_filename = format!("output/{}.exr", filename_str);
    let png_filename = format!("output/{}.png", filename_str);

    let srgb_tonemapper = sRGB::new(film, 1.0);
    srgb_tonemapper.write_to_files(film, &exr_filename, &png_filename);
}

pub fn hero_from_range(x: f32, bounds: Bounds1D) -> f32x4 {
    let hero = x * bounds.span();
    let delta = bounds.span() / 4.0;
    let mult = f32x4::new(0.0, 1.0, 2.0, 3.0);
    let wavelengths = bounds.lower + (hero + mult * delta);
    let sub: f32x4 = wavelengths
        .gt(f32x4::splat(bounds.upper))
        .select(f32x4::splat(bounds.span()), f32x4::splat(0.0));
    wavelengths - sub
}

fn main() {
    let num_cpus = num_cpus::get();
    let threads = num_cpus;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build_global()
        .unwrap();

    let h = 1024;
    let w = 1024;
    let wavelength_range = BOUNDED_VISIBLE_RANGE;

    let samples = 16;
    let bounces = 3;
    let mut film = Film::new(w, h, XYZColor::ZERO);
    let materials: Vec<MaterialEnum> = vec![
        MaterialEnum::ConstLambertian(ConstLambertian::new(SPD::Linear {
            signal: vec![1.0],
            bounds: EXTENDED_VISIBLE_RANGE,
            mode: InterpolationMode::Linear,
        })),
        MaterialEnum::ConstDiffuseEmitter(ConstDiffuseEmitter::new(
            SPD::Linear {
                signal: vec![1.0],
                bounds: EXTENDED_VISIBLE_RANGE,
                mode: InterpolationMode::Linear,
            },
            SPD::Linear {
                signal: vec![10.0],
                bounds: EXTENDED_VISIBLE_RANGE,
                mode: InterpolationMode::Linear,
            },
        )),
    ];
    let scene: Vec<Sphere> = vec![
        Sphere::new(1.0, Point3::ORIGIN, 0),
        Sphere::new(10.0, Point3::new(0.0, 0.0, 12.0), 0),
        Sphere::new(100.0, Point3::new(0.0, 0.0, -103.0), 1),
    ];
    let camera = ProjectiveCamera::new(
        Point3::new(-10.0, 0.0, 0.0),
        Point3::ORIGIN,
        Vec3::Z,
        30.0,
        1.0,
        10.0,
        0.01,
        0.0,
        1.0,
    );

    film.buffer.par_iter_mut().enumerate().for_each(|(i, e)| {
        let x = i % w;
        let y = i / w;

        for _ in 0..samples {
            let lambdas = hero_from_range(rand::random::<f32>(), wavelength_range);
            let (s, t) = (
                (x as f32 + rand::random::<f32>()) / (w as f32),
                (y as f32 + rand::random::<f32>()) / (h as f32),
            );
            let aperture_sample = Sample2D::new_random_sample();
            let mut ray = camera.get_ray(aperture_sample, s, t);

            let mut s = f32x4::splat(0.0);
            let mut throughput = f32x4::splat(1.0);

            for _ in 0..bounces {
                let mut nearest_intersection: Option<IntersectionData> = None;
                let mut nearest_intersection_time = INFINITY;

                for prim in scene.iter() {
                    if let Some(intersection) = prim.intersect(ray, 0.0, INFINITY) {
                        if intersection.time < nearest_intersection_time {
                            nearest_intersection_time = intersection.time;
                            nearest_intersection = Some(intersection);
                        }
                    }
                }

                if nearest_intersection.is_none() {
                    s += throughput * 1.0; // hit env
                    break;
                }
                let nearest_intersection = nearest_intersection.unwrap();
                let frame = TangentFrame::from_normal(nearest_intersection.normal);
                let wi = frame.to_local(&-ray.direction);
                let mat = &materials[nearest_intersection.material_id];

                let wo = mat.sample(lambdas.extract(0), wi, Sample2D::new_random_sample());

                let cos_i = wo.z();

                let (mut bsdf, mut pdf) = (f32x4::splat(0.0), f32x4::splat(0.0));
                let mut emission = f32x4::splat(0.0);
                // match mat {
                //     MaterialEnum::ConstDiffuseEmitter(_) => {
                //         println!("hit diffuse emitter")
                //     }
                //     _ => {}
                // }
                for i in 0..4 {
                    let (local_bsdf, local_pdf) = mat.bsdf(lambdas.extract(i), wi, wo);
                    bsdf = bsdf.replace(i, local_bsdf);
                    pdf = pdf.replace(i, local_pdf);
                    emission = emission.replace(i, mat.emission(lambdas.extract(i), wi));
                }
                if emission.gt(f32x4::splat(0.0)).any() {
                    s += throughput * emission * cos_i;
                }
                if pdf.extract(0) == 0.0 {
                    break;
                }

                throughput *= bsdf * cos_i.abs() / pdf;
                ray = Ray::new(
                    nearest_intersection.point + nearest_intersection.normal * 0.001,
                    frame.to_world(&wo).normalized(),
                );
            }
            *e += XYZColor::from_wavelength_and_energy(
                lambdas.extract(0),
                s.extract(0) / (samples as f32 * 4.0),
            );
            *e += XYZColor::from_wavelength_and_energy(
                lambdas.extract(1),
                s.extract(1) / (samples as f32 * 4.0),
            );
            *e += XYZColor::from_wavelength_and_energy(
                lambdas.extract(2),
                s.extract(2) / (samples as f32 * 4.0),
            );
            *e += XYZColor::from_wavelength_and_energy(
                lambdas.extract(3),
                s.extract(3) / (samples as f32 * 4.0),
            );
        }
    });
    output_film(None, &film);
}
