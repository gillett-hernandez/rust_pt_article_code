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
use material::{
    ConstDiffuseEmitter, ConstFilm, ConstLambertian, HenyeyGreensteinHomogeneous, Material,
    MaterialEnum, Medium, MediumEnum,
};
use math::*;
use primitive::{
    IntersectionData, MediumIntersectionData, Primitive, Sphere, SurfaceIntersectionData,
};
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
        .num_threads(1 as usize)
        // .num_threads(threads as usize)
        .build_global()
        .unwrap();

    let h = 1024;
    let w = 1024;
    let wavelength_range = BOUNDED_VISIBLE_RANGE;

    let samples = 16;
    let bounces = 10;
    let mut film = Film::new(w, h, XYZColor::ZERO);
    let white = SPD::Linear {
        signal: vec![1.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let black_ish = SPD::Linear {
        signal: vec![0.01],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let materials: Vec<MaterialEnum> = vec![
        MaterialEnum::ConstLambertian(ConstLambertian::new(white.clone())),
        MaterialEnum::ConstDiffuseEmitter(ConstDiffuseEmitter::new(
            white.clone(),
            SPD::Linear {
                signal: vec![10.0],
                bounds: EXTENDED_VISIBLE_RANGE,
                mode: InterpolationMode::Linear,
            },
        )),
        MaterialEnum::ConstFilm(ConstFilm::new(white.clone())),
    ];
    let mediums: Vec<MediumEnum> = vec![MediumEnum::HenyeyGreensteinHomogeneous(
        HenyeyGreensteinHomogeneous {
            g: 0.0,
            sigma_s: white.clone(),
            sigma_t: white.clone(),
        },
    )];
    let scene: Vec<Sphere> = vec![
        Sphere::new(1.0, Point3::ORIGIN, 0, 1, 0), // subject sphere
        Sphere::new(10.0, Point3::new(0.0, 0.0, 15.0), 0, 0, 0), // light
        Sphere::new(100.0, Point3::new(0.0, 0.0, -103.0), 1, 0, 0), // floor
        Sphere::new(3.0, Point3::new(0.0, 0.0, 0.0), 2, 0, 1), // bubble of scattering
    ];
    let camera = ProjectiveCamera::new(
        Point3::new(-25.0, 0.0, 0.0),
        Point3::ORIGIN,
        Vec3::Z,
        20.0,
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
            // somehow determine what medium the camera ray starts in. assume vacuum for now
            let mut tracked_mediums: Vec<usize> = Vec::new();

            for _ in 0..bounces {
                // if tracked_mediums.len() > 0 {
                //     println!("\n{:?}", tracked_mediums);
                // }
                tracked_mediums.dedup();
                let mut nearest_intersection: Option<IntersectionData> = None;
                let mut nearest_intersection_time = INFINITY;

                for prim in scene.iter() {
                    if let Some(IntersectionData::Surface(intersection)) =
                        prim.intersect(ray, 0.0, INFINITY)
                    {
                        if intersection.time < nearest_intersection_time {
                            nearest_intersection_time = intersection.time;
                            nearest_intersection = Some(IntersectionData::Surface(intersection));
                        }
                    }
                }

                if nearest_intersection.is_none() {
                    s += throughput * 1.0; // hit env
                    break;
                }
                // iterate through volumes and sample each, choosing the closest medium scattering (or randomly sampling?)
                let mut intersection = nearest_intersection.unwrap();
                let mut closest_p = if let IntersectionData::Surface(s) = intersection {
                    s.point
                } else {
                    panic!();
                };
                // let mut any_scatter = false;
                for medium_id in tracked_mediums.iter() {
                    let medium = &mediums[*medium_id - 1];
                    let (p, _tr, scatter) =
                        medium.sample(lambdas.extract(0), ray, Sample1D::new_random_sample());
                    if scatter {
                        // any_scatter = true;
                        let t = (p - ray.origin).norm();
                        if t < nearest_intersection_time {
                            nearest_intersection_time = t;
                            closest_p = p;
                            intersection = IntersectionData::Medium(MediumIntersectionData {
                                time: t,
                                point: p,
                                wi: -ray.direction,
                                medium_id: *medium_id,
                            });
                        }
                    }
                }
                for i in 0..4 {
                    let mut combined_throughput = 1.0;
                    for medium_id in tracked_mediums.iter() {
                        let medium = &mediums[*medium_id - 1];
                        combined_throughput *= medium.tr(lambdas.extract(i), ray.origin, closest_p);
                    }
                    throughput = throughput.replace(i, throughput.extract(i) * combined_throughput);
                }
                match intersection {
                    IntersectionData::Surface(isect) => {
                        let frame = TangentFrame::from_normal(isect.normal);
                        let wi = frame.to_local(&-ray.direction);
                        let mat = &materials[isect.material_id];

                        let outer = isect.outer_medium_id;
                        let inner = isect.inner_medium_id;

                        let wo = mat.sample(lambdas.extract(0), wi, Sample2D::new_random_sample());

                        let cos_i = wo.z();

                        let (mut bsdf, mut pdf) = (f32x4::splat(0.0), f32x4::splat(0.0));
                        let mut emission = f32x4::splat(0.0);

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
                        if wi.z() * wo.z() > 0.0 {
                            // scattering, so don't mess with volumes
                            // println!("reflect, {}, {}", outer, inner);
                        } else {
                            // transmitting, so remove appropriate medium from list and add new one. only applicable if inner != outer
                            if inner != outer {
                                // println!("transmit, {}, {}, {:?}", outer, inner, isect.normal);
                                // print!("{} ", isect.material_id);
                                if wo.z() < 0.0 {
                                    // println!("wo.z < 0, wi: {:?}, wo: {:?}", wi, wo);
                                    // transmitting from outer to inner. thus remove outer and add inner
                                    if outer != 0 {
                                        // only remove outer if it's not the Vacuum index.
                                        let index = tracked_mediums
                                            .iter()
                                            .position(|e| *e == outer)
                                            .expect("should have found correct medium.");
                                        tracked_mediums.remove(index);
                                    }
                                    if inner != 0 {
                                        // let insertion_index = tracked_mediums.binary_search(&inner);
                                        tracked_mediums.push(inner);
                                        tracked_mediums.sort_unstable();
                                    }
                                } else {
                                    // println!("wo.z > 0, wi: {:?}, wo: {:?}", wi, wo);
                                    // transmitting from inner to outer. thus remove inner and add outer, unless outer is vacuum.
                                    // also don't do anything if inner is vacuum.
                                    if inner != 0 {
                                        let index = tracked_mediums
                                            .iter()
                                            .position(|e| *e == inner)
                                            .expect("should have found correct medium.");
                                        tracked_mediums.remove(index);
                                    }
                                    if outer != 0 {
                                        tracked_mediums.push(outer);
                                        tracked_mediums.sort_unstable();
                                    }
                                }
                            }
                        }
                        ray = Ray::new(
                            isect.point
                                + (if wo.z() > 0.0 { 1.0 } else { -1.0 }) * isect.normal * 0.000001,
                            frame.to_world(&wo).normalized(),
                        );
                    }
                    IntersectionData::Medium(isect) => {
                        // println!("medium interaction {}", isect.medium_id);
                        let medium = &mediums[isect.medium_id - 1];
                        let wi = -ray.direction;
                        let (wo, f_and_pdf) =
                            medium.sample_p(lambdas.extract(0), wi, Sample2D::new_random_sample());
                        throughput = throughput.replace(0, throughput.extract(0) * f_and_pdf);
                        for i in 1..4 {
                            let f_and_pdf = medium.p(lambdas.extract(i), wi, wo);
                            throughput = throughput.replace(i, throughput.extract(i) * f_and_pdf);
                        }
                        ray = Ray::new(isect.point + wo * 0.000001, wo);
                    }
                }
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
