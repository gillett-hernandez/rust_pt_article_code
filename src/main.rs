#[macro_use]
extern crate packed_simd;

use std::f32::INFINITY;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use packed_simd::f32x4;
use pbr::ProgressBar;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub mod camera;
pub mod film;
pub mod geometry;
pub mod materials;
pub mod math;
pub mod parsing;
pub mod tonemap;

use crate::math::spectral::{SingleWavelength, SpectralPowerDistributionFunction};
use crate::math::{Sample1D, Sample2D};

use camera::ProjectiveCamera;
use film::Film;
use geometry::{
    IntersectionData, MediumIntersectionData, Primitive, Sphere, SurfaceIntersectionData,
};
use materials::{
    ConstDiffuseEmitter, ConstLambertian, ConstPassthrough, HenyeyGreensteinHomogeneous, Material,
    MaterialEnum, Medium, MediumEnum, GGX,
};
use math::spectral::{BOUNDED_VISIBLE_RANGE, EXTENDED_VISIBLE_RANGE};
use math::*;
use parsing::*;
use tonemap::{sRGB, Tonemapper};

fn output_film(filename: Option<&String>, film: &Film<XYZColor>) {
    let filename_str = filename.cloned().unwrap_or_else(|| String::from("output"));
    let exr_filename = format!("output/{}.exr", filename_str);
    let png_filename = format!("output/{}.png", filename_str);

    let srgb_tonemapper = sRGB::new(film, 1.0);
    srgb_tonemapper.write_to_files(film, &exr_filename, &png_filename);
}

fn generate_random_curve(max_spike_size: f32, num_bins: usize, bounds: Bounds1D) -> Curve {
    // generate using the linear form of the curve
    let mut data = Vec::new();
    for _ in 0..num_bins {
        data.push(max_spike_size * rand::random::<f32>());
    }
    Curve::Linear {
        signal: data,
        bounds,
        mode: InterpolationMode::Cubic,
    }
}

fn generate_and_push_random_materials(
    materials: &mut Vec<MaterialEnum>,
    num: usize,
    wavelength_bounds: Bounds1D,
    generate_nonphysical_metals_and_dielectrics: bool,
) {
    let flat_zero = Curve::Linear {
        signal: vec![0.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let mut ggx_db = Vec::new();

    // convert micrometers to nanometers
    let ior_and_kappa = load_ior_and_kappa("data/brass_90Cu10Zn.csv", |x| x * 1000.0).unwrap();
    ggx_db.push((ior_and_kappa.0, ior_and_kappa.1, 0.0f32));

    let ior_and_kappa = load_ior_and_kappa("data/gold.csv", |x| x * 1000.0).unwrap();
    ggx_db.push((ior_and_kappa.0, ior_and_kappa.1, 0.0f32));

    let ior_and_kappa = load_ior_and_kappa("data/copper-mcpeak.csv", |x| x * 1000.0).unwrap();
    ggx_db.push((ior_and_kappa.0, ior_and_kappa.1, 0.0f32));

    ggx_db.push((Curve::Cauchy { a: 1.5, b: 10000.0 }, flat_zero.clone(), 1.0));
    ggx_db.push((Curve::Cauchy { a: 1.4, b: 30000.0 }, flat_zero.clone(), 1.0));
    ggx_db.push((
        Curve::Cauchy {
            a: 1.5,
            b: 100000.0,
        },
        flat_zero.clone(),
        1.0,
    ));

    let max_roughness = 0.1;

    for i in 0..num {
        let sample = (rand::random::<f32>() * 4.0) as usize;
        match sample {
            3 => {
                // emissive
                // generate random curve
                let bounce_color = generate_random_curve(1.0, 20, wavelength_bounds);
                let emit_color = generate_random_curve(2.0, 20, wavelength_bounds);
                materials.push(MaterialEnum::ConstDiffuseEmitter(ConstDiffuseEmitter::new(
                    bounce_color,
                    emit_color,
                )));
            }
            2 if generate_nonphysical_metals_and_dielectrics => {
                // metal

                let eta = generate_random_curve(1.5, 5, wavelength_bounds);
                let kappa = generate_random_curve(5.0, 5, wavelength_bounds);
                let roughness = Bounds1D::new(0.0, max_roughness).sample(rand::random());
                materials.push(MaterialEnum::GGX(GGX::new(
                    roughness, eta, 1.0, kappa, 0.0, 0,
                )));
            }
            1 if generate_nonphysical_metals_and_dielectrics => {
                // dielectric

                let ior_base = Bounds1D::new(1.05, 2.2).sample(rand::random());
                let cauchy_coef = Bounds1D::new(300.0, 100000.0).sample(rand::random());
                let curve = Curve::Cauchy {
                    a: ior_base,
                    b: cauchy_coef,
                };
                let roughness = Bounds1D::new(0.0, max_roughness).sample(rand::random());
                materials.push(MaterialEnum::GGX(GGX::new(
                    roughness,
                    curve,
                    1.0,
                    flat_zero.clone(),
                    1.0,
                    0,
                )));
            }
            1 | 2 => {
                // choose from a list of iors + kappas, randomize roughness
                let roughness = Bounds1D::new(0.0, max_roughness).sample(rand::random());
                let choice = (rand::random::<f32>() * ggx_db.len() as f32) as usize;
                let (eta, kappa, perm) = ggx_db[choice].clone();
                materials.push(MaterialEnum::GGX(GGX::new(
                    roughness, eta, 1.0, kappa, perm, 0,
                )));
            }
            _ => {
                // lambertian
                // generate random curve.
                let color = generate_random_curve(1.0, 20, wavelength_bounds);
                materials.push(MaterialEnum::ConstLambertian(ConstLambertian::new(color)));
            }
        }
        // generate random color
    }
}

fn main() {
    let threads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(if threads > 3 { threads - 2 } else { 1 })
        // .num_threads(threads as usize)
        .build_global()
        .unwrap();

    // rendering constants, i.e. film size and wavelength bounds
    let h = 1024;
    let w = 1024;
    let wavelength_range = BOUNDED_VISIBLE_RANGE;
    let samples = 256;
    let bounces = 12;

    let mut film = Film::new(w, h, XYZColor::ZERO);

    // commonly used colors
    let white = Curve::Linear {
        signal: vec![1.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let off_white = Curve::Linear {
        signal: vec![0.95],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let blueish = Curve::Linear {
        signal: vec![0.9, 0.7, 0.5, 0.4, 0.2, 0.1],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let mut rayleigh = Vec::new();
    for i in 0..100 {
        let lambda = EXTENDED_VISIBLE_RANGE.sample(i as f32 / 100.0);
        rayleigh.push(100000000.0 * lambda.powf(-4.0));
    }
    println!("{:?}", rayleigh);
    let rayleigh_color = Curve::Linear {
        signal: rayleigh,
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Cubic,
    };
    let grey = Curve::Linear {
        signal: vec![0.2],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };
    let black_ish = Curve::Linear {
        signal: vec![0.01],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };

    let env_color = blueish.clone();

    let glass = Curve::Cauchy {
        a: 1.5,
        b: 100000.0,
    };
    let flat_zero = Curve::Linear {
        signal: vec![0.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };

    let ggx_glass = GGX::new(0.001, glass, 1.0, flat_zero, 1.0, 0);

    let mut materials: Vec<MaterialEnum> = vec![
        MaterialEnum::ConstLambertian(ConstLambertian::new(grey.clone())),
        MaterialEnum::ConstDiffuseEmitter(ConstDiffuseEmitter::new(white.clone(), white.clone())),
        MaterialEnum::GGX(ggx_glass.clone()),
        // MaterialEnum::ConstPassthrough(ConstPassthrough::new(white.clone())),
    ];

    let new_materials_count = 30;
    generate_and_push_random_materials(
        &mut materials,
        new_materials_count,
        EXTENDED_VISIBLE_RANGE,
        true,
    );
    let bag_size_of_random_materials = new_materials_count + 3;

    let bright_emitter_id = materials.len();
    materials.push(MaterialEnum::ConstDiffuseEmitter(ConstDiffuseEmitter::new(
        white.clone(),
        Curve::Linear {
            signal: vec![20.0],
            bounds: EXTENDED_VISIBLE_RANGE,
            mode: InterpolationMode::Linear,
        },
    )));

    let mediums: Vec<MediumEnum> = vec![
        MediumEnum::HenyeyGreensteinHomogeneous(HenyeyGreensteinHomogeneous {
            g: 0.1,
            sigma_s: off_white.clone(),
            sigma_t: rayleigh_color.clone(),
        }),
        MediumEnum::HenyeyGreensteinHomogeneous(HenyeyGreensteinHomogeneous {
            g: 0.0,
            sigma_s: off_white.clone(),
            sigma_t: black_ish.clone(),
        }),
    ];

    // the actual scene, only spheres in this case.
    
    let mut scene: Vec<Sphere> = vec![
        Sphere::new(3.0, Point3::new(0.0, 0.0, 10.0), bright_emitter_id, 0, 0), // light
        Sphere::new(1000.0, Point3::new(0.0, 0.0, -1000.0), 0, 0, 0),           // floor
    ];
    // Sphere::new(3.0, Point3::new(0.0, 0.0, 0.0), 2, 1, 2), // smaller bubble of scattering. inner medium is `2`. outer medium is `1`. surface is transparent shell.
    // Sphere::new(20.0, Point3::new(0.0, 0.0, 0.0), 2, 0, 1), // large bubble of scattering. inner medium is `1`. outer medium is `0`. surface is transparent shell.

    let mut spheres_to_add: Vec<Sphere> = Vec::new();
    for i in 0..100 {
        let material_id = (rand::random::<f32>() * bag_size_of_random_materials as f32) as usize;
        loop {
            let x = (rand::random::<f32>() - 0.5) * 50.0;
            let y = (rand::random::<f32>() - 0.5) * 50.0;
            let mut overlapping = false;
            let candidate_sphere = Sphere::new(1.0, Point3::new(x, y, 1.1), material_id, 0, 0);

            for sphere in &spheres_to_add {
                if (candidate_sphere.origin - sphere.origin).norm() < 2.0 {
                    // spheres are overlapping
                    overlapping = true;
                }
            }
            if !overlapping {
                println!("added sphere {}", i);
                spheres_to_add.push(candidate_sphere);
                break;
            }
        }
    }
    scene.extend(spheres_to_add.drain(..));
    let camera = ProjectiveCamera::new(
        Point3::new(-10.0, 10.0, 5.0),
        Point3::ORIGIN,
        Vec3::Z,
        60.0,
        1.0,
        5.0,
        0.01,
        0.0,
        1.0,
    );

    let total_pixels = w * h;
    let mut pb = ProgressBar::new((total_pixels) as u64);
    // progress tracking thread

    let pixel_count = Arc::new(AtomicUsize::new(0));
    let pixel_count_clone = pixel_count.clone();

    let thread = thread::spawn(move || {
        let mut local_index = 0;
        while local_index < total_pixels {
            let pixels_to_increment = pixel_count_clone.load(Ordering::Relaxed) - local_index;
            pb.add(pixels_to_increment as u64);
            local_index += pixels_to_increment;

            thread::sleep(Duration::from_millis(250));
        }

        pb.finish();
    });

    let pixel_count_clone = pixel_count.clone();
    film.buffer.par_iter_mut().enumerate().for_each(|(i, e)| {
        let x = i % w;
        let y = i / w;

        for _ in 0..samples {
            let mut sum = SingleWavelength::new_from_range(rand::random::<f32>(), wavelength_range);
            let (s, t) = (
                (x as f32 + rand::random::<f32>()) / (w as f32),
                (y as f32 + rand::random::<f32>()) / (h as f32),
            );
            let aperture_sample = Sample2D::new_random_sample();
            let mut ray = camera.get_ray(aperture_sample, s, t);

            let mut throughput = 1.0f32;

            // somehow determine what medium the camera ray starts in. assume vacuum for now
            let mut tracked_mediums: Vec<usize> = Vec::new();
            // tracked_mediums.push(1);

            for _ in 0..bounces {
                // if tracked_mediums.len() > 0 {
                //     println!("\n{:?}", tracked_mediums);
                // }
                // tracked_mediums.dedup();
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

                // TODO: handle case where there's still a tracked medium to potentially scatter off of. instead of just going straight to the environment. maybe this implicitly handles it since global volumes shouldn't be a thing? idk.
                // i.e. transmittance along a ray that travels infinitely would almost always be 0.0
                if nearest_intersection.is_none() {
                    sum.energy.0 += throughput * env_color.evaluate_power(sum.lambda); // hit env
                    assert!(s.is_finite(), "{:?}, {:?}", s, throughput);
                    assert!(throughput.is_finite(), "{:?}", throughput);
                    break;
                }
                // iterate through volumes and sample each, choosing the closest medium scattering (or randomly sampling?)
                let mut intersection = nearest_intersection.unwrap();
                let mut closest_p = if let IntersectionData::Surface(sid) = intersection {
                    sid.point
                } else {
                    panic!();
                };
                // let mut any_scatter = false;
                for medium_id in tracked_mediums.iter() {
                    let medium = &mediums[*medium_id - 1];
                    let (p, _tr, scatter) =
                        medium.sample(sum.lambda, ray, Sample1D::new_random_sample());
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

                    let mut combined_throughput = 1.0;
                    for medium_id in tracked_mediums.iter() {
                        let medium = &mediums[*medium_id - 1];
                        combined_throughput *= medium.tr(sum.lambda, ray.origin, closest_p);
                    }
                    throughput *= combined_throughput;

                    assert!(throughput.is_finite(), "{:?}", throughput);



                match intersection {
                    IntersectionData::Surface(isect) => {
                        let frame = TangentFrame::from_normal(isect.normal);
                        let wi = frame.to_local(&-ray.direction);
                        let mat = &materials[isect.material_id];
                        // let skip_throughput_mod = if let MaterialEnum::ConstFilm(e) = mat {
                        //     true
                        // } else {
                        //     false
                        // };

                        let outer = isect.outer_medium_id;
                        let inner = isect.inner_medium_id;

                        let wo = mat.sample(sum.lambda, wi, Sample2D::new_random_sample());

                        let cos_i = wo.z();


                        let (bsdf, pdf) = mat.bsdf(sum.lambda, wi, wo);
                        let emission = mat.emission(sum.lambda, wi);

                        if emission > 0.0 {
                            sum.energy.0 += throughput * emission * cos_i;
                            assert!(sum.energy.0.is_finite(), "{:?}, {:?}, {:?}, {:?}", s, throughput, emission, cos_i);
                        }
                        if pdf == 0.0 {
                            break;
                        }

                        throughput *= bsdf * cos_i.abs() / pdf;
                        assert!(throughput.is_finite(), "{:?}, {:?}, {:?}, {:?}", throughput, bsdf, cos_i, pdf);
                        if wi.z() * wo.z() > 0.0 {
                            // scattering, so don't mess with volumes
                            // println!("reflect, {}, {}", outer, inner);
                        } else {
                            // transmitting, so remove appropriate medium from list and add new one. only applicable if inner != outer

                            if inner != outer {
                                // println!(
                                // "transmit, {}, {}, {:?}, {:?}, {:?}",
                                // outer, inner, wo, isect.normal, tracked_mediums
                                // );
                                // print!("{} ", isect.material_id);
                                if wo.z() < 0.0 {
                                    // println!("wo.z < 0, wi: {:?}, wo: {:?}", wi, wo);
                                    // transmitting from outer to inner. thus remove outer and add inner
                                    if outer != 0 {
                                        // only remove outer if it's not the Vacuum index.
                                        match tracked_mediums.iter().position(|e| *e == outer) {
                                            Some(index) => {
                                                tracked_mediums.remove(index);
                                            }
                                            None => {
                                                println!(
                                                    "warning: attempted to transition out of a medium that was not being tracked. tracked mediums already was {:?}. transmit from {} to {}, {:?}, {:?}.",
                                                    tracked_mediums, outer, inner, wi, wo
                                                );
                                            }
                                        }
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
                                        match tracked_mediums.iter().position(|e| *e == inner) {
                                            Some(index) => {
                                                tracked_mediums.remove(index);
                                            }
                                            None => {
                                                println!(
                                                    "warning: attempted to transition out of a medium that was not being tracked. tracked mediums already was {:?}. transmit from {} to {}, {:?}, {:?}.",
                                                     tracked_mediums, inner,outer, wi, wo
                                                );
                                            }
                                        }
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
                                + (if wo.z() > 0.0 { 1.0 } else { -1.0 }) * isect.normal * 0.0001,
                            frame.to_world(&wo).normalized(),
                        );
                    }
                    IntersectionData::Medium(isect) => {
                        let medium = &mediums[isect.medium_id - 1];
                        let wi = -ray.direction;
                        let (wo, f_and_pdf) =
                            medium.sample_p(sum.lambda, wi, Sample2D::new_random_sample());

                        // println!("medium interaction {}, wi = {:?}, wo = {:?}", isect.medium_id, wi, wo);
                        throughput *= f_and_pdf;

                        ray = Ray::new(isect.point, wo);
                        assert!(!throughput.is_nan(), "{:?}", throughput);
                    }
                }
            }

            assert!(!sum.energy.0.is_nan(), "{:?}", s);



            *e += XYZColor::from(sum);

        }
        pixel_count_clone.fetch_add(1, Ordering::Relaxed);

    });
    thread.join().unwrap();
    output_film(None, &film);
}
