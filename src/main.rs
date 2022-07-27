#[macro_use]
extern crate packed_simd;
#[macro_use]
extern crate serde;

use std::f32::INFINITY;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use parsing::curves::load_ior_and_kappa;
use pbr::ProgressBar;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use structopt::StructOpt;

pub mod camera;
pub mod film;
pub mod geometry;
pub mod materials;
pub mod math;
pub mod parsing;
pub mod tonemap;

use geometry::SurfaceIntersectionData;

use camera::ProjectiveCamera;
use film::Film;
use geometry::{Intersect, Sphere};
use materials::{ConstDiffuseEmitter, ConstLambertian, Material, MaterialEnum, GGX};

use math::spectral::{
    SingleWavelength, SpectralPowerDistributionFunction, BOUNDED_VISIBLE_RANGE,
    EXTENDED_VISIBLE_RANGE,
};
use math::{
    Bounds1D, Curve, InterpolationMode, Point3, Ray, Sample2D, TangentFrame, Vec3, XYZColor,
};
use parsing::*;
use tonemap::{sRGB, Tonemapper};

fn output_film(filename: Option<&String>, film: &Film<XYZColor>, key_value: f32) {
    let filename_str = filename.cloned().unwrap_or_else(|| String::from("output"));
    let exr_filename = format!("output/{}.exr", filename_str);
    let png_filename = format!("output/{}.png", filename_str);

    let srgb_tonemapper = sRGB::new(film, key_value, true);
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
    max_light_strength: f32,
) {
    let flat_zero = Curve::Linear {
        signal: vec![0.0],
        bounds: EXTENDED_VISIBLE_RANGE,
        mode: InterpolationMode::Linear,
    };

    let air_ior = Curve::Cauchy {
        a: 1.0002724293,
        b: 1.64748969205,
    };

    let mut ggx_db = Vec::new();

    // convert micrometers to nanometers
    let ior_and_kappa =
        load_ior_and_kappa("data/curves/brass_90Cu10Zn.csv", |x| x * 1000.0).unwrap();
    ggx_db.push((ior_and_kappa.0, ior_and_kappa.1, 0.0f32));

    let ior_and_kappa = load_ior_and_kappa("data/curves/gold.csv", |x| x * 1000.0).unwrap();
    ggx_db.push((ior_and_kappa.0, ior_and_kappa.1, 0.0f32));

    let ior_and_kappa =
        load_ior_and_kappa("data/curves/copper-mcpeak.csv", |x| x * 1000.0).unwrap();
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

    for _ in 0..num {
        let sample = (rand::random::<f32>() * 4.0) as usize;
        match sample {
            3 => {
                // emissive
                // generate random curve
                let bounce_color = generate_random_curve(1.0, 20, wavelength_bounds);
                let emit_color = generate_random_curve(max_light_strength, 20, wavelength_bounds);
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
                    roughness,
                    eta,
                    air_ior.clone(),
                    kappa,
                    0.0,
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
                    air_ior.clone(),
                    flat_zero.clone(),
                    1.0,
                )));
            }
            1 | 2 => {
                // choose from a list of iors + kappas, randomize roughness
                let roughness = Bounds1D::new(0.0, max_roughness).sample(rand::random());
                let choice = (rand::random::<f32>() * ggx_db.len() as f32) as usize;
                let (eta, kappa, perm) = ggx_db[choice].clone();
                materials.push(MaterialEnum::GGX(GGX::new(
                    roughness,
                    eta,
                    air_ior.clone(),
                    kappa,
                    perm,
                )));
            }
            _ => {
                // lambertian
                // generate random curve.
                let color = generate_random_curve(1.0, 20, wavelength_bounds);
                materials.push(MaterialEnum::ConstLambertian(ConstLambertian::new(color)));
            }
        }
    }
}

#[derive(StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    #[structopt(long, default_value = "1080")]
    pub width: usize,
    #[structopt(long, default_value = "1080")]
    pub height: usize,
    #[structopt(long, default_value = "128")]
    pub samples: usize,
    #[structopt(long, default_value = "8")]
    pub bounces: usize,
    #[structopt(long, default_value = "0.18")]
    pub key_value: f32,
    pub scene_file: String,
}

const NORMAL_OFFSET: f32 = 0.0001;

fn main() {
    let threads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(if threads > 3 { threads - 2 } else { 1 })
        // .num_threads(threads as usize)
        .build_global()
        .unwrap();

    // rendering constants, i.e. film size and wavelength bounds
    let opts = Opt::from_args();
    let h = opts.height;
    let w = opts.width;
    let samples = opts.samples;
    let bounces = opts.bounces;
    let wavelength_range = BOUNDED_VISIBLE_RANGE;

    let mut film = Film::new(w, h, XYZColor::ZERO);

    let mut scene: Scene = load_json::<SceneData>(PathBuf::from(opts.scene_file))
        .expect("failed to parse scene")
        .into();

    // let ggx_glass = GGX::new(0.001, glass, air_ior, flat_zero, 1.0);

    // let mut materials: Vec<MaterialEnum> = vec![
    //     MaterialEnum::ConstLambertian(ConstLambertian::new(grey.clone())),
    //     MaterialEnum::ConstDiffuseEmitter(ConstDiffuseEmitter::new(white.clone(), white.clone())),
    //     MaterialEnum::GGX(ggx_glass.clone()),
    //     // MaterialEnum::ConstPassthrough(ConstPassthrough::new(white.clone())),
    // ];

    let new_materials_count = 30;
    let mut new_materials = Vec::new();
    generate_and_push_random_materials(
        &mut new_materials,
        new_materials_count,
        EXTENDED_VISIBLE_RANGE,
        false,
        1.0,
    );

    let bag_size_of_random_materials = new_materials_count
        + scene
            .materials
            .iter()
            .filter(|e| matches!(e, MaterialEnum::ConstLambertian(_) | MaterialEnum::GGX(_)))
            .count();
    let mut spheres_to_add: Vec<Sphere> = Vec::new();
    spheres_to_add.push(Sphere::new(4.0, Point3::new(0.0, 0.0, 4.1), 2));
    for _ in 0..100 {
        let material_id = (rand::random::<f32>() * bag_size_of_random_materials as f32) as usize;
        loop {
            let x = (rand::random::<f32>() - 0.5) * 50.0;
            let y = (rand::random::<f32>() - 0.5) * 50.0;
            let mut overlapping = false;
            let candidate_sphere = Sphere::new(1.0, Point3::new(x, y, 1.1), material_id);

            for sphere in &spheres_to_add {
                if (candidate_sphere.origin - sphere.origin).norm()
                    < sphere.radius + candidate_sphere.radius
                {
                    // spheres are overlapping
                    overlapping = true;
                    break;
                }
            }
            if !overlapping {
                spheres_to_add.push(candidate_sphere);
                break;
            }
        }
    }
    // scene
    //     .primitives
    //     .extend(spheres_to_add.drain(..).map(|e| e.into()));

    let camera = ProjectiveCamera::new(
        Point3::new(-10.0, 10.0, 5.0),
        Point3::new(0.0, 0.0, 2.0),
        Vec3::Z,
        60.0,
        1.0,
        14.0,
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
        println!();
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

            for _ in 0..bounces {
                let mut nearest_intersection: Option<SurfaceIntersectionData> = None;
                let mut nearest_intersection_time = INFINITY;

                for prim in scene.primitives.iter() {
                    if let Some(intersection) = prim.intersect(ray, 0.0, INFINITY) {
                        if intersection.time < nearest_intersection_time {
                            nearest_intersection_time = intersection.time;
                            nearest_intersection = Some(intersection);
                        }
                    }
                }

                if nearest_intersection.is_none() {
                    sum.energy.0 += throughput * scene.env_color.evaluate_power(sum.lambda); // hit env
                    assert!(s.is_finite(), "{:?}, {:?}", s, throughput);
                    assert!(throughput.is_finite(), "{:?}", throughput);
                    break;
                }

                let intersection = nearest_intersection.unwrap();

                assert!(throughput.is_finite(), "{:?}", throughput);

                let frame = TangentFrame::from_normal(intersection.normal);
                let wi = frame.to_local(&-ray.direction);
                let mat = &scene.materials[intersection.material_id];

                let wo = mat.sample(sum.lambda, wi, Sample2D::new_random_sample());

                let cos_i = wo.z();

                let (bsdf, pdf) = mat.bsdf(sum.lambda, wi, wo);
                let emission = mat.emission(sum.lambda, wi);

                if emission > 0.0 {
                    sum.energy.0 += throughput * emission * cos_i;
                    assert!(
                        sum.energy.0.is_finite(),
                        "{:?}, {:?}, {:?}, {:?}",
                        s,
                        throughput,
                        emission,
                        cos_i
                    );
                }
                if pdf == 0.0 {
                    break;
                }

                throughput *= bsdf * cos_i.abs() / pdf;
                assert!(
                    throughput.is_finite(),
                    "{:?}, {:?}, {:?}, {:?}",
                    throughput,
                    bsdf,
                    cos_i,
                    pdf
                );

                ray = Ray::new(
                    intersection.point
                        + (if wo.z() > 0.0 { 1.0 } else { -1.0 })
                            * intersection.normal
                            * NORMAL_OFFSET,
                    frame.to_world(&wo).normalized(),
                );
            }

            assert!(!sum.energy.0.is_nan(), "{:?}", s);

            *e += XYZColor::from(sum);
        }
        pixel_count_clone.fetch_add(1, Ordering::Relaxed);
    });
    thread.join().unwrap();
    output_film(None, &film, opts.key_value);
}
