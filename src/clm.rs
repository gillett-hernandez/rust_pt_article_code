// use math::XYZColor;
use crate::material::refract;
use crate::material::{Material, TransportMode, GGX};
use math::*;
use std::f32::consts::PI;

pub fn balance(f: f32, g: f32) -> f32 {
    f / (f + g)
}

#[derive(Clone)]
pub enum Layer {
    Diffuse { color: SPD },
    Dielectric(GGX),
    Emissive { color: SPD },
    None,
}

impl Layer {
    pub fn bsdf(
        &self,
        lambda: f32,
        wi: Vec3,
        wo: Vec3,
        transport_mode: TransportMode,
    ) -> (f32, f32) {
        match self {
            Layer::Diffuse { color } => {
                let cosine = wo.z();
                if cosine * wi.z() > 0.0 {
                    (color.evaluate(lambda).min(1.0) / PI, (cosine / PI))
                } else {
                    (0.0, 0.0)
                }
            }
            Layer::Dielectric(ggx) => ggx.bsdf(lambda, wi, wo),
            Layer::Emissive { color } => (0.0, 0.0),
            Layer::None => (0.0, 0.0),
        }
    }
    pub fn sample(
        &self,
        lambda: f32,
        wi: Vec3,
        sample: Sample2D,
        transport_mode: TransportMode,
    ) -> Vec3 {
        match self {
            Layer::Diffuse { color: _ } => random_cosine_direction(sample),
            Layer::Dielectric(ggx) => ggx.sample(lambda, wi, sample),
            Layer::Emissive { color } => random_cosine_direction(sample),
            Layer::None => panic!(),
        }
    }
    /*fn transmit(&self, data: &LayerData, wo: Vector3) -> Option<Vector3> {
        let eta = Self::eta(data, wo)?;
        Vector3::new(0.0, 0.0, 1.0).refract(wo, 1.0 / eta.extract(0))
    }*/
    pub fn perfect_transmission(&self, lambda: f32, wo: Vec3) -> Option<Vec3> {
        match self {
            Layer::Dielectric(ggx) => {
                // println!("ggx perfect transmission");
                let eta = ggx.eta.evaluate(lambda);
                refract(wo, Vec3::Z, 1.0 / eta)
            }
            Layer::Diffuse { .. } => {
                // println!("diffuse perfect transmission (no transmission)");
                None
            }
            Layer::Emissive { color } => None,
            Layer::None => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CLMVertex {
    pub wi: Vec3,
    pub wo: Vec3,
    pub throughput: f32,
    pub path_pdf: f32,
    pub index: usize,
}
#[derive(Clone, Debug)]
pub struct CLMPath(pub Vec<CLMVertex>);

#[derive(Clone)]
pub struct CLM {
    // 0 is base layer, other layers are on top
    pub layers: Vec<Layer>,
    pub bounce_limit: usize,
}

impl CLM {
    pub fn new(layers: Vec<Layer>, bounce_limit: usize) -> Self {
        CLM {
            layers,
            bounce_limit,
        }
    }
    pub fn generate_short(
        &self,
        lambda: f32,
        mut wo: Vec3,
        transport_mode: TransportMode,
    ) -> CLMPath {
        let mut path = Vec::new();
        let num_layers = self.layers.len();
        let (mut index, direction) = if wo.z() > 0.0 {
            // println!("index {}-1 case", num_layers);
            (num_layers - 1, -1)
        } else {
            // println!("index 0 case");
            (0, 1)
        };
        let mut throughput = 1.0;
        let mut path_pdf = 1.0;
        loop {
            let layer = &self.layers[index];
            // println!(
            //     "calling perfect transmission on layer {} with wo = {:?}",
            //     index, wo
            // );
            let wi = match layer.perfect_transmission(lambda, wo) {
                Some(wi) => {
                    path.push(CLMVertex {
                        wi,
                        wo,
                        throughput,
                        path_pdf,
                        index,
                    });
                    wi
                }
                None => {
                    path.push(CLMVertex {
                        wi: Vec3::Z,
                        wo,
                        throughput: 0.0,
                        path_pdf: 0.0,
                        index,
                    });
                    break;
                }
            };

            if (index == 0 && direction == -1) || (index + 1 == num_layers && direction == 1) {
                // println!("broke2");
                break;
            }
            let (f, pdf) = layer.bsdf(lambda, wi, wo, transport_mode);
            throughput *= f;
            path_pdf *= pdf;
            // println!("gs {:?} {:?}", throughput, path_pdf);

            index = (index as isize + direction) as usize;

            wo = -wi;
        }

        CLMPath(path)
    }
    pub fn generate(
        &self,
        lambda: f32,
        mut wi: Vec3,
        sampler: &mut dyn Sampler,
        transport_mode: TransportMode,
    ) -> CLMPath {
        let mut path = Vec::new();
        let num_layers = self.layers.len();
        let mut index = if wi.z() > 0.0 { num_layers - 1 } else { 0 };

        for _ in 0..self.bounce_limit {
            let layer = &self.layers[index];
            let wo = layer.sample(lambda, wi, sampler.draw_2d(), transport_mode);

            // println!("g {:?} {:?}", wi, wo);

            path.push(CLMVertex {
                wi,
                wo,
                throughput: 0.0,
                path_pdf: 0.0,
                index,
            });

            let is_up = wo.z() > 0.0;

            if !is_up && index > 0 {
                index -= 1;
            } else if is_up && index + 1 < num_layers {
                index += 1;
            } else {
                break;
            }

            wi = -wo;
        }

        CLMPath(path)
    }
    pub fn bsdf_eval(
        &self,
        lambda: f32,
        long_path: &CLMPath,
        short_path: &CLMPath,
        _sampler: &mut dyn Sampler,
        transport_mode: TransportMode,
    ) -> (f32, f32) {
        let _wi = long_path.0.first().unwrap().wi;
        let _wo = short_path.0.first().unwrap().wo;
        // let num_layers = self.layers.len();
        self.eval_path_full(lambda, long_path, short_path, transport_mode)
    }
    pub fn eval_path_full(
        &self,
        lambda: f32,
        long_path: &CLMPath,
        short_path: &CLMPath,
        transport_mode: TransportMode,
    ) -> (f32, f32) {
        let _opposite_mode = match transport_mode {
            TransportMode::Importance => TransportMode::Radiance,
            TransportMode::Radiance => TransportMode::Importance,
        };
        let mut sum = 0.0;
        let mut pdf_sum = 0.0;
        let wo = short_path.0.first().unwrap().wo;

        let nee_direction = if wo.z() > 0.0 { 1 } else { -1 };

        let mut throughput = 1.0;
        let mut path_pdf = 1.0;
        let num_samples = long_path.0.len();
        let nee_distance = short_path.0.len();

        for vert in long_path.0.iter() {
            let index = vert.index;
            let layer = &self.layers[index];
            let nee_index = index as isize + nee_direction;
            let wi = Vec3::ZERO; // TODO: fix this

            if nee_index < 0 || nee_index as usize >= self.layers.len() {
                let nee_wo = if nee_index < 0 {
                    short_path.0.first().unwrap().wo
                } else {
                    short_path.0.last().unwrap().wo
                };
                let (left_f, left_path_pdf) = (throughput, path_pdf);

                let (left_connection_f, left_connection_pdf) =
                    layer.bsdf(lambda, vert.wi, nee_wo, transport_mode);

                let (total_throughput, total_path_pdf) = (
                    left_f * left_connection_f,
                    left_path_pdf * left_connection_pdf,
                );

                let (f, pdf) = layer.bsdf(lambda, wi, wo, transport_mode);

                let weight = balance(left_connection_pdf, pdf);

                if total_path_pdf > 0.0 {
                    let addend = weight * total_throughput / total_path_pdf;
                    sum += addend;
                    pdf_sum += total_path_pdf;
                    // println!("a {} {}", addend, total_path_pdf);
                }

                throughput *= (1.0 - weight) * f;
                path_pdf *= pdf
            } else {
                let nee_index = nee_index as usize;
                let nee_layer = &self.layers[nee_index];
                let nee_vert = short_path.0[short_path.0.len() - nee_index];

                let (left_f, left_path_pdf) = (throughput, path_pdf);

                let (right_f, right_path_pdf) = (nee_vert.throughput, nee_vert.path_pdf);

                let (left_connection_f, left_connection_pdf) =
                    layer.bsdf(lambda, vert.wi, -nee_vert.wi, transport_mode);
                let (right_connection_f, right_connection_pdf) =
                    nee_layer.bsdf(lambda, nee_vert.wi, nee_vert.wo, transport_mode);

                let (total_throughput, total_path_pdf) = (
                    left_f * left_connection_f * right_connection_f * right_f,
                    left_path_pdf * left_connection_pdf * right_connection_pdf * right_path_pdf,
                );

                let (f, pdf) = layer.bsdf(lambda, wi, wo, transport_mode);

                let weight = balance(
                    left_connection_pdf * right_connection_pdf * right_path_pdf,
                    pdf,
                );

                if total_path_pdf > 0.0 {
                    let addend = weight * total_throughput / total_path_pdf;
                    sum += addend;
                    pdf_sum += total_path_pdf;
                    // println!("a2 {} {}", addend, total_path_pdf);
                }

                throughput *= (1.0 - weight) * f;
                path_pdf *= pdf;
            }
        }
        (
            (sum / num_samples as f32).into(),
            (pdf_sum / num_samples as f32).into(),
        )
    }
}

impl Material for CLM {
    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32) {
        let mut sampler = RandomSampler::new();
        let path = self.generate(
            lambda,
            Vec3::new(1.0, 0.0, 10.0).normalized(),
            &mut sampler,
            TransportMode::Importance,
        );
        // println!("long path finished");

        let wo = path.0.last().unwrap().wo;

        let short_path = self.generate_short(lambda, wo, TransportMode::Radiance);

        let (f, pdf) = self.bsdf_eval(
            lambda,
            &path,
            &short_path,
            &mut sampler,
            TransportMode::Importance,
        );
        (f, pdf)
    }
    fn sample(&self, lambda: f32, wi: Vec3, sample: Sample2D) -> Vec3 {
        let mut sampler = RandomSampler::new();
        let path = self.generate(
            lambda,
            Vec3::new(1.0, 0.0, 10.0).normalized(),
            &mut sampler,
            TransportMode::Importance,
        );
        path.0.last().unwrap().wo
    }
    fn emission(&self, _lambda: f32, _wo: Vec3) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_clm() {
        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());

        let glass = SPD::Cauchy { a: 1.5, b: 10000.0 };
        let flat_zero = SPD::Linear {
            signal: vec![0.0],
            bounds: Bounds1D::new(390.0, 750.0),
            mode: InterpolationMode::Linear,
        };
        let ggx_glass = GGX::new(0.00001, glass, 1.0, flat_zero, 1.0, 0);

        let white = SPD::Linear {
            signal: vec![0.9],
            bounds: Bounds1D::new(390.0, 750.0),
            mode: InterpolationMode::Linear,
        };
        let clm = CLM::new(
            vec![
                Layer::Diffuse { color: white },
                Layer::Dielectric(ggx_glass.clone()),
                // Layer::Dielectric(ggx_glass.clone()),
            ],
            20,
        );

        let lambda = 500.0;

        let path = clm.generate(
            lambda,
            Vec3::new(1.0, 0.0, 10.0).normalized(),
            &mut *sampler,
            TransportMode::Importance,
        );
        println!("long path finished");

        let wo = path.0.last().unwrap().wo;

        let short_path = clm.generate_short(lambda, wo, TransportMode::Radiance);

        let (f, pdf) = clm.bsdf_eval(
            lambda,
            &path,
            &short_path,
            &mut *sampler,
            TransportMode::Importance,
        );
        println!("{}, {}", f, pdf);
    }

    const WINDOW_HEIGHT: usize = 800;
    const WINDOW_WIDTH: usize = 800;
    #[test]
    fn visualize_clm() {
        use crate::Film;
        use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
        use ordered_float::OrderedFloat;
        use packed_simd::f32x4;
        use rand::prelude::*;
        use rayon::prelude::*;
        rayon::ThreadPoolBuilder::new()
            .num_threads(22 as usize)
            .build_global()
            .unwrap();
        let mut window = Window::new(
            "Lens",
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            WindowOptions {
                scale: Scale::X1,
                ..WindowOptions::default()
            },
        )
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });

        let mut film = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, XYZColor::BLACK);
        let mut window_pixels = Film::new(WINDOW_WIDTH, WINDOW_HEIGHT, 0u32);
        window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
        let width = film.width;
        let height = film.height;
        let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());

        let glass = SPD::Cauchy { a: 1.5, b: 10000.0 };
        let flat_zero = SPD::Linear {
            signal: vec![0.0],
            bounds: Bounds1D::new(390.0, 750.0),
            mode: InterpolationMode::Linear,
        };
        let ggx_glass = GGX::new(0.00001, glass, 1.0, flat_zero, 1.0, 0);

        let white = SPD::Linear {
            signal: vec![0.9],
            bounds: Bounds1D::new(390.0, 750.0),
            mode: InterpolationMode::Linear,
        };
        let clm = CLM::new(
            vec![
                Layer::Diffuse { color: white },
                Layer::Dielectric(ggx_glass.clone()),
                // Layer::Dielectric(ggx_glass.clone()),
            ],
            20,
        );
        while window.is_open() && !window.is_key_down(Key::Escape) {
            let keys = window.get_keys_pressed(KeyRepeat::No);

            for key in keys.unwrap_or(vec![]) {
                match key {
                    _ => {}
                }
            }
            std::thread::sleep(std::time::Duration::new(0, 16000000));

            window
                .update_with_buffer(&window_pixels.buffer, WINDOW_WIDTH, WINDOW_HEIGHT)
                .unwrap();
        }
    }
}
