use std::f32::consts::PI;

use super::Material;
use crate::math::{
    random_cosine_direction, spectral::SpectralPowerDistributionFunction, Curve, Sample2D, Vec3,
};

#[derive(Clone)]
pub struct ConstDiffuseEmitter {
    pub bounce_color: Curve,
    pub emission_color: Curve,
}

impl ConstDiffuseEmitter {
    pub fn new(bounce_color: Curve, emission_color: Curve) -> ConstDiffuseEmitter {
        ConstDiffuseEmitter {
            bounce_color,
            emission_color,
        }
    }
    pub const NAME: &'static str = "Lambertian";
}

impl Material for ConstDiffuseEmitter {
    fn sample(&self, _lambda: f32, _wi: Vec3, s: Sample2D) -> Vec3 {
        random_cosine_direction(s)
    }

    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32) {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            (self.bounce_color.evaluate(lambda) / PI, cosine / PI)
        } else {
            (0.0, 0.0)
        }
    }
    fn emission(&self, lambda: f32, _wo: Vec3) -> f32 {
        self.emission_color.evaluate_power(lambda) / PI
    }
}

unsafe impl Send for ConstDiffuseEmitter {}
unsafe impl Sync for ConstDiffuseEmitter {}
