use std::f32::consts::PI;

use super::Material;
use crate::math::{random_cosine_direction, Curve, Sample2D, Vec3};

#[derive(Clone)]
pub struct ConstLambertian {
    pub color: Curve,
}

impl ConstLambertian {
    pub fn new(color: Curve) -> ConstLambertian {
        ConstLambertian { color }
    }
    pub const NAME: &'static str = "Lambertian";
}

impl Material for ConstLambertian {
    fn sample(&self, _lambda: f32, _wi: Vec3, s: Sample2D) -> Vec3 {
        random_cosine_direction(s)
    }

    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32) {
        let cosine = wo.z();
        if cosine * wi.z() > 0.0 {
            (self.color.evaluate(lambda) / PI, cosine / PI)
        } else {
            (0.0, 0.0)
        }
    }
    // fn emission(&self, _lambda: f32, _wo: Vec3) -> f32 {
    //     0.0
    // }
}

unsafe impl Send for ConstLambertian {}
unsafe impl Sync for ConstLambertian {}
