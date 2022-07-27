use super::Material;
use crate::math::{Curve, Sample2D, Vec3};

#[derive(Clone)]
pub struct ConstPassthrough {
    pub color: Curve,
}

impl ConstPassthrough {
    pub fn new(color: Curve) -> ConstPassthrough {
        ConstPassthrough { color }
    }
    pub const NAME: &'static str = "Film";
}

impl Material for ConstPassthrough {
    fn sample(&self, _lambda: f32, wi: Vec3, _s: Sample2D) -> Vec3 {
        -wi
    }

    fn bsdf(&self, lambda: f32, _wi: Vec3, wo: Vec3) -> (f32, f32) {
        (self.color.evaluate(lambda) / wo.z().abs(), 1.0)
    }
}

unsafe impl Send for ConstPassthrough {}
unsafe impl Sync for ConstPassthrough {}
