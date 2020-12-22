use crate::math::{Sample2D, SpectralPowerDistributionFunction, Vec3, SPD};
use crate::random::random_cosine_direction;
use std::f32::consts::PI;
pub trait Material {
    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32);
    fn sample(&self, lambda: f32, wi: Vec3, sample: Sample2D) -> Vec3;
    fn emission(&self, _lambda: f32, _wo: Vec3) -> f32 {
        0.0
    }
}

#[derive(Clone)]
pub struct ConstLambertian {
    pub color: SPD,
}

impl ConstLambertian {
    pub fn new(color: SPD) -> ConstLambertian {
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
    // fn emission(&self, _hit: &HitRecord, _wi: Vec3, _wo: Option<Vec3>) -> SingleEnergy {
    //     SingleEnergy::ZERO
    // }
}

unsafe impl Send for ConstLambertian {}
unsafe impl Sync for ConstLambertian {}

#[derive(Clone)]
pub struct ConstDiffuseEmitter {
    pub bounce_color: SPD,
    pub emission_color: SPD,
}

impl ConstDiffuseEmitter {
    pub fn new(bounce_color: SPD, emission_color: SPD) -> ConstDiffuseEmitter {
        ConstDiffuseEmitter {
            bounce_color,
            emission_color,
        }
    }
    pub const NAME: &'static str = "Lambertian";
}

impl Material for ConstDiffuseEmitter {
    fn sample(&self, _lambda: f32, wi: Vec3, s: Sample2D) -> Vec3 {
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
    fn emission(&self, lambda: f32, wo: Vec3) -> f32 {
        let cosine = wo.z();
        // if cosine > 0.0 {
        //     self.emission_color.evaluate_power(lambda) / PI
        // } else {
        //     0.0
        // }
        self.emission_color.evaluate_power(lambda) / PI
    }
}

unsafe impl Send for ConstDiffuseEmitter {}
unsafe impl Sync for ConstDiffuseEmitter {}

pub enum MaterialEnum {
    ConstLambertian(ConstLambertian),
    ConstDiffuseEmitter(ConstDiffuseEmitter),
}

impl Material for MaterialEnum {
    fn sample(&self, lambda: f32, wi: Vec3, sample: Sample2D) -> Vec3 {
        match self {
            MaterialEnum::ConstLambertian(mat) => mat.sample(lambda, wi, sample),
            MaterialEnum::ConstDiffuseEmitter(mat) => mat.sample(lambda, wi, sample),
        }
    }
    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32) {
        match self {
            MaterialEnum::ConstLambertian(mat) => mat.bsdf(lambda, wi, wo),
            MaterialEnum::ConstDiffuseEmitter(mat) => mat.bsdf(lambda, wi, wo),
        }
    }
    fn emission(&self, lambda: f32, wo: Vec3) -> f32 {
        match self {
            MaterialEnum::ConstLambertian(mat) => mat.emission(lambda, wo),
            MaterialEnum::ConstDiffuseEmitter(mat) => mat.emission(lambda, wo),
        }
    }
}

unsafe impl Send for MaterialEnum {}
unsafe impl Sync for MaterialEnum {}
