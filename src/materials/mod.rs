use crate::math::{Sample2D, Vec3};

mod diffuse_emitter;
mod ggx;
mod lambertian;
mod passthrough;

pub use diffuse_emitter::ConstDiffuseEmitter;
pub use ggx::GGX;
pub use lambertian::ConstLambertian;
pub use passthrough::ConstPassthrough;

pub trait Material {
    fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32);
    fn sample(&self, lambda: f32, wi: Vec3, sample: Sample2D) -> Vec3;
    fn emission(&self, _lambda: f32, _wo: Vec3) -> f32 {
        0.0
    }
}

#[macro_export]
macro_rules! generate_enum {
    ( $name:ident, $( $s:ident),+) => {

        #[derive(Clone)]
        pub enum $name {
            $(
                $s($s),
            )+
        }
        $(
            impl From<$s> for $name {
                fn from(value: $s) -> Self {
                    $name::$s(value)
                }
            }
        )+

        impl $name {
            pub fn get_name(&self) -> &str {
                match self {
                    $($name::$s(_) => $s::NAME,)+
                }
            }
        }

        impl Material for $name {
            fn sample(&self, lambda: f32, wi: Vec3, sample: Sample2D) -> Vec3 {
                match self {
                    $($name::$s(mat) => mat.sample(lambda, wi, sample),)+
                }
            }
            fn bsdf(&self, lambda: f32, wi: Vec3, wo: Vec3) -> (f32, f32) {
                match self {
                    $($name::$s(mat) => mat.bsdf(lambda, wi, wo),)+
                }
            }
            fn emission(&self, lambda: f32, wo: Vec3) -> f32 {
                match self {
                    $($name::$s(mat) => mat.emission(lambda, wo),)+
                }
            }
        }
    };
}

generate_enum!(
    MaterialEnum,
    ConstLambertian,
    ConstDiffuseEmitter,
    ConstPassthrough,
    GGX
);

unsafe impl Send for MaterialEnum {}
unsafe impl Sync for MaterialEnum {}
