use std::collections::HashMap;

use crate::{
    materials::{ConstDiffuseEmitter, ConstLambertian, ConstPassthrough, MaterialEnum, GGX},
    math::Curve,
};

use super::curves::CurveDataOrReference;

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MaterialData {
    GGX {
        ior: CurveDataOrReference,
        ior_outer: CurveDataOrReference,
        kappa: CurveDataOrReference,
        permeability: f32,
        roughness: f32,
    },
    Lambertian {
        color: CurveDataOrReference,
    },
    Passthrough {
        color: CurveDataOrReference,
    },
    DiffuseEmitter {
        bounce_color: CurveDataOrReference,
        emit_color: CurveDataOrReference,
    },
}

impl MaterialData {
    pub fn transform(self, curves: &HashMap<String, Curve>) -> Option<MaterialEnum> {
        match self {
            MaterialData::DiffuseEmitter {
                bounce_color,
                emit_color,
            } => {
                if let (Some(bounce), Some(emit)) =
                    (bounce_color.resolve(curves), emit_color.resolve(curves))
                {
                    Some(MaterialEnum::ConstDiffuseEmitter(ConstDiffuseEmitter::new(
                        bounce, emit,
                    )))
                } else {
                    None
                }
            }
            MaterialData::GGX {
                ior,
                kappa,
                ior_outer,
                permeability,
                roughness,
            } => {
                if let (Some(ior), Some(eta_o), Some(kappa)) = (
                    ior.resolve(curves),
                    ior_outer.resolve(curves),
                    kappa.resolve(curves),
                ) {
                    Some(MaterialEnum::GGX(GGX::new(
                        roughness,
                        ior,
                        eta_o,
                        kappa,
                        permeability,
                    )))
                } else {
                    None
                }
            }
            MaterialData::Lambertian { color } => {
                if let Some(reflectance) = color.resolve(curves) {
                    Some(MaterialEnum::ConstLambertian(ConstLambertian::new(
                        reflectance,
                    )))
                } else {
                    None
                }
            }
            MaterialData::Passthrough { color } => {
                if let Some(reflectance) = color.resolve(curves) {
                    Some(MaterialEnum::ConstPassthrough(ConstPassthrough::new(
                        reflectance,
                    )))
                } else {
                    None
                }
            }
        }
    }
}
