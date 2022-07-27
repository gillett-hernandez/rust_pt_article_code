use std::collections::HashMap;

use crate::{
    geometry::{PrimitiveEnum, Sphere},
    math::Point3,
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PrimitiveData {
    Sphere {
        origin: [f32; 3],
        radius: f32,
        material: String,
    },
}

impl PrimitiveData {
    pub fn transform(self, material_mapping: &HashMap<String, usize>) -> PrimitiveEnum {
        match self {
            Self::Sphere {
                origin,
                radius,
                material,
            } => PrimitiveEnum::Sphere(Sphere::new(
                radius,
                Point3::new(origin[0], origin[1], origin[2]),
                *material_mapping.get(&material).expect(
                    format!("material {} not present in material mapping", material).as_str(),
                ),
            )),
        }
    }
}
