pub mod curves;
mod material;
mod primitives;

use std::{collections::HashMap, error::Error, fs::File, io::Read, path::PathBuf};

use serde::{de::DeserializeOwned, Deserialize};

use curves::*;
use material::*;
use primitives::*;

use crate::{geometry::PrimitiveEnum, materials::MaterialEnum, math::Curve};

// #[derive(Clone, Serialize, Deserialize)]
// struct _LiteralCurves(HashMap<String, CurveData>);

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CurvesData {
    Path(PathBuf),
    Literal(HashMap<String, CurveData>),
}
impl CurvesData {
    pub fn loaded(self) -> Result<HashMap<String, CurveData>, Box<dyn Error>> {
        match self {
            Self::Path(path) => load_json(path),
            // Self::Path(path) => _LiteralCurves::from_json(path).map(|e| e.0),
            Self::Literal(inner) => Ok(inner),
        }
    }
}

// #[derive(Clone, Serialize, Deserialize)]
// struct _LiteralMaterials(HashMap<String, MaterialData>);
#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MaterialsData {
    Path(PathBuf),
    Literal(HashMap<String, MaterialData>),
}
impl MaterialsData {
    pub fn loaded(self) -> Result<HashMap<String, MaterialData>, Box<dyn Error>> {
        match self {
            Self::Path(path) => load_json(path),
            // Self::Path(path) => _LiteralMaterials::from_json(path).map(|e| e.0),
            Self::Literal(inner) => Ok(inner),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SceneData {
    pub env_color: CurveDataOrReference,
    pub curves_lib: CurvesData,
    pub materials_lib: MaterialsData,
    pub primitives: Vec<PrimitiveData>,
}

pub struct Scene {
    pub env_color: Curve,
    pub materials: Vec<MaterialEnum>,
    pub primitives: Vec<PrimitiveEnum>,
}

impl From<SceneData> for Scene {
    fn from(mut data: SceneData) -> Self {
        let curves_map: HashMap<String, Curve> = data
            .curves_lib
            .loaded()
            .expect("failed to parse curves_map from disk")
            .drain()
            .map(|(k, v)| (k.clone(), v.into()))
            .collect();

        // i'm curious if the below code snippet can be rewritten using iterators
        // i.e. something like  let (materials, material_mapping) = materials_map.iter().map(|(k, v)| (...))

        let mut material_name_to_id = HashMap::new();
        let mut materials = Vec::new();
        for (name, material) in data
            .materials_lib
            .loaded()
            .expect("failed to parse materials_map from disk")
            .drain()
            .map(|(k, v)| (k.clone(), v.transform(&curves_map).unwrap()))
        {
            let id = materials.len();
            materials.push(material);
            material_name_to_id.insert(name, id);
        }

        let primitives = data
            .primitives
            .drain(..)
            .map(|e| e.transform(&material_name_to_id))
            .collect::<Vec<_>>();

        Scene {
            env_color: data
                .env_color
                .resolve(&curves_map)
                .expect("couldn't parse env map color"),
            materials,
            primitives,
        }
    }
}

pub fn load_json<T>(path: PathBuf) -> Result<T, Box<dyn Error>>
where
    T: DeserializeOwned,
{
    let mut input = String::new();
    File::open(path)
        .and_then(|mut f| f.read_to_string(&mut input))
        .unwrap();

    let data: T = serde_json::from_str(&input)?;
    Ok(data.into())
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_loading_simple_scene() {
        let scene: Scene =
            load_json::<SceneData>(PathBuf::from("data/scenes/cornell_box_simple.json"))
                .expect("failed to parse scene")
                .into();
        println!("{}", scene.materials.len());
        println!("{}", scene.primitives.len());
    }
    #[test]
    fn test_loading_complex_scene() {
        let scene: Scene = load_json::<SceneData>(PathBuf::from("data/scenes/cornell_box.json"))
            .expect("failed to parse scene")
            .into();
        println!("{}", scene.materials.len());
        println!("{}", scene.primitives.len());
    }
}
