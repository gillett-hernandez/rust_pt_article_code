use crate::math::{Point3, Ray, Vec3};

mod sphere;

pub use sphere::Sphere;

#[derive(Copy, Clone, Debug)]
pub struct SurfaceIntersectionData {
    pub time: f32,
    pub point: Point3,
    pub normal: Vec3,
    pub material_id: usize,
    pub outer_medium_id: usize,
    pub inner_medium_id: usize,
}

impl SurfaceIntersectionData {
    pub fn new(
        time: f32,
        point: Point3,
        normal: Vec3,
        material_id: usize,
        outer_medium_id: usize,
        inner_medium_id: usize,
    ) -> Self {
        SurfaceIntersectionData {
            time,
            point,
            normal,
            material_id,
            outer_medium_id,
            inner_medium_id,
        }
    }
}

type IntersectionData = SurfaceIntersectionData;

pub trait Primitive {
    fn intersect(&self, r: Ray, t0: f32, t1: f32) -> Option<IntersectionData>;
}
