use crate::math::{Point3, Ray};

#[derive(Copy, Clone)]
pub struct Intersection {
    pub point: Point3,
}

pub trait RayIntersection {
    fn intersects(&self, r: Ray) -> Intersection;
}
