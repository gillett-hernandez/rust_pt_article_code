use crate::math::{Point3, Ray, Vec3};

pub struct Camera {
    pub origin: Point3,
    pub direction: Vec3,
    pub side: Vec3,
    pub up: Vec3,
    pub factor: f32,
    pub aspect_ratio: f32,
}

impl Camera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        up: Vec3,
        factor: f32,
        aspect_ratio: f32,
    ) -> Camera {
        let direction = (look_at - look_from).normalized();
        let side = direction.cross(up).normalized();
        let real_up = direction.cross(side).normalized();
        Camera {
            origin: look_from,
            direction,
            side,
            up: real_up,
            factor,
            aspect_ratio,
        }
    }
}
