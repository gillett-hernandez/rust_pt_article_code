use crate::math::{Point3, Ray};

#[derive(Copy, Clone)]
pub struct Intersection {
    pub point: Point3,
}

pub trait RayIntersection {
    fn intersects(&self, r: Ray) -> Option<Intersection>;
}

#[derive(Copy, Clone)]
pub struct Sphere {
    pub origin: Point3,
    pub radius: f32,
}

impl RayIntersection for Sphere {
    fn intersects(&self, r: Ray) -> Option<Intersection> {
        let origin_diff = r.origin - self.origin;
        let d_d0 = origin_diff * r.direction;
        let d0_d0 = origin_diff * origin_diff;
        let r_2 = self.radius * self.radius;

        let discriminant = d_d0.powi(2) + r_2 - d0_d0;

        match discriminant {
            a if a > 0.0 => {
                let sqrt = discriminant.sqrt();
                let t0 = -d_d0 - sqrt;
                let t1 = -d_d0 + sqrt;

                if t0 > t1 {
                    Some(Intersection { point: r.at(t0) })
                } else {
                    Some(Intersection { point: r.at(t1) })
                }
            }
            a if a < 0.0 => None,
            _ => Some(Intersection { point: r.at(-d_d0) }),
        }
    }
}
