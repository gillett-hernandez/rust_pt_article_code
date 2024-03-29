use crate::geometry::{Intersect, IntersectionData, SurfaceIntersectionData};
use crate::math::{Point3, Ray, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    pub radius: f32,
    pub origin: Point3,
    pub material_id: usize,
}

impl Sphere {
    pub fn new(radius: f32, origin: Point3, material_id: usize) -> Sphere {
        Sphere {
            radius,
            origin,
            material_id,
        }
    }

    // fn solid_angle(&self, point: Point3, wi: Vec3) -> f32 {
    //     let cos_theta_max =
    //         (1.0 - self.radius * self.radius / (self.origin - point).norm_squared()).sqrt();
    //     2.0 * PI * (1.0 - cos_theta_max)
    // }
}

impl Intersect for Sphere {
    fn intersect(&self, r: Ray, t0: f32, t1: f32) -> Option<IntersectionData> {
        let oc: Vec3 = r.origin - self.origin;
        let a = r.direction * r.direction;
        let b = oc * r.direction;
        let c = oc * oc - self.radius * self.radius;
        let discriminant = b * b - a * c;
        let discriminant_sqrt = discriminant.sqrt();
        if discriminant > 0.0 {
            let mut time: f32;
            let point: Point3;
            let normal: Vec3;
            // time = r.time + (-b - discriminant_sqrt) / a;
            time = (-b - discriminant_sqrt) / a;
            if time < t1 && time > t0 && time < r.tmax {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                return Some(SurfaceIntersectionData::new(
                    time,
                    point,
                    normal.normalized(),
                    self.material_id,
                ));
            }
            // time = r.time + (-b + discriminant_sqrt) / a;
            time = (-b + discriminant_sqrt) / a;
            if time < t1 && time > t0 && time < r.tmax {
                point = r.point_at_parameter(time);
                debug_assert!((point.w() - 1.0).abs() < 0.000001, "{:?}", point);
                debug_assert!((self.origin.w() - 1.0).abs() < 0.000001);
                normal = (point - self.origin) / self.radius;
                return Some(SurfaceIntersectionData::new(
                    time,
                    point,
                    normal.normalized(),
                    self.material_id,
                ));
            }
        }
        None
    }
}

unsafe impl Send for Sphere {}
unsafe impl Sync for Sphere {}
