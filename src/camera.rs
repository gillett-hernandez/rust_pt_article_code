use crate::math::{random::random_in_unit_disk, *};

// #[derive(Debug, Clone)]
pub struct ProjectiveCamera {
    pub origin: Point3,
    pub direction: Vec3,
    focal_distance: f32,
    lower_left_corner: Point3,
    vfov: f32,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32,
    t0: f32,
    t1: f32,
}

impl ProjectiveCamera {
    pub fn new(
        look_from: Point3,
        look_at: Point3,
        v_up: Vec3,
        vertical_fov: f32,
        aspect_ratio: f32,
        focal_distance: f32,
        aperture: f32,
        t0: f32,
        t1: f32,
    ) -> ProjectiveCamera {
        let direction = (look_at - look_from).normalized();
        let lens_radius = aperture / 2.0;
        // vertical_fov should be given in degrees, since it is converted to radians
        let theta: f32 = vertical_fov.to_radians();
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        #[cfg(test)]
        {
            let aspect_ratio = half_width / half_height;
            println!("{}", aspect_ratio);
        }
        let w = -direction;
        let u = -v_up.cross(w).normalized();
        let v = w.cross(u).normalized();
        // println!(
        //     "constructing camera with point, direction, and uvw = {:?} {:?} {:?} {:?} {:?}",
        //     look_from, direction, u, v, w
        // );

        if lens_radius == 0.0 {
            println!("Warn: lens radius is 0");
        }

        ProjectiveCamera {
            origin: look_from,
            direction,
            focal_distance,
            lower_left_corner: look_from
                - u * half_width * focal_distance
                - v * half_height * focal_distance
                - w * focal_distance,
            vfov: vertical_fov,
            horizontal: u * 2.0 * half_width * focal_distance,
            vertical: v * 2.0 * half_height * focal_distance,
            u,
            v,
            w,
            lens_radius: aperture / 2.0,
            t0,
            t1,
        }
    }
}

impl ProjectiveCamera {
    pub fn get_ray(&self, sample: Sample2D, s: f32, t: f32) -> Ray {
        // circular aperture/lens
        let rd: Vec3 = self.lens_radius * random_in_unit_disk(sample);
        let offset = self.u * rd.x() + self.v * rd.y();
        let time: f32 = self.t0 + rand::random::<f32>() * (self.t1 - self.t0);
        let ray_origin: Point3 = self.origin + offset;

        let point_on_plane = self.lower_left_corner + s * self.horizontal + t * self.vertical;

        // println!("point on focal plane {:?}", point_on_plane);
        let ray_direction = (point_on_plane - ray_origin).normalized();
        debug_assert!(ray_origin.is_finite());
        debug_assert!(ray_direction.is_finite());
        Ray::new_with_time(ray_origin, ray_direction, time)
    }
    pub fn with_aspect_ratio(mut self, aspect_ratio: f32) -> Self {
        assert!(self.focal_distance > 0.0 && self.vfov > 0.0);
        let theta: f32 = self.vfov.to_radians();
        let half_height = (theta / 2.0).tan();
        let half_width = aspect_ratio * half_height;
        self.lower_left_corner = self.origin
            - self.u * half_width * self.focal_distance
            - self.v * half_height * self.focal_distance
            - self.w * self.focal_distance;
        self.horizontal = self.u * 2.0 * half_width * self.focal_distance;
        self.vertical = self.v * 2.0 * half_height * self.focal_distance;
        self
    }
}

unsafe impl Send for ProjectiveCamera {}
unsafe impl Sync for ProjectiveCamera {}
