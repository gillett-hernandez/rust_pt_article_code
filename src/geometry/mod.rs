use crate::math::{Point3, Ray, Vec3};

mod sphere;

pub use sphere::Sphere;

#[derive(Copy, Clone, Debug)]
pub struct SurfaceIntersectionData {
    pub time: f32,
    pub point: Point3,
    pub normal: Vec3,
    pub material_id: usize,
}

impl SurfaceIntersectionData {
    pub fn new(time: f32, point: Point3, normal: Vec3, material_id: usize) -> Self {
        SurfaceIntersectionData {
            time,
            point,
            normal,
            material_id,
        }
    }
}

type IntersectionData = SurfaceIntersectionData;

pub trait Intersect {
    fn intersect(&self, r: Ray, t0: f32, t1: f32) -> Option<IntersectionData>;
}

// pub enum PrimitiveEnum {
//     Sphere(Sphere),
//     // Rect or other primtives go here
// }

// impl Intersect for PrimitiveEnum {
//     fn intersect(&self, r: Ray, t0: f32, t1: f32) -> Option<IntersectionData> {
//         match self {
//             PrimitiveEnum::Sphere(inner) => inner.intersect(r, t0, t1),
//         }
//     }
// }

#[macro_export]
macro_rules! generate_primitive_enum {
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


        impl Intersect for $name {
            fn intersect(&self, r: Ray, t0: f32, t1: f32) -> Option<IntersectionData>{
                match self {
                    $($name::$s(inner) => inner.intersect(r, t0, t1),)+
                }
            }
        }
    };
}

generate_primitive_enum!(PrimitiveEnum, Sphere);
