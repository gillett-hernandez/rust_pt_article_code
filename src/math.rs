use std::{
    f32::INFINITY,
    fmt,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use packed_simd::f32x4;

pub struct Point3(pub f32x4);
pub struct Vec3(pub f32x4);

impl Point3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3(f32x4::new(x, y, z, 1.0))
    }
    pub const ZERO: Point3 = Point3(f32x4::new(0.0, 0.0, 0.0, 1.0));
    pub const ORIGIN: Point3 = Point3(f32x4::new(0.0, 0.0, 0.0, 1.0));
    pub const INFINITY: Point3 = Point3(f32x4::new(INFINITY, INFINITY, INFINITY, 1.0));
    pub const NEG_INFINITY: Point3 = Point3(f32x4::new(-INFINITY, -INFINITY, -INFINITY, 1.0));
    pub fn is_finite(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Point3 {
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
    pub fn normalize(mut self) -> Self {
        unsafe {
            self.0 = self.0 / self.0.extract_unchecked(3);
        }
        self
    }
    pub fn as_array(&self) -> [f32; 4] {
        self.0.into()
    }
}

impl Default for Point3 {
    fn default() -> Self {
        Point3::ORIGIN
    }
}

impl Add<Vec3> for Point3 {
    type Output = Point3;
    fn add(self, other: Vec3) -> Point3 {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        (self.0 + other.0).into()
    }
}

impl AddAssign<Vec3> for Point3 {
    fn add_assign(&mut self, other: Vec3) {
        // Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
        self.0 += other.0
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;
    fn sub(self, other: Vec3) -> Point3 {
        // Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        (self.0 - other.0).into()
    }
}

impl SubAssign<Vec3> for Point3 {
    fn sub_assign(&mut self, other: Vec3) {
        self.0 -= other.0
    }
}

impl Sub for Point3 {
    type Output = Vec3;
    fn sub(self, other: Point3) -> Vec3 {
        // Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
        Vec3((self.0 - other.0) * f32x4::new(1.0, 1.0, 1.0, 0.0))
    }
}

impl From<[f32; 3]> for Point3 {
    fn from(other: [f32; 3]) -> Point3 {
        Point3::new(other[0], other[1], other[2])
    }
}

impl From<f32x4> for Point3 {
    fn from(other: f32x4) -> Point3 {
        Point3(other)
    }
}

impl From<Vec3> for Point3 {
    fn from(v: Vec3) -> Point3 {
        Point3::ORIGIN + v
    }
}

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vec3")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3(f32x4::new(x, y, z, 0.0))
    }
    pub const ZERO: Vec3 = Vec3(f32x4::splat(0.0));
    pub const MASK: f32x4 = f32x4::new(1.0, 1.0, 1.0, 0.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    pub fn is_finite(&self) -> bool {
        !(self.0.is_nan().any() || self.0.is_infinite().any())
    }
}

impl Vec3 {
    #[inline(always)]
    pub fn x(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
    #[inline(always)]
    pub fn w(&self) -> f32 {
        unsafe { self.0.extract_unchecked(3) }
    }
    pub fn as_array(&self) -> [f32; 4] {
        self.0.into()
    }
    pub fn cross(&self, other: Vec3) -> Self {
        let (x1, y1, z1) = (self.x(), self.y(), self.z());
        let (x2, y2, z2) = (other.x(), other.y(), other.z());
        Vec3::new(y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - x2 * y1)
    }

    pub fn norm_squared(&self) -> f32 {
        (self.0 * self.0 * Vec3::MASK).sum()
    }

    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        Vec3(self.0 / norm)
    }
}

// dot product
impl Mul for Vec3 {
    type Output = f32;
    fn mul(self, other: Vec3) -> f32 {
        // self.x * other.x + self.y * other.y + self.z * other.z
        (self.0 * other.0).sum()
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: f32) -> Vec3 {
        Vec3(self.0 * other)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3(self * other.0)
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, other: f32) -> Vec3 {
        Vec3(self.0 / other)
    }
}

// don't implement adding or subtracting floats from Point3

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3(self.0 + other.0)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3(-self.0)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        self + (-other)
    }
}

impl From<f32> for Vec3 {
    fn from(s: f32) -> Vec3 {
        Vec3(f32x4::splat(s) * Vec3::MASK)
    }
}

impl From<Vec3> for f32x4 {
    fn from(v: Vec3) -> f32x4 {
        v.0
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(other: [f32; 3]) -> Vec3 {
        Vec3::new(other[0], other[1], other[2])
    }
}

impl From<[f32; 4]> for Vec3 {
    fn from(other: [f32; 4]) -> Vec3 {
        Vec3(f32x4::from(other))
    }
}

impl From<f32x4> for Vec3 {
    fn from(other: f32x4) -> Vec3 {
        Vec3(other)
    }
}

impl From<Point3> for Vec3 {
    fn from(p: Point3) -> Self {
        // Vec3::new(p.x, p.y, p.z)
        Vec3(p.0.replace(3, 0.0))
    }
}
