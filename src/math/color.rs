// use packed_simd::f32x4;

use nalgebra::{Matrix3, Vector3};

use packed_simd::f32x4;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

use super::Vec3;

#[derive(Copy, Clone, Debug)]
pub struct RGBColor(pub f32x4);

impl RGBColor {
    pub const fn new(r: f32, g: f32, b: f32) -> RGBColor {
        // RGBColor { x, y, z, w: 0.0 }
        RGBColor(f32x4::new(r, g, b, 0.0))
    }
    pub const fn from_raw(v: f32x4) -> RGBColor {
        RGBColor(v)
    }
    pub const ZERO: RGBColor = RGBColor::from_raw(f32x4::splat(0.0));
    pub const BLACK: RGBColor = RGBColor::from_raw(f32x4::splat(0.0));
}

impl RGBColor {
    #[inline(always)]
    pub fn r(&self) -> f32 {
        unsafe { self.0.extract_unchecked(0) }
    }
    #[inline(always)]
    pub fn g(&self) -> f32 {
        unsafe { self.0.extract_unchecked(1) }
    }
    #[inline(always)]
    pub fn b(&self) -> f32 {
        unsafe { self.0.extract_unchecked(2) }
    }
}

impl Mul for RGBColor {
    type Output = Self;
    fn mul(self, other: RGBColor) -> Self {
        // self.x * other.x + self.y * other.y + self.z * other.z
        RGBColor::from_raw(self.0 * other.0)
    }
}

impl MulAssign for RGBColor {
    fn mul_assign(&mut self, other: RGBColor) {
        // self.x *= other.x;
        // self.y *= other.y;
        // self.z *= other.z;
        self.0 = self.0 * other.0
    }
}

impl Mul<f32> for RGBColor {
    type Output = RGBColor;
    fn mul(self, other: f32) -> RGBColor {
        RGBColor::from_raw(self.0 * other)
    }
}

impl Mul<RGBColor> for f32 {
    type Output = RGBColor;
    fn mul(self, other: RGBColor) -> RGBColor {
        RGBColor::from_raw(self * other.0)
    }
}

impl Div<f32> for RGBColor {
    type Output = RGBColor;
    fn div(self, other: f32) -> RGBColor {
        RGBColor::from_raw(self.0 / other)
    }
}

impl DivAssign<f32> for RGBColor {
    fn div_assign(&mut self, other: f32) {
        self.0 = self.0 / other;
    }
}

// impl Div for RGBColor {
//     type Output = RGBColor;
//     fn div(self, other: RGBColor) -> RGBColor {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         RGBColor::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for RGBColor {
//     type Output = RGBColor;
//     fn add(self, other: f32) -> RGBColor {
//         RGBColor::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for RGBColor {
//     type Output = RGBColor;
//     fn sub(self, other: f32) -> RGBColor {
//         RGBColor::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for RGBColor {
    type Output = RGBColor;
    fn add(self, other: RGBColor) -> RGBColor {
        RGBColor::from_raw(self.0 + other.0)
    }
}

impl AddAssign for RGBColor {
    fn add_assign(&mut self, other: RGBColor) {
        self.0 = self.0 + other.0
    }
}

impl From<f32> for RGBColor {
    fn from(s: f32) -> RGBColor {
        RGBColor::from_raw(f32x4::splat(s) * f32x4::new(1.0, 1.0, 1.0, 0.0))
    }
}

impl From<RGBColor> for f32x4 {
    fn from(v: RGBColor) -> f32x4 {
        v.0
    }
}

impl Mul<RGBColor> for Vec3 {
    type Output = RGBColor;
    fn mul(self, other: RGBColor) -> RGBColor {
        // RGBColor::new(self.x() * other.r, self.y() * other.g, self.z() * other.b)
        RGBColor::from_raw(self.0 * other.0)
    }
}

impl From<RGBColor> for Vec3 {
    fn from(c: RGBColor) -> Vec3 {
        Vec3(c.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct XYZColor(pub f32x4);

impl XYZColor {
    pub const fn new(x: f32, y: f32, z: f32) -> XYZColor {
        // XYZColor { x, y, z, w: 0.0 }
        XYZColor(f32x4::new(x, y, z, 0.0))
    }
    pub const fn from_raw(v: f32x4) -> XYZColor {
        XYZColor(v)
    }
    pub const BLACK: XYZColor = XYZColor::from_raw(f32x4::splat(0.0));
    pub const ZERO: XYZColor = XYZColor::from_raw(f32x4::splat(0.0));
}

impl XYZColor {
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
}

// impl Mul for XYZColor {
//     type Output = Self;
//     fn mul(self, other: XYZColor) -> Self {
//         // self.x * other.x + self.y * other.y + self.z * other.z
//         XYZColor::from_raw(self.0 * other.0)
//     }
// }

// impl MulAssign for XYZColor {
//     fn mul_assign(&mut self, other: XYZColor) {
//         // self.x *= other.x;
//         // self.y *= other.y;
//         // self.z *= other.z;
//         self.0 = self.0 * other.0
//     }
// }

impl Mul<f32> for XYZColor {
    type Output = XYZColor;
    fn mul(self, other: f32) -> XYZColor {
        XYZColor::from_raw(self.0 * other)
    }
}

impl Mul<XYZColor> for f32 {
    type Output = XYZColor;
    fn mul(self, other: XYZColor) -> XYZColor {
        XYZColor::from_raw(other.0 * self)
    }
}

impl Div<f32> for XYZColor {
    type Output = XYZColor;
    fn div(self, other: f32) -> XYZColor {
        XYZColor::from_raw(self.0 / other)
    }
}

impl DivAssign<f32> for XYZColor {
    fn div_assign(&mut self, other: f32) {
        self.0 = self.0 / other;
    }
}

// impl Div for XYZColor {
//     type Output = XYZColor;
//     fn div(self, other: XYZColor) -> XYZColor {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         XYZColor::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for XYZColor {
//     type Output = XYZColor;
//     fn add(self, other: f32) -> XYZColor {
//         XYZColor::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for XYZColor {
//     type Output = XYZColor;
//     fn sub(self, other: f32) -> XYZColor {
//         XYZColor::new(self.x - other, self.y - other, self.z - other)
//     }
// }

impl Add for XYZColor {
    type Output = XYZColor;
    fn add(self, other: XYZColor) -> XYZColor {
        XYZColor::from_raw(self.0 + other.0)
    }
}

impl AddAssign for XYZColor {
    fn add_assign(&mut self, other: XYZColor) {
        self.0 = self.0 + other.0
        // self.0 = (*self + other).0
    }
}

impl From<XYZColor> for f32x4 {
    fn from(v: XYZColor) -> f32x4 {
        v.0
    }
}

impl From<XYZColor> for RGBColor {
    fn from(xyz: XYZColor) -> Self {
        let xyz_to_rgb: Matrix3<f32> = Matrix3::new(
            0.41847, -0.15866, -0.082835, -0.091169, 0.25243, 0.015708, 0.00092090, -0.0025498,
            0.17860,
        );
        let [a, b, c, _]: [f32; 4] = xyz.0.into();
        let intermediate = xyz_to_rgb * Vector3::new(a, b, c);
        RGBColor::new(intermediate[0], intermediate[1], intermediate[2])
    }
}

impl From<RGBColor> for XYZColor {
    fn from(rgb: RGBColor) -> Self {
        let rgb_to_xyz: Matrix3<f32> = Matrix3::new(
            0.490, 0.310, 0.200, 0.17697, 0.8124, 0.01063, 0.0, 0.01, 0.99,
        );
        let [a, b, c, _]: [f32; 4] = rgb.0.into();
        let intermediate = rgb_to_xyz * Vector3::new(a, b, c) / 0.17697;
        XYZColor::new(intermediate[0], intermediate[1], intermediate[2])
    }
}
