// use packed_simd::{f32x4, f32x8};
use crate::Point3;

use packed_simd::f32x4;

use std::fmt;
use std::ops::{Add, Div, Mul, MulAssign, Neg, Sub};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Copy, Clone, PartialEq, Default)]
pub struct Vec3(pub f32x4);

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
    // #[]
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        // Vec3 { x, y, z, w: 0.0 }
        Vec3(f32x4::new(x, y, z, 0.0))
    }
    pub const ZERO: Vec3 = Vec3(f32x4::splat(0.0));
    pub const MASK: f32x4 = f32x4::new(1.0, 1.0, 1.0, 0.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub fn from_axis(axis: Axis) -> Vec3 {
        match axis {
            Axis::X => Vec3::X,
            Axis::Y => Vec3::Y,
            Axis::Z => Vec3::Z,
        }
    }
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
    pub fn as_tuple(&self) -> (f32, f32, f32) {
        (self.x(), self.y(), self.z())
    }
}

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

// impl Div for Vec3 {
//     type Output = Vec3;
//     fn div(self, other: Vec3) -> Vec3 {
//         // by changing other.w to 1.0, we prevent a divide by 0.
//         Vec3::from_raw(self.0 / other.normalized().0.replace(3, 1.0))
//     }
// }

// don't implement adding or subtracting floats from Point3
// impl Add<f32> for Vec3 {
//     type Output = Vec3;
//     fn add(self, other: f32) -> Vec3 {
//         Vec3::new(self.x + other, self.y + other, self.z + other)
//     }
// }
// impl Sub<f32> for Vec3 {
//     type Output = Vec3;
//     fn sub(self, other: f32) -> Vec3 {
//         Vec3::new(self.x - other, self.y - other, self.z - other)
//     }
// }

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

impl Vec3 {
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

impl From<[f32; 3]> for Vec3 {
    fn from(other: [f32; 3]) -> Vec3 {
        Vec3::new(other[0], other[1], other[2])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec() {
        let v = Vec3::new(100.0, 0.2, 1.0);
        assert!(v.norm() > 100.0);
        assert!(v.norm_squared() > 10000.0);
        assert!(v.normalized().norm() - 1.0 < 0.000001);
    }
}
