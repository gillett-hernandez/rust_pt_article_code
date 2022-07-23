pub mod bounds;
pub mod color;
pub mod curve;
pub mod random;
pub mod sample;
pub mod spectral;

mod misc;
mod point;
mod ray;
mod transform;
mod vec;

pub use bounds::*;
pub use color::*;
pub use curve::*;
pub use misc::*;
pub use point::*;
pub use random::*;
pub use ray::*;
pub use sample::*;
pub use transform::*;
pub use vec::*;

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign};

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct PDF(pub f32);
impl PDF {
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
}

impl From<f32> for PDF {
    fn from(val: f32) -> Self {
        PDF(val)
    }
}

impl From<PDF> for f32 {
    fn from(val: PDF) -> Self {
        val.0
    }
}

impl Add for PDF {
    type Output = PDF;
    fn add(self, rhs: PDF) -> Self::Output {
        PDF::from(self.0 + rhs.0)
    }
}
impl AddAssign for PDF {
    fn add_assign(&mut self, rhs: PDF) {
        self.0 += rhs.0;
    }
}

impl Mul<f32> for PDF {
    type Output = PDF;
    fn mul(self, rhs: f32) -> Self::Output {
        PDF::from(self.0 * rhs)
    }
}
impl Mul<PDF> for f32 {
    type Output = PDF;
    fn mul(self, rhs: PDF) -> Self::Output {
        PDF::from(self * rhs.0)
    }
}

impl Mul for PDF {
    type Output = PDF;
    fn mul(self, rhs: PDF) -> Self::Output {
        PDF::from(self.0 * rhs.0)
    }
}

impl MulAssign for PDF {
    fn mul_assign(&mut self, other: PDF) {
        self.0 = self.0 * other.0
    }
}
impl Div<f32> for PDF {
    type Output = PDF;
    fn div(self, rhs: f32) -> Self::Output {
        PDF::from(self.0 / rhs)
    }
}
