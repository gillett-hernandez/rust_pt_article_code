use packed_simd::f32x4;

use crate::math::color::XYZColor;
use crate::math::misc::*;
use crate::math::Bounds1D;
use crate::math::*;

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

pub const EXTENDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(370.0, 790.0);
pub const BOUNDED_VISIBLE_RANGE: Bounds1D = Bounds1D::new(380.0, 780.0);

pub fn x_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.056, 5998.0, 379.0, 310.0)
        + gaussian(angstroms.into(), 0.362, 4420.0, 160.0, 267.0)
        + gaussian(angstroms.into(), -0.065, 5011.0, 204.0, 262.0)) as f32
}

pub fn y_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 0.821, 5688.0, 469.0, 405.0)
        + gaussian(angstroms.into(), 0.286, 5309.0, 163.0, 311.0)) as f32
}

pub fn z_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.217, 4370.0, 118.0, 360.0)
        + gaussian(angstroms.into(), 0.681, 4590.0, 260.0, 138.0)) as f32
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct SingleEnergy(pub f32);

impl SingleEnergy {
    pub fn new(energy: f32) -> Self {
        SingleEnergy { 0: energy }
    }
    pub const ZERO: SingleEnergy = SingleEnergy { 0: 0.0 };
    pub const ONE: SingleEnergy = SingleEnergy { 0: 1.0 };
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
}
impl Add for SingleEnergy {
    type Output = SingleEnergy;
    fn add(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self.0 + rhs.0)
    }
}
impl AddAssign for SingleEnergy {
    fn add_assign(&mut self, rhs: SingleEnergy) {
        self.0 += rhs.0;
    }
}

impl Mul<f32> for SingleEnergy {
    type Output = SingleEnergy;
    fn mul(self, rhs: f32) -> Self::Output {
        SingleEnergy::new(self.0 * rhs)
    }
}
impl Mul<SingleEnergy> for f32 {
    type Output = SingleEnergy;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self * rhs.0)
    }
}

impl Mul for SingleEnergy {
    type Output = SingleEnergy;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self.0 * rhs.0)
    }
}

impl MulAssign for SingleEnergy {
    fn mul_assign(&mut self, other: SingleEnergy) {
        self.0 = self.0 * other.0
    }
}

impl MulAssign<f32> for SingleEnergy {
    fn mul_assign(&mut self, other: f32) {
        self.0 = self.0 * other
    }
}

impl Div<f32> for SingleEnergy {
    type Output = SingleEnergy;
    fn div(self, rhs: f32) -> Self::Output {
        SingleEnergy::new(self.0 / rhs)
    }
}

impl From<f32> for SingleEnergy {
    fn from(value: f32) -> Self {
        SingleEnergy::new(value)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SingleWavelength {
    pub lambda: f32,
    pub energy: SingleEnergy,
}

impl SingleWavelength {
    pub const fn new(lambda: f32, energy: SingleEnergy) -> SingleWavelength {
        SingleWavelength { lambda, energy }
    }

    pub fn new_from_range(x: f32, bounds: Bounds1D) -> Self {
        SingleWavelength::new(bounds.lower + x * bounds.span(), SingleEnergy::ZERO)
    }

    pub fn with_energy(&self, energy: SingleEnergy) -> Self {
        SingleWavelength::new(self.lambda, energy)
    }

    pub fn replace_energy(&self, energy: f32) -> Self {
        self.with_energy(SingleEnergy::new(energy))
    }

    pub const BLACK: SingleWavelength = SingleWavelength::new(0.0, SingleEnergy::ZERO);
}

impl Mul<f32> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, other: f32) -> SingleWavelength {
        self.with_energy(self.energy * other)
    }
}

impl Mul<SingleWavelength> for f32 {
    type Output = SingleWavelength;
    fn mul(self, other: SingleWavelength) -> SingleWavelength {
        other.with_energy(self * other.energy)
    }
}

impl Mul<XYZColor> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, _xyz: XYZColor) -> SingleWavelength {
        // let lambda = other.wavelength;
        // let other_as_color: XYZColor = other.into();
        // other_as_color gives us the x y and z values for other
        // self.with_energy(self.energy * xyz.y())
        unimplemented!()
    }
}

impl Div<f32> for SingleWavelength {
    type Output = SingleWavelength;
    fn div(self, other: f32) -> SingleWavelength {
        self.with_energy(self.energy / other)
    }
}

impl DivAssign<f32> for SingleWavelength {
    fn div_assign(&mut self, other: f32) {
        self.energy = self.energy / other;
    }
}

impl Mul<SingleEnergy> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        self.with_energy(self.energy * rhs)
    }
}

impl Mul<SingleWavelength> for SingleEnergy {
    type Output = SingleWavelength;
    fn mul(self, rhs: SingleWavelength) -> Self::Output {
        rhs.with_energy(self * rhs.energy)
    }
}

// traits

pub trait SpectralPowerDistributionFunction {
    // range: [0, infinty)
    fn evaluate_power(&self, lambda: f32) -> f32;
    // range: [0, 1]
    fn evaluate_clamped(&self, lambda: f32) -> f32;

    fn sample_power_and_pdf(
        &self,
        wavelength_range: Bounds1D,
        sample: Sample1D,
    ) -> (SingleWavelength, PDF);

    fn convert_to_xyz(
        &self,
        integration_bounds: Bounds1D,
        step_size: f32,
        clamped: bool,
    ) -> XYZColor {
        let iterations = (integration_bounds.span() / step_size) as usize;
        let mut sum: XYZColor = XYZColor::ZERO;
        for i in 0..iterations {
            let lambda = integration_bounds.lower + (i as f32) * step_size;
            let angstroms = lambda * 10.0;
            let val = if clamped {
                self.evaluate_clamped(lambda)
            } else {
                self.evaluate_power(lambda)
            };
            sum.0 += f32x4::new(
                val * x_bar(angstroms),
                val * y_bar(angstroms),
                val * z_bar(angstroms),
                0.0,
            ) * step_size;
        }
        sum
    }
}

impl From<SingleWavelength> for XYZColor {
    fn from(swss: SingleWavelength) -> Self {
        // convert to Angstroms. 10 Angstroms == 1nm
        let angstroms = swss.lambda * 10.0;

        XYZColor::new(
            swss.energy.0 * x_bar(angstroms),
            swss.energy.0 * y_bar(angstroms),
            swss.energy.0 * z_bar(angstroms),
        )
    }
}

const HCC2: f32 = 1.1910429723971884140794892e-29;
const HKC: f32 = 1.438777085924334052222404423195819240925e-2;

pub fn blackbody(temperature: f32, lambda: f32) -> f32 {
    let lambda = lambda * 1e-9;

    lambda.powi(-5) * HCC2 / ((HKC / (lambda * temperature)).exp() - 1.0)
}

pub fn blackbody_f32x4(temperature: f32, lambda: f32x4) -> f32x4 {
    let lambda = lambda * 1e-9;

    lambda.powf(f32x4::splat(-5.0)) * HCC2 / ((HKC / (lambda * temperature)).exp() - 1.0)
}

pub fn max_blackbody_lambda(temp: f32) -> f32 {
    2.8977721e-3 / (temp * 1e-9)
}
