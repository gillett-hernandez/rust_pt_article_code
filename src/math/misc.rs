use super::Vec3;
use packed_simd::f32x4;
use std::f32::consts::PI;

pub fn power_heuristic(a: f32, b: f32) -> f32 {
    (a * a) / (a * a + b * b)
}

pub fn power_heuristic_hero(a: f32x4, b: f32x4) -> f32x4 {
    (a * a) / (a * a + b * b)
}

pub fn gaussianf32(x: f32, alpha: f32, mu: f32, sigma1: f32, sigma2: f32) -> f32 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

pub fn gaussian(x: f64, alpha: f64, mu: f64, sigma1: f64, sigma2: f64) -> f64 {
    let sqrt = (x - mu) / (if x < mu { sigma1 } else { sigma2 });
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

pub fn gaussian_f32x4(x: f32x4, alpha: f32, mu: f32, sigma1: f32, sigma2: f32) -> f32x4 {
    let sqrt = (x - mu)
        / x.lt(f32x4::splat(mu))
            .select(f32x4::splat(sigma1), f32x4::splat(sigma2));
    alpha * (-(sqrt * sqrt) / 2.0).exp()
}

pub fn w(x: f32, mul: f32, offset: f32, sigma: f32) -> f32 {
    mul * (-(x - offset).powi(2) / sigma).exp() / (sigma * PI).sqrt()
}

//----------------------------------------------------------------------
// theta = azimuthal angle
// phi = inclination, i.e. angle measured from +Z. the elevation angle would be pi/2 - phi

pub fn uv_to_direction(uv: (f32, f32)) -> Vec3 {
    let theta = (uv.0 - 0.5) * 2.0 * PI;
    let phi = uv.1 * PI;

    let (sin_theta, cos_theta) = theta.sin_cos();
    let (sin_phi, cos_phi) = phi.sin_cos();

    let (x, y, z) = (sin_phi * cos_theta, sin_phi * sin_theta, cos_phi);
    Vec3::new(x, y, z)
}

pub fn direction_to_uv(direction: Vec3) -> (f32, f32) {
    let theta = direction.y().atan2(direction.x());
    let phi = direction.z().acos();
    let u = theta / 2.0 / PI + 0.5;
    let v = phi / PI;
    (u, v)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::random_on_unit_sphere;
    use crate::math::sample::Sample2D;
    use crate::random::*;

    #[test]
    fn test_direction_to_uv() {
        let direction = random_on_unit_sphere(Sample2D::new_random_sample());
        let uv = direction_to_uv(direction);
        println!("{:?} {:?}", direction, uv);
    }

    #[test]
    fn test_uv_to_direction() {
        let mut center = Vec3::ZERO;
        let n = 100;
        for _ in 0..n {
            let uv = (rand::random::<f32>(), rand::random::<f32>());
            let direction = uv_to_direction(uv);
            println!("{:?} {:?}", direction, uv);
            center = center + direction / n as f32;
        }
        println!("{:?}", center);
    }

    #[test]
    fn test_bijectiveness_of_uv_direction() {
        let uv = (rand::random::<f32>(), rand::random::<f32>());
        let direction = uv_to_direction(uv);
        let uv2 = direction_to_uv(direction);
        assert!(uv == uv2, "{:?} {:?}", uv, uv2);

        let direction = random_on_unit_sphere(Sample2D::new_random_sample());
        let uv = direction_to_uv(direction);
        let direction2 = uv_to_direction(uv);
        assert!(
            (direction - direction2).norm() < 0.000001,
            "{:?} {:?}",
            direction,
            direction2
        );
    }
}
