//! # Fuzz
//!
//! This is an implementation of the fuzz (also known as sheen) lobe.
//!
//! As recommended by the OpenPBR specification, and following the official reference
//! implementation and the Adobe implementation, it's an implementation of the model proposed
//! by the paper "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines"
//! by Zeltner, Burley, and Chiang (2022).

use crate::{
    consts::DENOM_TOLERANCE, material::Material, math::SphericalCoordinates,
    sampling::cosine_hemisphere_sample,
};
use glam::{Mat3, Vec3};
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

fn albedo(x: f32, y: f32) -> f32 {
    let s = y * (0.0206607 + 1.58491 * y) / (0.0379424 + y * (1.32227 + y));
    let m = y * (-0.193854 + y * (-1.14885 + y * (1.7932 - 0.95943 * y * y))) / (0.046391 + y);
    let o = y * (0.000654023 + (-0.0207818 + 0.119681 * y) * y) / (1.26264 + y * (-1.92021 + y));
    (-0.5 * ((x - m) / s).powi(2)).exp() / (s * (2.0 * PI).sqrt()) + o
}

fn ltc_inverse_coeffs(x: f32, y: f32) -> (f32, f32) {
    let a = (2.58126 * x + 0.813703 * y) * y / (1.0 + 0.310327 * x * x + 2.60994 * x * y);
    let b = (1.0 - x).sqrt() * (y - 1.0) * y * y * y
        / (0.0000254053 + 1.71228 * x - 1.71506 * x * y + 1.34174 * y * y);
    (a, b)
}

fn orthonormal_basis_ltc(normal: Vec3) -> Mat3 {
    let xy_len_sq = normal.x * normal.x + normal.y * normal.y;
    let tangent = if xy_len_sq > 0.0 {
        Vec3::new(normal.x, normal.y, 0.0) * xy_len_sq.sqrt().recip()
    } else {
        Vec3::X
    };
    let bitangent = Vec3::new(-tangent.y, tangent.x, 0.0);
    Mat3::from_cols(tangent, bitangent, Vec3::Z)
}

fn brdf_and_density(
    color: Vec3,
    direction: Vec3,
    wi: Vec3,
    a_inverse: f32,
    roughness: f32,
) -> (Vec3, f32) {
    let jacobian = (a_inverse / direction.length_squared()).powi(2);
    let density = direction.z.max(0.0) / PI * jacobian;
    let albedo = albedo(wi.cos_theta(), roughness);
    (color * albedo / PI * jacobian, density)
}

pub struct Fuzz {
    pub color: Vec3,
    pub roughness: f32,
}

impl From<&Material> for Fuzz {
    fn from(m: &Material) -> Self {
        Self {
            color: m.fuzz_color,
            roughness: m.fuzz_roughness,
        }
    }
}

impl Lobe for Fuzz {
    fn incidence_is_valid(&self, wi: Vec3) -> bool {
        wi.cos_theta() >= DENOM_TOLERANCE
    }

    fn eval(&self, wi: Vec3, wo: Vec3) -> Throughput {
        if !self.incidence_is_valid(wi) || wo.cos_theta() < DENOM_TOLERANCE {
            return Throughput::ZERO;
        }

        let roughness = self.roughness.clamp(0.01, 1.0);

        let basis = orthonormal_basis_ltc(wi);
        let w = basis.transpose() * wo;
        let (a_inv, b) = ltc_inverse_coeffs(wi.cos_theta(), roughness);

        let direction = Vec3::new(a_inv * w.x + b * w.z, a_inv * w.y, w.z);
        let (brdf, _) = brdf_and_density(self.color, direction, wi, a_inv, roughness);

        Throughput::from_diffuse(brdf)
    }

    fn sample(&self, random: Vec3, wi: Vec3) -> Option<Sample> {
        if !self.incidence_is_valid(wi) {
            return None;
        }

        let roughness = self.roughness.clamp(0.01, 1.0);

        let direction = cosine_hemisphere_sample(random.truncate());
        let (a_inv, b) = ltc_inverse_coeffs(wi.cos_theta(), roughness);

        let w = Vec3::new(
            direction.x / a_inv - direction.z * b / a_inv,
            direction.y / a_inv,
            direction.z,
        );

        let basis = orthonormal_basis_ltc(wi);
        let wo = basis * w.normalize();
        let (brdf, density) = brdf_and_density(self.color, direction, wi, a_inv, roughness);

        Some(Sample {
            lobe_type: LobeType::Fuzz,
            throughput: Throughput::from_diffuse(brdf),
            density,
            wo,
        })
    }

    fn density(&self, wi: Vec3, wo: Vec3) -> f32 {
        if !self.incidence_is_valid(wi) || wo.cos_theta() < DENOM_TOLERANCE {
            return 0.0;
        }

        let roughness = self.roughness.clamp(0.01, 1.0);

        let w = orthonormal_basis_ltc(wi).transpose() * wo;

        let (a_inv, b) = ltc_inverse_coeffs(wi.cos_theta(), roughness);
        let direction = Vec3::new(a_inv * w.x + b * w.z, a_inv * w.y, w.z);
        let (_, density) = brdf_and_density(self.color, direction, wi, a_inv, roughness);

        density
    }

    fn estimate_directional_albedo(&self, wi: Vec3, _: &[Vec3]) -> Vec3 {
        if !self.incidence_is_valid(wi) {
            return Vec3::ZERO;
        }

        Vec3::splat(albedo(wi.cos_theta(), self.roughness)) * self.color
    }
}
