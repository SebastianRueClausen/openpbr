use crate::{
    material::Material,
    math::SphericalCoordinates,
    sampling::{cosine_hemisphere_density, cosine_hemisphere_sample},
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

fn oren_nayar(cos_theta: f32, sigma_squared: f32) -> f32 {
    let a = 1.0 - 0.5 * (sigma_squared / (sigma_squared + 0.33));
    let b = 0.45 * sigma_squared / (sigma_squared + 0.09);
    let s = (1.0 - cos_theta.powi(2)).sqrt();
    let g = s * (cos_theta.acos() - s * cos_theta)
        + (s / f32::max(1e-7, cos_theta)) * (1.0 - s.powi(3)) * 2.0 / 3.0;
    return a + (b / PI) * g;
}

fn energy_compensated_oren_nayar(rho: Vec3, sigma: f32, wi: Vec3, wo: Vec3) -> Vec3 {
    let sigma_squared = sigma.powi(2);
    let s = wi.dot(wo) - wi.cos_theta() * wo.cos_theta();
    let s_over_t = if s > 0.0 {
        s / f32::max(wi.cos_theta(), wo.cos_theta())
    } else {
        0.0
    };
    let a = 1.0 - 0.5 * (sigma_squared / (sigma_squared + 0.33));
    let b = 0.45 * sigma_squared / (sigma_squared + 0.09);
    let on_o = oren_nayar(wo.cos_theta(), sigma_squared);
    let on_i = oren_nayar(wi.cos_theta(), sigma_squared);
    let average_albedo = a + (2.0 / 3.0 - 64.0 / (45.0 * PI)) * b;
    let rho_ms =
        (rho * rho) * average_albedo / (Vec3::ONE - rho * f32::max(0.0, 1.0 - average_albedo));
    return (rho / PI) * (a + b * s_over_t)
        + (rho_ms / PI) * f32::max(1e-7, 1.0 - on_o) * f32::max(1e-7, 1.0 - on_i)
            / f32::max(1e-7, 1.0 - average_albedo);
}

/// # The Diffuse Lobe
///
/// As by the OpenPBR specifications, this implements the energy-preserving Oren-Nayar model
/// proposed in "EON: A practical energy-preserving rough diffuse BRDF" by Jamie Portsmouth,
/// Peter Kutz and Stephen Hill (2024).
pub struct Diffuse {
    pub weight: f32,
    pub color: Vec3,
    pub roughness: f32,
}

impl From<&Material> for Diffuse {
    fn from(m: &Material) -> Self {
        Self {
            weight: m.base_weight,
            color: m.base_color,
            roughness: m.base_diffuse_roughness,
        }
    }
}

impl Lobe for Diffuse {
    fn wo_is_valid(&self, wi: Vec3) -> bool {
        wi.is_in_upper_hemisphere()
    }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput {
        if !wi.is_in_upper_hemisphere() || !wo.is_in_upper_hemisphere() {
            return Throughput::ZERO;
        }

        Throughput::from_diffuse(energy_compensated_oren_nayar(
            self.weight * self.color,
            PI / 2.0 * self.roughness,
            wi,
            wo,
        ))
    }

    fn sample(&self, random: Vec3, wo: Vec3) -> Option<Sample> {
        if !wo.is_in_upper_hemisphere() {
            return None;
        }

        let wi = cosine_hemisphere_sample(random.truncate());
        let throughput = self.eval(wo, wi);
        let density = self.density(wo, wi);

        Some(Sample {
            lobe_type: LobeType::Diffuse,
            throughput,
            density,
            wi,
        })
    }

    fn density(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !wo.is_in_upper_hemisphere() {
            return 0.0;
        }

        cosine_hemisphere_density(wi.cos_theta())
    }

    fn estimate_directional_albedo(&self, wi: Vec3, _: &[Vec3]) -> Vec3 {
        if !wi.is_in_upper_hemisphere() {
            return Vec3::ZERO;
        }

        self.color * self.weight
    }
}
