use crate::{
    consts::{DENOM_TOLERANCE, DENSITY_EPSILON},
    fresnel::schlick,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::Microfacet,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, Sample, Throughput};

/// F82-tint Schlick model. OpenPBR Eq. (30).
/// `f0` is the normal-incidence reflectance (base_weight * base_color).
/// `tint` is the specular tint (specular_weight * specular_color).
fn fresnel_metal(cos_theta: f32, f0: Vec3, tint: Vec3) -> Vec3 {
    const MU_BAR: f32 = 1.0 / 7.0;
    let denom = MU_BAR * (1.0 - MU_BAR).powi(6);
    let f_at_mu = schlick(f0, MU_BAR);
    schlick(f0, cos_theta)
        - cos_theta * (1.0 - cos_theta).powi(6) * (Vec3::ONE - tint) * f_at_mu / denom
}

fn brdf_and_density(
    microfacet: &Microfacet,
    microfacet_normal: Vec3,
    wi_rotated: Vec3,
    wo_rotated: Vec3,
    wi: Vec3,
    wo: Vec3,
    f0: Vec3,
    tint: Vec3,
) -> (Vec3, f32) {
    let wi_dot_n = wi_rotated.dot(microfacet_normal);
    let d = microfacet.distribution(microfacet_normal);
    let visible_normals = d * microfacet.masking(wi_rotated) * wi_dot_n.max(0.0)
        / wi_rotated.cos_theta().max(DENOM_TOLERANCE);
    let jacobian = 1.0 / (4.0 * wi_dot_n).abs().max(DENOM_TOLERANCE);
    let density = (visible_normals * jacobian).max(DENSITY_EPSILON);
    let fresnel = fresnel_metal(wi_dot_n.abs(), f0, tint);
    let brdf = fresnel * d * microfacet.visibility(wi_rotated, wo_rotated)
        / (4.0 * wo.cos_theta().abs() * wi.cos_theta().abs()).max(DENOM_TOLERANCE);
    (brdf, density)
}

pub struct Metal {
    pub base_weight: f32,
    pub base_color: Vec3,
    pub specular_weight: f32,
    pub specular_color: Vec3,
    pub roughness: f32,
    pub roughness_anisotropy: f32,
    pub rotation: f32,
}

impl Metal {
    fn f0_tint(&self) -> (Vec3, Vec3) {
        let f0 = self.base_weight * self.base_color;
        let tint = self.specular_weight * self.specular_color;
        (f0, tint)
    }
}

impl Lobe for Metal {
    fn incidence_is_valid(&self, wi: Vec3) -> bool {
        wi.cos_theta() >= DENOM_TOLERANCE
    }

    fn eval(&self, wi: Vec3, wo: Vec3) -> Throughput {
        if wi.cos_theta() < DENOM_TOLERANCE || wo.cos_theta() < DENOM_TOLERANCE {
            return Throughput::ZERO;
        }
        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wi_rotated = rotation.rotate(wi);
        let wo_rotated = rotation.rotate(wo);
        let microfacet_normal = (wi_rotated + wo_rotated).normalize();
        let (f0, tint) = self.f0_tint();
        let (brdf, _) = brdf_and_density(
            &microfacet,
            microfacet_normal,
            wi_rotated,
            wo_rotated,
            wi,
            wo,
            f0,
            tint,
        );
        Throughput::from_specular(brdf)
    }

    fn sample(&self, random: Vec3, wi: Vec3) -> Sample {
        if wi.cos_theta() < DENOM_TOLERANCE {
            return Sample::ZERO;
        }
        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wi_rotated = rotation.rotate(wi);
        let microfacet_normal = microfacet.sample(wi_rotated, random.truncate());
        let wo_rotated = -wi_rotated.reflect(microfacet_normal);
        if !wi_rotated.in_same_hemisphere(&wo_rotated) {
            return Sample::ZERO;
        }
        let wo = rotation.inverse_rotate(wo_rotated);
        let (f0, tint) = self.f0_tint();
        let (brdf, density) = brdf_and_density(
            &microfacet,
            microfacet_normal,
            wi_rotated,
            wo_rotated,
            wi,
            wo,
            f0,
            tint,
        );
        Sample {
            wo,
            throughput: Throughput::from_specular(brdf),
            density,
        }
    }

    fn density(&self, wi: Vec3, wo: Vec3) -> f32 {
        if wi.cos_theta() < DENOM_TOLERANCE || wo.cos_theta() < DENOM_TOLERANCE {
            return DENSITY_EPSILON;
        }
        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wi_rotated = rotation.rotate(wi);
        let wo_rotated = rotation.rotate(wo);
        let microfacet_normal = (wi_rotated + wo_rotated).normalize();
        let (f0, tint) = self.f0_tint();
        let (_, density) = brdf_and_density(
            &microfacet,
            microfacet_normal,
            wi_rotated,
            wo_rotated,
            wi,
            wo,
            f0,
            tint,
        );
        density
    }
}
