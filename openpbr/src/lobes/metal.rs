use crate::{
    consts::{DENOM_TOLERANCE, DENSITY_EPSILON},
    fresnel::schlick,
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::Microfacet,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

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
    wo_rotated: Vec3,
    wi_rotated: Vec3,
    wo: Vec3,
    wi: Vec3,
    f0: Vec3,
    tint: Vec3,
) -> (Vec3, f32) {
    let wo_dot_n = wo_rotated.dot(microfacet_normal);
    let d = microfacet.distribution(microfacet_normal);
    let visible_normals = d * microfacet.masking(wo_rotated) * wo_dot_n.max(0.0)
        / wo_rotated.cos_theta().max(DENOM_TOLERANCE);
    let jacobian = 1.0 / (4.0 * wo_dot_n).abs().max(DENOM_TOLERANCE);
    let density = (visible_normals * jacobian).max(DENSITY_EPSILON);
    let fresnel = fresnel_metal(wo_dot_n.abs(), f0, tint);
    let brdf = fresnel * d * microfacet.visibility(wo_rotated, wi_rotated)
        / (4.0 * wi.cos_theta().abs() * wo.cos_theta().abs()).max(DENOM_TOLERANCE);
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

impl From<&Material> for Metal {
    fn from(m: &Material) -> Self {
        Self {
            base_weight: m.base_weight,
            base_color: m.base_color,
            specular_weight: m.specular_weight,
            specular_color: m.specular_color,
            roughness: m.specular_roughness,
            roughness_anisotropy: m.specular_roughness_anisotropy,
            rotation: m.specular_rotation,
        }
    }
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

    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput {
        if wo.cos_theta() < DENOM_TOLERANCE || wi.cos_theta() < DENOM_TOLERANCE {
            return Throughput::ZERO;
        }
        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let wi_rotated = rotation.rotate(wi);

        let microfacet_normal = (wo_rotated + wi_rotated).normalize();
        let (f0, tint) = self.f0_tint();

        let (brdf, _) = brdf_and_density(
            &microfacet,
            microfacet_normal,
            wo_rotated,
            wi_rotated,
            wo,
            wi,
            f0,
            tint,
        );

        Throughput::from_specular(brdf)
    }

    fn sample(&self, random: Vec3, wo: Vec3) -> Option<Sample> {
        if wo.cos_theta() < DENOM_TOLERANCE {
            return None;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let microfacet_normal = microfacet.sample(wo_rotated, random.truncate());
        let wi_rotated = -wo_rotated.reflect(microfacet_normal);

        if !wo_rotated.in_same_hemisphere(&wi_rotated) {
            return None;
        }

        let wi = rotation.inverse_rotate(wi_rotated);
        let (f0, tint) = self.f0_tint();

        let (brdf, density) = brdf_and_density(
            &microfacet,
            microfacet_normal,
            wo_rotated,
            wi_rotated,
            wo,
            wi,
            f0,
            tint,
        );

        Some(Sample {
            lobe_type: LobeType::Metal,
            throughput: Throughput::from_specular(brdf),
            density,
            wi,
        })
    }

    fn density(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.cos_theta() < DENOM_TOLERANCE || wi.cos_theta() < DENOM_TOLERANCE {
            return DENSITY_EPSILON;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let wi_rotated = rotation.rotate(wi);

        let microfacet_normal = (wo_rotated + wi_rotated).normalize();
        let (f0, tint) = self.f0_tint();
        let (_, density) = brdf_and_density(
            &microfacet,
            microfacet_normal,
            wo_rotated,
            wi_rotated,
            wo,
            wi,
            f0,
            tint,
        );

        density
    }
}
