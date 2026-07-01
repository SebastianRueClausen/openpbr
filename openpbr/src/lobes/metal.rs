use crate::{
    consts::DENSITY_EPSILON,
    fresnel::schlick,
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::{self, Microfacet},
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
    let fresnel = schlick(f0, cos_theta)
        - cos_theta * (1.0 - cos_theta).powi(6) * (Vec3::ONE - tint) * schlick(f0, MU_BAR) / denom;
    fresnel.clamp(Vec3::ZERO, Vec3::ONE)
}

fn brdf_and_density(
    microfacet: &Microfacet,
    microfacet_normal: Vec3,
    wo: Vec3,
    wi: Vec3,
    f0: Vec3,
    tint: Vec3,
) -> (Vec3, f32) {
    let fresnel = fresnel_metal(wo.dot(microfacet_normal).abs(), f0, tint);
    microfacet::torrance_sparrow(microfacet, wo, wi, microfacet_normal, fresnel)
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
    fn wo_is_valid(&self, wo: Vec3) -> bool {
        wo.is_in_upper_hemisphere()
    }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput {
        if !wo.is_in_upper_hemisphere() || !wi.is_in_upper_hemisphere() {
            return Throughput::ZERO;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);
        let wi = rotation.rotate(wi);

        let microfacet_normal = (wo + wi).normalize();
        let (f0, tint) = self.f0_tint();

        let (brdf, _) = brdf_and_density(&microfacet, microfacet_normal, wo, wi, f0, tint);

        Throughput::from_specular(brdf)
    }

    fn sample(&self, random: Vec3, wo: Vec3) -> Option<Sample> {
        if !wo.is_in_upper_hemisphere() {
            return None;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);
        let microfacet_normal = microfacet.sample(wo, random.truncate());
        let wi = -wo.reflect(microfacet_normal);

        if !wo.is_in_same_hemisphere(&wi) {
            return None;
        }

        let (f0, tint) = self.f0_tint();
        let (brdf, density) = brdf_and_density(&microfacet, microfacet_normal, wo, wi, f0, tint);

        Some(Sample {
            lobe_type: LobeType::Metal,
            throughput: Throughput::from_specular(brdf),
            wi: rotation.inverse_rotate(wi),
            density,
        })
    }

    fn density(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !wo.is_in_upper_hemisphere() || !wi.is_in_upper_hemisphere() {
            return DENSITY_EPSILON;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);
        let wi = rotation.rotate(wi);

        let microfacet_normal = (wo + wi).normalize();
        let (f0, tint) = self.f0_tint();
        let (_, density) = brdf_and_density(&microfacet, microfacet_normal, wo, wi, f0, tint);

        density
    }

    fn estimate_directional_albedo(&self, wo: Vec3, _: &[Vec3]) -> Vec3 {
        if !wo.is_in_upper_hemisphere() {
            return Vec3::ZERO;
        }

        let (f0, tint) = self.f0_tint();
        fresnel_metal(wo.cos_theta().abs(), f0, tint)
    }
}
