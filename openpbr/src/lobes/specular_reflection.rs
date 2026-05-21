use crate::{
    consts::{DENOM_TOLERANCE, IOR_EPSILON},
    fresnel::{f0_from_ior, fresnel_dielectric},
    math::{LocalRotation, SphericalCoordinates},
    microfacet::Microfacet,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, Sample, Throughput};

/// Computes the effective specular IOR accounting for the coat layer. OpenPBR Eq. (60).
fn specular_ior(specular_ior: f32, coat_ior: f32, coat_weight: f32) -> f32 {
    specular_ior / (1.0 + coat_weight * (coat_ior - 1.0))
}

/// Computes the IOR ratio that yields the correct Fresnel response given the
/// specular weight. OpenPBR Eq. (26).
fn specular_ior_ratio(s_ior: f32, coat_ior: f32, coat_weight: f32, specular_weight: f32) -> f32 {
    let ior = specular_ior(s_ior, coat_ior, coat_weight);
    let f0 = f0_from_ior(ior);
    let clamped_weight = specular_weight.clamp(0.0, 1.0 / f0.max(DENOM_TOLERANCE));
    let epsilon = (ior - 1.0).signum() * (clamped_weight * f0).sqrt();
    (1.0 + epsilon) / (1.0 - epsilon).max(DENOM_TOLERANCE)
}

/// Fresnel reflectance accounting for the coat layer refraction. OpenPBR Eq. (75).
fn specular_fresnel_with_coat(
    s_ior: f32,
    coat_ior: f32,
    coat_weight: f32,
    cos_theta: f32,
    ior: f32,
) -> Vec3 {
    let coat_refract_angle = (1.0 - (1.0 - cos_theta * cos_theta) / (coat_ior * coat_ior)).sqrt();
    let ior_mix = s_ior + coat_weight * (ior - s_ior);
    Vec3::splat(fresnel_dielectric(ior_mix, coat_refract_angle))
}

fn brdf_and_density(
    microfacet: &Microfacet,
    wi_rotated: Vec3,
    wo_rotated: Vec3,
    microfacet_normal: Vec3,
    wi: Vec3,
    wo: Vec3,
    ior: f32,
    s_ior: f32,
    coat_ior: f32,
    coat_weight: f32,
    specular_color: Vec3,
) -> (Vec3, f32) {
    let wi_dot_n = wi_rotated.dot(microfacet_normal);
    let d = microfacet.distribution(microfacet_normal);
    let visible_normals = d * microfacet.masking(wi_rotated) * wi_dot_n.max(0.0)
        / wi_rotated.cos_theta().max(DENOM_TOLERANCE);
    let jacobian = 1.0 / (4.0 * wi_dot_n).abs().max(DENOM_TOLERANCE);
    let density = visible_normals * jacobian;
    let fresnel = if wi_rotated.cos_theta() > 0.0 {
        specular_fresnel_with_coat(s_ior, coat_ior, coat_weight, wi_dot_n.abs(), ior)
    } else {
        Vec3::splat(fresnel_dielectric(ior, wi_dot_n.abs()))
    };
    let brdf = fresnel * d * microfacet.visibility(wi_rotated, wo_rotated)
        / (4.0 * wo.cos_theta().abs() * wi.cos_theta().abs()).max(DENOM_TOLERANCE)
        * specular_color;
    (brdf, density)
}

pub struct SpecularReflection {
    pub specular_ior: f32,
    pub specular_weight: f32,
    pub specular_color: Vec3,
    pub coat_ior: f32,
    pub coat_weight: f32,
    pub roughness: f32,
    pub roughness_anisotropy: f32,
    pub rotation: f32,
}

impl SpecularReflection {
    fn ior_ratio(&self) -> f32 {
        specular_ior_ratio(
            self.specular_ior,
            self.coat_ior,
            self.coat_weight,
            self.specular_weight,
        )
    }

    fn ior(&self, cos_theta: f32) -> f32 {
        let ior_ratio = self.ior_ratio();
        if cos_theta > 0.0 {
            ior_ratio
        } else {
            1.0 / ior_ratio
        }
    }
}

impl Lobe for SpecularReflection {
    fn incidence_is_valid(&self, wi: Vec3) -> bool {
        (self.ior(wi.cos_theta()) - 1.0).abs() >= IOR_EPSILON
    }

    fn eval(&self, wi: Vec3, wo: Vec3) -> Throughput {
        if !wi.in_same_hemisphere(&wo) {
            return Throughput::ZERO;
        }

        let ior = self.ior(wi.cos_theta());
        if (ior - 1.0).abs() < IOR_EPSILON {
            return Throughput::ZERO;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wi_rotated = rotation.rotate(wi);
        let wo_rotated = rotation.rotate(wo);

        let microfacet_normal = (wi_rotated + wo_rotated).normalize();
        if wi_rotated.dot(microfacet_normal) * wi_rotated.cos_theta() < 0.0
            || wo_rotated.dot(microfacet_normal) * wo_rotated.cos_theta() < 0.0
        {
            return Throughput::ZERO;
        }

        let (brdf, _) = brdf_and_density(
            &microfacet,
            wi_rotated,
            wo_rotated,
            microfacet_normal,
            wi,
            wo,
            ior,
            self.specular_ior,
            self.coat_ior,
            self.coat_weight,
            self.specular_color,
        );

        Throughput::from_specular(brdf)
    }

    fn sample(&self, random: Vec3, wi: Vec3) -> Sample {
        let ior = self.ior(wi.cos_theta());
        if (ior - 1.0).abs() < IOR_EPSILON {
            return Sample::ZERO;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wi_rotated = rotation.rotate(wi);

        let microfacet_normal = if wi_rotated.cos_theta() > 0.0 {
            microfacet.sample(wi_rotated, random.truncate())
        } else {
            let wi_flipped = Vec3::new(wi_rotated.x, wi_rotated.y, -wi_rotated.z);
            let mut n = microfacet.sample(wi_flipped, random.truncate());
            n.z = -n.z;
            n
        };

        let wo_rotated = -wi_rotated.reflect(microfacet_normal);
        if !wi_rotated.in_same_hemisphere(&wo_rotated) {
            return Sample::ZERO;
        }

        let wo = rotation.inverse_rotate(wo_rotated);

        let (brdf, density) = brdf_and_density(
            &microfacet,
            wi_rotated,
            wo_rotated,
            microfacet_normal,
            wi,
            wo,
            ior,
            self.specular_ior,
            self.coat_ior,
            self.coat_weight,
            self.specular_color,
        );

        Sample {
            wo,
            throughput: Throughput::from_specular(brdf),
            density,
        }
    }

    fn density(&self, wi: Vec3, wo: Vec3) -> f32 {
        if !wi.in_same_hemisphere(&wo) {
            return 0.0;
        }

        let ior = self.ior(wi.cos_theta());
        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wi_rotated = rotation.rotate(wi);
        let wo_rotated = rotation.rotate(wo);

        let microfacet_normal = (wi_rotated + wo_rotated).normalize();
        let (_, density) = brdf_and_density(
            &microfacet,
            wi_rotated,
            wo_rotated,
            microfacet_normal,
            wi,
            wo,
            ior,
            self.specular_ior,
            self.coat_ior,
            self.coat_weight,
            self.specular_color,
        );

        density
    }
}
