use crate::{
    consts::{DENOM_TOLERANCE, IOR_EPSILON},
    fresnel::{f0_from_ior, fresnel_dielectric},
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::Microfacet,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

/// Computes the effective specular IOR as seen from outside the coat layer. OpenPBR Eq. (60).
///
/// The coat sits above the specular interface, so light refracts through it before reaching
/// the specular layer. This makes the specular surface appear less refractive from the outside.
fn specular_ior(specular_ior: f32, coat_ior: f32, coat_weight: f32) -> f32 {
    specular_ior / (1.0 + coat_weight * (coat_ior - 1.0))
}

/// Converts specular weight to an IOR ratio that produces the correct Fresnel at normal
/// incidence. OpenPBR Eq. (26).
///
/// In other words, since `specular_weight` is an artistic property as opposed to a physical
/// property, this finds the IOR ratio that would produce the reflectance corresponding to
/// `specular_weight`.
fn specular_ior_ratio(s_ior: f32, coat_ior: f32, coat_weight: f32, specular_weight: f32) -> f32 {
    let ior = specular_ior(s_ior, coat_ior, coat_weight);
    let f0 = f0_from_ior(ior);
    let clamped_weight = specular_weight.clamp(0.0, 1.0 / f0.max(DENOM_TOLERANCE));
    let epsilon = (ior - 1.0).signum() * (clamped_weight * f0).sqrt();
    (1.0 + epsilon) / (1.0 - epsilon).max(DENOM_TOLERANCE)
}

/// Evaluates the Fresnel reflectance at the specular interface accounting for the outer medium
/// above it. OpenPBR Eq. (75).
///
/// In particular, the outer medium refracts the ray before it hits the specular surface
/// (Snell's law), so the effective angle of incidence is shallower than the geometric angle.
fn specular_fresnel(outer_ior: f32, fresnel_ior: f32, cos_theta: f32) -> Vec3 {
    let refracted_cos_theta = (1.0 - (1.0 - cos_theta.powi(2)) / outer_ior.powi(2)).sqrt();
    Vec3::splat(fresnel_dielectric(fresnel_ior, refracted_cos_theta))
}

fn brdf_and_density(
    microfacet: &Microfacet,
    wi_rotated: Vec3,
    wo_rotated: Vec3,
    microfacet_normal: Vec3,
    wi: Vec3,
    wo: Vec3,
    ior_ratio: f32,
    outer_ior: f32,
    fresnel_ior: f32,
    specular_color: Vec3,
) -> (Vec3, f32) {
    let wi_dot_n = wi_rotated.dot(microfacet_normal);
    let d = microfacet.distribution(microfacet_normal);
    let visible_normals = d * microfacet.masking(wi_rotated) * wi_dot_n.max(0.0)
        / wi_rotated.cos_theta().max(DENOM_TOLERANCE);
    let jacobian = 1.0 / (4.0 * wi_dot_n).abs().max(DENOM_TOLERANCE);
    let density = visible_normals * jacobian;
    let fresnel = if wi_rotated.cos_theta() > 0.0 {
        specular_fresnel(outer_ior, fresnel_ior, wi_dot_n.abs())
    } else {
        Vec3::splat(fresnel_dielectric(1.0 / ior_ratio, wi_dot_n.abs()))
    };
    let brdf = fresnel * d * microfacet.visibility(wi_rotated, wo_rotated)
        / (4.0 * wo.cos_theta().abs() * wi.cos_theta().abs()).max(DENOM_TOLERANCE)
        * specular_color;
    (brdf, density)
}

pub struct SpecularReflection {
    /// The IOR ratio η_above / η_below across the specular interface.
    ///
    /// Encodes the full OpenPBR parameterization: `specular_weight` sets the target F0 at
    /// normal incidence, and the coat compresses the apparent IOR seen from outside. A ratio
    /// of exactly 1.0 means no index mismatch and therefore no reflection at all.
    pub ior_ratio: f32,

    /// IOR of the medium directly above the specular interface — in OpenPBR, the coat layer.
    ///
    /// Used to refract the incoming ray before evaluating Fresnel at the specular surface. A
    /// denser outer medium (higher IOR) bends rays toward the normal, reducing the effective
    /// angle of incidence and softening the grazing-angle reflections that would otherwise
    /// dominate.
    pub outer_ior: f32,

    /// Effective IOR for evaluating Fresnel at the specular interface, blending `specular_ior`
    /// toward `ior_ratio` as coat weight increases.
    ///
    /// In OpenPBR, it's computed as `specular_ior + coat_weight * (ior_ratio - specular_ior)`.
    ///
    /// So without a coat, this equals `specular_ior`. With a full coat, it shifts toward `ior_ratio`.
    pub fresnel_ior: f32,

    pub specular_color: Vec3,
    pub roughness: f32,
    pub roughness_anisotropy: f32,
    pub rotation: f32,
}

impl From<&Material> for SpecularReflection {
    fn from(m: &Material) -> Self {
        let ior_ratio =
            specular_ior_ratio(m.specular_ior, m.coat_ior, m.coat_weight, m.specular_weight);
        Self {
            ior_ratio,
            outer_ior: m.coat_ior,
            fresnel_ior: m.specular_ior + m.coat_weight * (ior_ratio - m.specular_ior),
            specular_color: m.specular_color,
            roughness: m.specular_roughness,
            roughness_anisotropy: m.specular_roughness_anisotropy,
            rotation: m.specular_rotation,
        }
    }
}

impl Lobe for SpecularReflection {
    fn incidence_is_valid(&self, _wi: Vec3) -> bool {
        (self.ior_ratio - 1.0).abs() >= IOR_EPSILON
    }

    fn eval(&self, wi: Vec3, wo: Vec3) -> Throughput {
        if !wi.in_same_hemisphere(&wo) {
            return Throughput::ZERO;
        }

        if (self.ior_ratio - 1.0).abs() < IOR_EPSILON {
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
            self.ior_ratio,
            self.outer_ior,
            self.fresnel_ior,
            self.specular_color,
        );

        Throughput::from_specular(brdf)
    }

    fn sample(&self, random: Vec3, wi: Vec3) -> Option<Sample> {
        if (self.ior_ratio - 1.0).abs() < IOR_EPSILON {
            return None;
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
            return None;
        }

        let wo = rotation.inverse_rotate(wo_rotated);

        let (brdf, density) = brdf_and_density(
            &microfacet,
            wi_rotated,
            wo_rotated,
            microfacet_normal,
            wi,
            wo,
            self.ior_ratio,
            self.outer_ior,
            self.fresnel_ior,
            self.specular_color,
        );

        Some(Sample {
            lobe_type: LobeType::SpecularReflection,
            throughput: Throughput::from_specular(brdf),
            density,
            wo,
        })
    }

    fn density(&self, wi: Vec3, wo: Vec3) -> f32 {
        if !wi.in_same_hemisphere(&wo) {
            return 0.0;
        }

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
            self.ior_ratio,
            self.outer_ior,
            self.fresnel_ior,
            self.specular_color,
        );

        density
    }
}
