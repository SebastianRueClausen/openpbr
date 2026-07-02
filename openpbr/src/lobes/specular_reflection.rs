use crate::{
    consts::{DENOM_TOLERANCE, IOR_EPSILON},
    fresnel::{f0_from_ior, fresnel_dielectric},
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::{self, Microfacet},
    Sampler,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

/// Computes the effective specular IOR as seen from outside the coat layer. OpenPBR Eq. (60).
///
/// The coat sits above the specular interface, so light refracts through it before reaching
/// the specular layer. This makes the specular surface appear less refractive from the outside.
pub(crate) fn effective_specular_ior(specular_ior: f32, coat_ior: f32, coat_weight: f32) -> f32 {
    specular_ior / (1.0 + coat_weight * (coat_ior - 1.0))
}

/// Converts specular weight to an IOR ratio that produces the correct Fresnel at normal
/// incidence. OpenPBR Eq. (26).
///
/// In other words, since `specular_weight` is an artistic property as opposed to a physical
/// property, this finds the IOR ratio that would produce the reflectance corresponding to
/// `specular_weight`.
fn specular_ior_ratio(
    specular_ior: f32,
    coat_ior: f32,
    coat_weight: f32,
    specular_weight: f32,
) -> f32 {
    let ior = effective_specular_ior(specular_ior, coat_ior, coat_weight);
    let f0 = f0_from_ior(ior);
    let clamped_weight = specular_weight.clamp(0.0, 1.0 / f0.max(DENOM_TOLERANCE));
    let epsilon = (ior - 1.0).signum() * (clamped_weight * f0).sqrt();
    (1.0 + epsilon) / (1.0 - epsilon).max(DENOM_TOLERANCE)
}

/// Evaluates the Fresnel reflectance at the specular interface accounting for the outer medium
/// above it. OpenPBR Eq. (75).
///
/// In particular, the outer medium refracts the ray before it hits the specular surface, so the
/// effective angle of incidence is shallower than the geometric angle.
fn specular_fresnel(outer_ior: f32, fresnel_ior: f32, cos_theta: f32) -> Vec3 {
    let refracted_cos_theta = (1.0 - (1.0 - cos_theta.powi(2)) / outer_ior.powi(2)).sqrt();
    Vec3::splat(fresnel_dielectric(fresnel_ior, refracted_cos_theta))
}

fn brdf_and_density(
    microfacet: &Microfacet,
    wo: Vec3,
    wi: Vec3,
    microfacet_normal: Vec3,
    ior_ratio: f32,
    outer_ior: f32,
    fresnel_ior: f32,
    specular_color: Vec3,
) -> (Vec3, f32) {
    let fresnel = if wo.is_in_upper_hemisphere() {
        specular_fresnel(outer_ior, fresnel_ior, wo.dot(microfacet_normal).abs())
    } else {
        Vec3::splat(fresnel_dielectric(
            1.0 / ior_ratio,
            wo.dot(microfacet_normal).abs(),
        ))
    };
    let (brdf, density) =
        microfacet::torrance_sparrow(microfacet, wo, wi, microfacet_normal, fresnel);
    (brdf * specular_color, density)
}

/// # The Specular Reflection Lobe
///
/// The lobe is more or less the standard PBR implementation of dielectric materials. However,
/// because we don't model layers physically, we have to handle the possibility of a coat layer on
/// top. If a ray hits the dielectric layer, it might have refracted on the coat. This effectively
/// changes the incident direction and can cause issues with total internal reflections. The
/// OpenPBR specifications propose to ways to handle this. This implementation uses the second.
/// In particular, it approximates the resulting Fresnel factor. See OpenPBR Eq. (75).
pub struct SpecularReflection {
    /// The IOR ratio across the specular interface.
    pub ior_ratio: f32,

    /// IOR of the medium directly above the specular surface, i.e. the coat layer.
    pub outer_ior: f32,

    /// Effective IOR for evaluating Fresnel, blending `specular_ior` toward `ior_ratio` as coat
    /// weight increases.
    ///
    /// In OpenPBR, it's computed as `specular_ior + coat_weight * (ior_ratio - specular_ior)`.
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
            outer_ior: 1.0 + m.coat_weight * (m.coat_ior - 1.0),
            fresnel_ior: m.specular_ior + m.coat_weight * (ior_ratio - m.specular_ior),
            specular_color: m.specular_color,
            roughness: m.specular_roughness,
            roughness_anisotropy: m.specular_roughness_anisotropy,
            rotation: m.specular_rotation,
        }
    }
}

impl SpecularReflection {
    fn wo_is_valid(&self, _: Vec3) -> bool {
        (self.ior_ratio - 1.0).abs() >= IOR_EPSILON
    }
}

impl Lobe for SpecularReflection {
    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput {
        if !wo.is_in_same_hemisphere(&wi) {
            return Throughput::ZERO;
        }

        if (self.ior_ratio - 1.0).abs() < IOR_EPSILON {
            return Throughput::ZERO;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);
        let wi = rotation.rotate(wi);

        let microfacet_normal = (wo + wi).normalize();

        if wo.dot(microfacet_normal) * wo.cos_theta() < 0.0
            || wi.dot(microfacet_normal) * wi.cos_theta() < 0.0
        {
            return Throughput::ZERO;
        }

        let (brdf, _) = brdf_and_density(
            &microfacet,
            wo,
            wi,
            microfacet_normal,
            self.ior_ratio,
            self.outer_ior,
            self.fresnel_ior,
            self.specular_color,
        );

        Throughput::from_specular(brdf)
    }

    fn sample<S: Sampler>(&self, rng: &mut S, wo: Vec3) -> Option<Sample> {
        if (self.ior_ratio - 1.0).abs() < IOR_EPSILON {
            return None;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);

        let microfacet_normal = if wo.is_in_upper_hemisphere() {
            microfacet.sample(wo, rng.next_vec2())
        } else {
            microfacet
                .sample(wo.flip_hemisphere(), rng.next_vec2())
                .flip_hemisphere()
        };

        let wi = -wo.reflect(microfacet_normal);
        if !wo.is_in_same_hemisphere(&wi) {
            return None;
        }

        let (brdf, density) = brdf_and_density(
            &microfacet,
            wo,
            wi,
            microfacet_normal,
            self.ior_ratio,
            self.outer_ior,
            self.fresnel_ior,
            self.specular_color,
        );

        Some(Sample {
            lobe_type: LobeType::SpecularReflection,
            throughput: Throughput::from_specular(brdf),
            wi: rotation.inverse_rotate(wi),
            density,
        })
    }

    fn density(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !wo.is_in_same_hemisphere(&wi) {
            return 0.0;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);
        let wi = rotation.rotate(wi);

        let (_, density) = brdf_and_density(
            &microfacet,
            wo,
            wi,
            (wo + wi).normalize(),
            self.ior_ratio,
            self.outer_ior,
            self.fresnel_ior,
            self.specular_color,
        );

        density
    }

    fn estimate_directional_albedo(&self, wo: Vec3) -> Vec3 {
        if !self.wo_is_valid(wo) {
            return Vec3::ZERO;
        }

        let cos_theta = wo.cos_theta().abs();
        let fresnel = if wo.is_in_upper_hemisphere() {
            specular_fresnel(self.outer_ior, self.fresnel_ior, cos_theta)
        } else {
            Vec3::splat(fresnel_dielectric(1.0 / self.ior_ratio, cos_theta))
        };

        self.specular_color * fresnel
    }
}
