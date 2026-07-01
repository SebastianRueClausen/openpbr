use crate::{
    consts::DENOM_TOLERANCE,
    fresnel::fresnel_dielectric,
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::Microfacet,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

fn tint(transmission_color: Vec3, transmission_depth: f32) -> Vec3 {
    if transmission_depth == 0.0 {
        transmission_color
    } else {
        Vec3::ONE
    }
}

// Recover the generalized half-vector, i.e. the microfacet normal that would have refracted `wo`
// into `wi`. Returns `None` if `wi` could not have come from a valid refraction of `wo` through
// any microfacet.
fn microfacet_normal(wo: Vec3, wi: Vec3, ior: f32) -> Option<Vec3> {
    if wo.cos_theta() == 0.0 || wi.cos_theta() == 0.0 {
        return None;
    }

    let etap = 1.0 / ior;
    let normal = (wi * etap + wo).try_normalize()?;
    let normal = if normal.cos_theta() < 0.0 {
        -normal
    } else {
        normal
    };

    // Discard backfacing microfacets. A half-vector that isn't on the same side as both `wo`
    // and `wi` could not actually have produced this refraction.
    if normal.dot(wi) * wi.cos_theta() < 0.0 || normal.dot(wo) * wo.cos_theta() < 0.0 {
        return None;
    }

    Some(normal)
}

fn bsdf_and_density(
    microfacet: &Microfacet,
    microfacet_normal: Vec3,
    wo: Vec3,
    wi: Vec3,
    ior: f32,
    transmission_color: Vec3,
    transmission_depth: f32,
) -> (Vec3, f32) {
    let wo_dot_normal = wo.dot(microfacet_normal);
    let wi_dot_normal = wi.dot(microfacet_normal);

    let distribution = microfacet.distribution(microfacet_normal);
    let microfacet_density = distribution * microfacet.masking(wo) * wo_dot_normal.max(0.0)
        / wo.cos_theta().abs().max(DENOM_TOLERANCE);

    let denom = (wi_dot_normal + wo_dot_normal * ior)
        .powi(2)
        .max(DENOM_TOLERANCE);
    let density = microfacet_density * wi_dot_normal.abs() / denom;

    let fresnel = fresnel_dielectric(1.0 / ior, wo_dot_normal.abs());
    let transmission = (1.0 - fresnel).clamp(0.0, 1.0);
    let visibility = microfacet.visibility(wo, wi);

    let cos_product = (wi.cos_theta() * wo.cos_theta()).abs().max(DENOM_TOLERANCE);
    let btdf = transmission
        * distribution
        * visibility
        * (wi_dot_normal * wo_dot_normal / (cos_product * denom)).abs();

    (btdf * tint(transmission_color, transmission_depth), density)
}

pub struct SpecularTransmission {
    pub specular_ior: f32,
    pub transmission_color: Vec3,
    pub transmission_depth: f32,
    pub roughness: f32,
    pub roughness_anisotropy: f32,
    pub rotation: f32,
}

impl From<&Material> for SpecularTransmission {
    fn from(m: &Material) -> Self {
        Self {
            specular_ior: m.specular_ior,
            transmission_color: m.transmission_color,
            transmission_depth: m.transmission_depth,
            roughness: m.specular_roughness,
            roughness_anisotropy: m.specular_roughness_anisotropy,
            rotation: m.specular_rotation,
        }
    }
}

impl SpecularTransmission {
    // Compute the index of refraction based on the whether the ray is entering or leaving surface.
    fn ior(&self, wo: Vec3) -> f32 {
        if wo.is_in_upper_hemisphere() {
            1.0 / self.specular_ior
        } else {
            self.specular_ior
        }
    }
}

impl Lobe for SpecularTransmission {
    fn wo_is_valid(&self, _: Vec3) -> bool {
        true
    }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput {
        if wo.is_in_same_hemisphere(&wi) {
            return Throughput::ZERO;
        }

        let ior = self.ior(wo);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);
        let wi = rotation.rotate(wi);

        let Some(microfacet_normal) = microfacet_normal(wo, wi, ior) else {
            return Throughput::ZERO;
        };

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let (btdf, _) = bsdf_and_density(
            &microfacet,
            microfacet_normal,
            wo,
            wi,
            ior,
            self.transmission_color,
            self.transmission_depth,
        );

        Throughput::from_specular(btdf)
    }

    fn sample(&self, random: Vec3, wo: Vec3) -> Option<Sample> {
        let ior = self.ior(wo);

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);

        let microfacet_normal = if wo.is_in_upper_hemisphere() {
            microfacet.sample(wo, random.truncate())
        } else {
            microfacet
                .sample(wo.flip_hemisphere(), random.truncate())
                .flip_hemisphere()
        };

        let wi = (-wo).refract(microfacet_normal, ior).try_normalize()?;
        let (btdf, density) = bsdf_and_density(
            &microfacet,
            microfacet_normal,
            wo,
            wi,
            ior,
            self.transmission_color,
            self.transmission_depth,
        );

        Some(Sample {
            lobe_type: LobeType::SpecularTransmission,
            throughput: Throughput::from_specular(btdf),
            wi: rotation.inverse_rotate(wi),
            density,
        })
    }

    fn density(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.is_in_same_hemisphere(&wi) {
            return 0.0;
        }

        let ior = self.ior(wo);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo = rotation.rotate(wo);
        let wi = rotation.rotate(wi);

        let Some(microfacet_normal) = microfacet_normal(wo, wi, ior) else {
            return 0.0;
        };

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let (_, density) = bsdf_and_density(
            &microfacet,
            microfacet_normal,
            wo,
            wi,
            ior,
            self.transmission_color,
            self.transmission_depth,
        );

        density
    }

    fn estimate_directional_albedo(&self, wo: Vec3, _: &[Vec3]) -> Vec3 {
        let ior = self.ior(wo);
        let cos_theta = wo.cos_theta().abs();
        let transmittance = (1.0 - fresnel_dielectric(1.0 / ior, cos_theta)).clamp(0.0, 1.0);

        tint(self.transmission_color, self.transmission_depth) * transmittance
    }
}
