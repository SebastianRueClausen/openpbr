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

// Recover the microfacet normal. Returns `None` if the normal could not have been sampled.
fn microfacet_normal(wo: Vec3, wi: Vec3, ior: f32) -> Option<Vec3> {
    let n = (-wi - ior * wo).try_normalize()?;

    let h = if n.dot(wo) > 0.0 {
        n
    } else {
        if ior > 1.0 {
            -n
        } else {
            return None;
        }
    };

    h.in_same_hemisphere(&wo).then_some(h)
}

fn bsdf_and_density(
    microfacet: &Microfacet,
    microfacet_normal: Vec3,
    wo_rotated: Vec3,
    wi_rotated: Vec3,
    wo: Vec3,
    wi: Vec3,
    ior: f32,
    transmission_color: Vec3,
    transmission_depth: f32,
) -> (Vec3, f32) {
    let wo_dot_normal = wo_rotated.dot(microfacet_normal);
    let d = microfacet.distribution(microfacet_normal);
    let microfacet_density = d * microfacet.masking(wo_rotated) * wo_dot_normal.max(0.0)
        / wo_rotated.cos_theta().abs().max(DENOM_TOLERANCE);

    let cos_tangent_squared = 1.0 - ior.powi(2) * (1.0 - wo_dot_normal.max(0.0).powi(2)).max(0.0);
    if cos_tangent_squared < 0.0 {
        // Total internal reflection.
        return (Vec3::ZERO, 0.0);
    }

    let cos_tangent = cos_tangent_squared.sqrt();

    let refraction_denom = (ior * wo_dot_normal.max(0.0) - cos_tangent)
        .powi(2)
        .max(DENOM_TOLERANCE);

    // See Walter et al. (2007) Eq. 17.
    let density = microfacet_density * cos_tangent / refraction_denom;
    let visibility = microfacet.visibility(wo_rotated, wi_rotated);
    let transmission =
        (1.0 - fresnel_dielectric(1.0 / ior, wo_dot_normal.max(0.0))).clamp(0.0, 1.0);
    let btdf = transmission * wo_dot_normal.max(0.0) * cos_tangent * visibility * d
        / (wi.cos_theta().abs() * wo.cos_theta().abs() * refraction_denom).max(DENOM_TOLERANCE);
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
        if wo.in_upper_hemisphere() {
            1.0 / self.specular_ior
        } else {
            self.specular_ior
        }
    }
}

impl Lobe for SpecularTransmission {
    fn incidence_is_valid(&self, _: Vec3) -> bool {
        true
    }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput {
        if wo.in_same_hemisphere(&wi) {
            return Throughput::ZERO;
        }

        let ior = self.ior(wo);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let wi_rotated = rotation.rotate(wi);

        let Some(microfacet_normal) = microfacet_normal(wo_rotated, wi_rotated, ior) else {
            return Throughput::ZERO;
        };

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let (btdf, _) = bsdf_and_density(
            &microfacet,
            microfacet_normal,
            wo_rotated,
            wi_rotated,
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
        let wo_rotated = rotation.rotate(wo);

        let microfacet_normal = if wo_rotated.in_upper_hemisphere() {
            microfacet.sample(wo_rotated, random.truncate())
        } else {
            microfacet
                .sample(wo_rotated.flip_hemisphere(), random.truncate())
                .flip_hemisphere()
        };

        let wi_rotated = (-wo_rotated)
            .refract(microfacet_normal, ior)
            .try_normalize()?;
        let wi = rotation.inverse_rotate(wi_rotated);

        let (btdf, density) = bsdf_and_density(
            &microfacet,
            microfacet_normal,
            wo_rotated,
            wi_rotated,
            wo,
            wi,
            ior,
            self.transmission_color,
            self.transmission_depth,
        );

        Some(Sample {
            lobe_type: LobeType::SpecularTransmission,
            throughput: Throughput::from_specular(btdf),
            density,
            wi,
        })
    }

    fn density(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.in_same_hemisphere(&wi) {
            return 0.0;
        }

        let ior = self.ior(wo);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let wi_rotated = rotation.rotate(wi);

        let Some(microfacet_normal) = microfacet_normal(wo_rotated, wi_rotated, ior) else {
            return 0.0;
        };

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let (_, density) = bsdf_and_density(
            &microfacet,
            microfacet_normal,
            wo_rotated,
            wi_rotated,
            wo,
            wi,
            ior,
            self.transmission_color,
            self.transmission_depth,
        );

        density
    }
}
