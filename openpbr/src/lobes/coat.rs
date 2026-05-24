use crate::{
    consts::{DENOM_TOLERANCE, IOR_EPSILON},
    fresnel::fresnel_dielectric,
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::Microfacet,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

fn brdf_and_density(
    microfacet: &Microfacet,
    wo_rotated: Vec3,
    wi_rotated: Vec3,
    wo: Vec3,
    wi: Vec3,
    ior: f32,
) -> (Vec3, f32) {
    let microfacet_normal = (wi_rotated + wo_rotated).normalize();
    let wo_dot_n = wo_rotated.dot(microfacet_normal);
    let d = microfacet.distribution(microfacet_normal);
    let visible_normals =
        d * microfacet.masking(wo_rotated) * wo_dot_n.max(0.0) / wo_rotated.cos_theta();
    let jacobian = 1.0 / (4.0 * wo_dot_n).abs().max(DENOM_TOLERANCE);
    let density = visible_normals * jacobian;
    let fresnel = fresnel_dielectric(ior, wo_dot_n.abs());
    let brdf = Vec3::splat(fresnel) * d * microfacet.visibility(wo_rotated, wi_rotated)
        / (4.0 * wi.cos_theta().abs() * wo.cos_theta().abs()).max(DENOM_TOLERANCE);
    (brdf, density)
}

pub struct Coat {
    pub ior: f32,
    pub roughness: f32,
    pub roughness_anisotropy: f32,
    pub rotation: f32,
}

impl From<&Material> for Coat {
    fn from(m: &Material) -> Self {
        Self {
            ior: m.coat_ior,
            roughness: m.coat_roughness,
            roughness_anisotropy: m.coat_roughness_anisotropy,
            rotation: m.coat_rotation,
        }
    }
}

impl Coat {
    fn ior(&self, cos_theta: f32) -> f32 {
        if cos_theta > 0.0 {
            self.ior
        } else {
            1.0 / self.ior
        }
    }
}

impl Lobe for Coat {
    fn incidence_is_valid(&self, wi: Vec3) -> bool {
        (self.ior(wi.cos_theta()) - 1.0).abs() >= IOR_EPSILON
    }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput {
        if !wo.in_same_hemisphere(&wi) {
            return Throughput::ZERO;
        }

        let ior = self.ior(wo.cos_theta());
        if (ior - 1.0).abs() < IOR_EPSILON {
            return Throughput::ZERO;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let wi_rotated = rotation.rotate(wi);

        let (brdf, _) = brdf_and_density(&microfacet, wo_rotated, wi_rotated, wo, wi, ior);

        Throughput::from_specular(brdf)
    }

    fn sample(&self, random: Vec3, wo: Vec3) -> Option<Sample> {
        if !self.incidence_is_valid(wo) {
            return None;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        if wo_rotated.cos_theta() <= 0.0 {
            return None;
        }
        let microfacet_normal = microfacet.sample(wo_rotated, random.truncate());
        let wi_rotated = -wo_rotated.reflect(microfacet_normal);
        if !wo_rotated.in_same_hemisphere(&wi_rotated) {
            return None;
        }
        let wi = rotation.inverse_rotate(wi_rotated);

        let ior = self.ior(wo.cos_theta());
        let (brdf, density) = brdf_and_density(&microfacet, wo_rotated, wi_rotated, wo, wi, ior);

        Some(Sample {
            lobe_type: LobeType::Coat,
            throughput: Throughput::from_specular(brdf),
            density,
            wi,
        })
    }

    fn density(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !wo.in_same_hemisphere(&wi) || !self.incidence_is_valid(wo) {
            return 0.0;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let wi_rotated = rotation.rotate(wi);

        let ior = self.ior(wo.cos_theta());
        let (_, density) = brdf_and_density(&microfacet, wo_rotated, wi_rotated, wo, wi, ior);

        density
    }
}
