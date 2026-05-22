use crate::{
    consts::{DENOM_TOLERANCE, IOR_EPSILON},
    fresnel::fresnel_dielectric,
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::Microfacet,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, Sample, Throughput};

fn brdf_and_density(
    microfacet: &Microfacet,
    wi_rotated: Vec3,
    wo_rotated: Vec3,
    wi: Vec3,
    wo: Vec3,
    ior: f32,
) -> (Vec3, f32) {
    let microfacet_normal = (wo_rotated + wi_rotated).normalize();
    let wi_dot_n = wi_rotated.dot(microfacet_normal);
    let d = microfacet.distribution(microfacet_normal);
    let visible_normals =
        d * microfacet.masking(wi_rotated) * wi_dot_n.max(0.0) / wi_rotated.cos_theta();
    let jacobian = 1.0 / (4.0 * wi_dot_n).abs().max(DENOM_TOLERANCE);
    let density = visible_normals * jacobian;
    let fresnel = fresnel_dielectric(ior, wi_dot_n.abs());
    let brdf = Vec3::splat(fresnel) * d * microfacet.visibility(wi_rotated, wo_rotated)
        / (4.0 * wo.cos_theta().abs() * wi.cos_theta().abs()).max(DENOM_TOLERANCE);
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

        let (brdf, _) = brdf_and_density(&microfacet, wi_rotated, wo_rotated, wi, wo, ior);

        Throughput::from_specular(brdf)
    }

    fn sample(&self, random: Vec3, wi: Vec3) -> Sample {
        if !self.incidence_is_valid(wi) {
            return Sample::ZERO;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wi_rotated = rotation.rotate(wi);
        if wi_rotated.cos_theta() <= 0.0 {
            return Sample::ZERO;
        }
        let microfacet_normal = microfacet.sample(wi_rotated, random.truncate());
        let wo_rotated = -wi_rotated.reflect(microfacet_normal);
        if !wi_rotated.in_same_hemisphere(&wo_rotated) {
            return Sample::ZERO;
        }
        let wo = rotation.inverse_rotate(wo_rotated);

        let ior = self.ior(wi.cos_theta());
        let (brdf, density) = brdf_and_density(&microfacet, wi_rotated, wo_rotated, wi, wo, ior);

        Sample {
            wo,
            throughput: Throughput::from_specular(brdf),
            density,
        }
    }

    fn density(&self, wi: Vec3, wo: Vec3) -> f32 {
        if !wi.in_same_hemisphere(&wo) || !self.incidence_is_valid(wi) {
            return 0.0;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wi_rotated = rotation.rotate(wi);
        let wo_rotated = rotation.rotate(wo);

        let ior = self.ior(wi.cos_theta());
        let (_, density) = brdf_and_density(&microfacet, wi_rotated, wo_rotated, wi, wo, ior);

        density
    }
}
