use crate::{
    consts::IOR_EPSILON,
    fresnel::fresnel_dielectric,
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::{self, Microfacet},
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

fn brdf_and_density(
    microfacet: &Microfacet,
    wo: Vec3,
    wi: Vec3,
    microfacet_normal: Vec3,
    ior: f32,
) -> (Vec3, f32) {
    let fresnel = Vec3::splat(fresnel_dielectric(ior, wo.dot(microfacet_normal).abs()));
    microfacet::torrance_sparrow(microfacet, wo, wi, microfacet_normal, fresnel)
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
    fn ior(&self, wi: Vec3) -> f32 {
        if wi.is_in_upper_hemisphere() {
            self.ior
        } else {
            1.0 / self.ior
        }
    }
}

impl Lobe for Coat {
    fn wo_is_valid(&self, wo: Vec3) -> bool {
        (self.ior(wo) - 1.0).abs() >= IOR_EPSILON
    }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput {
        if !wo.is_in_same_hemisphere(&wi) {
            return Throughput::ZERO;
        }

        let ior = self.ior(wo);
        if (ior - 1.0).abs() < IOR_EPSILON {
            return Throughput::ZERO;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);

        let (brdf, _) = brdf_and_density(
            &microfacet,
            rotation.rotate(wo),
            rotation.rotate(wi),
            (wi + wo).normalize(),
            ior,
        );

        Throughput::from_specular(brdf)
    }

    fn sample(&self, random: Vec3, wo: Vec3) -> Option<Sample> {
        if !self.wo_is_valid(wo) {
            return None;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);

        let wo = rotation.rotate(wo);

        if !wo.is_in_upper_hemisphere() {
            return None;
        }

        let microfacet_normal = microfacet.sample(wo, random.truncate());

        let wi = -wo.reflect(microfacet_normal);
        if !wo.is_in_same_hemisphere(&wi) {
            return None;
        }

        let ior = self.ior(wo);
        let (brdf, density) = brdf_and_density(&microfacet, wo, wi, microfacet_normal, ior);

        Some(Sample {
            lobe_type: LobeType::Coat,
            throughput: Throughput::from_specular(brdf),
            wi: rotation.inverse_rotate(wi),
            density,
        })
    }

    fn density(&self, wo: Vec3, wi: Vec3) -> f32 {
        if !wo.is_in_same_hemisphere(&wi) || !self.wo_is_valid(wo) {
            return 0.0;
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);

        let (_, density) = brdf_and_density(
            &microfacet,
            rotation.rotate(wo),
            rotation.rotate(wi),
            (wi + wo).normalize(),
            self.ior(wo),
        );

        density
    }
}
