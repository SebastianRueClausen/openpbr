use crate::{
    consts::{DENOM_TOLERANCE, DENSITY_EPSILON, IOR_EPSILON},
    fresnel::fresnel_dielectric,
    material::Material,
    math::{LocalRotation, SphericalCoordinates},
    microfacet::Microfacet,
};
use glam::Vec3;
use std::f32::consts::PI;

use super::{Lobe, LobeType, Sample, Throughput};

/// Computes the refracted direction. Returns None on total internal reflection.
/// `ior` is n_i/n_t (incident over transmitted index of refraction).
fn refraction_direction(normal: Vec3, ior: f32, wo: Vec3) -> Option<Vec3> {
    let cos_theta_in = wo.dot(normal);

    let sin_theta_in_sq = (1.0 - cos_theta_in.powi(2)).max(0.0);
    let sin_theta_tr_sq = ior.powi(2) * sin_theta_in_sq;

    if sin_theta_tr_sq >= 1.0 {
        return None;
    }

    let cos_theta_tr = (1.0 - sin_theta_tr_sq).sqrt();

    Some(ior * (-wo) + (ior * cos_theta_in - cos_theta_tr) * normal)
}

fn tint(transmission_color: Vec3, transmission_depth: f32) -> Vec3 {
    if transmission_depth == 0.0 {
        transmission_color
    } else {
        Vec3::ONE
    }
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
    let wo_dot_n = wo_rotated.dot(microfacet_normal);
    let d = microfacet.distribution(microfacet_normal);
    let visible_normals = d * microfacet.masking(wo_rotated) * wo_dot_n.max(0.0)
        / wo_rotated.cos_theta().abs().max(DENOM_TOLERANCE);
    let jacobian = ior.powi(2) * wo.cos_theta().abs()
        / (wi.cos_theta() + ior * wo.cos_theta())
            .powi(2)
            .max(DENOM_TOLERANCE);
    let density = visible_normals * jacobian;
    let visibility = microfacet.visibility(wo_rotated, wi_rotated);
    let transmission = (1.0 - fresnel_dielectric(1.0 / ior, wo_dot_n.abs())).max(0.0);
    let btdf = transmission * wo_dot_n.abs() * jacobian * visibility * d
        / (wi.cos_theta().abs() * wo.cos_theta().abs()).max(DENOM_TOLERANCE);
    (
        Vec3::splat(btdf) * tint(transmission_color, transmission_depth),
        density,
    )
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
    fn ior(&self, wo: Vec3) -> f32 {
        if wo.cos_theta() > 0.0 {
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
        if (ior - 1.0).abs() < IOR_EPSILON {
            let density = 1.0 / DENSITY_EPSILON;
            let value = tint(self.transmission_color, self.transmission_depth) * density
                / wi.cos_theta().abs().max(DENOM_TOLERANCE);
            return Throughput::from_specular(value);
        }

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let wi_rotated = rotation.rotate(wi);

        let microfacet_normal_raw = -wi_rotated - ior * wo_rotated;
        if microfacet_normal_raw.length_squared() == 0.0 {
            return Throughput::ZERO;
        }

        let microfacet_normal = {
            let n = microfacet_normal_raw.normalize_or_zero();
            if n.cos_theta() <= 0.0 {
                -n
            } else {
                n
            }
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

        if (ior - 1.0).abs() < IOR_EPSILON {
            let wi = -wo;
            let density = 1.0 / DENSITY_EPSILON;
            let value = tint(self.transmission_color, self.transmission_depth) * density
                / wi.cos_theta().abs().max(DENOM_TOLERANCE);
            return Some(Sample {
                lobe_type: LobeType::SpecularTransmission,
                throughput: Throughput::from_specular(value),
                density,
                wi,
            });
        }

        let microfacet = Microfacet::new(self.roughness, self.roughness_anisotropy);
        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);

        let microfacet_normal = if wo_rotated.cos_theta() > 0.0 {
            microfacet.sample(wo_rotated, random.truncate())
        } else {
            let wo_flipped = Vec3::new(wo_rotated.x, wo_rotated.y, -wo_rotated.z);
            let mut n = microfacet.sample(wo_flipped, random.truncate());
            n.z = -n.z;
            n
        };

        let refract_dir = refraction_direction(microfacet_normal, ior, wo_rotated)?;

        let wi_rotated = refract_dir.normalize_or_zero();
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
        if (ior - 1.0).abs() < IOR_EPSILON {
            return 1.0 / DENSITY_EPSILON;
        }

        let rotation = LocalRotation::new(2.0 * PI * self.rotation);
        let wo_rotated = rotation.rotate(wo);
        let wi_rotated = rotation.rotate(wi);

        let microfacet_normal_raw = -wi_rotated - ior * wo_rotated;
        if microfacet_normal_raw.length_squared() == 0.0 {
            return 0.0;
        }

        let microfacet_normal = {
            let n = microfacet_normal_raw.normalize_or_zero();
            if n.cos_theta() <= 0.0 {
                -n
            } else {
                n
            }
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
