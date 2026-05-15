use crate::{
    dispersion::rgb_iors,
    fresnel::{f0_from_ior, fresnel_dielectric, metal_fresnel_f82_tint, refract, schlick},
    math::SphericalCoordinates,
    microfacet::Microfacet,
};
use glam::{Vec3, Vec3Swizzles};

use super::{Lobe, Sample, Throughput};

pub struct Specular {
    pub weight: f32,
    pub color: Vec3,
    pub roughness: f32,
    pub anisotropy: f32,
    pub ior: f32,
    pub metalness: f32,
    pub f82_tint: Vec3,
    pub transmission: Vec3,
    pub dispersion: f32,
}

impl Specular {
    fn microfacet(&self) -> Microfacet {
        Microfacet::new(self.roughness, self.anisotropy)
    }

    /// Per-channel IORs.
    fn iors(&self) -> Vec3 {
        rgb_iors(self.ior, self.dispersion)
    }

    /// Fresnel reflectance for the dielectric/metallic blend.
    fn fresnel_refl(&self, cos_theta_h: f32) -> Vec3 {
        let cos_theta = cos_theta_h.abs();
        let f0_dielectric = self.color * f0_from_ior(self.ior);
        let f_dielectric = schlick(f0_dielectric, cos_theta);
        let f_metallic = metal_fresnel_f82_tint(self.color, self.f82_tint, cos_theta);
        f_dielectric.lerp(f_metallic, self.metalness) * self.weight
    }

    /// Fresnel transmittance using the reference IOR.
    fn fresnel_trans(&self, cos_theta_h: f32) -> Vec3 {
        let f = fresnel_dielectric(self.ior, cos_theta_h.abs());
        self.transmission * (1.0 - f) * (1.0 - self.metalness) * self.weight
    }

    /// Reflection and transmission selection probabilities.
    fn probs(&self, cos_theta_h: f32) -> (f32, f32) {
        let r = self.fresnel_refl(cos_theta_h).max_element();
        let t = self.fresnel_trans(cos_theta_h).max_element();
        let sum = r + t;
        if sum <= 0.0 {
            return (1.0, 0.0);
        }
        (r / sum, t / sum)
    }
}

impl Lobe for Specular {
    fn eval(&self, wi: Vec3, wo: Vec3) -> Throughput {
        if wi.cos_theta() < 1e-10 {
            return Throughput::ZERO;
        }

        let microfacet = self.microfacet();

        if wi.in_same_hemisphere(&wo) {
            // Reflection.
            if wo.cos_theta() < 1e-10 {
                return Throughput::ZERO;
            }
            let h = (wi + wo).normalize();
            let cos_theta_h = wi.dot(h).abs();
            let f = self.fresnel_refl(cos_theta_h);
            let d = microfacet.distribution(h);
            let g2 = microfacet.visibility(wi, wo);
            Throughput::from_specular(f * d * g2 / (4.0 * wi.cos_theta() * wo.cos_theta()))
        } else {
            // Transmission.
            if self.transmission == Vec3::ZERO {
                return Throughput::ZERO;
            }

            let etas = self.iors();
            let g2 = microfacet.visibility(wi, wo);

            let channel = |index| {
                trans_btdf_channel(
                    wi,
                    wo,
                    &microfacet,
                    etas[index],
                    self.transmission[index],
                    self.metalness,
                    self.weight,
                )
            };

            Throughput::from_specular(Vec3::new(channel(0), channel(1), channel(2)) * g2)
        }
    }

    fn sample(&self, random: Vec3, wi: Vec3) -> Sample {
        if wi.cos_theta() < 1e-10 {
            return Sample::ZERO;
        }

        let microfacet = self.microfacet();
        let cos_theta_i = wi.cos_theta();

        if microfacet.is_mirror() {
            // Delta reflection/transmission.
            let f_refl = self.fresnel_refl(cos_theta_i);
            let f_trans = self.fresnel_trans(cos_theta_i);
            let (p_refl, p_trans) = self.probs(cos_theta_i);

            if random.z < p_refl {
                let wo = Vec3::new(-wi.x, -wi.y, wi.z);
                Sample {
                    wo,
                    throughput: Throughput::from_specular(f_refl / (wo.cos_theta() * p_refl)),
                    density: 1.0,
                }
            } else if p_trans > 0.0 {
                let eta = select_channel_eta(random.z, p_refl, p_trans, self.iors());
                match refract(wi, Vec3::Z, cos_theta_i, eta) {
                    None => Sample::ZERO,
                    Some(wo) => Sample {
                        wo,
                        throughput: Throughput::from_specular(
                            f_trans / (wo.cos_theta().abs() * p_trans),
                        ),
                        density: 1.0,
                    },
                }
            } else {
                Sample::ZERO
            }
        } else {
            // Microfacet reflection/refraction
            let h = microfacet.sample(wi, random.xy());
            let cos_h_i = wi.dot(h);
            if cos_h_i < 1e-10 {
                return Sample::ZERO;
            }

            let (p_refl, p_trans) = self.probs(cos_h_i);

            if random.z < p_refl {
                // Reflection.
                let wo = reflect(wi, h);
                if wo.cos_theta() < 1e-10 {
                    return Sample::ZERO;
                }
                Sample {
                    wo,
                    throughput: self.eval(wi, wo),
                    density: self.density(wi, wo),
                }
            } else if p_trans > 0.0 {
                // Transmission.
                let eta = select_channel_eta(random.z, p_refl, p_trans, self.iors());
                match refract(wi, h, cos_h_i, eta) {
                    None => Sample::ZERO,
                    Some(wo) => {
                        if wo.cos_theta() > -1e-10 {
                            return Sample::ZERO;
                        }
                        Sample {
                            wo,
                            throughput: self.eval(wi, wo),
                            density: self.density(wi, wo),
                        }
                    }
                }
            } else {
                Sample::ZERO
            }
        }
    }

    fn density(&self, wi: Vec3, wo: Vec3) -> f32 {
        if wi.cos_theta() < 1e-10 {
            return 0.0;
        }
        let microfacet = self.microfacet();
        if microfacet.is_mirror() {
            return 0.0;
        }

        if wi.in_same_hemisphere(&wo) {
            // Reflection.
            if wo.cos_theta() < 1e-10 {
                return 0.0;
            }
            let h = (wi + wo).normalize();
            let cos_h_i = wi.dot(h).abs();
            if cos_h_i < 1e-10 {
                return 0.0;
            }
            let (p_refl, _) = self.probs(cos_h_i);
            p_refl * microfacet.density(wi, h) / (4.0 * cos_h_i)
        } else {
            // Transmission.
            if self.transmission == Vec3::ZERO {
                return 0.0;
            }
            let etas = self.iors();

            // Use the reference (green) channel to evaluate the selection
            // probability. The channels are close enough that this is exact for
            // non-dispersive materials and a good approximation otherwise.
            let h_ref = -(wi + wo * etas.y).normalize();
            let cos_h_i_ref = wi.dot(h_ref);
            if cos_h_i_ref < 1e-10 {
                return 0.0;
            }
            let (_, p_trans) = self.probs(cos_h_i_ref);

            // Average the per-channel PDF contributions with equal channel weights.
            let pdf_r = trans_pdf_channel(wi, wo, &microfacet, etas.x);
            let pdf_g = trans_pdf_channel(wi, wo, &microfacet, etas.y);
            let pdf_b = trans_pdf_channel(wi, wo, &microfacet, etas.z);
            p_trans * (pdf_r + pdf_g + pdf_b) / 3.0
        }
    }
}

/// Evaluates the per-channel BTDF for one wavelength.
///
/// Returns 0.0 for geometrically invalid configurations (wrong hemisphere,
/// total internal reflection, etc.).
fn trans_btdf_channel(
    wi: Vec3,
    wo: Vec3,
    microfacet: &Microfacet,
    eta: f32,
    trans_c: f32,
    metalness: f32,
    weight: f32,
) -> f32 {
    let h = -(wi + wo * eta).normalize();
    if (eta >= 1.0 && h.cos_theta() <= 0.0) || (eta < 1.0 && h.cos_theta() >= 0.0) {
        return 0.0;
    }
    let cos_h_i = wi.dot(h);
    let cos_h_o = wo.dot(h);
    if cos_h_i * cos_h_o >= 0.0 {
        return 0.0;
    }
    let denom_sq = (cos_h_i + eta * cos_h_o).powi(2);
    if denom_sq < 1e-14 {
        return 0.0;
    }
    let f = trans_c * (1.0 - fresnel_dielectric(eta, cos_h_i.abs())) * (1.0 - metalness) * weight;
    let d = microfacet.distribution(h);
    let jacobian = eta * eta * cos_h_i.abs() * cos_h_o.abs() / denom_sq;
    f * d * jacobian / (wi.cos_theta() * wo.cos_theta().abs())
}

/// Evaluates the per-channel transmission PDF contribution for one wavelength.
/// Returns 0.0 for invalid geometries.
fn trans_pdf_channel(wi: Vec3, wo: Vec3, microfacet: &Microfacet, eta: f32) -> f32 {
    let h = -(wi + wo * eta).normalize();
    if (eta >= 1.0 && h.cos_theta() <= 0.0) || (eta < 1.0 && h.cos_theta() >= 0.0) {
        return 0.0;
    }
    let cos_h_i = wi.dot(h);
    let cos_h_o = wo.dot(h);
    if cos_h_i * cos_h_o >= 0.0 {
        return 0.0;
    }
    let denom_sq = (cos_h_i + eta * cos_h_o).powi(2);
    if denom_sq < 1e-14 {
        return 0.0;
    }

    microfacet.density(wi, h) * eta * eta * cos_h_o.abs() / denom_sq
}

fn reflect(wi: Vec3, h: Vec3) -> Vec3 {
    (2.0 * wi.dot(h) * h - wi).normalize()
}

/// Selects a per-channel IOR for transmission sampling using `random_z`.
///
/// `random_z` is remapped from [p_refl, 1) into [0, 1) and then used to pick
/// one of the three RGB channels with equal probability (1/3 each), matching
/// Adobe's channel selection for dispersive sampling. When `etas` is uniform
/// (no dispersion) the result is the same regardless of which channel is picked.
fn select_channel_eta(random_z: f32, p_refl: f32, p_trans: f32, etas: Vec3) -> f32 {
    let t = (random_z - p_refl) / p_trans;
    if t < 1.0 / 3.0 {
        etas.x
    } else if t < 2.0 / 3.0 {
        etas.y
    } else {
        etas.z
    }
}
