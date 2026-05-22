use glam::Vec3;

/// Full unpolarized Fresnel reflectance for a dielectric interface.
/// `eta` is the ratio n_t/n_i (transmitted over incident index of refraction).
pub fn fresnel_dielectric(ior: f32, cos_theta_i: f32) -> f32 {
    let sin_theta_i_sq = 1.0 - cos_theta_i.powi(2);
    let sin_theta_t_sq = sin_theta_i_sq / ior.powi(2);
    if sin_theta_t_sq >= 1.0 {
        return 1.0;
    }
    let cos_theta_t = (1.0 - sin_theta_t_sq).sqrt();
    let par = (ior * cos_theta_i - cos_theta_t) / (ior * cos_theta_i + cos_theta_t);
    let per = (cos_theta_i - ior * cos_theta_t) / (cos_theta_i + ior * cos_theta_t);
    0.5 * (per.powi(2) + par.powi(2))
}

/// Normal-incidence reflectance (F0) from an IOR ratio (n_t/n_i).
pub fn f0_from_ior(ior: f32) -> f32 {
    ((1.0 - ior) / (1.0 + ior)).powi(2)
}

fn dieletric_fresnel_factor(ior: f32) -> f32 {
    // OpenPBR Eq. (102)
    ((10893.0 * ior - 1438.2) / (-774.4 * ior.powi(2) + 10212.0 * ior + 1.0)).ln()
}

/// Cosine-weighted hemispherical average of the dielectric Fresnel reflectance.
pub fn average_dielectric_fresnel(ior: f32) -> f32 {
    if ior > 1.0 {
        dieletric_fresnel_factor(ior)
    } else if ior < 1.0 {
        1.0 - ior.powi(2) * (1.0 - dieletric_fresnel_factor(1.0 / ior)) // OpenPBR Eq. (103)
    } else {
        0.0
    }
}

/// Schlick Fresnel approximation.
pub fn schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let t = (1.0 - cos_theta).max(0.0);
    f0 + (Vec3::ONE - f0) * t.powi(5)
}
