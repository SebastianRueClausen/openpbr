use glam::Vec3;

/// Full unpolarized Fresnel reflectance for a dielectric interface.
/// `eta` is the ratio n_t/n_i (transmitted over incident index of refraction).
pub fn fresnel_dielectric(eta: f32, cos_theta_i: f32) -> f32 {
    let cos_theta_i = cos_theta_i.abs();
    let sin_theta_i_sq = (1.0 - cos_theta_i * cos_theta_i).max(0.0);
    let sin_theta_t_sq = sin_theta_i_sq / (eta * eta);
    if sin_theta_t_sq >= 1.0 {
        return 1.0;
    }
    let cos_theta_t = (1.0 - sin_theta_t_sq).sqrt();
    let r_s = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    let r_p = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    0.5 * (r_s * r_s + r_p * r_p)
}

/// Normal-incidence reflectance (F0) from an IOR ratio (n_t/n_i).
pub fn f0_from_ior(ior: f32) -> f32 {
    ((1.0 - ior) / (1.0 + ior)).powi(2)
}

/// Schlick Fresnel approximation.
pub fn schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let t = (1.0 - cos_theta).max(0.0);
    f0 + (Vec3::ONE - f0) * t.powi(5)
}
