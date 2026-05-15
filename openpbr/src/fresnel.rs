use glam::Vec3;

/// Full unpolarized Fresnel reflectance for a dielectric interface.
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

/// Normal-incidence reflectance (F0) from IOR.
pub fn f0_from_ior(ior: f32) -> f32 {
    ((1.0 - ior) / (1.0 + ior)).powi(2)
}

pub fn schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let t = (1.0 - cos_theta).max(0.0);
    f0 + (Vec3::ONE - f0) * t.powi(5)
}

pub fn refract(wi: Vec3, h: Vec3, cos_theta_i: f32, eta: f32) -> Option<Vec3> {
    let eta_inv = 1.0 / eta;
    let sin2_theta_t = eta_inv * eta_inv * (1.0 - cos_theta_i * cos_theta_i).max(0.0);
    if sin2_theta_t >= 1.0 {
        return None;
    }
    let cos_theta_t = (1.0 - sin2_theta_t).sqrt();
    Some((-wi * eta_inv + h * (eta_inv * cos_theta_i - cos_theta_t)).normalize())
}

/// Metallic Fresnel using the Lazanyi–Schlick F82-tint parametrization.
pub fn metal_fresnel_f82_tint(f0: Vec3, f82_tint: Vec3, cos_theta: f32) -> Vec3 {
    const COS_82: f32 = 1.0 / 7.0;
    let t = (1.0 - cos_theta).max(0.0);
    let t_82 = 1.0 - COS_82;
    let base_schlick = schlick(f0, cos_theta);
    let schlick_at_82 = schlick(f0, COS_82);
    let correction = (schlick_at_82 - f82_tint) / (COS_82 * t_82.powi(6));
    (base_schlick - correction * cos_theta * t.powi(6)).max(Vec3::ZERO)
}
