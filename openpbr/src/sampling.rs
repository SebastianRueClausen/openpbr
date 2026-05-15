use std::f32::consts::PI;

use glam::{Vec2, Vec3};

pub fn uniform_hemisphere(random: Vec2) -> Vec3 {
    let cos_theta = random.x;
    let radius = (1.0 - cos_theta.powi(2)).max(0.0).sqrt();
    let phi = random.y * 2.0 * PI;
    Vec3::new(radius * phi.cos(), radius * phi.sin(), cos_theta)
}

pub const UNIFORM_HEMISPHERE_DENSITY: f32 = 1.0 / (2.0 * PI);

pub fn uniform_sphere(random: Vec2) -> Vec3 {
    let cos_theta = 1.0 - 2.0 * random.x;
    let radius = (1.0 - cos_theta.powi(2)).max(0.0).sqrt();
    let phi = random.y * 2.0 * PI;
    Vec3::new(radius * phi.cos(), radius * phi.sin(), cos_theta)
}

pub const UNIFORM_SPHERE_DENSITY: f32 = 1.0 / (4.0 * PI);

pub fn uniform_disk_polar(random: Vec2) -> Vec2 {
    let radius = random.x.sqrt();
    let theta = 2.0 * PI * random.y;
    radius * Vec2::new(theta.cos(), theta.sin())
}

pub fn cosine_hemisphere_sample(random: Vec2) -> Vec3 {
    let r = random.x.sqrt();
    let theta = 2.0 * PI * random.y;
    let x = r * theta.cos();
    let y = r * theta.sin();
    let z = f32::max(0.0, 1.0 - x * x - y * y).sqrt();
    return Vec3::new(x, y, z);
}

pub fn cosine_hemisphere_density(normal_dot_scatter: f32) -> f32 {
    normal_dot_scatter.max(1e-6) / PI
}
