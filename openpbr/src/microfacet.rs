use crate::math::{SphericalCoordinates, SurfaceBasis};
use crate::sampling;
use glam::{FloatExt, Vec2, Vec3};
use std::f32::consts::PI;

/// The Throwbridge-Reitz microfacet distribution.
pub struct Microfacet {
    pub alpha: Vec2,
}

impl Microfacet {
    pub fn new(roughness: f32, anisotropy: f32) -> Self {
        Self {
            alpha: roughness_to_alpha(roughness, anisotropy),
        }
    }

    pub fn distribution(&self, microfacet_normal: Vec3) -> f32 {
        let tan_squared = microfacet_normal.tan_theta_squared();
        if tan_squared.is_infinite() {
            return 0.0;
        }
        let e = tan_squared
            * ((microfacet_normal.cos_phi() / self.alpha.x).powi(2)
                + (microfacet_normal.sin_phi() / self.alpha.y).powi(2));
        let denum = PI
            * self.alpha.element_product()
            * microfacet_normal.cos_theta_squared().powi(2)
            * (1.0 + e).powi(2);
        1.0 / denum
    }

    pub fn lambda(&self, wi: Vec3) -> f32 {
        let tan_theta_squared = wi.tan_theta_squared();
        if tan_theta_squared.is_infinite() {
            return 0.0;
        }
        let alpha_squared =
            (wi.cos_phi() * self.alpha.x).powi(2) + (wi.sin_phi() * self.alpha.y).powi(2);
        ((1.0 + alpha_squared * tan_theta_squared).sqrt() - 1.0) / 2.0
    }

    pub fn visibility(&self, wi: Vec3, wo: Vec3) -> f32 {
        1.0 / (1.0 + self.lambda(wo) + self.lambda(wi))
    }

    pub fn masking(&self, wi: Vec3) -> f32 {
        1.0 / (1.0 + self.lambda(wi))
    }

    #[allow(dead_code)]
    pub fn density(&self, view: Vec3, microfacet_normal: Vec3) -> f32 {
        self.masking(view)
            * view.dot(microfacet_normal).abs()
            * self.distribution(microfacet_normal)
            / view.cos_theta().abs()
    }

    pub fn sample(&self, wi: Vec3, random: Vec2) -> Vec3 {
        let mut view_hemisphere =
            Vec3::new(wi.x * self.alpha.x, wi.y * self.alpha.y, wi.z).normalize();
        if view_hemisphere.cos_theta() < 0.0 {
            view_hemisphere = -view_hemisphere;
        }
        let tangent = if view_hemisphere.cos_theta() < 0.99999 {
            Vec3::Z.cross(view_hemisphere).normalize()
        } else {
            Vec3::X
        };
        let basis = SurfaceBasis {
            normal: view_hemisphere,
            bitangent: view_hemisphere.cross(tangent),
            tangent,
        };
        let mut p = sampling::uniform_disk_polar(random);
        p.y = (1.0 - p.x.powi(2))
            .sqrt()
            .lerp(p.y, (1.0 + view_hemisphere.cos_theta()) / 2.0);
        let z = (1.0 - p.length_squared()).max(0.0).sqrt();
        let normal = basis.transform(p.extend(z)) * self.alpha.extend(1.0);
        normal.with_z(normal.z.max(1e-6)).normalize()
    }
}

fn roughness_to_alpha(roughness: f32, anisotropy: f32) -> Vec2 {
    let alpha_x = roughness.powi(2) * (2.0 / (1.0 + (1.0 - anisotropy).powi(2))).sqrt();
    let alpha_y = (1.0 - anisotropy) * alpha_x;
    Vec2::new(alpha_x.max(1e-4), alpha_y.max(1e-4))
}
