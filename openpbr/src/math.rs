use glam::Vec3;

#[derive(Debug)]
pub struct SurfaceBasis {
    pub normal: Vec3,
    pub tangent: Vec3,
    pub bitangent: Vec3,
}

impl SurfaceBasis {
    pub fn new(normal: Vec3, tangent: Vec3, bitangent: Vec3) -> Self {
        Self {
            normal,
            tangent,
            bitangent,
        }
    }

    pub fn with_y_up(normal: Vec3) -> Self {
        let tangent = Vec3::Y.cross(normal).normalize();
        let bitangent = tangent.cross(normal).normalize();
        Self::new(normal, tangent, bitangent)
    }

    pub fn any_with_normal(normal: Vec3) -> Self {
        let (tangent, bitangent) = normal.any_orthonormal_pair();
        Self::new(normal, tangent, bitangent)
    }

    pub fn transform(&self, local: Vec3) -> Vec3 {
        (self.tangent * local.x + self.bitangent * local.y + self.normal * local.z).normalize()
    }
}

pub trait SphericalCoordinates {
    fn from_spherical_coordinates(theta: f32, phi: f32) -> Self;
    fn to_spherical_coordinates(&self) -> (f32, f32);
    fn cos_theta(&self) -> f32;
    fn cos_theta_squared(&self) -> f32 {
        self.cos_theta().powi(2)
    }
    fn sin_theta_squared(&self) -> f32 {
        (1.0 - self.cos_theta_squared()).max(0.0)
    }
    fn sin_theta(&self) -> f32 {
        self.sin_theta_squared().sqrt()
    }
    fn cos_phi(&self) -> f32;
    fn sin_phi(&self) -> f32;
    fn tan_theta_squared(&self) -> f32 {
        self.sin_theta_squared() / self.cos_theta_squared()
    }
    fn in_same_hemisphere(&self, other: &Self) -> bool {
        self.cos_theta() * other.cos_theta() > 0.0
    }
}

impl SphericalCoordinates for Vec3 {
    fn from_spherical_coordinates(theta: f32, phi: f32) -> Self {
        let (sin_theta, cos_theta) = theta.sin_cos();
        let (sin_phi, cos_phi) = phi.sin_cos();
        Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
    }

    fn to_spherical_coordinates(&self) -> (f32, f32) {
        debug_assert!(self.is_normalized(), "vector must be normalized");
        (self.z.acos(), (self.y / self.x).atan())
    }

    fn cos_theta(&self) -> f32 {
        debug_assert!(self.is_normalized(), "vector must be normalized");
        self.z
    }

    fn cos_phi(&self) -> f32 {
        let sin_theta = self.sin_theta();
        if sin_theta.abs() < 1e-4 {
            1.0
        } else {
            (self.x / sin_theta).clamp(-1.0, 1.0)
        }
    }

    fn sin_phi(&self) -> f32 {
        let sin_theta = self.sin_theta();
        if sin_theta.abs() < 1e-4 {
            0.0
        } else {
            (self.y / sin_theta).clamp(-1.0, 1.0)
        }
    }
}
