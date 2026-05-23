use glam::Vec3;

pub struct Material {
    pub base_weight: f32,
    pub base_color: Vec3,
    pub base_diffuse_roughness: f32,
    pub base_metalness: f32,
    pub specular_weight: f32,
    pub specular_color: Vec3,
    pub specular_ior: f32,
    pub specular_roughness: f32,
    pub specular_roughness_anisotropy: f32,
    pub specular_rotation: f32,
    pub transmission_weight: f32,
    pub transmission_color: Vec3,
    pub transmission_depth: f32,
    pub coat_weight: f32,
    pub coat_color: Vec3,
    pub coat_ior: f32,
    pub coat_roughness: f32,
    pub coat_roughness_anisotropy: f32,
    pub coat_rotation: f32,
    pub coat_darkening: f32,
    pub fuzz_weight: f32,
    pub fuzz_color: Vec3,
    pub fuzz_roughness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_weight: 1.0,
            base_color: Vec3::splat(0.8),
            base_metalness: 0.0,
            base_diffuse_roughness: 0.0,
            specular_weight: 1.0,
            specular_color: Vec3::ONE,
            specular_roughness: 0.3,
            specular_roughness_anisotropy: 0.0,
            specular_ior: 1.5,
            transmission_weight: 0.0,
            transmission_color: Vec3::ONE,
            transmission_depth: 0.0,
            coat_weight: 0.0,
            coat_color: Vec3::ONE,
            coat_roughness: 0.0,
            coat_roughness_anisotropy: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,
            fuzz_weight: 0.0,
            fuzz_color: Vec3::ONE,
            fuzz_roughness: 0.5,
            specular_rotation: 0.0,
            coat_rotation: 0.0,
        }
    }
}
