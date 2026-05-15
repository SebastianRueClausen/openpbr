use glam::Vec3;

pub mod diffuse;
pub mod specular;

#[derive(Default, Clone, Copy)]
pub struct Throughput {
    pub diffuse: Vec3,
    pub specular: Vec3,
}

impl Throughput {
    pub const ZERO: Self = Self {
        diffuse: Vec3::ZERO,
        specular: Vec3::ZERO,
    };

    pub fn from_diffuse(diffuse: Vec3) -> Self {
        Self {
            diffuse,
            specular: Vec3::ZERO,
        }
    }

    pub fn from_specular(specular: Vec3) -> Self {
        Self {
            diffuse: Vec3::ZERO,
            specular,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct Sample {
    pub wo: Vec3,
    pub throughput: Throughput,
    pub density: f32,
}

impl Sample {
    pub const ZERO: Self = Self {
        wo: Vec3::ZERO,
        throughput: Throughput::ZERO,
        density: 0.0,
    };
}

pub trait Lobe {
    fn eval(&self, wi: Vec3, wo: Vec3) -> Throughput;
    fn sample(&self, random: Vec3, wi: Vec3) -> Sample;
    fn density(&self, wi: Vec3, wo: Vec3) -> f32;
}
