use crate::{
    consts::{DENSITY_EPSILON, RADIANCE_EPSILON},
    math::SphericalCoordinates,
};
use glam::Vec3;

pub mod coat;
pub mod diffuse;
pub mod fuzz;
pub mod metal;
pub mod specular_reflection;
pub mod specular_transmission;

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

    pub fn total(&self) -> Vec3 {
        self.diffuse + self.specular
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
    fn incidence_is_valid(&self, wi: Vec3) -> bool;

    fn estimate_directional_albedo(&self, wi: Vec3, samples: &[Vec3]) -> Vec3 {
        if !self.incidence_is_valid(wi) {
            return Vec3::ZERO;
        }

        let mut albedo = Vec3::ZERO;

        for random in samples {
            let sample = self.sample(*random, wi);

            if sample.throughput.total().length() > RADIANCE_EPSILON {
                albedo += sample.throughput.total() * sample.wo.cos_theta().abs()
                    / sample.density.max(DENSITY_EPSILON);
            }
        }

        return albedo / samples.len() as f32;
    }
}
