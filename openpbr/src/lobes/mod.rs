use crate::{consts::DENSITY_EPSILON, math::SphericalCoordinates};
use glam::Vec3;

pub mod bsdf;
pub mod coat;
pub mod diffuse;
pub mod fuzz;
pub mod metal;
pub mod specular_reflection;
pub mod specular_transmission;

#[cfg(test)]
mod test;

#[derive(Debug, Default, Clone, Copy)]
pub struct Throughput {
    pub diffuse: Vec3,
    pub specular: Vec3,
}

impl Throughput {
    pub const ZERO: Self = Self {
        diffuse: Vec3::ZERO,
        specular: Vec3::ZERO,
    };

    pub const ONE: Self = Self {
        diffuse: Vec3::ONE,
        specular: Vec3::ONE,
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

    pub fn channel_total(&self, channel: usize) -> f32 {
        self.diffuse[channel] + self.specular[channel]
    }
}

#[derive(Clone, Copy)]
pub struct Sample {
    pub wi: Vec3,
    pub throughput: Throughput,
    pub density: f32,
    pub lobe_type: LobeType,
}

pub trait Lobe {
    fn eval(&self, wo: Vec3, wi: Vec3) -> Throughput;
    fn sample(&self, random: Vec3, wo: Vec3) -> Option<Sample>;
    fn density(&self, wo: Vec3, wi: Vec3) -> f32;
    fn incidence_is_valid(&self, wo: Vec3) -> bool;

    fn estimate_directional_albedo(&self, wo: Vec3, samples: &[Vec3]) -> Vec3 {
        if !self.incidence_is_valid(wo) {
            return Vec3::ZERO;
        }

        let mut albedo = Vec3::ZERO;

        for random in samples {
            if let Some(sample) = self.sample(*random, wo) {
                albedo += sample.throughput.total() * sample.wi.cos_theta().abs()
                    / sample.density.max(DENSITY_EPSILON);
            };
        }

        return albedo / samples.len() as f32;
    }
}

#[derive(enum_map::Enum, Clone, Copy, PartialEq, Eq)]
pub enum LobeType {
    Fuzz = 0,
    Coat = 1,
    Metal = 2,
    SpecularReflection = 3,
    SpecularTransmission = 4,
    Diffuse = 5,
}

impl LobeType {
    pub fn is_specular(&self) -> bool {
        match self {
            LobeType::Fuzz | LobeType::Diffuse => false,
            _ => true,
        }
    }
}
