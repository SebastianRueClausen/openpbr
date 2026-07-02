use glam::Vec3;
use std::ops::{Index, IndexMut};

use crate::Sampler;

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
    fn sample<S: Sampler>(&self, random: &mut S, wo: Vec3) -> Option<Sample>;
    fn density(&self, wo: Vec3, wi: Vec3) -> f32;
    fn estimate_directional_albedo(&self, wo: Vec3) -> Vec3;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LobeType {
    Fuzz = 0,
    Coat = 1,
    Metal = 2,
    SpecularReflection = 3,
    SpecularTransmission = 4,
    Diffuse = 5,
}

impl LobeType {
    /// All lobe types, ordered by their discriminant. Used to index into [`PerLobe`].
    pub const ALL: [LobeType; 6] = [
        LobeType::Fuzz,
        LobeType::Coat,
        LobeType::Metal,
        LobeType::SpecularReflection,
        LobeType::SpecularTransmission,
        LobeType::Diffuse,
    ];

    pub fn is_specular(&self) -> bool {
        match self {
            LobeType::Fuzz | LobeType::Diffuse => false,
            _ => true,
        }
    }
}

/// A fixed-size map from each [`LobeType`] to a value of type `T`, backed by an array
/// indexed by the lobe's discriminant.
#[derive(Clone, Copy)]
pub struct PerLobe<T>([T; LobeType::ALL.len()]);

impl<T> PerLobe<T> {
    /// Build a map from values listed in [`LobeType::ALL`] order.
    pub fn new(values: [T; LobeType::ALL.len()]) -> Self {
        Self(values)
    }

    /// Build a map by calling `f` for each lobe type.
    pub fn from_fn(mut f: impl FnMut(LobeType) -> T) -> Self {
        Self(LobeType::ALL.map(|lobe| f(lobe)))
    }

    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }
}

impl<T> Index<LobeType> for PerLobe<T> {
    type Output = T;

    fn index(&self, lobe: LobeType) -> &T {
        &self.0[lobe as usize]
    }
}

impl<T> IndexMut<LobeType> for PerLobe<T> {
    fn index_mut(&mut self, lobe: LobeType) -> &mut T {
        &mut self.0[lobe as usize]
    }
}
