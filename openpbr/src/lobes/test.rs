use super::coat::Coat;
use super::diffuse::Diffuse;
use super::fuzz::Fuzz;
use super::metal::Metal;
use super::specular_reflection::SpecularReflection;
use super::specular_transmission::SpecularTransmission;
use super::Throughput;
use crate::material::Material;
use crate::math::SphericalCoordinates;
use crate::sampling;
use crate::{lobes::Lobe, lobes::Sample};
use glam::Vec3;
use integrate::romberg::romberg_method;
use rand::{RngExt, SeedableRng};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::{f32::consts::PI, ops::Range};

/// Test that the given lobe has some desired properties. In particular:
/// - `lobe.sample` correctly samples the distribution of `density`.
/// - `lobe.eval` follows Helmholtz reciprocity. That is, for any pair of vectors `view` and `light`,
///   `lobe.eval(wi, wo) == bsdf.eval(wo, wi)`.
/// - `bsdf.eval` is energy conserving, meaning that it integrates to at most 1 over the sphere.
/// - There are no NaNs and infinites.
pub struct LobeTest<'a, L> {
    pub lobe: &'a L,
    pub sample_count: usize,
    /// The amount of bins to use for the frequency table on the theta axis.
    pub theta_bin_count: usize,
    /// The amount of bins to use for the frequency table on the phi axis.
    pub phi_bin_count: usize,
    pub theta_integrate_dim: usize,
    pub phi_integrate_dim: usize,
    /// The minimum expected frequency for each bin. If the expected frequency in a bin is less
    /// than this, it is pooled into an outlier bin. This is because there will be a lot of
    /// variance in the extremes of the distribution unless `sample_count` is very higher.
    /// Therefore, if it is pooled together, the variance will be averaged out.
    pub min_expected_freq: f32,
}

impl<'a, B: Lobe + Sync> LobeTest<'a, B> {
    /// Generate a frequency table for the given view direction by sampling `sample` of the lobe.
    fn freq_table(&self, wi: Vec3, rng: &mut impl rand::Rng) -> Vec<f32> {
        let theta_factor = self.theta_bin_count as f32 / PI;
        let phi_factor = self.phi_bin_count as f32 / (2.0 * PI);

        let mut bins = vec![0.0f32; self.theta_bin_count * self.phi_bin_count];

        for _ in 0..self.sample_count {
            let Some(Sample { wo, .. }) = self.lobe.sample(rng.random(), wi) else {
                continue;
            };

            let theta = wo.cos_theta().acos() * theta_factor;

            let mut phi = f32::atan2(wo.y, wo.x) * phi_factor;
            if phi < 0.0 {
                phi += 2.0 * PI * phi_factor;
            }

            let theta_bin = (theta.floor() as usize).min(self.theta_bin_count - 1);
            let phi_bin = (phi.floor() as usize).min(self.phi_bin_count - 1);

            bins[theta_bin * self.phi_bin_count + phi_bin] += 1.0;
        }

        bins
    }

    fn integrated_freq_table(&self, view: Vec3) -> Vec<f32> {
        let theta_factor = PI / self.theta_bin_count as f32;
        let phi_factor = 2.0 * PI / self.phi_bin_count as f32;
        let mut bins = Vec::with_capacity(self.theta_bin_count * self.phi_bin_count);

        for theta_bin in 0..self.theta_bin_count {
            let theta_range =
                theta_factor * theta_bin as f32..theta_factor * (theta_bin + 1) as f32;

            for phi_bin in 0..self.phi_bin_count {
                let phi_range = phi_factor * phi_bin as f32..phi_factor * (phi_bin + 1) as f32;
                let integral = integrate(
                    theta_range.clone(),
                    phi_range,
                    self.theta_integrate_dim,
                    self.phi_integrate_dim,
                    |theta, phi| {
                        let density = self
                            .lobe
                            .density(view, Vec3::from_spherical_coordinates(theta, phi));
                        density * theta.sin()
                    },
                );
                bins.push(integral * self.sample_count as f32);
            }
        }

        bins
    }

    fn chi_squared_test(&self, freqs: &[f32], expected_freqs: &[f32]) -> f32 {
        let mut cells: Vec<_> = expected_freqs
            .iter()
            .copied()
            .zip(freqs.iter().copied())
            .collect();
        cells.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());

        let (mut pooled_freqs, mut pooled_expected_freqs) = (0.0, 0.0);
        let (mut chi_squared, mut dof) = (0.0, 0);

        let mut update = |expected_freq: f32, freq: f32| {
            chi_squared += (freq - expected_freq).powi(2) / expected_freq;
            dof += 1;
        };

        for (expected_freq, freq) in cells.into_iter() {
            if expected_freq == 0.0 {
                assert!(
                    freq <= self.sample_count as f32 * 1e-3,
                    "expected frequency is zero, but observed frequency is {freq}",
                );
            } else if expected_freq < self.min_expected_freq
                || pooled_expected_freqs > 0.0 && pooled_expected_freqs < self.min_expected_freq
            {
                pooled_freqs += freq;
                pooled_expected_freqs += expected_freq;
            } else {
                update(expected_freq, freq);
            }
        }

        if pooled_expected_freqs > 0.0 || pooled_freqs > 0.0 {
            update(pooled_expected_freqs, pooled_freqs);
        }

        1.0 - ChiSquared::new(dof as f64 - 1.0)
            .expect("failed to create distribution")
            .cdf(chi_squared.into()) as f32
    }

    pub fn run(&self, rng: &mut impl rand::Rng) {
        let significance_level: f32 = 0.01;

        let view = sampling::uniform_hemisphere(rng.random());

        let expected_freqs = self.integrated_freq_table(view);

        // `expected_freqs.iter().sum::<f32>()` should be close to `self.sample_count`, but the
        // integration does not seem to accurate enough to reliably test it.

        let freqs = self.freq_table(view, rng);

        let pval = self.chi_squared_test(&freqs, &expected_freqs);
        assert!(
            pval > significance_level,
            "p-value is too small: {pval} <= {significance_level}, `bsdf.distribution` does not follow the distribution of `bsdf.density`"
        );

        // Test Helmholtz reciprocity.

        for _ in 0..self.sample_count {
            let view = sampling::uniform_hemisphere(rng.random());
            let light = sampling::uniform_hemisphere(rng.random());
            let a = self.lobe.eval(view, light);
            let b = self.lobe.eval(light, view);
            let error = throughput_error(&a, &b);
            for error in error.to_array().into_iter() {
                assert!(error <= 1e-3, "bsdf is not reciprocal: {a:?} != {b:?}");
            }
        }

        // Test energy conservation.
        //
        // Uses |cos_theta| so the integral is correct for both BRDFs (upper hemisphere, where
        // cos_theta is naturally positive) and BTDFs (lower hemisphere, where raw cos_theta would
        // be negative and cancel the positive eval, making the test trivially pass).
        for channel in 0..=2 {
            let energy = integrate(
                0.0..PI,
                0.0..2.0 * PI,
                self.theta_integrate_dim,
                self.phi_integrate_dim,
                |theta, phi| {
                    let light = Vec3::from_spherical_coordinates(theta, phi);
                    self.lobe.eval(view, light).channel_total(channel)
                        * light.cos_theta().abs()
                        * light.sin_theta()
                },
            );
            assert!(
                energy < 1.0 + 1e-3,
                "bsdf does not conserve energy: {energy} (should be less than 1)"
            );
        }
    }
}

/// Relative error between two throughputs. Uses `total()` so it works for lobes returning
/// specular throughput. Adds a small epsilon to avoid 0/0 when both sides are exactly zero
/// (which would produce NaN and a spurious assertion failure).
fn throughput_error(a: &Throughput, b: &Throughput) -> Vec3 {
    let a = a.total();
    let b = b.total();
    let denom = a.abs().max(b.abs()) + Vec3::splat(1e-7);
    (a - b).abs() / denom
}

fn integrate<F: Fn(f32, f32) -> f32 + Sync + Send + Copy>(
    xs: Range<f32>,
    ys: Range<f32>,
    x_dim: usize,
    y_dim: usize,
    f: F,
) -> f32 {
    let outer = move |y| romberg_method(move |x| f(x, y) as f64, xs.start, xs.end, x_dim);
    romberg_method(outer, ys.start, ys.end, y_dim) as f32
}

/// Standard test parameters suitable for broadly-distributed lobes.
fn standard_test<'a, L: Lobe + Sync>(lobe: &'a L) -> LobeTest<'a, L> {
    LobeTest {
        lobe,
        sample_count: 100_000,
        theta_bin_count: 60,
        phi_bin_count: 120,
        theta_integrate_dim: 7,
        phi_integrate_dim: 7,
        min_expected_freq: 2.0,
    }
}

// ─── Diffuse ──────────────────────────────────────────────────────────────────

#[test]
fn diffuse_smooth() {
    let m = Material {
        base_diffuse_roughness: 0.0,
        ..Material::default()
    };
    standard_test(&Diffuse::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(1));
}

#[test]
fn diffuse_rough() {
    let m = Material {
        base_diffuse_roughness: 0.8,
        ..Material::default()
    };
    standard_test(&Diffuse::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(2));
}

#[test]
fn diffuse_colored() {
    let m = Material {
        base_color: Vec3::new(0.8, 0.3, 0.1),
        base_diffuse_roughness: 0.4,
        ..Material::default()
    };
    standard_test(&Diffuse::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(3));
}

// ─── Fuzz ─────────────────────────────────────────────────────────────────────

#[test]
fn fuzz_low_roughness() {
    let m = Material {
        fuzz_color: Vec3::ONE,
        fuzz_roughness: 0.2,
        ..Material::default()
    };
    standard_test(&Fuzz::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(10));
}

#[test]
fn fuzz_mid_roughness() {
    let m = Material {
        fuzz_color: Vec3::ONE,
        fuzz_roughness: 0.5,
        ..Material::default()
    };
    standard_test(&Fuzz::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(11));
}

#[test]
fn fuzz_high_roughness() {
    let m = Material {
        fuzz_color: Vec3::ONE,
        fuzz_roughness: 0.9,
        ..Material::default()
    };
    standard_test(&Fuzz::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(12));
}

// ─── Metal ────────────────────────────────────────────────────────────────────

#[test]
fn metal_isotropic() {
    let m = Material {
        specular_roughness: 0.3,
        ..Material::default()
    };
    standard_test(&Metal::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(20));
}

#[test]
fn metal_rough() {
    let m = Material {
        specular_roughness: 0.7,
        ..Material::default()
    };
    standard_test(&Metal::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(21));
}

#[test]
fn metal_anisotropic() {
    let m = Material {
        specular_roughness: 0.3,
        specular_roughness_anisotropy: 0.5,
        ..Material::default()
    };
    standard_test(&Metal::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(22));
}

#[test]
fn metal_anisotropic_rotated() {
    let m = Material {
        specular_roughness: 0.3,
        specular_roughness_anisotropy: 0.5,
        specular_rotation: 0.25,
        ..Material::default()
    };
    standard_test(&Metal::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(23));
}

// ─── Coat ─────────────────────────────────────────────────────────────────────

#[test]
fn coat_smooth() {
    let m = Material {
        coat_ior: 1.6,
        coat_roughness: 0.2,
        ..Material::default()
    };
    standard_test(&Coat::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(30));
}

#[test]
fn coat_rough() {
    let m = Material {
        coat_ior: 1.6,
        coat_roughness: 0.7,
        ..Material::default()
    };
    standard_test(&Coat::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(31));
}

#[test]
fn coat_anisotropic() {
    let m = Material {
        coat_ior: 1.6,
        coat_roughness: 0.3,
        coat_roughness_anisotropy: 0.5,
        ..Material::default()
    };
    standard_test(&Coat::from(&m)).run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(32));
}

// ─── Specular Reflection ──────────────────────────────────────────────────────

#[test]
fn specular_reflection_smooth() {
    let m = Material {
        specular_roughness: 0.3,
        ..Material::default()
    };
    standard_test(&SpecularReflection::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(40));
}

#[test]
fn specular_reflection_rough() {
    let m = Material {
        specular_roughness: 0.7,
        ..Material::default()
    };
    standard_test(&SpecularReflection::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(41));
}

#[test]
fn specular_reflection_with_coat() {
    // The coat layer changes the effective IOR seen by the specular surface (OpenPBR Eq. 60),
    // exercising a materially different code path from the uncoated cases.
    let m = Material {
        specular_roughness: 0.3,
        coat_weight: 0.8,
        coat_ior: 1.6,
        ..Material::default()
    };
    standard_test(&SpecularReflection::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(42));
}

#[test]
fn specular_reflection_anisotropic() {
    let m = Material {
        specular_roughness: 0.3,
        specular_roughness_anisotropy: 0.5,
        ..Material::default()
    };
    standard_test(&SpecularReflection::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(43));
}

#[test]
fn specular_reflection_anisotropic_rotated() {
    let m = Material {
        specular_roughness: 0.3,
        specular_roughness_anisotropy: 0.5,
        specular_rotation: 0.25,
        ..Material::default()
    };
    standard_test(&SpecularReflection::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(44));
}

// ─── Specular Transmission ────────────────────────────────────────────────────
//
// Note: the reciprocity sub-test samples both directions from the upper hemisphere, so
// eval() returns ZERO for same-hemisphere pairs and reciprocity passes trivially. The
// chi-squared sub-test (sampling vs. density) is the meaningful check for this lobe.

#[test]
fn specular_transmission_standard() {
    let m = Material {
        specular_roughness: 0.3,
        specular_ior: 1.5,
        ..Material::default()
    };
    standard_test(&SpecularTransmission::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(50));
}

#[test]
fn specular_transmission_rough() {
    let m = Material {
        specular_roughness: 0.7,
        specular_ior: 1.5,
        ..Material::default()
    };
    standard_test(&SpecularTransmission::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(51));
}

#[test]
fn specular_transmission_high_ior() {
    // Higher IOR → stronger total internal reflection → fewer transmitted samples. Tests
    // that the sampler and density agree even when many microfacet orientations are rejected.
    let m = Material {
        specular_roughness: 0.3,
        specular_ior: 2.0,
        ..Material::default()
    };
    standard_test(&SpecularTransmission::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(52));
}

#[test]
fn specular_transmission_colored() {
    let m = Material {
        specular_roughness: 0.3,
        specular_ior: 1.5,
        transmission_color: Vec3::new(0.5, 0.8, 1.0),
        ..Material::default()
    };
    standard_test(&SpecularTransmission::from(&m))
        .run(&mut rand::rngs::Xoshiro256PlusPlus::seed_from_u64(53));
}
