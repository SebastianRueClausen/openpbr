use crate::{
    consts::DENOM_TOLERANCE,
    fresnel::{average_dielectric_fresnel, f0_from_ior, fresnel_dielectric},
    material::Material,
    math::SphericalCoordinates,
    Sampler,
};
use enum_map::EnumMap;
use glam::Vec3;

use super::{
    coat::Coat, diffuse::Diffuse, fuzz::Fuzz, metal::Metal,
    specular_reflection::SpecularReflection, specular_transmission::SpecularTransmission, Lobe,
    LobeType, Sample, Throughput,
};

pub struct Bsdf {
    fuzz: Fuzz,
    coat: Coat,
    metal: Metal,
    spec_refl: SpecularReflection,
    spec_trans: SpecularTransmission,
    diffuse: Diffuse,
    weights: EnumMap<LobeType, Vec3>,
    probs: EnumMap<LobeType, f32>,
}

impl Bsdf {
    /// Build lobes and compute sampling weights from `material` for incident direction `wi`.
    /// One sample per active lobe is drawn from `rng` for directional-albedo estimation.
    pub fn new<S: Sampler>(material: &Material, wo: Vec3, rng: &mut S) -> Self {
        let fuzz = Fuzz::from(material);
        let coat = Coat::from(material);
        let metal = Metal::from(material);
        let spec_refl = SpecularReflection::from(material);
        let spec_trans = SpecularTransmission::from(material);
        let diffuse = Diffuse::from(material);

        let (weights, probs) = compute_weights(
            material,
            wo,
            rng,
            &fuzz,
            &coat,
            &metal,
            &spec_refl,
            &spec_trans,
            &diffuse,
        );

        Self {
            fuzz,
            coat,
            metal,
            spec_refl,
            spec_trans,
            diffuse,
            weights,
            probs,
        }
    }

    pub fn eval(&self, wo: Vec3, wi: Vec3) -> (Throughput, f32) {
        let (densities, throughput) = self.eval_lobes(wo, wi, None);
        let density = total_density(&self.probs, &densities);
        (throughput, density)
    }

    /// Sample a lobe proportional to its weight, then evaluate all other lobes.
    pub fn sample<S: Sampler>(&self, wo: Vec3, rng: &mut S) -> Option<Sample> {
        let random_lobe = rng.next_f32();
        let random_sample = rng.next_vec3();

        let mut cumulative = 0.0f32;

        for lobe in ALL_LOBES {
            cumulative += self.probs[*lobe];
            if random_lobe < cumulative {
                let Some(Sample {
                    wi,
                    density,
                    throughput: lobe_throughput,
                    ..
                }) = self.sample_lobe(*lobe, wo, random_sample)
                else {
                    continue;
                };

                let (mut densities, throughput) = self.eval_lobes(wo, wi, Some(*lobe));
                densities[*lobe] = density;

                return Some(Sample {
                    throughput: Throughput {
                        diffuse: throughput.diffuse + self.weights[*lobe] * lobe_throughput.diffuse,
                        specular: throughput.specular
                            + self.weights[*lobe] * lobe_throughput.specular,
                    },
                    density: total_density(&self.probs, &densities),
                    lobe_type: *lobe,
                    wi,
                });
            }
        }

        None
    }

    fn eval_lobes(
        &self,
        wo: Vec3,
        wi: Vec3,
        skip: Option<LobeType>,
    ) -> (EnumMap<LobeType, f32>, Throughput) {
        let mut throughput = Throughput::ONE;

        macro_rules! eval {
            ($i:expr, $lobe:expr) => {
                if skip != Some($i) && self.probs[$i] > 0.0 {
                    let t = $lobe.eval(wo, wi);
                    throughput.diffuse += self.weights[$i] * t.diffuse;
                    throughput.specular += self.weights[$i] * t.specular;
                    $lobe.density(wo, wi)
                } else {
                    0.0
                }
            };
        }

        let densities = enum_map::enum_map! {
            LobeType::Fuzz => eval!(LobeType::Fuzz, self.fuzz),
            LobeType::Coat => eval!(LobeType::Coat, self.coat),
            LobeType::Metal => eval!(LobeType::Metal, self.metal),
            LobeType::SpecularReflection => eval!(LobeType::SpecularReflection, self.spec_refl),
            LobeType::SpecularTransmission => eval!(LobeType::SpecularTransmission, self.spec_trans),
            LobeType::Diffuse => eval!(LobeType::Diffuse, self.diffuse),
        };

        (densities, throughput)
    }

    fn sample_lobe(&self, lobe: LobeType, wo: Vec3, random: Vec3) -> Option<Sample> {
        match lobe {
            LobeType::Fuzz => self.fuzz.sample(random, wo),
            LobeType::Coat => self.coat.sample(random, wo),
            LobeType::Metal => self.metal.sample(random, wo),
            LobeType::SpecularReflection => self.spec_refl.sample(random, wo),
            LobeType::SpecularTransmission => self.spec_trans.sample(random, wo),
            LobeType::Diffuse => self.diffuse.sample(random, wo),
        }
    }
}

fn total_density(probs: &EnumMap<LobeType, f32>, densities: &EnumMap<LobeType, f32>) -> f32 {
    probs
        .values()
        .zip(densities.values())
        .map(|(p, d)| p * d)
        .sum()
}

fn compute_weights<S: Sampler>(
    material: &Material,
    wo: Vec3,
    rng: &mut S,
    fuzz: &Fuzz,
    coat: &Coat,
    metal: &Metal,
    spec_refl: &SpecularReflection,
    spec_trans: &SpecularTransmission,
    diffuse: &Diffuse,
) -> (EnumMap<LobeType, Vec3>, EnumMap<LobeType, f32>) {
    let fully_metallic = material.base_metalness == 1.0;
    let fully_transmissive = material.transmission_weight == 1.0;

    let albedos = enum_map::enum_map! {
        LobeType::Fuzz => if material.fuzz_weight > 0.0 {
            fuzz.estimate_directional_albedo(wo, &[rng.next_vec3()])
        } else {
            Vec3::ZERO
        },
        LobeType::Coat => if material.coat_weight > 0.0 {
            coat.estimate_directional_albedo(wo, &[rng.next_vec3()])
        } else {
            Vec3::ZERO
        },
        LobeType::Metal => if material.base_metalness > 0.0 {
            metal.estimate_directional_albedo(wo, &[rng.next_vec3()])
        } else {
            Vec3::ZERO
        },
        LobeType::SpecularReflection => if !fully_metallic {
            spec_refl.estimate_directional_albedo(wo, &[rng.next_vec3()])
        } else {
            Vec3::ZERO
        },
        LobeType::SpecularTransmission => if !fully_metallic && material.transmission_weight > 0.0 {
            spec_trans.estimate_directional_albedo(wo, &[rng.next_vec3()])
        } else {
            Vec3::ZERO
        },
        LobeType::Diffuse => if !fully_metallic && !fully_transmissive {
            diffuse.estimate_directional_albedo(wo, &[rng.next_vec3()])
        } else {
            Vec3::ZERO
        },
    };

    let mut weights: EnumMap<LobeType, Vec3> = EnumMap::from_fn(|_| Vec3::ZERO);

    // OpenPBR Eq. (81): fuzz attenuates everything below it.
    weights[LobeType::Fuzz] = Vec3::splat(material.fuzz_weight);
    let coated_base_weight =
        Vec3::ONE.lerp(Vec3::ONE - albedos[LobeType::Fuzz], material.fuzz_weight);
    weights[LobeType::Coat] = coated_base_weight * material.coat_weight;

    // OpenPBR Eq. (65–71): coat darkening of the base.
    let mut base_darkening = Vec3::ONE;
    if material.coat_weight > 0.0 && material.coat_darkening > 0.0 {
        // Adjusted specular IOR through the coat, OpenPBR Eq. (60).
        let adjusted_ior =
            material.specular_ior / (1.0 + material.coat_weight * (material.coat_ior - 1.0));

        // OpenPBR Eq. (70), (69).
        let fresnel_weight = (material.specular_weight * f0_from_ior(adjusted_ior)).clamp(0.0, 1.0);
        let dielectric_roughness = 1.0 + (material.specular_roughness - 1.0) * fresnel_weight;
        let base_roughness = dielectric_roughness
            + (material.specular_roughness - dielectric_roughness) * material.base_metalness;

        // OpenPBR Eq. (66): internal coat average Fresnel.
        let avg_fresnel = 1.0
            - (1.0 - average_dielectric_fresnel(material.coat_ior))
                / (material.coat_ior * material.coat_ior);

        // OpenPBR Eq. (68): blend between directional and average Fresnel.
        let fresnel = fresnel_dielectric(material.coat_ior, wo.cos_theta().abs());
        let diffuse_refl_coeff = fresnel + (avg_fresnel - fresnel) * base_roughness;

        // OpenPBR Eq. (65): darkening factor.
        let dielectric_base_albedo = albedos[LobeType::Diffuse].lerp(
            albedos[LobeType::SpecularTransmission],
            material.transmission_weight,
        );
        let base_albedo =
            dielectric_base_albedo.lerp(albedos[LobeType::Metal], material.base_metalness);
        let denom =
            (Vec3::ONE - base_albedo * diffuse_refl_coeff).max(Vec3::splat(DENOM_TOLERANCE));
        let darkening_factor = Vec3::splat(1.0 - diffuse_refl_coeff) / denom;

        // OpenPBR Eq. (71).
        base_darkening = Vec3::ONE.lerp(
            darkening_factor,
            material.coat_weight * material.coat_darkening,
        );
    }

    // OpenPBR Eq. (92): base weight accounting for coat absorption and darkening.
    let base_weight = coated_base_weight
        * Vec3::ONE.lerp(
            base_darkening * material.coat_color * (Vec3::ONE - albedos[LobeType::Coat]),
            material.coat_weight,
        );

    weights[LobeType::Metal] = base_weight * material.base_metalness;

    let dielectric_base_weight = base_weight * (1.0 - material.base_metalness);
    weights[LobeType::SpecularReflection] = dielectric_base_weight;
    weights[LobeType::SpecularTransmission] = dielectric_base_weight * material.transmission_weight;

    let opaque_dielectric = dielectric_base_weight * (1.0 - material.transmission_weight);
    weights[LobeType::Diffuse] =
        opaque_dielectric * (Vec3::ONE - albedos[LobeType::SpecularReflection]);

    // Normalize weights into sampling probabilities.
    let mut probs = EnumMap::from_fn(|i| (weights[i] * albedos[i]).length());
    let total = probs.values().sum::<f32>().max(DENOM_TOLERANCE);

    for p in probs.values_mut() {
        *p /= total;
    }

    (weights, probs)
}

const ALL_LOBES: &[LobeType] = &[
    LobeType::Fuzz,
    LobeType::Coat,
    LobeType::Metal,
    LobeType::SpecularReflection,
    LobeType::SpecularTransmission,
    LobeType::Diffuse,
];
