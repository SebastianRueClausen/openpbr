mod ffi;

pub type Vec3 = [f32; 3];
pub type Vec2 = [f32; 2];

#[derive(Clone, Copy, Debug)]
pub struct Basis {
    pub t: Vec3,
    pub b: Vec3,
    pub n: Vec3,
}

impl Basis {
    pub const IDENTITY: Self = Self {
        t: [1.0, 0.0, 0.0],
        b: [0.0, 1.0, 0.0],
        n: [0.0, 0.0, 1.0],
    };
}

#[derive(Clone, Debug)]
pub struct ResolvedInputs {
    pub base_weight: f32,
    pub base_color: Vec3,
    pub base_diffuse_roughness: f32,
    pub base_metalness: f32,

    pub subsurface_weight: f32,
    pub subsurface_color: Vec3,
    pub subsurface_radius: f32,
    pub subsurface_radius_scale: Vec3,
    pub subsurface_scatter_anisotropy: f32,

    pub specular_weight: f32,
    pub specular_color: Vec3,
    pub specular_roughness: f32,
    pub specular_roughness_anisotropy: f32,
    pub specular_ior: f32,
    pub specular_anisotropy_rotation_cos_sin: Vec2,

    pub coat_weight: f32,
    pub coat_color: Vec3,
    pub coat_roughness: f32,
    pub coat_roughness_anisotropy: f32,
    pub coat_ior: f32,
    pub coat_darkening: f32,
    pub coat_anisotropy_rotation_cos_sin: Vec2,

    pub fuzz_weight: f32,
    pub fuzz_color: Vec3,
    pub fuzz_roughness: f32,

    pub transmission_weight: f32,
    pub transmission_color: Vec3,
    pub transmission_depth: f32,
    pub transmission_scatter: Vec3,
    pub transmission_scatter_anisotropy: f32,
    pub transmission_dispersion_scale: f32,
    pub transmission_dispersion_abbe_number: f32,

    pub thin_film_weight: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,

    pub emission_luminance: f32,
    pub emission_color: Vec3,

    pub geometry_opacity: f32,
    pub geometry_thin_walled: bool,
    pub geometry_basis: Basis,
    pub geometry_coat_basis: Basis,
}

impl Default for ResolvedInputs {
    fn default() -> Self {
        let raw = unsafe { ffi::openpbr_bridge_make_default_inputs() };
        from_ffi_inputs(&raw)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DiffuseSpecular {
    pub diffuse: Vec3,
    pub specular: Vec3,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HomogeneousVolume {
    pub extinction_coefficient: Vec3,
    pub albedo: Vec3,
    pub anisotropy: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BsdfLobeType(pub u32);

impl BsdfLobeType {
    pub const NONE: Self = Self(0);
    pub const REFLECTION: Self = Self(1 << 0);
    pub const TRANSMISSION: Self = Self(1 << 1);
    pub const DIFFUSE: Self = Self(1 << 2);
    pub const GLOSSY: Self = Self(1 << 3);
    pub const SPECULAR: Self = Self(1 << 4);
    pub const VOLUME: Self = Self(1 << 5);

    pub fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 != 0
    }
}

#[derive(Clone, Debug)]
pub struct SampleResult {
    pub light_direction: Vec3,
    pub weight: DiffuseSpecular,
    pub pdf: f32,
    pub lobe_type: BsdfLobeType,
}

pub struct PreparedBsdf {
    ptr: *mut ffi::FfiBridgePreparedBsdf,
    pub emission: Vec3,
    pub volume: HomogeneousVolume,
}

unsafe impl Send for PreparedBsdf {}
unsafe impl Sync for PreparedBsdf {}

impl Drop for PreparedBsdf {
    fn drop(&mut self) {
        unsafe { ffi::openpbr_bridge_free(self.ptr) };
    }
}

impl PreparedBsdf {
    pub fn prepare(
        inputs: &ResolvedInputs,
        path_throughput: Vec3,
        rgb_wavelengths_nm: Vec3,
        exterior_ior: f32,
        view_direction: Vec3,
    ) -> Self {
        let ffi_inputs = to_ffi_inputs(inputs);
        let ptr = unsafe {
            ffi::openpbr_bridge_prepare(
                &ffi_inputs,
                path_throughput.as_ptr(),
                rgb_wavelengths_nm.as_ptr(),
                exterior_ior,
                view_direction.as_ptr(),
            )
        };
        assert!(!ptr.is_null(), "openpbr_bridge_prepare returned null");

        let emission = unsafe {
            let mut e = [0f32; 3];
            ffi::openpbr_bridge_get_emission(ptr, e.as_mut_ptr());
            e
        };
        let volume = unsafe {
            let mut v = ffi::FfiHomogeneousVolume::default();
            ffi::openpbr_bridge_get_volume(ptr, &mut v);
            HomogeneousVolume {
                extinction_coefficient: [
                    v.extinction_coefficient.x,
                    v.extinction_coefficient.y,
                    v.extinction_coefficient.z,
                ],
                albedo: [v.albedo.x, v.albedo.y, v.albedo.z],
                anisotropy: v.anisotropy,
            }
        };

        Self {
            ptr,
            emission,
            volume,
        }
    }

    pub fn eval(&self, light_direction: Vec3) -> DiffuseSpecular {
        let r = unsafe { ffi::openpbr_bridge_eval(self.ptr, light_direction.as_ptr()) };
        DiffuseSpecular {
            diffuse: [r.diffuse.x, r.diffuse.y, r.diffuse.z],
            specular: [r.specular.x, r.specular.y, r.specular.z],
        }
    }

    pub fn sample(&self, rand: Vec3) -> SampleResult {
        let mut light_dir = [0f32; 3];
        let mut weight = ffi::FfiDiffuseSpecular::default();
        let mut pdf = 0f32;
        let mut lobe_type = 0u32;

        unsafe {
            ffi::openpbr_bridge_sample(
                self.ptr,
                rand.as_ptr(),
                light_dir.as_mut_ptr(),
                &mut weight,
                &mut pdf,
                &mut lobe_type,
            );
        }

        SampleResult {
            light_direction: light_dir,
            weight: DiffuseSpecular {
                diffuse: [weight.diffuse.x, weight.diffuse.y, weight.diffuse.z],
                specular: [weight.specular.x, weight.specular.y, weight.specular.z],
            },
            pdf,
            lobe_type: BsdfLobeType(lobe_type),
        }
    }

    pub fn pdf(&self, light_direction: Vec3) -> f32 {
        unsafe { ffi::openpbr_bridge_pdf(self.ptr, light_direction.as_ptr()) }
    }
}

pub const RGB_WAVELENGTHS_NM: Vec3 = [620.0, 540.0, 450.0];
pub const VACUUM_IOR: f32 = 1.0;

fn convert_vec3(v: Vec3) -> ffi::FfiVec3 {
    ffi::FfiVec3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

fn convert_vec2(v: Vec2) -> ffi::FfiVec2 {
    ffi::FfiVec2 { x: v[0], y: v[1] }
}

fn convert_basis(b: &Basis) -> ffi::FfiBasis {
    ffi::FfiBasis {
        t: convert_vec3(b.t),
        b: convert_vec3(b.b),
        n: convert_vec3(b.n),
    }
}

fn to_ffi_inputs(i: &ResolvedInputs) -> ffi::FfiResolvedInputs {
    ffi::FfiResolvedInputs {
        base_weight: i.base_weight,
        base_color: convert_vec3(i.base_color),
        base_diffuse_roughness: i.base_diffuse_roughness,
        base_metalness: i.base_metalness,

        subsurface_weight: i.subsurface_weight,
        subsurface_color: convert_vec3(i.subsurface_color),
        subsurface_radius: i.subsurface_radius,
        subsurface_radius_scale: convert_vec3(i.subsurface_radius_scale),
        subsurface_scatter_anisotropy: i.subsurface_scatter_anisotropy,

        specular_weight: i.specular_weight,
        specular_color: convert_vec3(i.specular_color),
        specular_roughness: i.specular_roughness,
        specular_roughness_anisotropy: i.specular_roughness_anisotropy,
        specular_ior: i.specular_ior,
        specular_anisotropy_rotation_cos_sin: convert_vec2(i.specular_anisotropy_rotation_cos_sin),

        coat_weight: i.coat_weight,
        coat_color: convert_vec3(i.coat_color),
        coat_roughness: i.coat_roughness,
        coat_roughness_anisotropy: i.coat_roughness_anisotropy,
        coat_ior: i.coat_ior,
        coat_darkening: i.coat_darkening,
        coat_anisotropy_rotation_cos_sin: convert_vec2(i.coat_anisotropy_rotation_cos_sin),

        fuzz_weight: i.fuzz_weight,
        fuzz_color: convert_vec3(i.fuzz_color),
        fuzz_roughness: i.fuzz_roughness,

        transmission_weight: i.transmission_weight,
        transmission_color: convert_vec3(i.transmission_color),
        transmission_depth: i.transmission_depth,
        transmission_scatter: convert_vec3(i.transmission_scatter),
        transmission_scatter_anisotropy: i.transmission_scatter_anisotropy,
        transmission_dispersion_scale: i.transmission_dispersion_scale,
        transmission_dispersion_abbe_number: i.transmission_dispersion_abbe_number,

        thin_film_weight: i.thin_film_weight,
        thin_film_thickness: i.thin_film_thickness,
        thin_film_ior: i.thin_film_ior,

        emission_luminance: i.emission_luminance,
        emission_color: convert_vec3(i.emission_color),

        geometry_opacity: i.geometry_opacity,
        geometry_thin_walled: i.geometry_thin_walled as u32,
        geometry_basis: convert_basis(&i.geometry_basis),
        geometry_coat_basis: convert_basis(&i.geometry_coat_basis),
    }
}

fn from_ffi_inputs(f: &ffi::FfiResolvedInputs) -> ResolvedInputs {
    ResolvedInputs {
        base_weight: f.base_weight,
        base_color: { [f.base_color.x, f.base_color.y, f.base_color.z] },
        base_diffuse_roughness: f.base_diffuse_roughness,
        base_metalness: f.base_metalness,

        subsurface_weight: f.subsurface_weight,
        subsurface_color: {
            let v = f.subsurface_color;
            [v.x, v.y, v.z]
        },
        subsurface_radius: f.subsurface_radius,
        subsurface_radius_scale: {
            let v = f.subsurface_radius_scale;
            [v.x, v.y, v.z]
        },
        subsurface_scatter_anisotropy: f.subsurface_scatter_anisotropy,

        specular_weight: f.specular_weight,
        specular_color: {
            let v = f.specular_color;
            [v.x, v.y, v.z]
        },
        specular_roughness: f.specular_roughness,
        specular_roughness_anisotropy: f.specular_roughness_anisotropy,
        specular_ior: f.specular_ior,
        specular_anisotropy_rotation_cos_sin: {
            let v = f.specular_anisotropy_rotation_cos_sin;
            [v.x, v.y]
        },

        coat_weight: f.coat_weight,
        coat_color: { [f.coat_color.x, f.coat_color.y, f.coat_color.z] },
        coat_roughness: f.coat_roughness,
        coat_roughness_anisotropy: f.coat_roughness_anisotropy,
        coat_ior: f.coat_ior,
        coat_darkening: f.coat_darkening,
        coat_anisotropy_rotation_cos_sin: {
            let v = f.coat_anisotropy_rotation_cos_sin;
            [v.x, v.y]
        },

        fuzz_weight: f.fuzz_weight,
        fuzz_color: { [f.fuzz_color.x, f.fuzz_color.y, f.fuzz_color.z] },
        fuzz_roughness: f.fuzz_roughness,

        transmission_weight: f.transmission_weight,
        transmission_color: {
            let v = f.transmission_color;
            [v.x, v.y, v.z]
        },
        transmission_depth: f.transmission_depth,
        transmission_scatter: {
            let v = f.transmission_scatter;
            [v.x, v.y, v.z]
        },
        transmission_scatter_anisotropy: f.transmission_scatter_anisotropy,
        transmission_dispersion_scale: f.transmission_dispersion_scale,
        transmission_dispersion_abbe_number: f.transmission_dispersion_abbe_number,

        thin_film_weight: f.thin_film_weight,
        thin_film_thickness: f.thin_film_thickness,
        thin_film_ior: f.thin_film_ior,

        emission_luminance: f.emission_luminance,
        emission_color: {
            let v = f.emission_color;
            [v.x, v.y, v.z]
        },

        geometry_opacity: f.geometry_opacity,
        geometry_thin_walled: f.geometry_thin_walled != 0,
        geometry_basis: Basis {
            t: {
                let v = f.geometry_basis.t;
                [v.x, v.y, v.z]
            },
            b: {
                let v = f.geometry_basis.b;
                [v.x, v.y, v.z]
            },
            n: {
                let v = f.geometry_basis.n;
                [v.x, v.y, v.z]
            },
        },
        geometry_coat_basis: Basis {
            t: {
                let v = f.geometry_coat_basis.t;
                [v.x, v.y, v.z]
            },
            b: {
                let v = f.geometry_coat_basis.b;
                [v.x, v.y, v.z]
            },
            n: {
                let v = f.geometry_coat_basis.n;
                [v.x, v.y, v.z]
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bsdf() -> PreparedBsdf {
        PreparedBsdf::prepare(
            &ResolvedInputs::default(),
            [1.0, 1.0, 1.0],
            RGB_WAVELENGTHS_NM,
            VACUUM_IOR,
            [0.0, 0.0, 1.0],
        )
    }

    #[test]
    fn prepare_default_material() {
        let bsdf = default_bsdf();
        assert_eq!(bsdf.emission, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn eval_non_negative() {
        let bsdf = default_bsdf();
        let ds = bsdf.eval([0.0, 0.0, 1.0]);
        for ch in ds.diffuse.iter().chain(ds.specular.iter()) {
            assert!(*ch >= 0.0, "negative BSDF value: {ch}");
        }
    }

    #[test]
    fn sample_valid_pdf() {
        let bsdf = default_bsdf();
        let result = bsdf.sample([0.1, 0.5, 0.9]);
        assert!(result.pdf >= 0.0, "negative PDF: {}", result.pdf);
        assert!(result.pdf.is_finite(), "non-finite PDF: {}", result.pdf);
    }
}
