#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FfiVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FfiVec2 {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FfiBasis {
    pub t: FfiVec3,
    pub b: FfiVec3,
    pub n: FfiVec3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FfiResolvedInputs {
    pub base_weight: f32,
    pub base_color: FfiVec3,
    pub base_diffuse_roughness: f32,
    pub base_metalness: f32,

    pub subsurface_weight: f32,
    pub subsurface_color: FfiVec3,
    pub subsurface_radius: f32,
    pub subsurface_radius_scale: FfiVec3,
    pub subsurface_scatter_anisotropy: f32,

    pub specular_weight: f32,
    pub specular_color: FfiVec3,
    pub specular_roughness: f32,
    pub specular_roughness_anisotropy: f32,
    pub specular_ior: f32,
    pub specular_anisotropy_rotation_cos_sin: FfiVec2,

    pub coat_weight: f32,
    pub coat_color: FfiVec3,
    pub coat_roughness: f32,
    pub coat_roughness_anisotropy: f32,
    pub coat_ior: f32,
    pub coat_darkening: f32,
    pub coat_anisotropy_rotation_cos_sin: FfiVec2,

    pub fuzz_weight: f32,
    pub fuzz_color: FfiVec3,
    pub fuzz_roughness: f32,

    pub transmission_weight: f32,
    pub transmission_color: FfiVec3,
    pub transmission_depth: f32,
    pub transmission_scatter: FfiVec3,
    pub transmission_scatter_anisotropy: f32,
    pub transmission_dispersion_scale: f32,
    pub transmission_dispersion_abbe_number: f32,

    pub thin_film_weight: f32,
    pub thin_film_thickness: f32,
    pub thin_film_ior: f32,

    pub emission_luminance: f32,
    pub emission_color: FfiVec3,

    pub geometry_opacity: f32,
    pub geometry_thin_walled: u32,
    pub geometry_basis: FfiBasis,
    pub geometry_coat_basis: FfiBasis,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FfiDiffuseSpecular {
    pub diffuse: FfiVec3,
    pub specular: FfiVec3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FfiHomogeneousVolume {
    pub extinction_coefficient: FfiVec3,
    pub albedo: FfiVec3,
    pub anisotropy: f32,
}

#[repr(C)]
pub struct FfiBridgePreparedBsdf {
    data: [u8; 0],
}

unsafe extern "C" {
    pub fn openpbr_bridge_make_default_inputs() -> FfiResolvedInputs;

    pub fn openpbr_bridge_prepare(
        inputs: *const FfiResolvedInputs,
        path_throughput: *const f32,
        rgb_wavelengths_nm: *const f32,
        exterior_ior: f32,
        view_direction: *const f32,
    ) -> *mut FfiBridgePreparedBsdf;

    pub fn openpbr_bridge_free(prepared: *mut FfiBridgePreparedBsdf);

    pub fn openpbr_bridge_eval(
        prepared: *const FfiBridgePreparedBsdf,
        light_direction: *const f32,
    ) -> FfiDiffuseSpecular;

    pub fn openpbr_bridge_sample(
        prepared: *const FfiBridgePreparedBsdf,
        rand: *const f32,
        out_light_direction: *mut f32,
        out_weight: *mut FfiDiffuseSpecular,
        out_pdf: *mut f32,
        out_lobe_type: *mut u32,
    );

    pub fn openpbr_bridge_pdf(
        prepared: *const FfiBridgePreparedBsdf,
        light_direction: *const f32,
    ) -> f32;

    pub fn openpbr_bridge_get_emission(
        prepared: *const FfiBridgePreparedBsdf,
        out_emission: *mut f32,
    );

    pub fn openpbr_bridge_get_volume(
        prepared: *const FfiBridgePreparedBsdf,
        out_volume: *mut FfiHomogeneousVolume,
    );
}
