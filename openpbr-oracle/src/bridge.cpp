#include <glm/glm.hpp>
#include "openpbr-bsdf/openpbr.h"
#include "bridge.h"

static inline glm::vec2 to_vec2(BridgeVec2 v) {
    return glm::vec2(v.x, v.y);
}

static inline glm::vec3 to_vec3(BridgeVec3 v) {
    return glm::vec3(v.x, v.y, v.z);
}

static inline glm::vec3 to_vec3(const float v[3]) {
    return glm::vec3(v[0], v[1], v[2]);
}

static inline BridgeVec3 from_vec3(glm::vec3 v) {
    return BridgeVec3{ v.x, v.y, v.z };
}

static inline OpenPBR_Basis to_basis(const BridgeBasis& b) {
    OpenPBR_Basis basis;
    basis.t = to_vec3(b.t);
    basis.b = to_vec3(b.b);
    basis.n = to_vec3(b.n);
    return basis;
}

static inline BridgeBasis from_basis(const OpenPBR_Basis& b) {
    return BridgeBasis{
        BridgeVec3{ b.t.x, b.t.y, b.t.z },
        BridgeVec3{ b.b.x, b.b.y, b.b.z },
        BridgeVec3{ b.n.x, b.n.y, b.n.z },
    };
}

static OpenPBR_ResolvedInputs to_resolved_inputs(const BridgeResolvedInputs* c) {
    OpenPBR_ResolvedInputs i;

    i.base_weight = c->base_weight;
    i.base_color = to_vec3(c->base_color);
    i.base_diffuse_roughness = c->base_diffuse_roughness;
    i.base_metalness = c->base_metalness;

    i.subsurface_weight = c->subsurface_weight;
    i.subsurface_color = to_vec3(c->subsurface_color);
    i.subsurface_radius = c->subsurface_radius;
    i.subsurface_radius_scale = to_vec3(c->subsurface_radius_scale);
    i.subsurface_scatter_anisotropy = c->subsurface_scatter_anisotropy;

    i.specular_weight = c->specular_weight;
    i.specular_color = to_vec3(c->specular_color);
    i.specular_roughness = c->specular_roughness;
    i.specular_roughness_anisotropy = c->specular_roughness_anisotropy;
    i.specular_ior = c->specular_ior;
    i.specular_anisotropy_rotation_cos_sin = to_vec2(c->specular_anisotropy_rotation_cos_sin);

    i.coat_weight = c->coat_weight;
    i.coat_color = to_vec3(c->coat_color);
    i.coat_roughness = c->coat_roughness;
    i.coat_roughness_anisotropy = c->coat_roughness_anisotropy;
    i.coat_ior = c->coat_ior;
    i.coat_darkening = c->coat_darkening;
    i.coat_anisotropy_rotation_cos_sin = to_vec2(c->coat_anisotropy_rotation_cos_sin);

    i.fuzz_weight = c->fuzz_weight;
    i.fuzz_color = to_vec3(c->fuzz_color);
    i.fuzz_roughness = c->fuzz_roughness;

    i.transmission_weight = c->transmission_weight;
    i.transmission_color = to_vec3(c->transmission_color);
    i.transmission_depth = c->transmission_depth;
    i.transmission_scatter = to_vec3(c->transmission_scatter);
    i.transmission_scatter_anisotropy = c->transmission_scatter_anisotropy;
    i.transmission_dispersion_scale = c->transmission_dispersion_scale;
    i.transmission_dispersion_abbe_number = c->transmission_dispersion_abbe_number;

    i.thin_film_weight = c->thin_film_weight;
    i.thin_film_thickness = c->thin_film_thickness;
    i.thin_film_ior = c->thin_film_ior;

    i.emission_luminance = c->emission_luminance;
    i.emission_color = to_vec3(c->emission_color);

    i.geometry_opacity = c->geometry_opacity;
    i.geometry_thin_walled = (c->geometry_thin_walled != 0u);
    i.geometry_basis = to_basis(c->geometry_basis);
    i.geometry_coat_basis = to_basis(c->geometry_coat_basis);

    return i;
}

static BridgeResolvedInputs from_resolved_inputs(const OpenPBR_ResolvedInputs& i) {
    BridgeResolvedInputs c{};

    c.base_weight = i.base_weight;
    c.base_color = from_vec3(i.base_color);
    c.base_diffuse_roughness = i.base_diffuse_roughness;
    c.base_metalness = i.base_metalness;

    c.subsurface_weight = i.subsurface_weight;
    c.subsurface_color = from_vec3(i.subsurface_color);
    c.subsurface_radius = i.subsurface_radius;
    c.subsurface_radius_scale = from_vec3(i.subsurface_radius_scale);
    c.subsurface_scatter_anisotropy = i.subsurface_scatter_anisotropy;

    c.specular_weight = i.specular_weight;
    c.specular_color = from_vec3(i.specular_color);
    c.specular_roughness = i.specular_roughness;
    c.specular_roughness_anisotropy = i.specular_roughness_anisotropy;
    c.specular_ior = i.specular_ior;
    c.specular_anisotropy_rotation_cos_sin = BridgeVec2{
        i.specular_anisotropy_rotation_cos_sin.x,
        i.specular_anisotropy_rotation_cos_sin.y
    };

    c.coat_weight = i.coat_weight;
    c.coat_color = from_vec3(i.coat_color);
    c.coat_roughness = i.coat_roughness;
    c.coat_roughness_anisotropy = i.coat_roughness_anisotropy;
    c.coat_ior = i.coat_ior;
    c.coat_darkening = i.coat_darkening;
    c.coat_anisotropy_rotation_cos_sin = BridgeVec2{
        i.coat_anisotropy_rotation_cos_sin.x,
        i.coat_anisotropy_rotation_cos_sin.y
    };

    c.fuzz_weight = i.fuzz_weight;
    c.fuzz_color = from_vec3(i.fuzz_color);
    c.fuzz_roughness = i.fuzz_roughness;

    c.transmission_weight = i.transmission_weight;
    c.transmission_color = from_vec3(i.transmission_color);
    c.transmission_depth = i.transmission_depth;
    c.transmission_scatter = from_vec3(i.transmission_scatter);
    c.transmission_scatter_anisotropy = i.transmission_scatter_anisotropy;
    c.transmission_dispersion_scale = i.transmission_dispersion_scale;
    c.transmission_dispersion_abbe_number = i.transmission_dispersion_abbe_number;

    c.thin_film_weight = i.thin_film_weight;
    c.thin_film_thickness = i.thin_film_thickness;
    c.thin_film_ior = i.thin_film_ior;

    c.emission_luminance = i.emission_luminance;
    c.emission_color = from_vec3(i.emission_color);

    c.geometry_opacity = i.geometry_opacity;
    c.geometry_thin_walled = i.geometry_thin_walled ? 1u : 0u;
    c.geometry_basis = from_basis(i.geometry_basis);
    c.geometry_coat_basis = from_basis(i.geometry_coat_basis);

    return c;
}

struct BridgePreparedBsdf {
    OpenPBR_PreparedBsdf inner;
};

extern "C" {

BridgeResolvedInputs openpbr_bridge_make_default_inputs(void) {
    return from_resolved_inputs(openpbr_make_default_resolved_inputs());
}

BridgePreparedBsdf* openpbr_bridge_prepare(
    const BridgeResolvedInputs* inputs,
    const float path_throughput[3],
    const float rgb_wavelengths_nm[3],
    float exterior_ior,
    const float view_direction[3]
) {
    auto* result = new BridgePreparedBsdf();
    result->inner = openpbr_prepare(
        to_resolved_inputs(inputs), to_vec3(path_throughput), to_vec3(rgb_wavelengths_nm), exterior_ior, to_vec3(view_direction)
    );
    return result;
}

void openpbr_bridge_free(BridgePreparedBsdf* prepared) {
    delete prepared;
}

BridgeDiffuseSpecular openpbr_bridge_eval(const BridgePreparedBsdf* prepared, const float light_direction[3]) {
    const OpenPBR_DiffuseSpecular r = openpbr_eval(prepared->inner, to_vec3(light_direction));
    return BridgeDiffuseSpecular{ from_vec3(r.diffuse), from_vec3(r.specular) };
}

void openpbr_bridge_sample(
    const BridgePreparedBsdf* prepared,
    const float rand[3],
    float out_light_direction[3],
    BridgeDiffuseSpecular* out_weight,
    float* out_pdf,
    uint32_t* out_lobe_type
) {
    glm::vec3 light_dir;
    OpenPBR_DiffuseSpecular weight;
    float pdf;
    OpenPBR_BsdfLobeType lobe_type;

    openpbr_sample(prepared->inner, to_vec3(rand), light_dir, weight, pdf, lobe_type);

    out_light_direction[0] = light_dir.x;
    out_light_direction[1] = light_dir.y;
    out_light_direction[2] = light_dir.z;

    *out_weight = BridgeDiffuseSpecular{from_vec3(weight.diffuse), from_vec3(weight.specular)};
    *out_pdf = pdf;
    *out_lobe_type = static_cast<uint32_t>(lobe_type);
}

float openpbr_bridge_pdf(const BridgePreparedBsdf* prepared, const float light_direction[3]) {
    return openpbr_pdf(prepared->inner, to_vec3(light_direction));
}

void openpbr_bridge_get_emission(const BridgePreparedBsdf* prepared, float out_emission[3]) {
    const glm::vec3& e = prepared->inner.emission;
    out_emission[0] = e.x;
    out_emission[1] = e.y;
    out_emission[2] = e.z;
}

void openpbr_bridge_get_volume(const BridgePreparedBsdf* prepared, BridgeHomogeneousVolume* out_volume) {
    const OpenPBR_HomogeneousVolume& v = prepared->inner.volume;
    out_volume->extinction_coefficient = from_vec3(v.extinction_coefficient);
    out_volume->albedo = from_vec3(v.albedo);
    out_volume->anisotropy = v.anisotropy;
}

}
