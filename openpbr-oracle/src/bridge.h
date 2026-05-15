#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float x, y, z; } BridgeVec3;
typedef struct { float x, y; } BridgeVec2;

typedef struct {
    BridgeVec3 t;
    BridgeVec3 b;
    BridgeVec3 n;
} BridgeBasis;

typedef struct {
    float base_weight;
    BridgeVec3 base_color;
    float base_diffuse_roughness;
    float base_metalness;

    float subsurface_weight;
    BridgeVec3 subsurface_color;
    float subsurface_radius;
    BridgeVec3 subsurface_radius_scale;
    float subsurface_scatter_anisotropy;

    float specular_weight;
    BridgeVec3 specular_color;
    float specular_roughness;
    float specular_roughness_anisotropy;
    float specular_ior;
    BridgeVec2 specular_anisotropy_rotation_cos_sin;

    float coat_weight;
    BridgeVec3 coat_color;
    float coat_roughness;
    float coat_roughness_anisotropy;
    float coat_ior;
    float coat_darkening;
    BridgeVec2 coat_anisotropy_rotation_cos_sin;

    float fuzz_weight;
    BridgeVec3 fuzz_color;
    float fuzz_roughness;

    float transmission_weight;
    BridgeVec3 transmission_color;
    float transmission_depth;
    BridgeVec3 transmission_scatter;
    float transmission_scatter_anisotropy;
    float transmission_dispersion_scale;
    float transmission_dispersion_abbe_number;

    float thin_film_weight;
    float thin_film_thickness;
    float thin_film_ior;

    float emission_luminance;
    BridgeVec3 emission_color;

    float geometry_opacity;
    uint32_t geometry_thin_walled;
    BridgeBasis geometry_basis;
    BridgeBasis geometry_coat_basis;
} BridgeResolvedInputs;

typedef struct {
    BridgeVec3 diffuse;
    BridgeVec3 specular;
} BridgeDiffuseSpecular;

typedef struct {
    BridgeVec3 extinction_coefficient;
    BridgeVec3 albedo;
    float anisotropy;
} BridgeHomogeneousVolume;

typedef struct BridgePreparedBsdf BridgePreparedBsdf;

BridgeResolvedInputs openpbr_bridge_make_default_inputs();

BridgePreparedBsdf* openpbr_bridge_prepare(
    const BridgeResolvedInputs* inputs,
    const float path_throughput[3],
    const float rgb_wavelengths_nm[3],
    float exterior_ior,
    const float view_direction[3]
);

void openpbr_bridge_free(BridgePreparedBsdf* prepared);

BridgeDiffuseSpecular openpbr_bridge_eval(const BridgePreparedBsdf* prepared, const float light_direction[3]);

void openpbr_bridge_sample(
    const BridgePreparedBsdf* prepared,
    const float rand[3],
    float out_light_direction[3],
    BridgeDiffuseSpecular* out_weight,
    float* out_pdf,
    uint32_t* out_lobe_type
);

float openpbr_bridge_pdf(const BridgePreparedBsdf* prepared, const float light_direction[3]);

void openpbr_bridge_get_emission(const BridgePreparedBsdf* prepared, float out_emission[3]);

void openpbr_bridge_get_volume(const BridgePreparedBsdf* prepared, BridgeHomogeneousVolume* out_volume);

#ifdef __cplusplus
}
#endif
