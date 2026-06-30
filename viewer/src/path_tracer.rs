use super::Model;
use crate::{bvh::Ray, Progress};
use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3, Vec4};
use openpbr::{math::SurfaceBasis, Bsdf, Material};
use rand::RngExt;

pub struct DirectionalLight {
    /// Normalized direction pointing *toward* the light in world space.
    pub direction: Vec3,
    /// Emitted radiance.
    pub radiance: Vec3,
}

#[derive(Clone)]
pub struct Config {
    pub width: usize,
    pub height: usize,
    pub samples: usize,
    pub bounces: usize,
}

pub struct Camera {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
}

struct Constants {
    inverse_view: Mat4,
    inverse_proj: Mat4,
}

impl Constants {
    fn new(camera: &Camera) -> Self {
        Self {
            inverse_proj: camera.proj.inverse(),
            inverse_view: camera.view.inverse(),
        }
    }
}

struct PathState {
    accumulated: Vec3,
    throughput: Vec3,
    ray: Ray,
}

fn sample_ndc(x: u32, y: u32, config: &Config, rng: &mut impl rand::Rng) -> Vec2 {
    let offset = Vec2::new(rng.random(), rng.random()) - 0.5;
    ((Vec2::new(x as f32, y as f32) + offset)
        / Vec2::new(config.width as f32, config.height as f32))
        * 2.0
        - 1.0
}

fn camera_ray(ndc: Vec2, camera: &Camera, constants: &Constants) -> Ray {
    let view_space_point = constants.inverse_proj * Vec4::new(ndc.x, -ndc.y, 1.0, 1.0);
    let direction = (constants.inverse_view * view_space_point.with_w(0.0))
        .truncate()
        .normalize();
    Ray {
        origin: camera.position,
        direction,
    }
}

fn next_bounce(
    path_state: &mut PathState,
    model: &Model,
    material: &Material,
    light: &DirectionalLight,
    rng: &mut impl rand::Rng,
) -> bool {
    let Some(hit) = model.bvh.hit(&path_state.ray) else {
        path_state.accumulated += path_state.throughput;
        return false;
    };

    let base_index = hit.index * 3;

    let i1 = model.obj.indices[base_index + 0] as usize;
    let i2 = model.obj.indices[base_index + 1] as usize;
    let i3 = model.obj.indices[base_index + 2] as usize;

    let v1 = model.obj.vertices[i1];
    let v2 = model.obj.vertices[i2];
    let v3 = model.obj.vertices[i3];

    let barycentric = Vec3::new(1.0 - hit.u - hit.v, hit.u, hit.v);

    let position = Vec3::from(v1.position) * barycentric.x
        + Vec3::from(v2.position) * barycentric.y
        + Vec3::from(v3.position) * barycentric.z;

    let normal = (Vec3::from(v1.normal) * barycentric.x
        + Vec3::from(v2.normal) * barycentric.y
        + Vec3::from(v3.normal) * barycentric.z)
        .normalize();

    let basis = SurfaceBasis::any_with_normal(normal);
    let wo = basis.inverse_transform(-path_state.ray.direction);

    let bsdf = Bsdf::new(material, wo, rng);

    /*
    let wi_light = basis.inverse_transform(light.direction);
    if wi_light.z > 0.0 {
        let shadow_ray = Ray {
            origin: position + normal * 1e-4,
            direction: light.direction,
        };
        if model.bvh.hit(&shadow_ray).is_none() {
            let (direct, _) = bsdf.eval(wo, wi_light);
            let nee = direct.total() * wi_light.z * light.radiance;
            path_state.accumulated += path_state.throughput * nee;
        }
    }
    */

    let Some(sample) = bsdf.sample(wo, rng) else {
        return false;
    };

    let wi_world = basis.transform(sample.wi);

    path_state.throughput *=
        (sample.throughput.diffuse + sample.throughput.specular) / sample.density;
    path_state.ray = Ray {
        origin: position + normal * 1e-4,
        direction: wi_world,
    };

    true
}

fn integrate_pixel(
    x: u32,
    y: u32,
    config: &Config,
    constants: &Constants,
    camera: &Camera,
    model: &Model,
    material: &Material,
    light: &DirectionalLight,
    rng: &mut impl rand::Rng,
) -> Vec3 {
    let mut accumulated = Vec3::ZERO;

    for _ in 0..config.samples {
        let ndc = sample_ndc(x, y, config, rng);
        let mut path_state = PathState {
            ray: camera_ray(ndc, camera, constants),
            accumulated: Vec3::ZERO,
            throughput: Vec3::ONE,
        };

        for _ in 0..config.bounces {
            if !next_bounce(&mut path_state, model, material, light, rng) {
                break;
            }
        }

        accumulated += path_state.accumulated;
    }

    accumulated / config.samples as f32
}

pub fn path_trace(
    config: Config,
    camera: Camera,
    model: Arc<Model>,
    material: Material,
    light: DirectionalLight,
    progress: Progress,
) -> Vec<Vec3> {
    let constants = Constants::new(&camera);
    let mut rng = rand::rng();

    let size = config.width * config.height;
    let mut output = vec![Vec3::ZERO; size];

    for y in 0..config.height {
        for x in 0..config.width {
            let offset = y * config.width + x;
            output[offset] = integrate_pixel(
                x as u32, y as u32, &config, &constants, &camera, &model, &material, &light,
                &mut rng,
            );

            progress.set(offset as f32 / size as f32);
        }
    }

    output
}
