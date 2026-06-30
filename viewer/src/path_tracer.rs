use super::Model;
use crate::bvh::Ray;
use rand::RngExt;

use glam::{Mat4, Vec2, Vec3, Vec4};

pub struct Config {
    pub width: usize,
    pub height: usize,
    pub samples: usize,
    pub bounces: usize,
    pub view: Mat4,
    pub proj: Mat4,
    pub camera_position: Vec3,
}

struct Constants {
    inverse_view: Mat4,
    inverse_proj: Mat4,
}

impl Constants {
    fn new(config: &Config) -> Self {
        Self {
            inverse_proj: config.proj.inverse(),
            inverse_view: config.view.inverse(),
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

fn camera_ray(ndc: Vec2, config: &Config, constants: &Constants) -> Ray {
    let view_space_point = constants.inverse_proj * Vec4::new(-ndc.x, -ndc.y, 1.0, 1.0);
    let direction = (constants.inverse_view * view_space_point.with_w(0.0))
        .truncate()
        .normalize();
    Ray {
        origin: config.camera_position,
        direction,
    }
}

fn next_bounce(path_state: &mut PathState, model: &Model, rng: &mut impl rand::Rng) -> bool {
    let Some(hit) = model.bvh.hit(&path_state.ray) else {
        return false;
    };

    let index = model.obj.indices[hit.index];

    true
}

fn integrate_pixel(
    x: u32,
    y: u32,
    config: &Config,
    constants: &Constants,
    model: &Model,
    rng: &mut impl rand::Rng,
) -> Vec3 {
    let mut accumulated = Vec3::ZERO;

    for sample in 0..config.samples {
        let ndc = sample_ndc(x, y, config, rng);
        let mut path_state = PathState {
            ray: camera_ray(ndc, config, constants),
            accumulated: Vec3::ZERO,
            throughput: Vec3::ONE,
        };

        for bounce in 0..config.bounces {
            if !next_bounce(&mut path_state, model, rng) {
                break;
            }
        }

        accumulated += path_state.accumulated;
    }

    accumulated / config.samples as f32
}

pub fn path_trace(config: &Config, model: &Model) -> Vec<Vec3> {
    let constants = Constants::new(config);
    let mut rng = rand::rng();

    let mut output = vec![Vec3::ZERO; config.width * config.height];

    for x in 0..config.width {
        for y in 0..config.height {
            let offset = y * config.width + x;
            output[offset] =
                integrate_pixel(x as u32, y as u32, config, &constants, model, &mut rng);
        }
    }

    output
}
