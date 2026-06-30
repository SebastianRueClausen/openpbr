use glam::Vec3;

pub trait Sampler {
    fn next_f32(&mut self) -> f32;

    fn next_vec3(&mut self) -> Vec3 {
        Vec3::new(self.next_f32(), self.next_f32(), self.next_f32())
    }
}

#[cfg(feature = "rand")]
impl<R: rand::Rng> Sampler for R {
    fn next_f32(&mut self) -> f32 {
        rand::RngExt::random::<f32>(self)
    }
}
