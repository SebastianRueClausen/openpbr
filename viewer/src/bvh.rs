use std::array;
use std::range::Range;

use glam::Vec3;

#[derive(Debug, Clone, Copy)]
struct Volume {
    min: Vec3,
    max: Vec3,
}

impl Volume {
    fn axis_length(&self, axis: usize) -> f32 {
        assert!(axis <= 2);
        self.max[axis] - self.min[axis]
    }

    // Slab test: returns true if the ray hits the volume closer than `t_max`.
    fn intersect(&self, ray: &Ray, t_max: f32) -> bool {
        let t1 = (self.min - ray.origin) / ray.direction;
        let t2 = (self.max - ray.origin) / ray.direction;
        let t_near = t1.min(t2).max_element();
        let t_far = t1.max(t2).min_element();
        t_near <= t_far && t_far > 0.0 && t_near < t_max
    }

    fn longest_axis(&self) -> usize {
        array::from_fn::<_, 3, _>(|i| self.axis_length(i))
            .iter()
            .enumerate()
            .fold((0, 0.0f32), |(max_axis, max_len), (axis, &len)| {
                if len > max_len {
                    (axis, len)
                } else {
                    (max_axis, max_len)
                }
            })
            .0
    }

    fn max_length(&self) -> f32 {
        self.axis_length(self.longest_axis())
    }
}

#[derive(Debug, Clone, Copy)]
struct Triangle {
    positions: [Vec3; 3],
    index: usize,
}

impl Triangle {
    fn centroid(&self) -> Vec3 {
        self.positions.iter().sum::<Vec3>() / 3.0
    }

    // Möller–Trumbore intersection. Returns `(t, u, v)` along the ray, or `None` on miss.
    fn intersect(&self, ray: &Ray) -> Option<(f32, f32, f32)> {
        let e1 = self.positions[1] - self.positions[0];
        let e2 = self.positions[2] - self.positions[0];
        let h = ray.direction.cross(e2);
        let det = e1.dot(h);

        if det.abs() < f32::EPSILON {
            return None;
        }

        let inv_det = 1.0 / det;
        let s = ray.origin - self.positions[0];
        let u = inv_det * s.dot(h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(e1);
        let v = inv_det * ray.direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = inv_det * e2.dot(q);
        (t > f32::EPSILON).then_some((t, u, v))
    }
}

#[derive(Debug, Clone, Copy)]
enum Node {
    Internal {
        volume: Volume,
        children: [usize; 2],
    },
    Leaf {
        triangles: Range<usize>,
    },
}

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

pub struct Hit {
    /// Distance along the ray.
    pub t: f32,
    /// Barycentric coordinates of the hit point (w = 1 - u - v).
    pub u: f32,
    pub v: f32,
    /// Index of the triangle in the original positions slice passed to `Bvh::new`.
    pub index: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    triangles: Vec<Triangle>,
}

impl Bvh {
    pub fn new(positions: &[Vec3]) -> Self {
        let (chunks, &[]) = positions.as_chunks::<3>() else {
            panic!("`positions.len()` not a multiple of 3");
        };

        let mut triangles: Vec<Triangle> = chunks
            .iter()
            .enumerate()
            .map(|(index, &positions)| Triangle { positions, index })
            .collect();

        let mut bvh = Self::default();

        if !triangles.is_empty() {
            build(&mut bvh, &mut triangles);
        }

        bvh
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn hit(&self, ray: &Ray) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best: Option<Hit> = None;
        let mut t_max = f32::INFINITY;
        let mut stack = vec![0usize];

        while let Some(node) = stack.pop() {
            match self.nodes[node] {
                Node::Internal { volume, children } => {
                    if volume.intersect(ray, t_max) {
                        stack.extend_from_slice(&children);
                    }
                }
                Node::Leaf { triangles } => {
                    for tri in &self.triangles[triangles.start..triangles.end] {
                        if let Some((t, u, v)) = tri.intersect(ray) && t < t_max {
                            t_max = t;
                            best = Some(Hit { t, u, v, index: tri.index });
                        }
                    }
                }
            }
        }

        best
    }
}

fn bounding_volume(triangles: &[Triangle]) -> Volume {
    let mut min = Vec3::INFINITY;
    let mut max = Vec3::NEG_INFINITY;

    for tri in triangles {
        for &pos in &tri.positions {
            min = min.min(pos);
            max = max.max(pos);
        }
    }

    Volume { min, max }
}

fn triangle_max_length(tri: &Triangle) -> f32 {
    let min = tri
        .positions
        .iter()
        .copied()
        .reduce(|a, b| a.min(b))
        .unwrap();
    let max = tri
        .positions
        .iter()
        .copied()
        .reduce(|a, b| a.max(b))
        .unwrap();
    Volume { min, max }.max_length()
}

fn build(bvh: &mut Bvh, triangles: &mut [Triangle]) -> usize {
    let volume = bounding_volume(triangles);

    let max_tri_len = triangles
        .iter()
        .map(triangle_max_length)
        .fold(0.0f32, f32::max);

    if volume.max_length() <= 2.0 * max_tri_len || triangles.len() == 1 {
        let start = bvh.triangles.len();
        bvh.triangles.extend_from_slice(triangles);
        let end = bvh.triangles.len();
        let node = bvh.nodes.len();
        bvh.nodes.push(Node::Leaf {
            triangles: (start..end).into(),
        });
        return node;
    }

    let axis = volume.longest_axis();
    let mid = (volume.min[axis] + volume.max[axis]) / 2.0;

    let mut split = 0;
    for i in 0..triangles.len() {
        if triangles[i].centroid()[axis] < mid {
            triangles.swap(split, i);
            split += 1;
        }
    }

    // If all centroids landed on one side, split evenly to avoid infinite recursion.
    if split == 0 || split == triangles.len() {
        split = triangles.len() / 2;
    }

    // Push a placeholder so children get higher indices.
    let node = bvh.nodes.len();
    bvh.nodes.push(Node::Internal {
        volume,
        children: [0, 0],
    });

    let (left, right) = triangles.split_at_mut(split);
    let left_child = build(bvh, left);
    let right_child = build(bvh, right);

    bvh.nodes[node] = Node::Internal {
        volume,
        children: [left_child, right_child],
    };

    node
}
