mod bvh;
mod path_tracer;

use crate::bvh::Bvh;

use eframe::{
    egui_wgpu::wgpu::util::DeviceExt as _,
    egui_wgpu::{self, wgpu},
    egui,
};
use egui_file_dialog::FileDialog;
use obj::{load_obj, Obj};
use glam::{Vec3, Mat4};

use std::path::PathBuf;
use std::io::BufReader;
use std::fs::File;
use std::sync::{Mutex, Arc};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view: [f32; 16],
    proj: [f32; 16],
}

struct ModelGeometry {
    /// This acts as the ID of the model.
    path: PathBuf,
    index_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_count: u32,
}

impl ModelGeometry {
    fn new(device: &wgpu::Device, model: &Model) -> Self {
        let vertices: Vec<Vertex> = model
            .obj
            .vertices
            .iter()
            .map(|vertex| Vertex {
                position: vertex.position,
                normal: vertex.normal,
            })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview index buffer"),
            contents: bytemuck::cast_slice(&model.obj.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            path: model.path.clone(),
            index_buffer,
            vertex_buffer,
            index_count: model.obj.indices.len() as u32,
        }
    }
}

struct Preview {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    geometry: Mutex<Option<ModelGeometry>>,
}

impl Preview {
    const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint16;

    const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    };

    fn new(render_state: &egui_wgpu::RenderState) -> Self {
        let device = &render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preview shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview uniforms"),
            size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Self::VERTEX_LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(render_state.target_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            uniform_buffer,
            geometry: Mutex::new(None),
        }
    }
}

struct Model {
    path: PathBuf,
    bvh: Bvh,
    obj: Obj,
    /// Bounding sphere of the mesh, used to frame the camera.
    center: Vec3,
    radius: f32,
}

impl Model {
    fn from_obj(path: PathBuf, obj: Obj) -> Self {
        let positions: Vec<Vec3> = obj
            .indices
            .iter()
            .map(|index| {
                Vec3::from_array(obj.vertices[*index as usize].position)
            })
            .collect();
        let bvh = Bvh::new(&positions);

        let mut min = Vec3::INFINITY;
        let mut max = Vec3::NEG_INFINITY;

        for position in &positions {
            min = min.min(*position);
            max = max.max(*position);
        }

        let center = (min + max) * 0.5;
        let radius = (max - center).length().max(f32::MIN_POSITIVE);

        Self { path, bvh, obj, center, radius }
    }
}

struct ModelCallback {
    model: Arc<Model>,
    proj: Mat4,
    view: Mat4,
}

impl egui_wgpu::CallbackTrait for ModelCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let preview: &Preview = resources.get().unwrap();

        let uniforms = Uniforms {
            view: self.view.to_cols_array(),
            proj: self.proj.to_cols_array(),
        };
        queue.write_buffer(&preview.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut geometry = preview.geometry.lock().unwrap();
        let stale = geometry
            .as_ref()
            .is_none_or(|geometry| geometry.path != self.model.path);
        if stale {
            *geometry = Some(ModelGeometry::new(device, &self.model));
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let preview: &Preview = resources.get().unwrap();

        let geometry = preview.geometry.lock().unwrap();
        let Some(geometry) = geometry.as_ref() else {
            return;
        };

        render_pass.set_pipeline(&preview.pipeline);
        render_pass.set_bind_group(0, &preview.bind_group, &[]);
        render_pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
        render_pass.set_index_buffer(geometry.index_buffer.slice(..), Preview::INDEX_FORMAT);
        render_pass.draw_indexed(0..geometry.index_count, 0, 0..1);
    }
}

struct Viewer {
    file_dialog: FileDialog,
    error: Option<String>,
    model: Option<Arc<Model>>,
    material: openpbr::Material,
    /// Orbit camera angles in radians, driven by dragging the preview.
    yaw: f32,
    pitch: f32,
}

impl Viewer {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("the wgpu backend must be enabled");
        let preview = Preview::new(render_state);
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(preview);

        Self {
            file_dialog: FileDialog::new(),
            material: openpbr::Material::default(),
            error: None,
            model: None,
            yaw: 0.6,
            pitch: 0.4,
        }
    }

    fn show_preview(&mut self, ui: &mut egui::Ui, model: &Arc<Model>) {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        let drag = response.drag_delta();
        self.yaw += drag.x * 0.01;
        self.pitch = (self.pitch + drag.y * 0.01)
            .clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);

        let fov = 45f32.to_radians();
        let distance = model.radius / (fov * 0.5).sin() * 1.2;
        let direction = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );
        let eye = model.center + direction * distance;

        let view = Mat4::look_at_rh(eye, model.center, Vec3::Y);
        let aspect = (rect.width() / rect.height()).max(f32::MIN_POSITIVE);
        let near = (distance - model.radius).max(distance * 0.01);
        let far = distance + model.radius;
        let proj = Mat4::perspective_rh(fov, aspect, near, far);

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            ModelCallback { model: Arc::clone(model), proj, view },
        ));
    }
}

fn main() -> eframe::Result {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        depth_buffer: 32,
        ..Default::default()
    };

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|cc| Ok(Box::new(Viewer::new(cc)))),
    )
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("control_panel").show(ui, |ui| {
            if ui.button("Load Object").clicked() {
                self.error = None;
                self.file_dialog.pick_file();
            }

            self.file_dialog.update(ui.ctx());

            if let Some(path) = self.file_dialog.take_picked() {
                let path = path.to_path_buf();

                if let Ok(file) = File::open(&path) {
                    match load_obj(BufReader::new(file)) {
                        Ok(obj) => self.model = Some(Arc::new(Model::from_obj(path.clone(), obj))),
                        Err(err) => self.error = Some(err.to_string()),
                    }
                }
            }

            if let Some(error) = &self.error {
                ui.label(format!("Failed to load obj: {error}"));
            }

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                let m = &mut self.material;

                ui.collapsing("Base", |ui| {
                    ui.add(egui::Slider::new(&mut m.base_weight, 0.0..=1.0).text("Weight"));
                    ui.horizontal(|ui| {
                        let mut c = m.base_color.to_array();
                        ui.color_edit_button_rgb(&mut c);
                        ui.label("Color");
                        m.base_color = Vec3::from_array(c);
                    });
                    ui.add(egui::Slider::new(&mut m.base_diffuse_roughness, 0.0..=1.0).text("Diffuse Roughness"));
                    ui.add(egui::Slider::new(&mut m.base_metalness, 0.0..=1.0).text("Metalness"));
                });

                ui.collapsing("Specular", |ui| {
                    ui.add(egui::Slider::new(&mut m.specular_weight, 0.0..=1.0).text("Weight"));
                    ui.horizontal(|ui| {
                        let mut c = m.specular_color.to_array();
                        ui.color_edit_button_rgb(&mut c);
                        ui.label("Color");
                        m.specular_color = Vec3::from_array(c);
                    });
                    ui.add(egui::Slider::new(&mut m.specular_ior, 1.0..=3.0).text("IOR"));
                    ui.add(egui::Slider::new(&mut m.specular_roughness, 0.0..=1.0).text("Roughness"));
                    ui.add(egui::Slider::new(&mut m.specular_roughness_anisotropy, 0.0..=1.0).text("Anisotropy"));
                    ui.add(egui::Slider::new(&mut m.specular_rotation, 0.0..=1.0).text("Rotation"));
                });

                ui.collapsing("Transmission", |ui| {
                    ui.add(egui::Slider::new(&mut m.transmission_weight, 0.0..=1.0).text("Weight"));
                    ui.horizontal(|ui| {
                        let mut c = m.transmission_color.to_array();
                        ui.color_edit_button_rgb(&mut c);
                        ui.label("Color");
                        m.transmission_color = Vec3::from_array(c);
                    });
                    ui.add(egui::Slider::new(&mut m.transmission_depth, 0.0..=10.0).text("Depth"));
                });

                ui.collapsing("Coat", |ui| {
                    ui.add(egui::Slider::new(&mut m.coat_weight, 0.0..=1.0).text("Weight"));
                    ui.horizontal(|ui| {
                        let mut c = m.coat_color.to_array();
                        ui.color_edit_button_rgb(&mut c);
                        ui.label("Color");
                        m.coat_color = Vec3::from_array(c);
                    });
                    ui.add(egui::Slider::new(&mut m.coat_ior, 1.0..=3.0).text("IOR"));
                    ui.add(egui::Slider::new(&mut m.coat_roughness, 0.0..=1.0).text("Roughness"));
                    ui.add(egui::Slider::new(&mut m.coat_roughness_anisotropy, 0.0..=1.0).text("Anisotropy"));
                    ui.add(egui::Slider::new(&mut m.coat_rotation, 0.0..=1.0).text("Rotation"));
                    ui.add(egui::Slider::new(&mut m.coat_darkening, 0.0..=1.0).text("Darkening"));
                });

                ui.collapsing("Fuzz", |ui| {
                    ui.add(egui::Slider::new(&mut m.fuzz_weight, 0.0..=1.0).text("Weight"));
                    ui.horizontal(|ui| {
                        let mut c = m.fuzz_color.to_array();
                        ui.color_edit_button_rgb(&mut c);
                        ui.label("Color");
                        m.fuzz_color = Vec3::from_array(c);
                    });
                    ui.add(egui::Slider::new(&mut m.fuzz_roughness, 0.0..=1.0).text("Roughness"));
                });

                ui.separator();

                if ui.button("Render").clicked() {
                    todo!();
                }
            });
        });

        if let Some(model) = self.model.clone() {
            self.show_preview(ui, &model);
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Load an object to preview it.");
            });
        }
    }
}

const SHADER: &str = r#"
struct Uniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_position: vec3<f32>,
    @location(1) view_normal: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    let view_position = uniforms.view * vec4<f32>(position, 1.0);
    out.view_position = view_position.xyz;
    out.view_normal = (uniforms.view * vec4<f32>(normal, 0.0)).xyz;
    out.clip_position = uniforms.proj * view_position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = vec3<f32>(0.8, 0.8, 0.85);

    // Fall back to a flat (per-face) normal if the mesh has no vertex normals.
    var normal = in.view_normal;
    if (dot(normal, normal) < 1e-8) {
        normal = cross(dpdx(in.view_position), dpdy(in.view_position));
    }
    normal = normalize(normal);

    if (dot(normal, in.view_position) > 0.0) {
        normal = -normal;
    }

    let light = normalize(-in.view_position);
    let half = light;
    let diffuse = max(dot(normal, light), 0.0);
    let specular = pow(max(dot(normal, half), 0.0), 32.0);

    let ambient = 0.15;
    let color = base_color * (ambient + 0.8 * diffuse) + vec3<f32>(0.3 * specular);
    return vec4<f32>(color, 1.0);
}
"#;
