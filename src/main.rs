#![allow(non_snake_case)]

use bevy::{
    app::ScheduleRunnerPlugin, prelude::*, /*time::common_conditions::on_timer,*/
    window::ExitCondition, winit::WinitPlugin,
};
use bevy_shader_plugin::prelude::*;
use std::time::{Duration, Instant};

const SHADER_PATH: &str = "shaders/shader.wgsl";
const COURANT: f32 = 0.9;
//const REDOX_WG_SIZE: u32 = 256;
const NUM_REDOX_WG: u32 = 256;
const T_MAX: f32 = 6.9541e-4; //seconds; matches t = 0.2 in the unitless Sod problem
const SUBSTEPS: u32 = 10;
const GAMMA: f32 = 1.4;

//name tags
struct States1;
struct States2;
struct Partials;
struct DecodedBuffer;
///[0] = elapsed time, [1] = frame count, [2] = error code, [3] = finished flag
struct TimeBuffer;


#[derive(Resource, ShaderType, Clone, Default)]
struct Params {
    courant: f32,
    t_max: f32,
    total_cells: u32,
    ///Pa
    atmos_p: f32,
    ///Celcius
    atmos_T: f32,
    #[shader(size(runtime))]
    pipes: Box<[Pipe]>,
}

#[derive(ShaderType, Clone, Copy, Default)]
struct Decoded {
    q: Vec3,
    rho: f32,
    u: f32,
    p: f32,
    h: f32,
    euler: Vec3,
    a: f32,
}

#[derive(Resource)]
struct TimeData {
    done: bool,
    state_num: usize,
    timer: Instant,
}

#[derive(ShaderType, Clone, Copy, Default)]
struct CellData {
    q1: f32,
    q2: f32,
    q3: f32,
    radius: f32,
    pipe_id: u32,
}

enum Boundary {
    ///1
    Transmissive,
    ///2
    Closed,
    ///3
    Exhaust
}

#[derive(ShaderType, Clone, Default)]
struct Pipe {
    ///Real cell count
    cell_count: u32,
    ///meters
    length: f32,
    dx: f32,
    ///density, velocity, pressure
    left_init_state: Vec3,
    ///meters
    radius: f32,
    ///density, velocity, pressure
    right_init_state: Vec3,
    start_index: u32,
    boundaries: [u32; 2]
}

impl Pipe {
    ///Creates a new Pipe instance; length and radius are in meters
    fn new(
        radius: f32,
        cell_count: u32,
        length: f32,
        start_index: u32,
        left_init_state: Vec3,
        right_init_state: Vec3,
        left_boundary: Boundary,
        right_boundary: Boundary
    ) -> Self {
        let left = match left_boundary {
            Boundary::Transmissive => 1,
            Boundary::Closed => 2,
            Boundary::Exhaust => 3
        };
        let right = match right_boundary {
            Boundary::Transmissive => 1,
            Boundary::Closed => 2,
            Boundary::Exhaust => 3
        };
        Self {
            cell_count,
            length,
            dx: length / cell_count as f32,
            radius,
            left_init_state,
            right_init_state,
            start_index,
            boundaries: [left, right]
        }
    }
}

struct PipeStore {
    pipes: Vec<Pipe>,
    total_cells: u32,
}

impl PipeStore {
    fn add_pipe(
        &mut self,
        radius: f32,
        cell_count: u32,
        length: f32,
        left_init_state: Vec3,
        right_init_state: Vec3,
        left_boundary: Boundary,
        right_boundary: Boundary
    ) {
        let start_index = self.total_cells;
        self.pipes.push(Pipe::new(
            radius,
            cell_count,
            length,
            start_index,
            left_init_state,
            right_init_state,
            left_boundary,
            right_boundary
        ));
        self.total_cells += cell_count + 2; //also include two ghost cells for each pipe
    }
}

impl Default for PipeStore {
    fn default() -> Self {
        Self {
            pipes: Vec::new(),
            total_cells: 0,
        }
    }
}

fn main() {
    let mut pipe_store = PipeStore::default();
    //add sod's problem pipe in different resolutions; left is sea-level air, in kg/m^3, m/s and Pa, with Sod's 8:1 density and 10:1 pressure ratios
    pipe_store.add_pipe(
        1.0 / 100.0,
        5000,
        1.0,
        Vec3::new(1.225, 0.0, 101325.0),
        Vec3::new(0.153125, 0.0, 10132.5),
        Boundary::Transmissive,
        Boundary::Transmissive
    );

    pipe_store.add_pipe(
        1.0 / 100.0,
        2500,
        1.0,
        Vec3::new(1.225, 0.0, 101325.0),
        Vec3::new(0.153125, 0.0, 10132.5),
        Boundary::Transmissive,
        Boundary::Transmissive
    );

    pipe_store.add_pipe(
        1.0 / 100.0,
        1000,
        1.0,
        Vec3::new(1.225, 0.0, 101325.0),
        Vec3::new(0.153125, 0.0, 10132.5),
        Boundary::Transmissive,
        Boundary::Transmissive
    );

    let mut compute =
        ComputeShaderPlugin::new(SHADER_PATH).init_pass("init", Workgroups::over::<States1>());

    for _ in 0..SUBSTEPS {
        compute = compute
            .pass("decode_pipe", Workgroups::over::<States1>())
            .pass("redox_pass_1", Workgroups::exact(uvec3(NUM_REDOX_WG, 1, 1)))
            .pass("redox_pass_2", Workgroups::exact(uvec3(1, 1, 1)))
            .pass("update", Workgroups::over::<States1>())
            //TODO: If apply_boundaries is wasting too many threads, refactor to use full size 64 workgroups
            .pass("apply_boundaries", Workgroups::exact(uvec3(pipe_store.pipes.len() as u32, 1, 1))); //1 workgroup per pipe
    }

    App::new()
        //make it headless
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    close_when_requested: false,
                    ..Default::default()
                })
                .disable::<WinitPlugin>(),
            //Uncapped calculation rate
            ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        ))
        //set up compute shaders
        .add_plugins(
            compute, /*
                        ComputeShaderPlugin::new(SHADER_PATH)
                            .init_pass("init", Workgroups::over::<States1>())
                            .pass("decode_pipe", Workgroups::over::<States1>())
                            .pass("redox_pass_1", Workgroups::exact(uvec3(NUM_REDOX_WG, 1, 1)))
                            .pass("redox_pass_2", Workgroups::exact(uvec3(1, 1, 1)))
                            .pass("update", Workgroups::over::<States1>())
                            .pass("apply_boundaries", Workgroups::exact(uvec3(1, 1, 1))),
                    */
        )
        //buffers
        //group 0
        .register_gpu_buffer::<States1, CellData>(0, 0, pipe_store.total_cells)
        .read_back_gpu_buffer::<States1, CellData>(ReadbackMode::OnRequest)
        .register_gpu_buffer::<States2, CellData>(0, 1, pipe_store.total_cells)
        .read_back_gpu_buffer::<States2, CellData>(ReadbackMode::OnRequest)
        .register_gpu_buffer::<DecodedBuffer, Decoded>(0, 2, pipe_store.total_cells)
        .register_gpu_buffer::<Partials, f32>(0, 3, NUM_REDOX_WG)
        //group 1
        .register_storage_buffer::<Params>(1, 0, false)
        .insert_resource(Params {
            courant: COURANT,
            t_max: T_MAX,
            total_cells: pipe_store.total_cells,
            pipes: pipe_store.pipes.into_boxed_slice(),
            atmos_p: 101325.0,
            atmos_T: 20.0
        })
        .register_gpu_buffer::<TimeBuffer, f32>(1, 1, 4)
        .read_back_gpu_buffer::<TimeBuffer, f32>(ReadbackMode::EveryFrame)
        //.read_back_gpu_buffer::<Partials, f32>(ReadbackMode::EveryFrame)
        //systems
        .add_systems(Update, stop_simulation)
        .insert_resource(TimeData {
            done: false,
            state_num: 0,
            timer: Instant::now(),
        })
        .run();
}

//stops the simulation once the time limit has been reached
fn stop_simulation(
    total_time: Res<GpuReadback<TimeBuffer, f32>>,
    states1: Res<GpuReadback<States1, CellData>>,
    states2: Res<GpuReadback<States2, CellData>>,
    mut time_data: ResMut<TimeData>,
    params: Res<Params>
) {
    if time_data.done {
        if states1.is_changed() || states2.is_changed() {
            if let Some(data) = match time_data.state_num {
                1 => states1.latest(),
                2 => states2.latest(),
                _ => None,
            } {
                println!("Data retrieved!");

                //pack data into pipes;
                let mut pipe_data: Vec<Vec<Vec3>> = Vec::new();
                for pipe in params.pipes.iter() { //don't include ghost cells
                    let start = pipe.start_index as usize + 1;
                    let end = pipe.start_index as usize + pipe.cell_count as usize;
                    pipe_data.push(data[start..=end]
                        .iter()
                        .map(|cell| {
                                let rho = cell.q1;
                                let u = cell.q2 / rho;
                                let p = (GAMMA - 1.0) * (cell.q3 - 0.5 * rho * u * u);
                                Vec3::new(rho, u, p)
                        })
                        .collect()
                    );
                }

                std::process::exit(0);
            } else {
                println!("Waiting for results.");
            }
        }
    } else if let Some(time) = total_time.latest() {
        if time[3] != 0.0 {
            let elapsed = time_data.timer.elapsed();
            let frames = time[1];
            println!("Elapsed time: {:.2?}", elapsed);
            println!("Average frame time: {:.6?}", elapsed / (frames as u32).max(1));
            println!("Average fps: {:.2?}", frames as f32 / elapsed.as_secs_f32());

            if time[2] != 0.0 {
                eprintln!("Error code: {}", time[2]);
            } else {
                println!("Finished at t = {} seconds in {} cycles", time[0], time[1]);
            }
            time_data.done = true;
            if time[1] % 2.0 == 0.0 {
                time_data.state_num = 1;
                states1.request();
            } else {
                time_data.state_num = 2;
                states2.request();
            }
        }
    }
}
