use bevy::{
    app::ScheduleRunnerPlugin, prelude::*, /*time::common_conditions::on_timer,*/
    window::ExitCondition, winit::WinitPlugin,
};
use bevy_shader_plugin::prelude::*;
use std::time::Duration;

const SHADER_PATH: &str = "shaders/shader.wgsl";
const CELL_COUNT: u32 = 2048;
const BUFFER_SIZE: u32 = CELL_COUNT + 2; //1 ghost cell on each edge
const COURANT: f32 = 0.5;
//const REDOX_WG_SIZE: u32 = 256;
const NUM_REDOX_WG: u32 = 256;
const T_MAX: f32 = 0.2;
//name tags
struct States1;
struct States2;
struct Partials;
struct DecodedBuffer;
struct TimeBuffer;

#[derive(Resource, ShaderType, Clone, Default)]
struct Params {
    cell_count: u32,
    courant: f32,
    dx: f32,
    t_max: f32
}

#[derive(ShaderType, Clone, Copy, Default)]
struct Decoded {
    q: Vec3,
    rho: f32,
    u: f32,
    p: f32,
    h: f32,
    a: f32,
    euler: Vec3,
}

#[derive(Resource)]
struct TimeData {
    done: bool,
    state_num: usize
}

fn main() {
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
        .add_plugins(ComputeShaderPlugin::new(SHADER_PATH)
            .init_pass("init", Workgroups::over::<States1>())
            .pass("decode_pipe", Workgroups::over::<States1>())
            .pass("redox_pass_1", Workgroups::exact(uvec3(NUM_REDOX_WG,1,1)))
            .pass("redox_pass_2", Workgroups::exact(uvec3(1, 1, 1)))
            .pass("update", Workgroups::over::<States1>())
            .pass("apply_boundaries", Workgroups::exact(uvec3(1, 1, 1)))
        )
        //buffers
        //group 0
        .register_gpu_buffer::<States1, Vec4>(0, 0, BUFFER_SIZE)
        .read_back_gpu_buffer::<States1, Vec4>(ReadbackMode::OnRequest)
        .register_gpu_buffer::<States2, Vec4>(0, 1, BUFFER_SIZE)
        .read_back_gpu_buffer::<States2, Vec4>(ReadbackMode::OnRequest)
        .register_gpu_buffer::<DecodedBuffer, Decoded>(0, 2, BUFFER_SIZE)
        .register_gpu_buffer::<Partials, f32>(0, 3, NUM_REDOX_WG)
        //group 1
        .register_uniform_buffer::<Params>(1, 0)
        .insert_resource(Params {
            cell_count: CELL_COUNT,
            courant: COURANT,
            dx: 1.0/ CELL_COUNT as f32,
            t_max: T_MAX
        })
        .register_gpu_buffer::<TimeBuffer, f32>(1, 1, 3)
        .read_back_gpu_buffer::<TimeBuffer, f32>(ReadbackMode::EveryFrame)
        //.read_back_gpu_buffer::<Partials, f32>(ReadbackMode::EveryFrame)

        //systems
        .add_systems(
            Update,
                stop_simulation
        )
        .insert_resource(TimeData {
            done: false,
            state_num: 0
        })
        .run();
}

//stops the simulation once the time limit or convergence has been reached
fn stop_simulation(
    total_time: Res<GpuReadback<TimeBuffer, f32>>,
    states1: Res<GpuReadback<States1, Vec4>>,
    states2: Res<GpuReadback<States2, Vec4>>,   
    mut time_data: ResMut<TimeData>
) {
    if time_data.done {
        if states1.is_changed() || states2.is_changed() {
            if let Some(_data) = match time_data.state_num {
                1 => states1.latest(),
                2 => states2.latest(),
                _ => None
            } {
                println!("Data retrieved!");
                std::process::exit(0);      
            } else {
                println!("Waiting for results.");
            }
        }
    } else if let Some(time) = total_time.latest() {
        if time[0] >= T_MAX {
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