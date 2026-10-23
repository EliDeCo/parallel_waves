use bevy::{
    app::ScheduleRunnerPlugin, prelude::*, /*time::common_conditions::on_timer,*/
    window::ExitCondition, winit::WinitPlugin,
};
use bevy_shader_plugin::prelude::*;
use std::time::Duration;

const SHADER_PATH: &str = "shaders/shader.wgsl";
const CELL_COUNT: u32 = 2048;
const COURANT: f32 = 0.5;
//const REDOX_WG_SIZE: u32 = 256;
const NUM_REDOX_WG: u32 = 256;
const T_MAX: f32 = 0.3;
//name tags
struct States;
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
    done: bool
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
            .init_pass("init", Workgroups::over::<States>())
            .pass("decode_pipe", Workgroups::over::<States>())
            .pass("redox_pass_1", Workgroups::exact(uvec3(NUM_REDOX_WG,1,1)))
            .pass("redox_pass_2", Workgroups::exact(uvec3(1, 1, 1)))
            .pass("update", Workgroups::over::<States>())
        )
        //buffers
        .register_gpu_buffer::<States, Vec4>(0, 0, CELL_COUNT)
        .read_back_gpu_buffer::<States, Vec4>(ReadbackMode::OnRequest)
        .register_uniform_buffer::<Params>(1, 0)
        .insert_resource(Params {
            cell_count: CELL_COUNT,
            courant: COURANT,
            dx: 1.0/ CELL_COUNT as f32,
            t_max: T_MAX
        })
        .register_gpu_buffer::<TimeBuffer, f32>(1, 1, 2)
        .read_back_gpu_buffer::<TimeBuffer, f32>(ReadbackMode::EveryFrame)
        .register_gpu_buffer::<Partials, f32>(0, 2, NUM_REDOX_WG)
        //.read_back_gpu_buffer::<Partials, f32>(ReadbackMode::EveryFrame)
        .register_gpu_buffer::<DecodedBuffer, Decoded>(0, 1, CELL_COUNT)
        //systems
        .add_systems(
            Update,
                stop_simulation
        )
        .insert_resource(TimeData {
            done: false
        })
        .run();
}

//stops the simulation once the time limit or convergence has been reached
fn stop_simulation(
    total_time: Res<GpuReadback<TimeBuffer, f32>>,
    states: Res<GpuReadback<States, Vec4>>,
    mut time_data: ResMut<TimeData>
) {
    if time_data.done {
        if states.is_changed() && let Some(_final_state) = states.latest() {
            println!("Data retrieved!");
            std::process::exit(0);
        } else {
            println!("Waiting for results.");
        }
    } else {
        if let Some(time) = total_time.latest() {
            //println!("Elapsed time: {}", time[0]);
            if time[0] >= T_MAX {
                println!("Finished at t = {} seconds in {} cycles", time[0], time[1]);
                states.request();
                time_data.done = true;
            }
        }
    }

}