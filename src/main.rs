use bevy::{
    app::ScheduleRunnerPlugin, prelude::*, time::common_conditions::on_timer,
    window::ExitCondition, winit::WinitPlugin,
};
use bevy_shader_plugin::prelude::*;
use std::time::Duration;

const SHADER_PATH: &str = "shaders/shader.wgsl";
const CELL_COUNT: u32 = 2048;
const COURANT: f32 = 0.5;

//name tag
struct States;

#[derive(Resource, ShaderType, Clone, Default)]
struct Params {
    cell_count: u32,
    courant: f32,
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
        .add_plugins(
            ComputeShaderPlugin::new(SHADER_PATH)
                .init_pass("init", Workgroups::over::<States>())
                .pass("update", Workgroups::over::<States>()),
                //may add substeps in the future if still cpu limited
        )
        //buffers
        .register_gpu_buffer::<States, Vec4>(0, 0, CELL_COUNT)
        .read_back_gpu_buffer::<States, Vec4>(ReadbackMode::OnRequest)
        .register_uniform_buffer::<Params>(0, 1)
        .insert_resource(Params {
            cell_count: CELL_COUNT,
            courant: COURANT,
        })
        //systems
        .add_systems(
            Update,
            (
                display_data,
                ask_readback.run_if(on_timer(Duration::from_secs(3))),
            ),
        )
        .run();
}

fn ask_readback(readback: Res<GpuReadback<States, Vec4>>) {
    readback.request();
}

fn display_data(readback: Res<GpuReadback<States, Vec4>>) {
    if !readback.is_changed() {
        return;
    }

    let Some(data) = readback.latest() else {
        return;
    };

    println!("{}", data.len());
}
