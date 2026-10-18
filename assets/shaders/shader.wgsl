const GAMMA: f32 = 1.4;

struct Params {
    cell_count: u32,
    courant: f32
}

@group(0) @binding(0) var<storage, read_write> states: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.cell_count {return;}

    //default to Sod's problem:
    //Left state: rho = 1, u = 0, p = 1
    //Right state: rho = 0.125, u = 0, p = 0.1
    if index < (params.cell_count >> 1) {
        states[index] = vec4<f32>(1.0, 0.0, 1.0, 0.0);
    } else {
        states[index] = vec4<f32>(0.125, 0.0, 0.1, 0.0);
    }
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.cell_count {return;}
    //do nothing for now   
    states[index].w = 1.0;
}