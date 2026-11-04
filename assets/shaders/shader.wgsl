const GAMMA: f32 = 1.4;
const WG_SIZE: u32 = 64;
//specifically for timestep reduction
const REDOX_WG_SIZE: u32 = 256; 
const NUM_REDOX_WG: u32 = 256;
const TOTAL_THREADS: u32 = REDOX_WG_SIZE * NUM_REDOX_WG;
const F32_MAX: f32 = 3.4028235e+38;
//Convergence mode stops when no cell's density, velocity or pressure has changed by more than
//these amounts (kg/m^3, m/s, Pa) over one window of CONVERGENCE_WINDOW seconds
const CONVERGENCE_TOL: vec3f = vec3f(1e-4, 0.1, 10.0);
const CONVERGENCE_WINDOW: f32 = 0.01;
//The f32 cycle counter stops incrementing at 2^24, so stop before reaching it
const MAX_CYCLES: f32 = 16000000.0;

struct Params {
    courant: f32,
    t_max: f32,
    total_cells: u32,
    pipes: array<Pipe>
}

struct Pipe {
    cell_count: u32,
    length: f32,
    dx: f32,
    ///density, velocity, pressure
    left_init_state: vec3f,
    radius: f32,
    ///density, velocity, pressure
    right_init_state: vec3f,
    start_index: u32,
}


struct Decoded {
    q: vec3f,     //conserved variables (rho, rho*u, E)
    rho: f32,    //density
    u: f32,      //velocity
    p: f32,      //pressure
    checkpoint: vec3f, //(rho, u, p) at the start of this convergence window
    h: f32,      //specific total enthalpy
    euler: vec3f, //physical euler flux
    a: f32,      //speed of sound
}

struct CellData {
    q1: f32,
    q2: f32,
    q3: f32,
    radius: f32,
    pipe_id: u32
}

//if step count % 2 == 0, read from states1 and write to states2; otherwise do the opposite
//first 3 are state values, last f32 is pipe radius
@group(0) @binding(0) var<storage, read_write> states1: array<CellData>;
@group(0) @binding(1) var<storage, read_write> states2: array<CellData>;
@group(0) @binding(2) var<storage, read_write> decoded: array<Decoded>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(0) @binding(4) var<storage, read_write> diff_partials: array<vec3f>;
@group(1) @binding(0) var<storage> params: Params;
@group(1) @binding(1) var<storage, read_write> time_buffer: array<f32>;

//Compiler wont allow using 2*REDOX_WG_SIZE here, make sure this matches ALWAYS
var<workgroup> scratch: array<f32, 256>;
var<workgroup> diff_scratch: array<vec3f, 256>;
//compiler won't allow using WG_SIZE + 1 here
var<workgroup> fluxes: array<vec3f, 65>;

@compute @workgroup_size(WG_SIZE)
fn init(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if index >= params.total_cells {return;}

    for (var i: u32 = 0; i < arrayLength(&params.pipes); i++) {
        let pipe = params.pipes[i];
        if index >= pipe.start_index && index <= (pipe.start_index + pipe.cell_count + 1) {
            //this cell belongs to this pipe
            let half = pipe.cell_count >> 1;
            let state: vec3f = select(pipe.right_init_state, pipe.left_init_state, index <= (pipe.start_index + half));
            states1[index] = CellData(state.x, state.y * state.x, state.z / (GAMMA - 1.0) + 0.5 * state.x * state.y * state.y, pipe.radius, i);
            states2[index] = states1[index];
            break;
        }
    }

    //ensure total time elapsed and total cycles are properly initialized
    if index == 0 {
        time_buffer[0] = 0;
        time_buffer[1] = 0;
        time_buffer[2] = 0;
        time_buffer[3] = 0;
        time_buffer[4] = CONVERGENCE_WINDOW; //end of the first convergence window
        time_buffer[5] = 1; //save the first checkpoint
        partials[0] = 1; //initialize dt to a nonzero number
    }

}

@compute @workgroup_size(WG_SIZE)
fn update(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,  
) {
    let global_index = gid.x;
    let in_bounds = global_index < params.total_cells;
    let pipe = params.pipes[states1[global_index].pipe_id];
    let pipe_index = global_index - pipe.start_index;
    let is_interior = in_bounds && pipe_index >= 1 && pipe_index <= pipe.cell_count;
    let is_right_ghost = in_bounds && pipe_index == pipe.cell_count + 1;

    // The right ghost computes the last interior cell's right face
    if is_interior || is_right_ghost {
        fluxes[lid.x] = hllc(global_index - 1, GAMMA); // F_{i-1/2}
    }
    if is_interior && lid.x == WG_SIZE - 1 {
        fluxes[WG_SIZE] = hllc(global_index, GAMMA);   // F_{i+1/2}
    }
    workgroupBarrier();

    let dt = partials[0];
    if !is_interior || dt == 0.0 { return; }

    let old_state = select(
        vec3f(states2[global_index].q1, states2[global_index].q2, states2[global_index].q3),
        vec3f(states1[global_index].q1, states1[global_index].q2, states1[global_index].q3),
        time_buffer[1] % 2 == 1,
    );

    let new_state = old_state - (dt / pipe.dx) * (fluxes[lid.x + 1] - fluxes[lid.x]);
    



    //write to the correct buffer (this will never branch)
    if time_buffer[1] % 2 == 1 {
        states2[global_index].q1 = new_state.x;
        states2[global_index].q2 = new_state.y;
        states2[global_index].q3 = new_state.z;
    } else {
        states1[global_index].q1 = new_state.x;
        states1[global_index].q2 = new_state.y;
        states1[global_index].q3 = new_state.z;
    }
}


@compute @workgroup_size(2)
fn apply_boundaries(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let pipe = params.pipes[wg_id.x]; //1 workgroup per pipe

    // lid == 0 means left boundary, lid == 1 means right boundary
    let left_ghost = pipe.start_index;
    let right_ghost = left_ghost + pipe.cell_count + 1;
    
    let ghost = select(right_ghost, left_ghost, lid.x == 0);
    let edge = select(right_ghost - 1, left_ghost + 1, lid.x == 0);


    if time_buffer[1] % 2 == 1 {
        states2[ghost] = states2[edge];
    } else {
        states1[ghost] = states1[edge];
    }
}

///Returns the HLLC flux between two decoded states, using the wave speeds of
///Batten, Clarke, Lambert & Causon (1997).
fn hllc(index: u32, gamma: f32) -> vec3f {
    let d_l = decoded[index];
    let d_r = decoded[index+1];

    //(1) Roe averages of velocity, total enthalpy and sound speed
    let k = sqrt(d_r.rho / d_l.rho);
    let roe_u = (k * d_r.u + d_l.u) / (k + 1.0);
    let roe_h = (k * d_r.h + d_l.h) / (k + 1.0);
    let roe_a = sqrt((gamma - 1.0) * (roe_h - 0.5 * roe_u * roe_u));

    //(2) outer wave speeds: the wider of the local and the Roe-averaged estimate
    let s_l = min(d_l.u - d_l.a, roe_u - roe_a);
    let s_r = max(d_r.u + d_r.a, roe_u + roe_a);

    //(3) contact speed: the one value that gives the same star pressure on both sides
    let m_l = d_l.rho * (s_l - d_l.u);
    let m_r = d_r.rho * (s_r - d_r.u);
    let s_m = (d_r.p - d_l.p + m_l * d_l.u - m_r * d_r.u) / (m_l - m_r);

    //(4) which side of the contact the interface (x/t = 0) lies on
    let left = s_m >= 0.0;

    //(5) pick that side's data, field by field
    let q_k = select(d_r.q, d_l.q, left);
    let vel_k = select(d_r.u, d_l.u, left);
    let p_k = select(d_r.p, d_l.p, left);
    let f_k = select(d_r.euler, d_l.euler, left);
    let m_k = select(m_r, m_l, left);
    let s_k = select(s_r, s_l, left);
    
    //(6) outer wave speed clamped to zero: supersonic cases then return F_L or F_R unchanged
    let s_k0 = select(max(s_r, 0.0), min(s_l, 0.0), left);

    //(7) star state next to the contact on side K
    let rho_s = m_k / (s_k - s_m);
    let p_s = p_k + m_k * (s_m - vel_k);
    let e_s = (q_k.z * (s_k - vel_k) - p_k * vel_k + p_s * s_m) / (s_k - s_m);
    let q_s = vec3f(rho_s, rho_s * s_m, e_s);

    //(8) F*_K = F_K + S_K (U*_K - U_K)
    return f_k + s_k0 * (q_s - q_k);
}

///Computes the physical euler flux
fn euler_flux(rho: f32, u: f32, p: f32, h:f32) -> vec3f {
    return vec3f(
        rho * u,                     //mass flux,
        rho * u * u + p,             //momentum flux
        rho * u * h,                 //energy flux
    );
}

///Decodes the conserved variables into primitive variables (ρ, u, p)
fn decode_conserved(q: vec3f, gamma: f32) -> vec4f {
    let rho = q.x; //first component is just density
    let u = q.y / rho; // u = (rho*u)/rho
    let p = (gamma - 1.0) * (q.z - 0.5 * rho * u * u); //p = (γ-1)*(rho*E - 0.5*rho*u^2)
    let h = (q.z + p) / rho; //specific total enthalpy
    return vec4f(rho, u, p, h);
}

//Decodes the conserved variables into primitive variables, sound speed and euler flux
fn decode_all(q: vec3f, gamma: f32) -> Decoded {
    let conserved = decode_conserved(q, gamma);

    let rho = conserved.x;
    let u = conserved.y;
    let p = conserved.z;
    let h = conserved.w;

    return Decoded(
        q,
        rho,
        u,
        p,
        vec3f(0),
        h,
        euler_flux(rho, u, p , h),
        sqrt(gamma * p / rho)
    );
}

//gets the local timestep for this cell
fn get_timestep(index: u32) -> f32 {
    let pipe = params.pipes[states1[index].pipe_id];
    return params.courant * pipe.dx / (abs(decoded[index].u) + decoded[index].a);
}

//reduce all cells to just 256 partials
@compute @workgroup_size(REDOX_WG_SIZE)
fn redox_pass_1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    if partials[0] == 0 { return; }
    let n = params.total_cells;

    //grid stride loop
    var local_min = F32_MAX;
    var local_diff = vec3f(0);
    for (var i = gid.x; i < n; i += TOTAL_THREADS) {
        let v = get_timestep(i);
        local_min = min(v, local_min);
        let primitives = vec3f(decoded[i].rho, decoded[i].u, decoded[i].p);
        local_diff = max(local_diff, abs(primitives - decoded[i].checkpoint));
    }

    scratch[lid.x] = local_min;
    diff_scratch[lid.x] = local_diff;
    workgroupBarrier();

    var stride = REDOX_WG_SIZE >> 1;
    while stride > 0 {
        if lid.x < stride {
            scratch[lid.x] = min(scratch[lid.x], scratch[lid.x + stride]);
            diff_scratch[lid.x] = max(diff_scratch[lid.x], diff_scratch[lid.x + stride]);
        }

        workgroupBarrier();
        stride >>= 1;
    }

    if lid.x == 0 {
        partials[wid.x] = scratch[0];
        diff_partials[wid.x] = diff_scratch[0];
    }
}

//reduces all 256 partials with a single workgroup to get the final minimum
@compute @workgroup_size(REDOX_WG_SIZE)
fn redox_pass_2(@builtin(local_invocation_id) lid: vec3<u32>) {
    //if simulation is already over, end early
    if partials[0] == 0 { return; }

    scratch[lid.x] = partials[lid.x];
    diff_scratch[lid.x] = diff_partials[lid.x];
    workgroupBarrier();

    var stride = REDOX_WG_SIZE >> 1;
    while stride > 0 {
        if lid.x < stride {
            scratch[lid.x] = min(scratch[lid.x], scratch[lid.x + stride]);
            diff_scratch[lid.x] = max(diff_scratch[lid.x], diff_scratch[lid.x + stride]);
        }

        workgroupBarrier();
        stride >>= 1;
    }

    //thread 0 gives final answers
    if lid.x == 0 {
        var dt = scratch[0];
        var t = time_buffer[0];

        if dt != dt {
            raise_error(2);
            return;
        } else if dt >= F32_MAX {
            raise_error(1);
            return;
        } 

        //check if simulation is finished, and set dt to 0 if it is
        time_buffer[5] = 0.0; //the checkpoint only saves on the decode right after a window ends
        var finished = false;
        if params.t_max >= 0.0 { //time mode
            finished = t >= params.t_max;
        } else if t >= time_buffer[4] { //convergence mode, at the end of a window
            finished = all(diff_scratch[0] < CONVERGENCE_TOL);
            time_buffer[4] = t + CONVERGENCE_WINDOW;
            time_buffer[5] = 1.0;
        }
        if finished {
            partials[0] = 0.0;
            time_buffer[3] = 1.0;
            return;
        }

        if time_buffer[1] >= MAX_CYCLES {
            raise_error(5);
            return;
        }

        //only if in timed mode
        if params.t_max >= 0.0 && (t + dt) > params.t_max {
            dt = params.t_max - t;
        }

        t += dt;

        time_buffer[0] = t; //update total elapsed time
        time_buffer[1] += 1; //update cycle counter
        partials[0] = dt; //update dt
    }
}
//1 -> dt == F32_MAX
//2 -> dt = NaN
//3 -> nonphysical density
//4 -> nonphysical pressure
//5 -> cycle limit reached
fn raise_error(error_id: f32) {
    time_buffer[2] = error_id;           //error flag for the CPU
    time_buffer[3] = 1.0;                //tell the CPU the simulation is over
    partials[0] = 0.0;                   //stop all passes
}

//Decodes the entire pipe and stores it in memory
@compute @workgroup_size(WG_SIZE)
fn decode_pipe(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let index = gid.x;
    if index >= params.total_cells //out of bounds
    || partials[0] == 0 //simulation over
    { return; }

    var q: vec3f;
    if time_buffer[1] % 2 == 0 {
        q = vec3f(states1[index].q1, states1[index].q2, states1[index].q3);
    } else {
        q = vec3f(states2[index].q1, states2[index].q2, states2[index].q3);
    }

    var d = decode_all(q, GAMMA);
    //save a new checkpoint at the start of a window, otherwise keep the current one
    if time_buffer[5] != 0.0 {
        d.checkpoint = vec3f(d.rho, d.u, d.p);
    } else {
        d.checkpoint = decoded[index].checkpoint;
    }


    if !(d.rho > 0.0) {
        raise_error(3);
    } else if !(d.p > 0.0) {
        raise_error(4);
    }

    decoded[index] = d;
}