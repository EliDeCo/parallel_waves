const GAMMA: f32 = 1.4;
const WG_SIZE: u32 = 64;
//specifically for timestep reduction
const REDOX_WG_SIZE: u32 = 256; 
const NUM_REDOX_WG: u32 = 256;
const TOTAL_THREADS: u32 = REDOX_WG_SIZE * NUM_REDOX_WG;
const F32_MAX: f32 = 3.4028235e+38; 

struct Params {
    cell_count: u32,
    courant: f32,
    dx: f32,
    t_max: f32
}

struct Decoded {
    q: vec3f,     //conserved variables (rho, rho*u, E)
    rho: f32,    //density
    u: f32,      //velocity
    p: f32,      //pressure
    h: f32,      //specific total enthalpy
    a: f32,      //speed of sound
    euler: vec3f, //physical euler flux
}

//if step count % 2 == 0, read from states1 and write to states2; otherwise do the opposite
//first 3 are state values, last f32 is pipe radius
@group(0) @binding(0) var<storage, read_write> states1: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> states2: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> decoded: array<Decoded>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;
@group(1) @binding(1) var<storage, read_write> time_buffer: array<f32>;

//Compiler wont allow using REDOX_WG_SIZE here, make sure this matches ALWAYS
var<workgroup> scratch: array<f32, 256>;
//compiler won't allow using WG_SIZE + 1 here
var<workgroup> fluxes: array<vec3f, 65>;

@compute @workgroup_size(WG_SIZE)
fn init(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if index >= params.cell_count + 2 {return;} //include ghosts in init

    //default to Sod's problem:
    //Left state: rho = 1, u = 0, p = 1
    //Right state: rho = 0.125, u = 0, p = 0.1
    let s = select(
        vec4f(0.125, 0.0, 0.1 / (GAMMA - 1.0), 0.0),  // (ρ, ρu, E) right
        vec4f(1.0,   0.0, 1.0 / (GAMMA - 1.0), 0.0),  // (ρ, ρu, E) left
        index <= (params.cell_count >> 1),
    );

    states1[index] = s;
    states2[index] = s;

    //ensure total time elapsed and total cycles are properly initialized
    if index == 0 {
        time_buffer[0] = 0;
        time_buffer[1] = 0;
        time_buffer[2] = 0;
    }

}

@compute @workgroup_size(WG_SIZE)
fn update(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,  
) {
    let i = gid.x + 1; // interior cell index (1 ghost cell at each end)
    
    
    let dt = partials[0];

    //flux stuff happens here
    
    let n = params.cell_count;

    // Each thread computes the face on its left; the last thread also does the right edge.
    fluxes[lid.x] = hllc(min(i-1,n), GAMMA); // F_{i-1/2}
    if lid.x == WG_SIZE - 1 {
        fluxes[WG_SIZE] = hllc(min(i,n), GAMMA); // F_{i+1/2}
    }
    workgroupBarrier();
    if i > params.cell_count || partials[0] == 0.0 { return; }
    
    let old_state = select(states2[i],states1[i],time_buffer[1] % 2 == 1).xyz;
    let new_state = old_state - (dt / params.dx) * (fluxes[lid.x + 1u] - fluxes[lid.x]);
    //write to the correct buffer (this will never branch)
    if time_buffer[1] % 2 == 1 {
        states2[i].x = new_state.x;
        states2[i].y = new_state.y;
        states2[i].z = new_state.z;
    } else {
        states1[i].x = new_state.x;
        states1[i].y = new_state.y;
        states1[i].z = new_state.z;
    }
}

@compute @workgroup_size(2)
fn apply_boundaries(
    @builtin(global_invocation_id) gid: vec3<u32>, 
) {
    let n = params.cell_count;
    let ghost = select(n + 1, 0u, gid.x == 0);
    let edge = select(n, 1u, gid.x == 0);

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
        h,
        sqrt(gamma * p / rho),
        euler_flux(rho, u, p , h)
    );
}

//gets the local timestep for this cell
fn get_timestep(index: u32) -> f32 {
    return params.courant * params.dx / (abs(decoded[index].u) + decoded[index].a);
}

//reduce all cells to just 256 partials
@compute @workgroup_size(REDOX_WG_SIZE)
fn redox_pass_1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    if time_buffer[0] >= params.t_max { return; }
    let n = params.cell_count;

    //grid stride loop
    var local_min = F32_MAX;
    for (var i = gid.x + 1; i <= n; i += TOTAL_THREADS) {
        let v = get_timestep(i);
        local_min = min(v, local_min);
    }

    scratch[lid.x] = local_min;
    workgroupBarrier();

    var stride = REDOX_WG_SIZE >> 1;
    while stride > 0 {
        if lid.x < stride {
            scratch[lid.x] = min(scratch[lid.x], scratch[lid.x + stride]);
        }

        workgroupBarrier();
        stride >>= 1;
    }

    if lid.x == 0 {
        partials[wid.x] = scratch[0];
    }
}

//reduces all 256 partials with a single workgroup to get the final minimum
@compute @workgroup_size(REDOX_WG_SIZE)
fn redox_pass_2(@builtin(local_invocation_id) lid: vec3<u32>) {
    if time_buffer[0] >= params.t_max { 
        if lid.x == 0 { partials[0] = 0; } //set dt to 0 when finished
        return; 
    }
    scratch[lid.x] = partials[lid.x];
    workgroupBarrier();

    var stride = REDOX_WG_SIZE >> 1;
    while stride > 0 {
        if lid.x < stride {
            scratch[lid.x] = min(scratch[lid.x], scratch[lid.x + stride]);
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

        if (t + dt) > params.t_max {
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
fn raise_error(error_id: f32) {
    time_buffer[0] = params.t_max;  //stop all passes
    time_buffer[2] = error_id;           //error flag for the CPU
    partials[0] = 0.0;    
}

//Decodes the entire pipe and stores it in memory
@compute @workgroup_size(WG_SIZE)
fn decode_pipe(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let index = gid.x;
    if index >= params.cell_count + 2 || time_buffer[0] >= params.t_max { return; }

    let d = decode_all(select(states2[index],states1[index],time_buffer[1] % 2 == 0).xyz, GAMMA);


    if !(d.rho > 0.0) {
        raise_error(3);
    } else if !(d.p > 0.0) {
        raise_error(4);
    }

    decoded[index] = d;
}