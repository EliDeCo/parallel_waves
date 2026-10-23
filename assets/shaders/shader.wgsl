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

@group(0) @binding(0) var<storage, read_write> states: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> decoded: array<Decoded>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
@group(1) @binding(0) var<uniform> params: Params;
@group(1) @binding(1) var<storage, read_write> time_buffer: array<f32>;

//Compiler wont allow using REDOX_WG_SIZE here, make sure this matches ALWAYS
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(WG_SIZE)
fn init(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if index >= params.cell_count {return;}

    //default to Sod's problem:
    //Left state: rho = 1, u = 0, p = 1
    //Right state: rho = 0.125, u = 0, p = 0.1
    states[index] = select(
        vec4f(0.125, 0.0, 0.1 / (GAMMA - 1.0), 0.0),  // (ρ, ρu, E) right
        vec4f(1.0,   0.0, 1.0 / (GAMMA - 1.0), 0.0),  // (ρ, ρu, E) left
        index < (params.cell_count >> 1),
    );

    //ensure total time elapsed and total cycles are properly initialized
    time_buffer[0] = 0;
    time_buffer[1] = 0;
}

@compute @workgroup_size(WG_SIZE)
fn update(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let index = gid.x;
    if index >= params.cell_count || time_buffer[0] >= params.t_max { return; }

    //flux stuff happens here
    let dt = partials[0];


    //temporarily use this to send back data for debugging
    
    states[index].w = dt; //sending total runtime
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
    for (var i = gid.x; i < n; i += TOTAL_THREADS) {
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
    if time_buffer[0] >= params.t_max { return; }
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
        if (t + dt) > params.t_max {
            dt = params.t_max - t;
        }
        t += dt;

        time_buffer[0] = t; //update total elapsed time
        partials[0] = dt; //update dt
        time_buffer[1] += 1; //update cycle counter
    }
}

//Decodes the entire pipe and stores it in memory
@compute @workgroup_size(WG_SIZE)
fn decode_pipe(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let index = gid.x;
    if index >= params.cell_count || time_buffer[0] >= params.t_max { return; }

    decoded[index] = decode_all(states[index].xyz, GAMMA);
}