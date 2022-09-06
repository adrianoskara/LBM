import jax.numpy as jnp

filename = "cfg.txt"
fileobj = open(filename)
params = {}
for line in fileobj:
    line = line.strip()
    data = line.split(" = ")
    params[data[0]] = data[1]


Nx = int(params['Nx'])
Ny = int(params['Ny'])
N_ITERATION = int(params['N_ITERATION'])
Re = int(params['Re'])

Cylinder_cx = Nx // 5
Cylinder_cy = Ny // 2
Cylinder_ri = Ny // 9

Inflow_vel = float(params['Inflow_vel'])

V_visualise = bool(params['V_visualise'])
V_vel_mag = bool(params['V_vel_mag'])
V_vort_mag = bool(params['V_vort_mag'])

V_freq = int(params['V_freq'])
V_skip = int(params['V_skip'])

D_Velocities = 9

L_Velocities = jnp.array([
        [0, 1, 0, -1, 0, 1, -1, -1, 1, ],
        [0, 0, 1, 0, -1, 1, 1, -1, -1, ]
])

x = L_Velocities.shape
print(x)

L_Indices = jnp.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8,
])

O_L_Indices = jnp.array([
    0, 3, 4, 1, 2, 7, 8, 5, 6,
])

L_Weight = jnp.array([
    4/9,                        # Center velocity
    1/9, 1/9, 1/9, 1/9,         # Axis-Aligned [1,2,3,4]
    1/36, 1/36, 1/36, 1/36      # 45 defg [5,6,7,8]
])
LEFT_VELOCITIES = jnp.array([3, 6, 7])
RIGHT_VELOCITIES = jnp.array([1, 5, 8])
UP_VELOCITIES = jnp.array([2, 5, 6])
DOWN_VELOCITIES = jnp.array([4, 7, 8])

Vertical_Velocities = jnp.array([0, 2, 4])
Horizontal_Velocities = jnp.array([0, 1, 3])

kinematic_viscosity = (
        Inflow_vel
        *
        Cylinder_ri
        /
        Re
    )
relaxation_omega = (
    (
        1.0
    ) /
    (
        3.0
        *
        kinematic_viscosity
        + 0.5
    )
)


