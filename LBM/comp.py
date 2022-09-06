import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

N_ITERATIONS = 15_000
REYNOLDS_NUMBER = 80

N_POINTS_X = 300
N_POINTS_Y = 50

Cylinder_cx = N_POINTS_X // 5
Cylinder_cy = N_POINTS_Y // 2
Cylinder_ri = N_POINTS_Y // 9

MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04

VISUALIZE = True
PLOT_EVERY_N_STEPS = 100
SKIP_FIRST_N_ITERATIONS = 5000

r"""
LBM Grid: D2Q9
    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8 
"""

D_Velocities = 9

L_Velocities = jnp.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1, ],
    [0, 0, 1, 0, -1, 1, 1, -1, -1, ]
])

L_Indices = jnp.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8,
])

OPPOSITE_L_Indices = jnp.array([
    0, 3, 4, 1, 2, 7, 8, 5, 6,
])

L_WeightS = jnp.array([
    4 / 9,  # Center Velocity [0,]
    1 / 9, 1 / 9, 1 / 9, 1 / 9,  # Axis-Aligned Velocities [1, 2, 3, 4]
    1 / 36, 1 / 36, 1 / 36, 1 / 36,  # 45 ° Velocities [5, 6, 7, 8]
])

RIGHT_VELOCITIES = jnp.array([1, 5, 8])
UP_VELOCITIES = jnp.array([2, 5, 6])
LEFT_VELOCITIES = jnp.array([3, 6, 7])
DOWN_VELOCITIES = jnp.array([4, 7, 8])
Vertical_Velocities = jnp.array([0, 2, 4])
Horizontal_Velocities = jnp.array([0, 1, 3])


def get_rho(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis=-1)

    return density


def get_macro_velocities(discrete_velocities, density):
    macroscopic_velocities = jnp.einsum(
        "NMQ,dQ->NMd",
        discrete_velocities,
        L_Velocities,
    ) / density[..., jnp.newaxis]

    return macroscopic_velocities


def get_eq_d_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = jnp.einsum(
        "dQ,NMd->NMQ",
        L_Velocities,
        macroscopic_velocities,
    )
    macroscopic_velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocities,
        axis=-1,
        ord=2,
    )
    equilibrium_discrete_velocities = (
            density[..., jnp.newaxis]
            *
            L_WeightS[jnp.newaxis, jnp.newaxis, :]
            *
            (
                    1
                    +
                    3 * projected_discrete_velocities
                    +
                    9 / 2 * projected_discrete_velocities ** 2
                    -
                    3 / 2 * macroscopic_velocity_magnitude[..., jnp.newaxis] ** 2
            )
    )

    return equilibrium_discrete_velocities


