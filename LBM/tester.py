import numpy as np

from cfg import *
import jax.numpy as jnp

velocity_profile = jnp.zeros((Nx, Ny, 2))
velocity_profile = velocity_profile.at[:, :, 0].set(Inflow_vel)


def get_eq_d_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = jnp.einsum(
        "dQ,NMd->NMQ",
        L_Velocities,
        macroscopic_velocities,
    )

    macroscopic_velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocities, axis=-1, ord=2
    )
    print(jnp.shape(macroscopic_velocity_magnitude))

    equilibrium_discrete_velocities = (
        density[..., jnp.newaxis]
        *
        L_Weight[jnp.newaxis, jnp.newaxis, :]
        *
        (
            1
            +
            3 * projected_discrete_velocities
            +
            9/2 * projected_discrete_velocities**2
            -
            3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2
        )
    )
    print(equilibrium_discrete_velocities)
    return equilibrium_discrete_velocities


d_velocities_prev = get_eq_d_velocities(velocity_profile, jnp.ones((Nx, Ny)))
