from cfg import *
import jax.numpy as jnp


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
        macroscopic_velocities, axis=-1, ord=2
    )
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
    return equilibrium_discrete_velocities
# Obstacle mask
# Define mesh


x = jnp.arange(Nx)
y = jnp.arange(Ny)
X, Y = jnp.meshgrid(x, y, indexing="ij")

obstacle_mask = (
    jnp.sqrt(
        (
            X
            -
            Cylinder_cx
        )**2
        +
        (
            Y
            -
            Cylinder_cy
        )**2
    )
    <
    Cylinder_ri
)
