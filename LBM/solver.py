from Visuals import *
from preliminaries import *
from cfg import *
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():

    jax.config.update("jax_enable_x64", True)

    velocity_profile = jnp.zeros((Nx, Ny, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(Inflow_vel)

    @jax.jit
    def update(d_velocities_prev):
        # (1) Prescribe the outflow BC on the right boundary
        d_velocities_prev = d_velocities_prev.at[-1, :, LEFT_VELOCITIES].set(
            d_velocities_prev[-2, :, LEFT_VELOCITIES]
        )

        # (2) Macroscopic Velocities
        density_prev = get_rho(d_velocities_prev)
        macro_velocities_prev = get_macro_velocities(
            d_velocities_prev,
            density_prev
        )

        # (3) Prescribe inflow Dirichlet using Zou/He scheme

        macro_velocities_prev =\
            macro_velocities_prev.at[0, 1:-1, :].set(
                velocity_profile[0, 1:-1, :]
            )

        density_prev = density_prev.at[0, :].set(
            (
                    get_rho(d_velocities_prev[0, :, Vertical_Velocities].T)
                    +
                    2 *
                    get_rho(d_velocities_prev[0, :, LEFT_VELOCITIES].T)
            ) / (
                1 - macro_velocities_prev[0, :, 0]
            )
        )

        # (4) Compute discrete equiliberia velocities
        equilibrium_discrete_velocities = get_eq_d_velocities(
            macro_velocities_prev,
            density_prev
        )

        # (3) Belongs to Zou/He scheme
        d_velocities_prev =\
            d_velocities_prev.at[0, :, RIGHT_VELOCITIES].set(
                equilibrium_discrete_velocities[0, :, RIGHT_VELOCITIES]
            )

        # (5) BGK
        d_velocities_before_coll = (
            d_velocities_prev
            -
            relaxation_omega
            *
            (
                d_velocities_prev
                -
                equilibrium_discrete_velocities
            )
        )

        # (6) Bounce Back Boundary Conditions (no slip)
        for i in range(D_Velocities):
            d_velocities_before_coll =\
                d_velocities_before_coll.at[obstacle_mask, L_Indices[i]].set(
                    d_velocities_prev[obstacle_mask, O_L_Indices[i]]
                )

        # (7) Stream alongside latice velocities
        d_velocities_st = d_velocities_before_coll
        for i in range(D_Velocities):
            d_velocities_st = d_velocities_st.at[:, :, i].set(
                jnp.roll(
                    jnp.roll(
                        d_velocities_before_coll[:, :, i],
                        L_Velocities[0, i],
                        axis=0
                    ),
                    L_Velocities[1, i],
                    axis=1
                )
            )

        return d_velocities_st

    d_velocities_prev = get_eq_d_velocities(
        velocity_profile,
        jnp.ones((Nx, Ny))
    )

    for iteration_index in tqdm(range(N_ITERATION)):
        d_velocities_next = update(d_velocities_prev)

        d_velocities_prev = d_velocities_next

        if iteration_index % V_freq == 0 and V_visualise and iteration_index > V_skip:
            density = get_rho(d_velocities_next)
            macroscopic_velocities = get_macro_velocities(
                d_velocities_next,
                density
            )
            if V_vel_mag:
                plot_Velocity_Magnitude(macroscopic_velocities)
            if V_vort_mag:
                plot_Vorticity_Magnitude(macroscopic_velocities)

            if V_visualise:
                plt.show()


if __name__ == "__main__":
    main()
