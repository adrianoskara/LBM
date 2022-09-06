from solver import *


def plot_Velocity_Magnitude(macroscopic_velocities):

    velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocities,
        axis=-1,
        ord=2
    )

    plt.subplot(211)
    plt.contourf(
        X,
        Y,
        velocity_magnitude,
        levels=50,
        cmap='viridis'
    )
    plt.colorbar().set_label("Velocity Magnitude")
    plt.gca().add_patch(plt.Circle((
        Cylinder_cx, Cylinder_cy),
        Cylinder_ri,
        color="darkgreen"
    ))


# Vorticity MAgnityde
def plot_Vorticity_Magnitude(macroscopic_velocities):
    d_u__d_x, d_u__d_y = jnp.gradient(macroscopic_velocities[..., 0])
    d_v__d_x, d_v__d_y = jnp.gradient(macroscopic_velocities[..., 1])
    curl = (d_u__d_y-d_v__d_x)
    plt.subplot(212)
    plt.contourf(
        X,
        Y,
        curl,
        levels=50,
        cmap='hsv',
        vmin=-0.02,
        vmax=0.02
    )
    plt.colorbar().set_label("Vorticity Magnitude")
    plt.gca().add_patch(plt.Circle((
        Cylinder_cx, Cylinder_cy),
        Cylinder_ri,
        color="darkgreen"
    ))

    plt.draw()
    plt.pause(0.0001)
    plt.clf()



