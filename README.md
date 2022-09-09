# LATTICE-BOLTZMANN METHOD SOLVER




This code models the incompressible* Navier Stokes equations using the Lattice-Boltzmann Method. The specific scenario studied is that of the flow around a cylinder in 2D which yields a Karman vortex trail.


- Written in Python using the Numpy Scientific Library
- Use of the Lattice-Boltzmann method for the fluid simulation of a bluff body.
- Valid for orthonomal mesh


An important caveat regarding this solver is that the LBM solves the Navier-Stokes equation in it's compressible version. However, by maintaing a low Mach regime (~0.3 Mach) the fluid can be considered almost incompressible due to the miniscule density fluctuations. Consequently, the stability of the scheme is linked the Mach number.

>


[1] Perumal D, Dr.Arumuga & Gundavarapu, Venkata Suresh & Dass, Anoop. (2014). Lattice Boltzmann simulation of flow over a circular cylinder at moderate Reynolds numbers. Thermal Science. 18. 1235-1246. 10.2298/TSCI110908093A.

[2] Tritton, D. J., Experiments on the Flow Past a Circular Cylinder at Low Reynolds Numbers, Journal of Fluid Mechanics, 6 (1959), 4, pp. 547-555

[3] Fornberg, B., A Numerical Study of Steady Viscous Flow Past a Circular Cylinder, Journal of Fluid Me- chanics, 98 (1980), 4, pp. 819-855

[4] Calhoun, D., A Cartesian Grid Method for Solving the Two-Dimensional Streamfunction-Vorticity Equa- tions in Irregular Regions, Journal of Computational Physics, 176 (2002), 2, pp. 231-275

