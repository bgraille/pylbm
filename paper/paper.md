---
title: '`pylbm`: A flexible Python package for lattice Boltzmann method'
tags:
  - Python
  - lattice Boltzmann Method
  - Computational Fluid Dynamics
  - High Performance Computing
  - Code Generator
authors:
  - name: Loïc Gouarin
    orcid: 0000-0003-4761-9989
    affiliation: 1
  - name: Benjamin Graille
    orcid: 0000-0002-6287-2627
    affiliation: 2
affiliations:
 - name: CMAP/CNRS, école polytechnique
   index: 1
 - name: Université Paris-Saclay, CNRS, Laboratoire de mathématiques d’Orsay, 91405, Orsay, France
   index: 2
date: June 16, 2020
bibliography: paper.bib
csl: paper.csl
link-citations: true
---

# Summary

`pylbm` is an open-source package written in Python [@pylbm]. It proposes a leading-edge implementation of the lattice Boltzmann method for 1D/2D/3D problems and has been created to address various communities:

- **Mathematicians**: `pylbm` provides a very pleasant and efficient environment in order to test new schemes, understand their mathematical properties, such as stability and consistency and open towards new frontiers in the field such as mesh refinement, equivalent equations...
- **Physicists**: `pylbm` offers an ideal framework to conduct numerical experiments for various levels of modeling and schemes, without having to dive into the mathematical machinery of LBM schemes. The high level of the proposed algorithm can also allow to go beyond the first test and easily conduct large scales simulations thanks to its parallel capability.
- **Computer scientists**: In `pylbm`, the lattice Boltzmann method is not hard-coded. Advanced tools of code generation, based on a large set of newly developed computer algebra libraries, allow a high-level entry by the user of scheme definition and boundary conditions. The `pylbm` software then generates the resulting numerical code. It is therefore possible to modify the code building kernels to test performance and code optimization on different architectures (AA pattern and pull algorithm); the code can also be generated in different languages (C, C++, openCL, ...).

The principle feature of `pylbm` is its great flexibility to build lattice Boltzmann schemes and generate efficient numerical simulations. Moreover, it has excellent parallel capabilities and uses MPI for distributed computing and openCL for GPUs.

The generalized multiple-relaxation-time framework is used to describe the schemes [@dhumiere_generalized_1992]. It's then easy to define your lattice Boltzmann scheme by providing the velocities, the moments, their equilibrium values, and the associated relaxation parameters. Moreover, multiple $D_dQ_q$ schemes can be coupled in the simulation  where $d$ is the dimension and $q$ the number of velocities. This formalism is used for example to simulate thermodynamic fluid flows as in the Rayleigh-Benard test case. But you can also experiment with new types of lattice Boltzmann schemes like vectorial schemes [@graille_approximation_2014] or with relative velocities [@dubois_lattice_2015].

`pylbm` will offer in the future releases more tools to help the user to design their lattice Boltzmann schemes and make large simulations with complex geometries.

- **More examples**: we want to give access to various lattice Boltzmann schemes you can find in the literature. We will add multi-component flows, multi-phase flows, ... in order to have a full gallery of what we can do with LBM. We hope this way the users can improve this list with their own schemes.
- **Equivalent equations**: the hard part with the LBM is that you never write the physical equations you want to solve but the lattice Boltzmann scheme associated. We will offer the possibility to retrieve the physical equations from the given scheme by doing a Chapman-Enskog expansion for nonlinear equations up to the second order.
- **Complex geometries**: the geometry in `pylbm` can be described by a union of simple geometry elements like circle, triangle, sphere,... It's not realistic for industrial challenges and we will offer the possibility to use various CAD formats like STL.

In the following sections, we describe the main features of the `pylbm` module: the formal description of the scheme, the tools of analysis (to compute the equivalent equations and to visualize the stability), the possibility to implement your own boundary conditions.

# A formal description of the lattice Boltzmann scheme

The greatest asset of `pylbm` is that the scheme is not hard-coded: it is described by a dictionary that contains all necessary information of the simulation. 
Moreover, schemes with multiple distribution functions are simply described with a list of elementary schemes that can be coupled by the equilibrium functions. 
Each elementary scheme is written in the MRT formalism proposed by d'Humière [@dhumiere_generalized_1992]. 

## MRT formalism

Let us first consider a regular lattice $L$ in dimension $d$ with a mesh size $dx$ and a time step $dt$. The scheme velocity $\lambda$ is then defined by $\lambda = dx/dt$. We introduce a set of $q$ velocities adapted to this lattice $\{v_0, \ldots, v_{q-1}\}$, that is to say that, if $x$ is a point of the lattice $L$, the point $x+v_jdt$ is also on the lattice for every $j\in\{0,\ldots,q{-}1\}$.

The aim of the $DdQq$ scheme is to compute a distribution function vector ${\boldsymbol f} = (f_0,\ldots,f_{q-1})$ on the lattice $L$ at discrete values of time. The scheme splits into two steps: the relaxation phase and the transport phase. That is, the passage from the time $t$ to the time $t+dt$ consists in the succession of these two phases.

__the relaxation phase__

This phase, also called collision, is local in space: on every site $x$ of the lattice, the values of the vector ${\boldsymbol f}$ are modified, the result after the collision being denoted by ${\boldsymbol f}^\star$. The collision operator is a linear operator of relaxation toward an equilibrium value denoted ${\boldsymbol f}^{\textrm{eq}}$.

This linear operator is diagonal in a peculiar basis called moments denoted by ${\boldsymbol m} = (m_0,\ldots,m_{q-1})$. The change-of-basis matrix $M$ is such that ${\boldsymbol m} = M{\boldsymbol f}$ and ${\boldsymbol f} = M^{-1}{\boldsymbol m}$. In the basis of the moments, the collision operator then just reads

$$ m_k^\star = m_k - s_k (m_k - m_k^{\textrm{eq}}), \qquad 0\leqslant k < q, $$

where $s_k$ is the relaxation parameter associated to the $k$-th moment. The $k$-th moment is said conserved during the collision if the associated relaxation parameter $s_k$ is zero. We introduce the diagonal matrix $S=\operatorname{diag}(s_0,\ldots,s_{q-1})$ and the collision operator reads
$$ m^\star = m - S (m-m^{\textrm{eq}}).$$
In this previous formula, the equilibrium value of the $k$-th moment, $m_k^{\textrm{eq}}$, $0\leq k < q$, is a given function of the conserved moments. 

By analogy with the kinetic theory, the change-of-basis matrix $M$ is defined by a set of polynomials with $d$ variables $(P_0,\ldots,P_{q-1})$ by

$$ M_{ij} = P_i(v_j). $$

The relaxation phase consists then to compute on every site $x$ of the lattice the new particle distribution functions $f^\star$:

- compute the vector of the moments $m=Mf$,
- compute the moments after the collision $m^\star = m - S (m-m^{\textrm{eq}})$,
- compute the vector of the particle distribution functions $f^\star=M^{-1}m^\star$.

__the transport phase__

This phase just consists in a shift of the indices and reads

$$ f_j(x, t+dt) = f_j^\star(x-v_jdt, t), \qquad 0\leqslant j < q. $$

## Build your first scheme

A lattice Boltzmann scheme is described in `pylbm` by a dictionary that contains all the necessary information. For example, the following dictionary can be used to build a scheme to simulate the Burgers equation $\partial_t U + \partial_x U^2/2 = 0$:
```python
scheme_cfg = {
    'dim': 1,
    'scheme_velocity': 1.,
    'schemes': [
        {
            'velocities': [1, 2],
            'conserved_moments': U,
            'polynomials': [1, X],
            'equilibrium': [U, U**2/2],
            'relaxation_parameters': [0, 1.5],
        },
    ],
}
```


Each elementary scheme is then given by
the **stencil** of velocities,
the **conserved moments**,
the **polynomials** that define the moments,
the **relaxation parameters**,
the **equilibrium values** that are functions of the conserved moments.

At our knowledge, `pylbm` is then the only lattice Boltzmann tool for which the user can implement is own scheme without to code inside the module. All the description of the simulation is given in a script file. 

## Symbolic variables



## relative velocities

# References
