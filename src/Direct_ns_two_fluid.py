from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import numpy as np
import ufl
from ufl.core.expr import Expr
from basix.ufl import element
from dolfinx import default_real_type, fem, cpp
from dolfinx.fem import (
    Constant, Function, dirichletbc, form,
    functionspace, locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    assemble_matrix_block, assemble_vector_block,
    create_vector, create_matrix
)
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner, sym, nabla_grad, dot

from utilities import (
    project, SmoothedHeaviside, LinearSolver, CFL_conditioning, define_facet_tags
)

y_min, y_max = 0.0, 2.0
x_min, x_max = 0.0, 1.0

def interface(x):
    return x[1] - (1. - 0.5*x[0])


# Function to mark y = 0 and y = 1 (bottom and top)
def noslip_boundary(x):
    return np.isclose(x[1], y_min) | np.isclose(x[1], y_max)


def slip_boundary(x):
    return np.isclose(x[0], x_min) | np.isclose(x[0], x_max)


# Function to mark the lid (y = 1)
def point(x):
    return np.isclose(x[1], y_max) & np.isclose(x[0], x_max)


# Parameter for Newton iterations
max_iterations = 30
i = 1
# Define temporal parameters
t = 0  # Start time
T = 10.  # Final time

# Define mesh
nx, ny = 30, 60
space_step = 1/ny
alpha = 30
# Time step size
dt = alpha * space_step
num_steps = int(T/dt)

# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([x_min, y_min]), np.array([x_max, y_max])], [nx, ny],
    CellType.triangle
)

P2 = element(
    "Lagrange", msh.basix_cell(), degree=2, shape=(msh.geometry.dim,),
    dtype=default_real_type
)
P1 = element("Lagrange", msh.basix_cell(), degree=1, dtype=default_real_type)
V, Q = functionspace(msh, P2), functionspace(msh, P1)
D = fem.functionspace(msh, ("DG", 0))

'''Boundary conditions'''
# No-slip condition on boundaries where y = 0 and y = 1
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V, 1, facets), V)
# Slip condition on boundaries where x = 0 and x = 1
boundary_facets = locate_entities_boundary(msh, msh.topology.dim - 1,
                                           slip_boundary)
boundary_dofs_x = locate_dofs_topological(V.sub(0), msh.topology.dim - 1,
                                          boundary_facets)
bcx = dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs_x, V.sub(0))
# You also have to set Neumann boundary condition for the stress tensor
n = ufl.FacetNormal(msh)
def tangential_proj(u: Expr, n: Expr):
    """
    See for instance:
    https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


boundaries = [
    (1, lambda x: np.isclose(x[0], x_min)),
    (2, lambda x: np.isclose(x[0], x_max)),
    (3, lambda x: np.isclose(x[1], y_min)),
    (4, lambda x: np.isclose(x[1], y_max))
    ]
facet_tag = define_facet_tags(msh, boundaries)
ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)
fric_coeff = 0
# Not needed for a Direct solver, easy to remove the null space
facets = locate_entities_boundary(msh, msh.topology.dim - 2, point)
bcp = dirichletbc(dolfinx.default_scalar_type(0),
                  locate_dofs_topological(Q, msh.topology.dim - 2, facets), Q)

# Collect Dirichlet boundary conditions
bcs = [bc0, bcx]

# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)

u_n = fem.Function(V)
u_n.x.petsc_vec.set(0)
###############################################################################
# Define the conductivity in the two media
phi_n = fem.Function(Q)
phi_n.interpolate(interface)
xdmf_prob = XDMFFile(msh.comm, "problem.xdmf", "w")
xdmf_prob.write_mesh(msh)
phi_n.name = "phi"
xdmf_prob.write_function(phi_n, t)

nu = fem.Function(D)
rho = fem.Function(D)
mu = fem.Function(D)

# rho [kg/m^3] = 1000 || mu [Pa s] =1e-3 || nu [m^2/s]
nu_water = 1.e-6
rho_water = 30
mu_water = 0.1
# rho [kg/m^3] = 800 || mu [Pa s] =5e-2 || nu [m^2/s]
nu_oil = 6.25e-5
rho_oil = 5
mu_oil = 10

# Retrieve the cells dimensions
tdim = msh.topology.dim
num_cells = msh.topology.index_map(tdim).size_local
cells = np.arange(num_cells, dtype=np.int32)
h = cpp.mesh.h(msh._cpp_object, tdim, cells)
epsilon = h.max()
'''
You cannot interpolate. A quadrature function space are point evaluations.
They do not have an underlying polynomial that can be used as the basis on
a single cell. You can project a quadrature function into any space though.
'''
phi_n_project = fem.Function(D)
project(phi_n, phi_n_project)
smoothed_heaviside = SmoothedHeaviside(phi_n_project.x.petsc_vec, epsilon)
nu.x.array[:] = np.add(nu_water,
                       (nu_oil - nu_water)
                       * smoothed_heaviside(phi_n_project.x.array))
rho.x.array[:] = np.add(rho_water,
                        (rho_oil - rho_water)
                        * smoothed_heaviside(phi_n_project.x.array))
mu.x.array[:] = np.add(mu_water,
                       (mu_oil - mu_water)
                       * smoothed_heaviside(phi_n_project.x.array))
nu.x.scatter_forward()
xdmf_nu = XDMFFile(msh.comm, "nu.xdmf", "w")
xdmf_nu.write_mesh(msh)
xdmf_nu.write_function(rho)
###############################################################################
# Define the variational problem for the stationary Navier-Stokes problem
a = form([
        [(2*mu*inner(sym(nabla_grad(u)), sym(nabla_grad(v)))
          + rho*dot(dot(u_n, nabla_grad(u)), v)
          + rho*dot(dot(u, nabla_grad(u_n)), v)) * dx
          + fric_coeff * inner(tangential_proj(u, n), tangential_proj(v, n)) * ds(1)
          + fric_coeff * inner(tangential_proj(u, n), tangential_proj(v, n)) * ds(2),
            - inner(p, div(v)) * dx],
        [-inner(q, div(u)) * dx, None],
    ])

f = Constant(msh, PETSc.ScalarType((0., -9.8)))
L = form([rho*(dot(dot(u_n, nabla_grad(u_n)), v)
               + inner(f, v)) * dx,
         inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])

a_p11 = form(inner(p, q) * dx)
a_p = [[a[0][0], None], [None, a_p11]]


def block_operators():
    """Return block operators and block RHS vector for the Stokes
    problem"""
    print("Linear system construction")
    # Assembler matrix operator, preconditioner and RHS vector into
    # single objects but preserving block structure
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    # Assemble the preconditioner
    P = assemble_matrix_block(a_p, bcs=bcs)
    P.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    return A, P, b


def block_direct_solver():
    """Solve the Navier-Stokes problem using blocked matrices and a direct
    solver."""

    # Assembler the operators and RHS vector
    A, _, b = block_operators()
    print("Stokes solver construction")
    # Create a solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    # Set the solver type to MUMPS (LU solver) and configure MUMPS to
    # handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")
    sys = PETSc.Sys()  # type: ignore
    use_superlu = PETSc.IntType == np.int64
    if sys.hasExternalPackage("mumps") and not use_superlu:
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    else:
        pc.setFactorSolverType("superlu_dist")

    return A, b, ksp


# Create a block vector (x) to store the full solution and solve Stokes
# to have an initial guess
A, b, ksp = block_direct_solver()

x = A.createVecRight()
ksp.solve(b, x)
# Create Functions to split u and p
u_h, p_h = Function(V), Function(Q)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u_h.x.array[:offset] = x.array_r[:offset]
p_h.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
u_n.x.array[:] = u_h.x.array
Re = rho_water*2*np.max(u_n.x.array[:])/mu_water
print("Reynolds number: ", Re)

u_n.x.scatter_forward()
P1 = element(
        "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,),
        dtype=default_real_type
)
u1 = Function(functionspace(msh, P1))
u1.interpolate(u_n)
u1.name = "velocity"
xdmf_prob.write_function(u1, t)
p_h.name = "pressure"
xdmf_prob.write_function(p_h, t)

###############################################################################
# Level set
print("Set up Level Set")
phi_h = fem.Function(Q)
phi_h.interpolate(interface)


def boundary_D(x):
    return np.isclose(x[1], 2)


dofs_D = fem.locate_dofs_geometrical(Q, boundary_D)
phi_D = fem.Function(Q)
phi_D.interpolate(interface)
BCs_phi = []

phi, csi = ufl.TrialFunction(Q), ufl.TestFunction(Q)


class DeltaFunc():
    def __init__(self, t, jh, h):
        self.t = t
        self.jh = jh
        self.h = h

    def __call__(self, x):
        average_curr = fem.form(inner(self.jh, self.jh) * dx)
        average = fem.assemble_scalar(average_curr)
        L2_average = np.sqrt(msh.comm.allreduce(average, op=MPI.SUM))
        assert L2_average > 0, "L2 average is zero"
        return self.h.max()/(2*L2_average)


delta = DeltaFunc(t, u1, h)
w = csi + delta(t) * dot(u1, grad(csi))
theta = 0.5
a_levelSet = (phi * w * dx +
              (dt * theta) * dot(u1, grad(phi)) * w * dx)
L_levelSet = (phi_n * w * dx -
              dt * (1-theta) * dot(u1, grad(phi_n)) * w * dx)
# Preparing linear algebra structures for time dependent problems.
bilinear_form = fem.form(a_levelSet)
linear_form = fem.form(L_levelSet)
# Since the left hand side depends on the potential that is recalculated each
# iteration, we have to update both the left hand side and the right hand side,
# which is dependent on the previous time step u_n.
A_LS = create_matrix(bilinear_form)
b_LS = create_vector(linear_form)
# Instantiate the unified solver for this system.
solver_phi = LinearSolver(msh.comm, A_LS, b_LS, bilinear_form,
                          linear_form, BCs_phi, True)
###############################################################################

for n in range(num_steps):
    print(f"Iteration {n}")
    print("CFL condition: ", CFL_conditioning(-1., u1, dt, h.max()))
    i = 0
    L2_stop = fem.form(dot(u_n - u_h, u_n - u_h) * dx)
    while i < max_iterations:
        A, b, ksp = block_direct_solver()

        x = A.createVecRight()
        ksp.solve(b, x)

        u_h.x.array[:offset] = x.array_r[:offset]
        p_h.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

        i += 1

        A.destroy()
        b.destroy()
        x.destroy()
        ksp.destroy()

        # Compute norm of update
        stopping_crit = np.sqrt(msh.comm.allreduce(
            fem.assemble_scalar(L2_stop), op=MPI.SUM))
        print(f"Iteration {i}: ||u_n+1 - u_n|| = {stopping_crit}")
        if stopping_crit < 1e-6:
            break

        u_n.x.array[:] = u_h.x.array

    delta.t = t
    u1.interpolate(u_n)
    delta.jh = u1
    # Calculate the new Level set
    solver_phi.solve(phi_h.x)
    # Update solution at previous time step (u_n)
    phi_n.x.array[:] = phi_h.x.array
    # Update the conductivity
    project(phi_n, phi_n_project)
    smoothed_heaviside.phi_n_project = phi_n_project.x.petsc_vec
    nu.x.array[:] = np.add(nu_water,
                           (nu_oil - nu_water)
                           * smoothed_heaviside(phi_n_project.x.array))
    rho.x.array[:] = np.add(rho_water,
                            (rho_oil - rho_water)
                            * smoothed_heaviside(phi_n_project.x.array))
    mu.x.array[:] = np.add(mu_water,
                           (mu_oil - mu_water)
                           * smoothed_heaviside(phi_n_project.x.array))
    t += dt
    xdmf_nu.write_function(rho, t)
    xdmf_prob.write_function(phi_n, t)
    xdmf_prob.write_function(u1, t)
    xdmf_prob.write_function(p_h, t)

xdmf_nu.close()
xdmf_prob.close()
