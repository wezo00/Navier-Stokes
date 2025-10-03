# # Stokes equations using Taylor-Hood elements

# ## Equation and problem definition
#
# ### Strong formulation
#
# $$
# \begin{align}
#   - \nabla \cdot (\nabla u + p I) &= f \quad {\rm in} \ \Omega,\\
#   \nabla \cdot u &= 0 \quad {\rm in} \ \Omega.
# \end{align}
# $$
#
# with conditions on the boundary $\partial \Omega = \Gamma_{D} \cup
# \Gamma_{N}$ of the form:
#
# $$
# \begin{align}
#   u &= u_0 \quad {\rm on} \ \Gamma_{D},\\
#   \nabla u \cdot n + p n &= g \,   \quad\;\; {\rm on} \ \Gamma_{N}.
# \end{align}
# $$
#
# ```{note}
# The sign of the pressure has been changed from the usual
# definition. This is to generate have a symmetric system
# of equations.
# ```
#
# ### Weak formulation
#
# The weak formulation reads: find $(u, p) \in V \times Q$ such that
#
# $$
# a((u, p), (v, q)) = L((v, q)) \quad \forall  (v, q) \in V \times Q
# $$
#
# where
#
# $$
# \begin{align}
#   a((u, p), (v, q)) &:= \int_{\Omega} \nabla u \cdot \nabla v -
#            \nabla \cdot v \ p + \nabla \cdot u \ q \, {\rm d} x,
#   L((v, q)) &:= \int_{\Omega} f \cdot v \, {\rm d} x + \int_{\partial
#            \Omega_N} g \cdot v \, {\rm d} s.
# \end{align}
# $$
#
# ### Domain and boundary conditions
#
# We consider the lid-driven cavity problem with the following
# domain and boundary conditions:
#
# - $\Omega := [0,1]\times[0,1]$ (a unit square)
# - $\Gamma_D := \partial \Omega$
# - $u_0 := (1, 0)^\top$ at $x_1 = 1$ and $u_0 = (0, 0)^\top$ otherwise
# - $f := (0, 0)^\top$
#
#
# ## Implementation
#
# The Stokes problem using Taylor-Hood elements is solved using:

# 1. [Block preconditioner with the `u` and `p` fields stored block-wise
#    in a single matrix](#monolithic-block-iterative-solver)
# 1. [Direct solver with the `u` and `p` fields stored block-wise in a
#    single matrix](#monolithic-block-direct-solver)
#

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import numpy as np

import ufl
from basix.ufl import element
from dolfinx import default_real_type, fem, la
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner

opts = PETSc.Options()
opts["mat_superlu_dist_iterrefine"] = True

# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [32, 32], CellType.triangle)

# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)


# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


# Two function spaces <dolfinx.fem.FunctionSpace> are
# defined using different finite elements. `P2` corresponds to a
# continuous piecewise quadratic basis (vector) and `P1` to a continuous
# piecewise linear basis (scalar).

P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=default_real_type)
P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
V, Q = functionspace(msh, P2), functionspace(msh, P1)

# Boundary conditions for the velocity field are defined:

# +
# No-slip condition on boundaries where x = 0, x = 1, and y = 0
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)  # type: ignore
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V, 1, facets), V)

# Driving (lid) velocity condition on top boundary (y = 1)
lid_velocity = Function(V)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, 1, facets))

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]
# -

# The bilinear and linear forms for the Stokes equations are defined
# using a a blocked structure:

# +
# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
f = Constant(msh, (PETSc.ScalarType(0), PETSc.ScalarType(0)))  # type: ignore

a = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx], [inner(div(u), q) * dx, None]])
L = form([inner(f, v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])  # type: ignore
# -

# A block-diagonal preconditioner will be used with the iterative
# solvers for this problem:

a_p11 = form(inner(p, q) * dx)
a_p = [[a[0][0], None], [None, a_p11]]


# ### Monolithic block iterative solver
#
# We now solve the same Stokes problem, but using monolithic
# (non-nested) matrices. We first create a helper function for
# assembling the linear operators and the RHS vector.


def block_operators():
    """Return block operators and block RHS vector for the Stokes
    problem"""

    # Assembler matrix operator, preconditioner and RHS vector into
    # single objects but preserving block structure
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    P = assemble_matrix_block(a_p, bcs=bcs)
    P.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    # Set the nullspace for pressure (since pressure is determined only
    # up to a constant)
    null_vec = A.createVecLeft()
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    null_vec.array[offset:] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    assert nsp.test(A)
    A.setNullSpace(nsp)

    return A, P, b


# The following function solves the Stokes problem using a
# block-diagonal preconditioner and monolithic PETSc matrices.


def block_iterative_solver():
    """Solve the Stokes problem using blocked matrices and an iterative
    solver."""

    # Assembler the operators and RHS vector
    A, P, b = block_operators()

    # Build PETSc index sets for each field (global dof indices for each
    # field)
    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
    offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
    is_u = PETSc.IS().createStride(
        V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF
    )
    is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setTolerances(rtol=1e-9)
    ksp.setType("minres")
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    # Configure velocity and pressure sub-solvers
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # The matrix A combined the vector velocity and scalar pressure
    # parts, hence has a block size of 1. Unlike the MatNest case, GAMG
    # cannot infer the correct near-nullspace from the matrix block
    # size. Therefore, we set block size on the top-left block of the
    # preconditioner so that GAMG can infer the appropriate near
    # nullspace.
    ksp.getPC().setUp()
    Pu, _ = ksp_u.getPC().getOperators()
    Pu.setBlockSize(msh.topology.dim)

    # Create a block vector (x) to store the full solution and solve
    x = A.createVecRight()
    ksp.solve(b, x)
    
    # Report iteration count and reason
    its = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    if MPI.COMM_WORLD.rank == 0:
        print(f"[MINRES] Converged in {its} iterations (reason={reason})")

    # Create Functions to split u and p
    u, p = Function(V), Function(Q)
    offset = V_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    p.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

    # Compute the $L^2$ norms of the solution vectors
    norm_u, norm_p = la.norm(u.x), la.norm(p.x)
    if MPI.COMM_WORLD.rank == 0:
        print(f"(B) Norm of velocity coefficient vector (blocked, iterative): {norm_u}")
        print(f"(B) Norm of pressure coefficient vector (blocked, iterative): {norm_p}")
    
    # Save solution to file in XDMF format for visualization, e.g. with
    # ParaView. Before writing to file, ghost values are updated using
    # `scatter_forward`.
    with XDMFFile(MPI.COMM_WORLD, "out_stokes/velocity.xdmf", "w") as ufile_xdmf:
        u.x.scatter_forward()
        P1 = element(
            "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,), dtype=default_real_type
        )
        u1 = Function(functionspace(msh, P1))
        u1.interpolate(u)
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u1)

    with XDMFFile(MPI.COMM_WORLD, "out_stokes/pressure.xdmf", "w") as pfile_xdmf:
        p.x.scatter_forward()
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p)

    return norm_u, norm_p


# ### Monolithic block direct solver
#
# We now solve the same Stokes problem again, but using monolithic
# (non-nested) matrices and a direct (LU) solver.


def block_direct_solver():
    """Solve the Stokes problem using blocked matrices and a direct
    solver."""

    # Assembler the block operator and RHS vector
    A, _, b = block_operators()

    # Create a solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    # Set the solver type to MUMPS (LU solver) and configure MUMPS to
    # handle pressure nullspace
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("superlu_dist")
    try:
        pc.setFactorSetUpSolverType()
    except PETSc.Error as e:
        if e.ierr == 92:
            print("The required PETSc solver/preconditioner is not available. Exiting.")
            print(e)
            exit(0)
        else:
            raise e
    # pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # For pressure nullspace
    # pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # For pressure nullspace

    # Create a block vector (x) to store the full solution, and solve
    x = A.createVecLeft()
    ksp.solve(b, x)

    # Create Functions and scatter x solution
    u, p = Function(V), Function(Q)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    p.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

    # Compute the $L^2$ norms of the u and p vectors
    norm_u, norm_p = la.norm(u.x), la.norm(p.x)
    if MPI.COMM_WORLD.rank == 0:
        print(f"(C) Norm of velocity coefficient vector (blocked, direct): {norm_u}")
        print(f"(C) Norm of pressure coefficient vector (blocked, direct): {norm_p}")

    return norm_u, norm_p


# Solve using PETSc block matrices and an iterative solver
norm_u_1, norm_p_1 = block_iterative_solver()

# Solve using PETSc block matrices and an LU solver
norm_u_2, norm_p_2 = block_direct_solver()
np.testing.assert_allclose(norm_u_2, norm_u_1, rtol=1e-4)
np.testing.assert_allclose(norm_p_2, norm_p_1, rtol=1e-4)

