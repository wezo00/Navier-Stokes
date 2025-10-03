from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import numpy as np
import math

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


# Analytical solution
# Analytical solutions as Python functions returning NumPy arrays
def u_analytical(x):
    # x has shape (2, n_points)
    u = np.zeros((2, x.shape[1]))
    u[0] = np.sin(np.pi*x[1])*np.cos(np.pi*x[1])*np.sin(np.pi*x[0])**2
    u[1] = -np.sin(np.pi*x[0])*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])**2
    return u

def p_analytical(x):
    return (x[0]*x[1]*(1-x[0]*x[1]))[np.newaxis, :]  # shape (1, n_points)

def f_analytical(x):
    # x has shape (2, n_points)
    f = np.zeros((2, x.shape[1]))
    # compute forcing term from -Δu + ∇p
    f[0] = (x[1]*(1 - 2*x[0]*x[1]) 
                 - np.sin(2*np.pi*x[1])*(2*np.cos(2*np.pi*x[0])-1)*np.pi**2)
    f[1] = (x[0]*(1 - 2*x[1]*x[0]) 
                 - np.sin(2*np.pi*x[0])*(1-2*np.cos(2*np.pi*x[1]))*np.pi**2)
    return f

y_min, y_max = 0.0, 1.0
x_min, x_max = 0.0, 1.0

# Poiseuille flow
#'''
def u_analytical(x):
    # x has shape (2, n_points)
    u = np.zeros((2, x.shape[1]))
    u[0] = 1 - x[1]**2
    u[1] = 0
    return u

def p_analytical(x):
    return -(2*x[0])[np.newaxis, :]  # shape (1, n_points)

def f_analytical(x):
    # x has shape (2, n_points)
    f = np.zeros((2, x.shape[1]))
    # compute forcing term from -Δu + ∇p
    return f

y_min, y_max = -1.0, 1.0
x_min, x_max = -1.0, 1.0
#'''
#####################################################

def get_L2_norm(u, msh):
    local = fem.assemble_scalar(form(inner(u, u) * dx))
    global_val = msh.comm.allreduce(local, op=MPI.SUM)
    return math.sqrt(global_val)


# Create mesh with different sizes
def create_mesh(nx, ny, x_min, y_min, x_max, y_max):
    msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([x_min, y_min]), np.array([x_max, y_max])], [nx, ny], CellType.triangle)
    return msh

def on_boundary(x, x_min, y_min, x_max, y_max):
        return np.isclose(x[0], x_min) | np.isclose(x[0], x_max) | np.isclose(x[1], y_min) | np.isclose(x[1], y_max)

def block_operators(a, a_p, L, bcs, V):
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
    assert nsp.test(A), "Nullspace vector does not test as nullspace for A"
    A.setNullSpace(nsp)

    return A, P, b


# The following function solves the Stokes problem using a
# block-diagonal preconditioner and monolithic PETSc matrices.

def block_iterative_solver(a, a_p, L, bcs, V, Q, msh):
    """Solve the Stokes problem using blocked matrices and an iterative
    solver."""

    # Assembler the operators and RHS vector
    A, P, b = block_operators(a, a_p, L, bcs, V)

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
    ksp.setType("gmres")
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
        print(f"[GMRES] Converged in {its} iterations (reason={reason})")

    # Create Functions to split u and p
    u_h, p_h = Function(V), Function(Q)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    offset_p = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    u_h.x.array[:offset] = x.array_r[:offset]
    p_h.x.array[:offset_p] = x.array_r[offset: (offset_p + offset)]

    # Compute the $L^2$ norms of the solution vectors
    norm_u, norm_p = get_L2_norm(u_h, msh), get_L2_norm(p_h, msh)

    if MPI.COMM_WORLD.rank == 0:
        print(f"(B) Norm of velocity coefficient vector (blocked, iterative): {norm_u}")
        print(f"(B) Norm of pressure coefficient vector (blocked, iterative): {norm_p}")
    
    # Save solution to file in XDMF format for visualization, e.g. with
    # ParaView. Before writing to file, ghost values are updated using
    # `scatter_forward`.
    with XDMFFile(MPI.COMM_WORLD, "v_stokes.xdmf", "w") as ufile_xdmf:
        u_h.x.scatter_forward()
        P1 = element(
            "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,),
            dtype=default_real_type
        )
        u1 = Function(functionspace(msh, P1))
        u1.interpolate(u_h)
        u1.name = "Velocity"
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u1)

    with XDMFFile(MPI.COMM_WORLD, "p_stokes.xdmf", "w") as pfile_xdmf:
        p_h.x.scatter_forward()
        p_h.name = "Pressure"
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p_h)

    return norm_u, norm_p, u_h, p_h

# ### Monolithic block direct solver
#
# We now solve the same Stokes problem again, but using monolithic
# (non-nested) matrices and a direct (LU) solver.


def block_direct_solver(a, a_p, L, bcs, V, Q, msh):
    """Solve the Stokes problem using blocked matrices and a direct
    solver."""

    # Assembler the block operator and RHS vector
    A, _, b = block_operators(a, a_p, L, bcs, V)

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
    u_n = fem.Function(V)
    u_n.x.petsc_vec.set(0.)

    u_h, p_h = Function(V), Function(Q)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u_h.x.array[:offset] = x.array_r[:offset]
    p_h.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]


    # Compute the $L^2$ norms of the u and p vectors
    norm_u, norm_p = get_L2_norm(u_h, msh), get_L2_norm(p_h, msh)

    if MPI.COMM_WORLD.rank == 0:
        print(f"(C) Norm of velocity coefficient vector (blocked, direct): {norm_u}")
        print(f"(C) Norm of pressure coefficient vector (blocked, direct): {norm_p}")
    return norm_u, norm_p, u_h, p_h

def run_on_mesh(nx, ny, solver=block_direct_solver, y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max):

    # Create mesh
    msh = create_mesh(nx, ny, x_min, y_min, x_max, y_max)

    # Function spaces
    P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=default_real_type)
    P1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)
    V, Q = functionspace(msh, P2), functionspace(msh, P1)

    # Dirichlet BC
    def boundary(x):
        return on_boundary(x, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    boundary_v = Function(V)
    boundary_v.interpolate(u_analytical)
    facets = locate_entities_boundary(msh, 1, boundary)
    bc = dirichletbc(boundary_v, locate_dofs_topological(V, 1, facets))
    bcs = [bc]

    # Variational problem
    (u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    (v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)

    # Forcing function
    f = fem.Function(V)
    f.interpolate(lambda x: f_analytical(x))

    a = form([[inner(grad(u), grad(v)) * dx, -inner(p, div(v)) * dx],
              [inner(div(u), q) * dx, None]])
    L = form([inner(f, v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])

    # Preconditioner
    a_p11 = form(inner(p, q) * dx)
    a_p = [[a[0][0], None], [None, a_p11]]

    # Solve
    norm_u, norm_p, u_h, p_h = solver(a, a_p, L, bcs, V, Q, msh)

    # Interpolate analytical solutions
    u_ana = fem.Function(V)
    u_ana.interpolate(u_analytical)

    p_ana = fem.Function(Q)
    p_ana.interpolate(p_analytical)

    norm_u_ana, norm_p_ana = get_L2_norm(u_ana, msh), get_L2_norm(p_ana, msh)

    # L2 error
    u_error_form = form(inner(u_h - u_ana, u_h - u_ana) * dx)
    err_u = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(u_error_form), op=MPI.SUM))

    p_error_form = form(inner(p_h - p_ana, p_h - p_ana) * dx)
    err_p = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(p_error_form), op=MPI.SUM))

    # simpler: compute L2 of strong residual components:
    res_form = form(inner(-div(grad(u_h)) + grad(p_h) - f, -div(grad(u_h)) + grad(p_h) - f) * dx)
    res_norm = math.sqrt(msh.comm.allreduce(fem.assemble_scalar(res_form), op=MPI.SUM))
    div_norm = math.sqrt(msh.comm.allreduce(fem.assemble_scalar(form(div(u_h)**2 * dx)), op=MPI.SUM))
    print("strong residual norm:", res_norm, "divergence norm:", div_norm)

    return norm_u, norm_p, norm_u_ana, norm_p_ana, err_u, err_p

mesh_sizes = [4, 8, 16, 32, 64]  # you can change or extend
#mesh_sizes = [4]
errs_u = []
errs_p = []
norms_u = []
norms_p = []
for n in mesh_sizes:
    nu, np_, norm_u_ana, norm_p_ana, err_u_L2, err_p_L2 = run_on_mesh(n, n, solver=block_iterative_solver)
    norms_u.append(nu) 
    norms_p.append(np_)
    errs_u.append(err_u_L2)
    errs_p.append(err_p_L2)

    print(f"Norm of velocity analitycal coefficient vector:  {norm_u_ana}")
    print(f"Norm of pressure analitycal coefficient vector:  {norm_p_ana}")
    
    print(f"Norm of velocity error:  {err_u_L2}")
    print(f"Norm of pressure error:  {err_p_L2}")

# compute convergence rates (log2 ratios)
if MPI.COMM_WORLD.rank == 0:
    print("\nConvergence rates (estimated):")
    for i in range(1, len(mesh_sizes)):
        rate_u = math.log(errs_u[i-1]/errs_u[i], 2)
        rate_p = math.log(errs_p[i-1]/errs_p[i], 2)
        print(f"  {mesh_sizes[i-1]} -> {mesh_sizes[i]} : rate u ~ {rate_u:.2f}, rate p ~ {rate_p:.2f}")



