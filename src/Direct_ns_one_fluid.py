from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import numpy as np
import ufl
from ufl.core.expr import Expr
from basix.ufl import element
from dolfinx import default_real_type, fem, la
from dolfinx.fem import (
    Constant, Function, dirichletbc, form,
    functionspace, locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    assemble_matrix_block, assemble_vector_block)
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, inner, sym, nabla_grad, dot

def analitical_velocity(x, alpha):
    values = np.zeros((2, x.shape[1]))
    values[0] = alpha*np.sin(np.pi*x[1])*np.cos(np.pi*x[1])*np.sin(np.pi*x[0])**2
    values[1] = -alpha*np.sin(np.pi*x[0])*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])**2
    return values


def analitical_pressure(x, gamma):
    return gamma*x[0]*x[1]*(1-x[0]-x[1])


def analitical_force(x, nu, alpha, gamma):
    values = np.zeros((2, x.shape[1]))
    values[0] = (gamma*x[1]*(1-2*x[0]*x[1])
                 - nu*alpha*np.sin(2*np.pi*x[1])*(2*np.cos(2*np.pi*x[0])-1)*np.pi**2
                 + (alpha*np.sin(np.pi*x[1])*np.cos(np.pi*x[1])*np.sin(np.pi*x[0])**2)
                 * (np.pi*alpha*np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[1])/2.
                    + np.pi*alpha*(np.sin(np.pi*x[0])**2)*np.cos(2*np.pi*x[1]))
    )
    values[1] = (gamma*x[0]*(1-2*x[1]*x[0])
                 - nu*alpha*np.sin(2*np.pi*x[0])*(1-2*np.cos(2*np.pi*x[1]))*np.pi**2
                 - (alpha*np.sin(np.pi*x[0])*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])**2)
                 * (-np.pi*alpha*np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[1])/2.
                    - np.pi*alpha*(np.sin(np.pi*x[1])**2)*np.cos(2*np.pi*x[0]))
    )
    return values

def analitical_force_stokes(x, nu, alpha, gamma):
    values = np.zeros((2, x.shape[1]))
    values[0] = (gamma*x[1]*(1-2*x[0]*x[1])
                 - nu*alpha*np.sin(2*np.pi*x[1])*(2*np.cos(2*np.pi*x[0])-1)*np.pi**2
    )
    values[1] = (gamma*x[0]*(1-2*x[1]*x[0])
                 - nu*alpha*np.sin(2*np.pi*x[0])*(1-2*np.cos(2*np.pi*x[1]))*np.pi**2
    )
    return values


y_min, y_max = 0.0, 1.0
x_min, x_max = 0.0, 1.0

# Function to mark y = 0 and y = 1 (bottom and top)
def noslip_boundary(x):
    return np.isclose(x[1], y_min) | np.isclose(x[1], y_max) | np.isclose(x[0], x_min) | np.isclose(x[0], x_max)


def slip_boundary(x):
    return np.isclose(x[0], x_min) | np.isclose(x[0], x_max)

# -----------------------------
# Dataset & sampling setup
# -----------------------------
# dataset sizes and evaluation grid
N_samples = 1      # Total number of samples 200

# Parameter for Newton iterations
max_iterations = 30
i = 1

# Define mesh
nx, ny = 30, 30

# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([x_min, y_min]), np.array([x_max, y_max])], [nx, ny],
    CellType.triangle
)

# helper: evaluate function `fn` (dolfinx Function) at points `coords` (shape (G,2))
# returns shape (G, value_dim)

P2 = element(
    "Lagrange", msh.basix_cell(), degree=2, shape=(msh.geometry.dim,),
    dtype=np.float64
)
P1 = element("Lagrange", msh.basix_cell(), degree=1, dtype=np.float64)
V, Q = functionspace(msh, P2), functionspace(msh, P1)

###############################################################################
seed = 196732
np.random.seed(seed)  # Set a seed for reproducibility
alpha = np.random.uniform(low=0.2, high=1.2, size=N_samples).astype(np.double)
beta = np.random.uniform(low=0, high=1, size=N_samples).astype(np.double)
nu = Constant(msh, PETSc.ScalarType(np.float64(1e-1)))

###############################################################################
'''Boundary conditions'''
# No-slip condition on boundaries where y = 0 and y = 1
noslip = np.zeros(msh.geometry.dim, dtype=np.float64)
u_fixed = Function(V)
u_fixed.interpolate(lambda x: analitical_velocity(x, alpha[0]))
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(u_fixed, locate_dofs_topological(V, 1, facets))

# Collect Dirichlet boundary conditions
bcs = [bc0]

# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)

u_n = fem.Function(V)
u_n.x.petsc_vec.set(0.)
###############################################################################

# Define the variational problem for the stationary Navier-Stokes problem
a = form([
        [(2*nu*inner(sym(nabla_grad(u)), sym(nabla_grad(v)))
          + dot(dot(u_n, nabla_grad(u)), v)
          + dot(dot(u, nabla_grad(u_n)), v)) * dx,
            - inner(p, div(v)) * dx],
        [-inner(q, div(u)) * dx, None],
    ])
f = fem.Function(V)
f.interpolate(lambda x: analitical_force_stokes(x, nu.value, alpha[0], beta[0]))
L = form([(dot(dot(u_n, nabla_grad(u_n)), v)
               + inner(f, v)) * dx,
         inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])


def block_operators():
    """Return block operators and block RHS vector for the Stokes
    problem"""
    print("Linear system construction")
    # Assembler matrix operator, preconditioner and RHS vector into
    # single objects but preserving block structure
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    return A, b

def block_direct_solver():
    """Solve the Navier-Stokes problem using blocked matrices and a direct
    solver."""

    # Assembler the operators and RHS vector
    A, b = block_operators()
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

def block_iterative_solver_iterative():
    """Iterative block solver for Navier--Stokes: GMRES + fieldsplit block-diag (A, Q)."""

    A, b = block_operators()   # assemble Jacobian and rhs as you already do

    # build index sets (use create_is if available, otherwise keep your manual stride logic)
    from dolfinx.fem.petsc import create_is
    # If using mixed W = V*Q (not here) you could do create_is(W.sub(0)), etc.
    # For your separate V,Q spaces, you can reuse your stride creation:
    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
    offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
    is_u = PETSc.IS().createStride(V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF)
    is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)              # single operator (no explicit P provided)
    ksp.setType("gmres")             # use GMRES for non-symmetric Jacobian
    ksp.setTolerances(rtol=1e-8, max_it=1000)

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)  # block-diagonal
    pc.setFieldSplitIS(("u", is_u), ("p", is_p))

    # configure subsolvers (preonly + local PC) — apply each block approx once
    ksp_u, ksp_p = pc.getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")   # AMG for vector velocity block
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi") # cheap approx for pressure mass

    # set up and give GAMG the block size hint for velocity
    pc.setUp()
    Pu, _ = ksp_u.getPC().getOperators()
    Pu.setBlockSize(msh.topology.dim)

    # Solve
    x = A.createVecRight()
    ksp.solve(b, x)

    # report iteration count + reason
    its = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    if MPI.COMM_WORLD.rank == 0:
        print(f"[GMRES] iterations = {its}, reason = {reason}")

    return A, b, ksp, x

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
u_stokes = fem.Function(V)
u_stokes.x.array[:] = u_h.x.array
Re = 2*np.max(u_n.x.array[:])/nu.value
print("Reynolds number: ", Re)

u_n.x.scatter_forward()
P1_vel = element(
        "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,),
        dtype=default_real_type
)

u1 = Function(functionspace(msh, P1_vel))
u1.interpolate(u_n)
u_ana = Function(functionspace(msh, P1_vel))
u_ana.interpolate(lambda x: analitical_velocity(x, alpha[0]))
u1.name = "velocity"
u_ana.name = "velocity_analytical"
p_h.name = "pressure"
###############################################################################

# ---------------------------------------------------------
# Per-sample loop: construct top&bottom Dirichlet profiles, solve, evaluate and save.
# ---------------------------------------------------------
sample_idx = 0
for i in range(N_samples):
    print(f"--- Generating Sample {i+1}/{N_samples} ---")
    # parameter vector for this sample: [nu, top_mode1..4, bottom_mode1..4]
    nu_i = nu.value  # if you want to vary viscosity, replace with sampled value
    f.interpolate(lambda x: analitical_force(x, nu_i, alpha[i], beta[i]))
    u_fixed.interpolate(lambda x: analitical_velocity(x, alpha[i]))
    # Solution of the Navier-Stokes problem
    j = 0
    L2_stop = fem.form(dot(u_n - u_h, u_n - u_h) * dx)
    u_n.x.petsc_vec.set(0.)
    u_n.x.array[:] = u_stokes.x.array
    while j < max_iterations:
        A, b, ksp = block_direct_solver()

        x = A.createVecRight()
        ksp.solve(b, x)

        u_h.x.array[:offset] = x.array_r[:offset]
        p_h.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

        j += 1

        A.destroy()
        b.destroy()
        x.destroy()
        ksp.destroy()

        # Compute norm of update
        stopping_crit = np.sqrt(msh.comm.allreduce(
            fem.assemble_scalar(L2_stop), op=MPI.SUM))
        print(f"Iteration {j}: ||u_n+1 - u_n|| = {stopping_crit}")
        if stopping_crit < 1e-6:
            break

        u_n.x.array[:] = u_h.x.array

    u1.interpolate(u_n)
    u_ana.interpolate(lambda x: analitical_velocity(x, alpha[i]))

     # --- Evaluate and Store Data ---
    # 1. Evaluate Branch Input: Forcing function at sensor locations
    # 2. Evaluate Output: Velocity and pressure at evaluation (trunk) coordinates
    # 3. Store in HDF5 file

norm_u, norm_p = la.norm(u_n.x), la.norm(p_h.x)
p_ana = Function(Q)
p_ana.interpolate(lambda x: analitical_pressure(x, beta[-1]))
norm_u_ana, norm_p_ana = la.norm(u_ana.x), la.norm(p_ana.x)
if MPI.COMM_WORLD.rank == 0:
    print(
        f"(D) Norm of velocity coefficient vector "
        f"(block, iterative): {norm_u}"
    )
    print(f"(D) Norm of pressure coefficient vector "
          f"(block, iterative): {norm_p}")
    print(f"(D) Norm of velocity analitical coefficient vector "
          f"(block, iterative): {norm_u_ana}")
    print(f"(D) Norm of pressure analitical coefficient vector "
          f"(block, iterative): {norm_p_ana}")

    
    
# ---------------------------------------------------------
# Compute true L2 errors between numerical and analytical solutions
# ---------------------------------------------------------

# Analytical velocity and pressure as Functions in same spaces
u_ana = Function(V)
u_ana.interpolate(lambda x: analitical_velocity(x, alpha[0]))

p_ana = Function(Q)
p_ana.interpolate(lambda x: analitical_pressure(x, beta[0]))

# Define error forms
error_u_L2 = fem.form(inner(u_n - u_ana, u_n - u_ana) * dx)
error_p_L2 = fem.form(inner(p_h - p_ana, p_h - p_ana) * dx)

# Assemble integrals (parallel safe)
err_u = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_u_L2), op=MPI.SUM))
err_p = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_p_L2), op=MPI.SUM))

# Also compute norms of the exact fields for relative error
norm_u_ana = np.sqrt(msh.comm.allreduce(
    fem.assemble_scalar(fem.form(inner(u_ana, u_ana) * dx)), op=MPI.SUM))
norm_p_ana = np.sqrt(msh.comm.allreduce(
    fem.assemble_scalar(fem.form(inner(p_ana, p_ana) * dx)), op=MPI.SUM))

# Relative errors
rel_err_u = err_u / norm_u_ana
rel_err_p = err_p / norm_p_ana

if MPI.COMM_WORLD.rank == 0:
    print(f"Velocity L2 error      = {err_u:.6e}")
    print(f"Pressure L2 error      = {err_p:.6e}")
    print(f"Velocity relative L2   = {rel_err_u:.6%}")
    print(f"Pressure relative L2   = {rel_err_p:.6%}")
