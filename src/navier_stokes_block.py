from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import default_real_type, fem, la
from dolfinx.fem import (
    Constant, Function, dirichletbc, form,
    functionspace, locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
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


# Function to mark x = 0, x = 1, y = 0 and y = 1
def noslip_boundary(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
            np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)


# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [20, 20],
    CellType.triangle
)

P2 = element(
    "Lagrange", msh.basix_cell(), degree=2, shape=(msh.geometry.dim,),
    dtype=default_real_type
)
P1 = element("Lagrange", msh.basix_cell(), degree=1, dtype=default_real_type)

V, Q = functionspace(msh, P2), functionspace(msh, P1)
D = fem.functionspace(msh, ("DG", 0))

# No-slip condition on boundaries where x = 0, x = 1, y = 0 and y=1
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)  # type: ignore
u_fixed = Function(V)
u_fixed.interpolate(lambda x: analitical_velocity(x, 1))
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(u_fixed, locate_dofs_topological(V, 1, facets))

# Driving (lid) velocity condition on top boundary (y = 1)
p_fixed = Function(Q)
p_fixed.interpolate(lambda x: analitical_pressure(x, 1))
facets = locate_entities_boundary(msh, 1, lid)
bc1 = dirichletbc(p_fixed, locate_dofs_topological(Q, 1, facets))

# Collect Dirichlet boundary conditions
bcs = [bc0]

# Define variational problem
(u, p, lam) = ufl.TrialFunction(V), ufl.TrialFunction(Q), ufl.TrialFunction(D)
(v, q, eta) = ufl.TestFunction(V), ufl.TestFunction(Q), ufl.TestFunction(D)

u_n = fem.Function(V)
u_n.x.petsc_vec.set(0)
nu = Constant(msh, PETSc.ScalarType(1e-1))

a = form([
        [(2*nu*inner(sym(nabla_grad(u)), sym(nabla_grad(v)))
          + dot(dot(u_n, nabla_grad(u)), v)
          + dot(dot(u, nabla_grad(u_n)), v)) * dx, - inner(p, div(v)) * dx],
        [-inner(q, div(u)) * dx, None],
    ])
f = fem.Function(V)
f.interpolate(lambda x: analitical_force(x, nu.value, 1, 1))
L = form([(dot(dot(u_n, nabla_grad(u_n)), v)
           + inner(f, v)) * dx,
         inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])


def block_operators():
    """Return block operators and block RHS vector for the Navier-Stokes
    problem"""

    # Assembler matrix operator, preconditioner and RHS vector into
    # single objects but preserving block structure
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    # If you set the pressure somewhere you don't need the null space
    # Set the nullspace for pressure (since pressure is determined only
    # up to a constant)
    null_vec = A.createVecLeft()
    # V.dofmap.index_map.size_local = local dof (index map that described
    # the parallel distribution of the dofmap)
    # V = [(N+1)*N]*[(N+1)*N] since P2 FE and Q = (N+1)*(N+1) since P1
    # V.dofmap.index_map_bs = Block size of the index map.
    # V = 2 since u has two components and Q = 1
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    null_vec.array[offset:] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    assert nsp.test(A)
    A.setNullSpace(nsp)

    return A, b


def block_iterative_solver():
    """Solve the Navier-Stokes problem using blocked matrices and an iterative
    solver."""

    # Assembler the operators and RHS vector
    A, b = block_operators()
    '''A_dense = A.convert("dense")
    A_dense.assemble()

    # Get the values as a NumPy array
    array = A_dense.getDenseArray()

    # Compute determinant
    det = np.linalg.cond(array)
    print("Determinant:", det)'''
    # Build PETSc index sets for each field (global dof indices for each
    # field)
    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    # V_map.local_range = Range of indices (global) owned by this process.
    # V = [0, (N+1)*N] and Q = [0,(N+1)]
    # Because dof u starts from zero
    offset_u = (V_map.local_range[0] * V.dofmap.index_map_bs +
                Q_map.local_range[0])
    # Because dof p starts after dof u
    offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
    is_u = PETSc.IS().createStride(
        V_map.size_local * V.dofmap.index_map_bs, offset_u, 1,
        comm=PETSc.COMM_SELF
    )
    is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1,
                                   comm=PETSc.COMM_SELF)

    # Create a GMRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setTolerances(rtol=1e-9)
    ksp.setType("gmres")
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    # Configure velocity and pressure sub-solvers
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()

    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
    ksp_u.setType("preonly")
    # Use HYPRE BoomerAMG for nonsymmetric systems
    ksp_u.getPC().setType("hypre")
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

    return A, b, ksp


# Create a block vector (x) to store the full solution and solve Stokes
# to have an initial guess
A, b, ksp = block_iterative_solver()

x = A.createVecRight()
ksp.solve(b, x)

# Create Functions to split u and p
u_h, p_h = Function(V), Function(Q)
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
offset_p = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
u_h.x.array[:offset] = x.array_r[:offset]
p_h.x.array[:offset_p] = x.array_r[offset: (offset_p + offset)]
u_n.x.array[:] = u_h.x.array

with XDMFFile(MPI.COMM_WORLD, "v_block_iterative_ns.xdmf", "w") as ufile_xdmf:
    u_h.x.scatter_forward()
    P1 = element(
        "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,),
        dtype=default_real_type
    )
    u1 = Function(functionspace(msh, P1))
    u1.interpolate(u_n)
    u1.name = "Velocity"
    ufile_xdmf.write_mesh(msh)
    ufile_xdmf.write_function(u1)

with XDMFFile(MPI.COMM_WORLD, "p_block_iterative_ns.xdmf", "w") as pfile_xdmf:
    p_h.x.scatter_forward()
    p_h.name = "Pressure"
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(p_h)

max_iterations = 30
i = 1

uD = fem.Function(V)
uD.interpolate(lambda x: analitical_velocity(x, 1))
L2_stop = fem.form(dot(u_n - u_h, u_n - u_h) * dx)

while i < max_iterations:
    A, b, ksp = block_iterative_solver()

    x = A.createVecRight()
    ksp.solve(b, x)

    u_h.x.array[:offset] = x.array_r[:offset]
    p_h.x.array[:offset_p] = x.array_r[offset: (offset_p + offset)]

    i += 1

    u1.interpolate(u_h)
    u1.x.scatter_forward()
    ufile_xdmf.write_function(u1, i)
    pfile_xdmf.write_function(p_h, i)

    A.destroy()
    b.destroy()
    x.destroy()
    ksp.destroy()

    # Compute norm of update
    stopping_crit = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(L2_stop),
                                               op=MPI.SUM))
    print(f"Iteration {i}: ||u_n+1 - u_n|| = {stopping_crit}")
    if stopping_crit < 1e-10:
        break

    u_n.x.array[:] = u_h.x.array

norm_u, norm_p = la.norm(u_n.x), la.norm(p_h.x)
if MPI.COMM_WORLD.rank == 0:
    print(
        f"(D) Norm of velocity coefficient vector "
        f"(block, iterative): {norm_u}"
    )
    print(f"(D) Norm of pressure coefficient vector "
          f"(block, iterative): {norm_p}")
