# manufactured_stokes_convergence.py
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import math

import ufl
from ufl import dx, inner, div, grad, as_vector, SpatialCoordinate
from basix.ufl import element

import dolfinx
from dolfinx import fem, la
from dolfinx.fem import Function, dirichletbc, form, locate_dofs_topological, functionspace
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.io import XDMFFile

# --- Manufactured solution (divergence-free velocity + pressure) ---
# Choose smooth analytic fields
pi = math.pi

def make_ufl_exact():
    x = SpatialCoordinate(msh)  # will be set after mesh creation
    # We'll set placeholders; caller will re-create these per-mesh
    return None

# We'll pick:
# u = [ sin(pi*x) sin(pi*y), - sin(pi*x) sin(pi*y) ]  (div = 0)
# p = cos(pi*x) * cos(pi*y)

def run_on_mesh(nx, ny, verbose=True):
    # create mesh
    global msh
    msh = create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
                           [nx, ny], CellType.triangle)

    # define finite elements (Taylor-Hood: P2 - P1)
    P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,), dtype=fem.default_float_type)
    P1 = element("Lagrange", msh.basix_cell(), 1, dtype=fem.default_float_type)
    V = functionspace(msh, P2)
    Q = functionspace(msh, P1)

    # Spatial coordinate and analytic functions as UFL expressions
    x = ufl.SpatialCoordinate(msh)
    u_exact_ufl = as_vector([ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]),
                             - ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])])
    p_exact_ufl = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Compute forcing: f = -div(grad(u)) + grad(p)
    # Note: for vector Laplacian using component-wise div(grad(u))
    lap_u = ufl.as_vector([div(grad(u_exact_ufl[i])) for i in range(msh.geometry.dim)])
    grad_p = grad(p_exact_ufl)
    f_ufl = -lap_u + grad_p

    # Convert analytic fields to Functions for BCs and error measurement
    u_exact_fun = Function(V)
    # interpolate expects a function mapping points -> values
    def u_exact_eval(x_points):
        # x_points shape (2, N)
        N = x_points.shape[1]
        vals = np.zeros((2, N), dtype=fem.default_float_type)
        vals[0, :] = np.sin(pi * x_points[0, :]) * np.sin(pi * x_points[1, :])
        vals[1, :] = -vals[0, :]
        return vals
    u_exact_fun.interpolate(u_exact_eval)

    p_exact_fun = Function(Q)
    def p_exact_eval(x_points):
        N = x_points.shape[1]
        vals = np.cos(pi * x_points[0, :]) * np.cos(pi * x_points[1, :])
        return vals
    p_exact_fun.interpolate(p_exact_eval)

    # Boundary condition: enforce u = u_exact on whole boundary
    def on_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)

    facets = dolfinx.mesh.locate_entities_boundary(msh, msh.topology.dim - 1, on_boundary)
    bdofs = locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dirichletbc(u_exact_fun, bdofs, V)
    bcs = [bc]

    # Define variational forms (Stokes): -div(grad(u)) + grad(p) = f, div u = 0
    (u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    (v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)

    a = form([[inner(grad(u), grad(v)) * dx, - inner(p, div(v)) * dx],
              [inner(div(u), q) * dx, None]])
    L = form([inner(f_ufl, v) * dx, inner(ufl.Constant(msh, 0.0), q) * dx])  # RHS for pressure eqn is 0

    # Block preconditioner matrix for pressure mass (for compatibility with iterative solvers)
    a_p11 = form(inner(p, q) * dx)
    a_p = [[a[0][0], None], [None, a_p11]]

    # Assemble block matrices
    A = assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    P = assemble_matrix_block(a_p, bcs=bcs)
    P.assemble()
    b = assemble_vector_block(L, a, bcs=bcs)

    # Pressure nullspace (pressure only indeterminate up to const)
    null_vec = A.createVecLeft()
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    null_vec.array[offset:] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    A.setNullSpace(nsp)

    # Solve with direct LU (robust for small meshes). Use KSP preonly + LU.
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    # try to use superlu_dist if available, else default
    try:
        pc.setFactorSolverType("superlu_dist")
    except Exception:
        pass
    ksp.setFromOptions()
    x = A.createVecRight()
    ksp.solve(b, x)

    # Extract solutions into Functions
    u_h = Function(V)
    p_h = Function(Q)
    offset_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u_h.x.array[:offset_local] = x.array_r[:offset_local]
    p_h.x.array[:] = x.array_r[offset_local:]

    # Compute L2 errors (parallel-safe)
    e_u_form = fem.form(inner(u_h - u_exact_fun, u_h - u_exact_fun) * dx)
    e_p_form = fem.form(inner(p_h - p_exact_fun, p_h - p_exact_fun) * dx)
    e_u_loc = fem.assemble_scalar(e_u_form)
    e_p_loc = fem.assemble_scalar(e_p_form)
    e_u = math.sqrt(msh.comm.allreduce(e_u_loc, op=MPI.SUM))
    e_p = math.sqrt(msh.comm.allreduce(e_p_loc, op=MPI.SUM))

    # Norms of exact solutions for relative error
    norm_u_loc = fem.assemble_scalar(fem.form(inner(u_exact_fun, u_exact_fun) * dx))
    norm_p_loc = fem.assemble_scalar(fem.form(inner(p_exact_fun, p_exact_fun) * dx))
    norm_u = math.sqrt(msh.comm.allreduce(norm_u_loc, op=MPI.SUM))
    norm_p = math.sqrt(msh.comm.allreduce(norm_p_loc, op=MPI.SUM))

    if MPI.COMM_WORLD.rank == 0 and verbose:
        print(f"mesh {nx}x{ny}: dofs (u) ~ {V.dofmap.index_map.size_global*V.dofmap.index_map_bs}, (p) ~ {Q.dofmap.index_map.size_global}")
        print(f"  L2 error u = {e_u:.6e}, rel = {e_u/norm_u:.4%}")
        print(f"  L2 error p = {e_p:.6e}, rel = {e_p/norm_p:.4%}")

    # cleanup PETSc objects
    A.destroy(); P.destroy(); b.destroy(); x.destroy(); ksp.destroy()
    return e_u, e_p, norm_u, norm_p

# --- Run convergence study on sequence of mesh refinements ---
if __name__ == "__main__":
    # meshes to test (number of cells per direction)
    mesh_sizes = [8, 16, 32, 64]  # you can change or extend
    errs_u = []
    errs_p = []
    norms_u = []
    norms_p = []
    for n in mesh_sizes:
        eu, ep, nu, np_ = run_on_mesh(n, n)
        errs_u.append(eu); errs_p.append(ep)
        norms_u.append(nu); norms_p.append(np_)

    # compute convergence rates (log2 ratios)
    if MPI.COMM_WORLD.rank == 0:
        print("\nConvergence rates (estimated):")
        for i in range(1, len(mesh_sizes)):
            rate_u = math.log(errs_u[i-1]/errs_u[i], 2)
            rate_p = math.log(errs_p[i-1]/errs_p[i], 2)
            print(f"  {mesh_sizes[i-1]} -> {mesh_sizes[i]} : rate u ~ {rate_u:.2f}, rate p ~ {rate_p:.2f}")
