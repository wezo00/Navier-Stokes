import ufl
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import (assemble_vector, assemble_matrix,
                               apply_lifting, set_bc)

from typing import Callable, List, Tuple


def project(e, target_func, bcs=[]):
    """Project UFL expression.
    Note
    ----
    This method solves a linear system (using KSP defaults).

    Parameters:
    e (function): function to be projected
    target_func (function): new projected function
    bcs (function): possible boundary conditions
    """

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    v = ufl.TrialFunction(V)
    a = fem.form(ufl.inner(v, w) * dx)
    L = fem.form(ufl.inner(e, w) * dx)

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    if bcs:
        apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    # solver.setType("bcgs")
    # solver.getPC().setType("bjacobi")
    solver.setType("gmres")
    solver.getPC().setType("hypre")  # or "ilu" for direct solve
    solver.rtol = 1.0e-05
    solver.setOperators(A)

    def monitor(ksp, its, rnorm):
        print(f"Iteration {its}, residual norm {rnorm}")

    solver.setMonitor(monitor)

    solver.solve(b, target_func.x.petsc_vec)
    r, c = A.getDiagonal().array.min(), A.getDiagonal().array.max()
    cond_num = c / r
    print(f"Condition number: {cond_num}")
    assert solver.reason > 0, f"Solver failed with reason: {solver.reason}"
    target_func.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()


def define_facet_tags(
    domain: mesh.Mesh,
    boundaries: List[Tuple[int, Callable[[np.ndarray], np.ndarray]]]
) -> mesh.meshtags:
    """
    Define facet tags for a 2D mesh using custom boundary locators.

    Parameters:
    ----------
    domain : dolfinx.mesh.Mesh
        The mesh of the domain.
    boundaries : List[Tuple[int, Callable[[np.ndarray], np.ndarray]]]
        A list of (marker, locator function) pairs. The locator function should
        return a boolean array indicating whether a point lies on the
        desired facet.

    Returns:
    -------
    dolfinx.mesh.meshtags
        A meshtag object containing the tagged facets.
    """
    fdim = domain.topology.dim - 1
    facet_indices = []
    facet_markers = []

    for marker, locator in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        if len(facets) > 0:
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker, dtype=np.int32))

    if not facet_indices:
        raise ValueError("No facets found for any of the "
                         "provided locator functions.")

    facet_indices = np.hstack(facet_indices)
    facet_markers = np.hstack(facet_markers)
    sorted_facets = np.argsort(facet_indices)

    return mesh.meshtags(domain, fdim, facet_indices[sorted_facets],
                         facet_markers[sorted_facets])


def error_L2_func(Vh, V_ex, degree_raise=3):
    """
    Function to calculate the L2 error

    Parameter:
    Vh (function): numerical solution
    V_ex (function): analytical dolution
    degree_raise (int): dimension of the higher order FE space
    """
    # Create higher order function space
    degree = 1  # Vh.function_space.ufl_element().degree
    family = Vh.function_space.ufl_element().family_name
    mesh = Vh.function_space.mesh
    Q = fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    V_W = fem.Function(Q)
    V_W.interpolate(Vh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    V_ex_W = fem.Function(Q)

    V_ex_W.interpolate(V_ex)

    # Compute the error in the higher order function space
    e_W = fem.Function(Q)
    e_W.x.array[:] = V_W.x.array - V_ex_W.x.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def error_gradient_distance(phi_h):
    """
    Function to calculate the L2 error |grad(phi)|-1

    Parameter:
    phi_h (function): numerical solution
    """
    mesh = phi_h.function_space.mesh

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    grad_phi_num_ex = fem.Constant(mesh, PETSc.ScalarType(1))

    diff = (ufl.sqrt(ufl.inner(ufl.grad(phi_h), ufl.grad(phi_h)))
            - grad_phi_num_ex)
    # Integrate the error
    error = fem.form(ufl.inner(diff, diff) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def CFL_conditioning(k, j, dt, h):
    """
    In two dimension CFL = dt* sum_i [(avg_vel_i)/h_i] for i = x,y
    """
    mesh = j.function_space.mesh

    # Integrate velocity
    integral_quantity = fem.form((j[0]+j[1]) * ufl.dx)
    integral = -k*fem.assemble_scalar(integral_quantity)
    integral_global = mesh.comm.allreduce(integral, op=MPI.SUM)

    # Compute domain volume
    const = fem.Constant(mesh, default_scalar_type(1.))
    domain_volume_form = fem.form(const * ufl.dx)
    volume = fem.assemble_scalar(domain_volume_form)
    volume_global = mesh.comm.allreduce(volume, op=MPI.SUM)

    # Average velocity
    avg_vel = integral_global / volume_global

    return dt * avg_vel / h


def extract_point_interface(domain, phi):
    # Extract mesh topology
    topology = domain.topology
    # Ensure we can access element-node connectivity
    topology.create_connectivity(2, 0)
    cells = topology.connectivity(2, 0)

    # Extract node coordinates
    x = domain.geometry.x

    interface_points = []

    for cell in range(domain.topology.index_map(2).size_local):
        # Get vertex indices of the triangle
        vertices = cells.links(cell)
        phi_vals = phi.x.array[vertices]
        coords = x[vertices]

        # Find edges where sign change occurs
        edge_intersections = []
        for i in range(3):
            # v1, v2 = vertices[i], vertices[(i + 1) % 3]
            phi1, phi2 = phi_vals[i], phi_vals[(i + 1) % 3]
            if phi1 * phi2 < 0:  # Sign change means interface crosses edge
                x1, x2 = coords[i], coords[(i + 1) % 3]
                alpha = phi1 / (phi1 - phi2)  # Interpolation factor
                x_intersect = x1 + alpha * (x2 - x1)
                edge_intersections.append(x_intersect)

        if len(edge_intersections) == 2:
            interface_points.append(edge_intersections)

    interface_points = np.array(interface_points)

    cells = topology.connectivity(domain.topology.dim, 0).array.reshape(-1, 3)
    # Plot the extracted interface
    plt.figure(figsize=(6, 6))
    plt.triplot(x[:, 0], x[:, 1], cells, color="gray", alpha=0.3)
    for seg in interface_points:
        plt.plot(seg[:, 0], seg[:, 1], "r-", linewidth=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Extracted Level Set Interface")
    plt.axis("equal")
    plt.show()


class SmoothedHeaviside:
    """
    Class to define the Heaviside function for the definition of the
    conductivities in the two media.

    Parameters:
    phi_n_project [petsc4py.PETSc.Vec] : Input vector
    epsilon [float] : Smoothing parameter
    """
    def __init__(self, phi_n_project, epsilon):
        self.phi_n_project = phi_n_project
        self.epsilon = epsilon

    def __call__(self, x):
        # Convert PETSc Vec to NumPy array
        phi_array = self.phi_n_project.getArray()

        # Create output array
        value = np.zeros_like(phi_array)

        # Apply Heaviside function with smoothing
        value[phi_array < -self.epsilon] = 0
        value[phi_array > self.epsilon] = 1
        mask = (phi_array >= -self.epsilon) & (phi_array <= self.epsilon)
        value[mask] = (0.5 *
                       (1 + (phi_array[mask] / self.epsilon) +
                        np.sin(np.pi * phi_array[mask] / self.epsilon)/
                        np.pi))

        return value
    

class SmoothedDelta:
    """
    Class to define the smoothed Dirac delta function, which is the
    derivative of the smoothed Heaviside function.

    Parameters:
    phi_n_project [petsc4py.PETSc.Vec] : Input vector
    epsilon [float] : Smoothing parameter
    """
    def __init__(self, phi_n_project, epsilon):
        self.phi_n_project = phi_n_project
        self.epsilon = epsilon

    def __call__(self):
        # Convert PETSc Vec to NumPy array
        phi_array = self.phi_n_project.getArray()

        # Create output array
        value = np.zeros_like(phi_array)

        # Apply smoothed delta function
        mask = (phi_array >= -self.epsilon) & (phi_array <= self.epsilon)
        value[mask] = (0.5 / self.epsilon) * (
            1 + np.cos(np.pi * phi_array[mask] / self.epsilon)
        )

        return value


class DerivativeSmoothedDelta:
    """
    Class to define the derivative of the smoothed delta function
    (second derivative of the smoothed Heaviside function).

    Parameters:
    phi_n_project [petsc4py.PETSc.Vec] : Input vector
    epsilon [float] : Smoothing parameter
    """
    def __init__(self, phi_n_project, epsilon):
        self.phi_n_project = phi_n_project
        self.epsilon = epsilon

    def __call__(self):
        # Convert PETSc Vec to NumPy array
        phi_array = self.phi_n_project.getArray()

        # Create output array
        value = np.zeros_like(phi_array)

        # Apply derivative of delta function
        mask = (phi_array >= -self.epsilon) & (phi_array <= self.epsilon)
        value[mask] = ((-np.pi / (2 * self.epsilon**2)) *
                       np.sin(np.pi * phi_array[mask] / self.epsilon))

        return value


class LinearSolver:
    """
    A unified class that encapsulates both the PETSc solver setup and the
    assembly/solve process for a linear system defined by a bilinear form and
    a linear form.

    Attributes:
        comm: The MPI communicator.
        A: The PETSc matrix.
        b: The PETSc vector (RHS).
        left_form: The bilinear form used for assembling A.
        right_form: The linear form used for assembling b.
        BCs: A list of Dirichlet boundary conditions.
        flag: A boolean flag that, if True, applies additional BC adjustments
          to the RHS.
        solver_type: The PETSc KSP solver type (default is PREONLY).
        pc_type: The PETSc preconditioner type (default is LU).
        rtol: Relative tolerance for iterative solvers.
        max_it: Maximum number of iterations.
        monitor: Whether to print iteration info.
        solver: The PETSc KSP solver instance.
    """

    def __init__(self, comm, A, b, left_form, right_form, BCs, flag,
                 solver_type=PETSc.KSP.Type.PREONLY, pc_type=PETSc.PC.Type.LU,
                 rtol=1e-5, max_it=1000, monitor=False, nullspace=None):
        self.comm = comm
        self.A = A
        self.b = b
        self.left_form = left_form
        self.right_form = right_form
        self.BCs = BCs
        self.flag = flag
        self.solver_type = solver_type
        self.pc_type = pc_type
        self.rtol = rtol
        self.max_it = max_it
        self.monitor = monitor
        self.nullspace = nullspace
        self.solver = None

    def setup_solver(self):
        """
        Set up the PETSc KSP solver with the provided matrix A.
        """
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(self.A)
        solver.setType(self.solver_type)
        solver.getPC().setType(self.pc_type)

        # If an iterative solver is used, set tolerances
        if self.solver_type != PETSc.KSP.Type.PREONLY:
            solver.setTolerances(rtol=self.rtol, max_it=self.max_it)
        else:
            # Set the solver type to MUMPS (LU solver) and configure MUMPS to
            # handle pressure nullspace
            pc = solver.getPC()
            pc.setType("lu")
            sys = PETSc.Sys()  # type: ignore
            use_superlu = PETSc.IntType == np.int64
            if sys.hasExternalPackage("mumps") and not use_superlu:
                pc.setFactorSolverType("mumps")
                pc.setFactorSetUpSolverType()
                # If you encounter a zero pivot, don’t abort; just continue.
                pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
                # Don’t return an error code for that zero pivot.
                pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
            else:
                pc.setFactorSolverType("superlu_dist")

        if self.nullspace is not None:
            if self.solver_type != PETSc.KSP.Type.PREONLY:
                self.A.setNearNullSpace(self.nullspace)
            else:
                self.A.setNullSpace(self.nullspace)
                self.nullspace.remove(self.b)

        # Optional: Monitor iterations
        if self.monitor:
            def monitor(ksp, its, rnorm):
                print(f"    Iteration {its}, residual norm {rnorm}")
            solver.setMonitor(monitor)

        return solver

    def assemble_system(self):
        """
        Assemble the matrix and right-hand side vector for the linear system.
        """
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.left_form, bcs=self.BCs)
        self.A.assemble()

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b, self.right_form)

        if self.flag:
            apply_lifting(self.b, [self.left_form], [self.BCs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                               mode=PETSc.ScatterMode.REVERSE)
            set_bc(self.b, self.BCs)

        self.solver = self.setup_solver()

    def solve(self, solution_vec):
        """
        Assemble the system and solve for the given solution vector.

        Parameters:
            solution_vec: The PETSc vector to store the solution.
        """
        self.assemble_system()
        self.solver.solve(self.b, solution_vec.petsc_vec)
        solution_vec.scatter_forward()

    def destroy(self):
        """
        Destroy PETSc objects to free memory.
        """
        if self.solver is not None:
            self.solver.destroy()
            self.solver = None

        if self.A is not None:
            self.A.destroy()
            self.A = None

        if self.b is not None:
            self.b.destroy()
            self.b = None

    def __enter__(self):
        """Return self when entering a context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup PETSc objects when exiting the context."""
        self.destroy()
