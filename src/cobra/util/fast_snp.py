"""Provides Fast-SNP algorithm implementation for finding nullspace basis of matrix."""

from typing import List, Tuple

import numpy as np
import optlang
from optlang.interface import OPTIMAL, Model, Variable
from optlang.symbolics import Zero


def _solve_snv(
    weights: np.ndarray, model: Model, v_list: List[Variable], positive: bool
) -> np.ndarray:
    dir = 1 if positive else -1

    model.constraints["nonzero_constraint"].set_linear_coefficients(
        {variable: weight * dir for variable, weight in zip(v_list, weights)}
    )

    model.optimize()
    if model.status != OPTIMAL:
        return None

    result = np.array([variable.primal for variable in v_list])
    result /= np.linalg.norm(result)

    return result


def _create_fast_snp_problem(
    solver: "optlang.interface",
    S: np.ndarray,
    directions: np.ndarray,
    v_bound: float,
    zero_cutoff: float,
    bias: float,
) -> Tuple[Model, List[Variable]]:
    n = S.shape[1]

    model = solver.Model()
    modulo_list = []
    v_list = []

    for i in range(n):
        v = solver.Variable(
            name=f"v_{i}",
            lb=v_bound * min(directions[i, 0], 0),
            ub=v_bound * max(directions[i, 1], 0),
            problem=model,
        )
        model.add([v])
        v_list.append(v)

        if directions[i, 0] == 0:
            modulo_list.append((v, 1))
        elif directions[i, 1] == 0:
            modulo_list.append((v, -1))
        else:
            x = solver.Variable(name=f"x_{i}", lb=0, problem=model)
            model.add([x])
            modulo_list.append((x, 1))

            constraint1 = solver.Constraint(
                Zero,
                lb=0.0,
                name=f"modulo_constraint1_{i}",
            )
            constraint2 = solver.Constraint(
                Zero,
                lb=0.0,
                name=f"modulo_constraint2_{i}",
            )
            model.add([constraint1, constraint2], sloppy=True)

            model.constraints[f"modulo_constraint1_{i}"].set_linear_coefficients(
                {x: 1.0, v: -1.0}
            )
            model.constraints[f"modulo_constraint2_{i}"].set_linear_coefficients(
                {x: 1.0, v: 1.0}
            )

    for idx, row in enumerate(S):
        nnz_list = np.flatnonzero(np.abs(row) > zero_cutoff)
        if len(nnz_list) == 0:
            continue

        constraint = solver.Constraint(Zero, lb=0.0, ub=0.0, name=f"row_{idx}")
        model.add([constraint], sloppy=True)
        model.constraints[f"row_{idx}"].set_linear_coefficients(
            {v_list[i]: row[i] for i in nnz_list}
        )

    model.add(
        [
            solver.Constraint(
                Zero,
                lb=bias,
                name="nonzero_constraint",
            )
        ],
        sloppy=True,
    )

    model.objective = solver.Objective(Zero)
    model.objective.set_linear_coefficients({x: coef for (x, coef) in modulo_list})
    model.objective.direction = "min"

    return model, v_list


def _project(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w - (N.T @ (N @ w))


def _get_condition_vector(N: np.ndarray) -> np.ndarray:
    return _project(N, np.random.normal(size=N.shape[1]))


def nullspace_fast_snp(
    solver: "optlang.interface",
    S: np.ndarray,
    directions: np.ndarray,
    v_bound: float = 1e4,
    zero_cutoff: float = 1e-6,
    bias: float = 1,
    required_stop_checks_num: int = 3,
) -> np.ndarray:
    """Compute an approximate basis for the nullspace of S with coordinate directions.

    The algorithm used by this function is described in [1]_.

    Parameters
    ----------
    solver : "optlang.interface"
        The solver interface to use for the optimization problem.
        You can use `model.problem` to get the solver interface.
    S : numpy.ndarray
        The matrix for which the nullspace is computed.
        `S` should be a 2-D array.
    directions : numpy.ndarray
        A 2-D array with shape (k, 2) where `k` is the number of columns in `S`.
        This array specifies the directions of coordinates.
        Each row should be:
            - [0, 0] for coordinates that can be only zero
            - [0, 1] for coordinates that can be only positive
            - [-1, 0] for coordinates that can be only negative
            - [-1, 1] for coordinates that can be both positive and negative
    v_bound : float, optional
        The bound for the variables in the optimization problem (default 1e4).
    zero_cutoff : float, optional
        The cutoff value to consider a coordinate value as zero (default 1e-6).
    bias : float, optional
        The bias for the non-zero constraint in the optimization problem
        (default 1).
    required_stop_checks_num : int, optional
        The number of random checks to pass to prove that basis
        could not be expanded (default 3).

    Returns
    -------
    numpy.ndarray
        If `S` is an array with shape (m, k), then an array
        with shape (k, n) will be returned, where `n` is the dimension of the
        nullspace of `S` with `directions`. Each column of this array is a basis
        vector for the nullspace; each element in numpy.dot(S, column) will be
        approximately zero. Each coordinate in the column will have an allowed
        sign according to the `directions` parameter.

    References
    ----------
    .. [1] Fast-SNP: a fast matrix pre-processing algorithm for efficient
       loopless flux optimization of metabolic models. Saa PA, Nielsen LK.
       Bioinformatics. 2016 Dec;32(24):3807–3814. doi: 10.1093/bioinformatics/btw555.
    """

    if len(S.shape) != 2 or S.shape[0] == 0 or S.shape[1] == 0:
        raise ValueError("Input matrix S must be a 2D array with non-zero dimensions.")

    problem, v_list = _create_fast_snp_problem(
        solver, S, directions, v_bound, zero_cutoff, bias
    )

    n = S.shape[1]
    N = np.zeros((0, n))
    U = np.zeros((0, n))
    stop_checks_num = 0

    while N.shape[0] < n and stop_checks_num < required_stop_checks_num:
        weights = _get_condition_vector(U)

        v1 = _solve_snv(weights, problem, v_list, True)
        v2 = _solve_snv(weights, problem, v_list, False)

        if v1 is None and v2 is None:
            stop_checks_num += 1
            continue

        stop_checks_num = 0
        v = v1
        if v1 is None:
            v = v2

        if v1 is not None and v2 is not None:
            nnz_v1 = np.sum(np.abs(v1) > zero_cutoff)
            nnz_v2 = np.sum(np.abs(v2) > zero_cutoff)
            if nnz_v1 > nnz_v2:
                v = v2

        N = np.vstack([N, v])

        v = _project(U, v)
        v /= np.linalg.norm(v)
        U = np.vstack([U, v])

    return N.T
