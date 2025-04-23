import numpy as np

from typing import Tuple, List

import optlang
from optlang.interface import Model, Variable, OPTIMAL
from optlang.symbolics import Zero


def _solve_snv(weights: np.ndarray, model: Model, v_list: List[Variable], positive: bool) -> np.ndarray:
    dir = 1 if positive else -1

    model.constraints["nonzero_constraint"].set_linear_coefficients({
        variable: weight * dir
        for variable, weight in zip(v_list, weights)
    })

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
    x_list = []
    v_list = []

    for i in range(n):
        x = solver.Variable(name=f'x_{i}', lb=0, problem=model)
        v = solver.Variable(name=f'v_{i}', 
                            lb=v_bound*min(directions[i, 0], 0),
                            ub=v_bound*max(directions[i, 1], 0),
                            problem=model)
        model.add([x, v])
        x_list.append(x)
        v_list.append(v)
        
        constraint1 = solver.Constraint(Zero, lb=0.0, problem=model, name=f'modulo_constraint1_{i}')
        constraint2 = solver.Constraint(Zero, lb=0.0, problem=model, name=f'modulo_constraint2_{i}')
        model.add([constraint1, constraint2], sloppy=True)
        
        constraint1.set_linear_coefficients({x: 1.0, v: -1.0})
        constraint2.set_linear_coefficients({x: 1.0, v: 1.0})

    for idx, row in enumerate(S):
        nnz_list = np.flatnonzero(np.abs(row) > zero_cutoff)
        if len(nnz_list) == 0:
            continue

        constraint = solver.Constraint(
            Zero,
            lb=0.0,
            ub=0.0,
            problem=model,
            name=f'row_{idx}'
        )
        model.add([constraint], sloppy=True)
        constraint.set_linear_coefficients({v_list[i]: row[i] for i in nnz_list})

    model.add([solver.Constraint(
        Zero,
        lb=bias,
        problem=model,
        name="nonzero_constraint",
    )], sloppy=True)

    model.objective = solver.Objective(Zero)
    model.objective.set_linear_coefficients({x: 1 for x in x_list})
    model.objective.direction = "min"

    return model, v_list


def _project(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w - (N.T @ (N @ w))


def _get_condition_vector(N: np.ndarray) -> np.ndarray:
    return _project(N, np.random.uniform(-1.0, 1.0, size=N.shape[1]))


def nullspace_fast_snp(
    solver: "optlang.interface",
    S: np.ndarray,
    directions: np.ndarray,
    v_bound: float = 1e4,
    zero_cutoff: float = 1e-6,
    bias: float = 1e-3,
) -> np.ndarray:
    """
        TODO: @isaf27
    """

    problem, v_list = _create_fast_snp_problem(solver, S, directions, v_bound, zero_cutoff, bias)

    n = S.shape[1]
    N = np.zeros((0, n))
    U = np.zeros((0, n))

    for _ in range(n):
        weights = _get_condition_vector(U)

        v1 = _solve_snv(weights, problem, v_list, True)
        v2 = _solve_snv(weights, problem, v_list, False)

        if v1 is None and v2 is None:
            break

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
