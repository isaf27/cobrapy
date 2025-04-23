from typing import TYPE_CHECKING, Optional, List, Tuple

import numpy as np
import optlang
from optlang.symbolics import Zero

from ..util import solver as sutil
from ..util import create_stoichiometric_matrix
from .helpers import normalize_cutoff


if TYPE_CHECKING:
    from cobra import Model


def _create_find_cyclic_reactions_problem(
    solver: "optlang.interface",
    s_int: np.ndarray,
    directions_int: np.ndarray,
    zero_cutoff: float,
    bound: float
) -> Tuple["optlang.interface.Model", List["optlang.interface.Variable"]]:
    model = solver.Model()
    
    q_list = []
    for i in range(s_int.shape[1]):
        q = solver.Variable(
            name=f'q_{i}',
            lb=bound*min(0, directions_int[i, 0]),
            ub=bound*max(0, directions_int[i, 1]),
            problem=model
        )
        q_list.append(q)
    model.add(q_list)

    for idx, row in enumerate(s_int):
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
        constraint.set_linear_coefficients({q_list[i]: row[i] for i in nnz_list})

    model.objective = solver.Objective(Zero)

    return model, q_list


def find_cyclic_reactions(
    model: "Model",
    zero_cutoff: Optional[float] = None,
    bound: float = 1e4,
    method: str = "optimized"
) -> List[str]:
    """
        TODO: @isaf27
    """
    zero_cutoff = normalize_cutoff(model, zero_cutoff)

    internal = [i for i, r in enumerate(model.reactions) if not r.boundary]
    s_int = create_stoichiometric_matrix(model)[:, np.array(internal)]
    n = s_int.shape[1]

    bounds_int = np.array([model.reactions[i].bounds for i in internal])
    directions_int = np.sign(bounds_int)

    lp, q_list = _create_find_cyclic_reactions_problem(
        model.problem,
        s_int,
        directions_int,
        zero_cutoff,
        bound
    )
    
    candidate_reactions = list(range(n))

    if method == "optimized":
        is_active = [False] * n

        random_checks_left_count = 3
        def set_random_weights():
            nonlocal random_checks_left_count
            random_checks_left_count -= 1
            
            weights = np.random.uniform(0.5, 1.0, size=n) * (2 * np.random.randint(low=0, high=2, size=n) - 1)
            weights /= np.sum(np.abs(weights))
            lp.objective.set_linear_coefficients({q_list[i]: weights[i] for i in range(n) if not is_active[i]})

        set_random_weights()
        dir_order = ["min", "max"]

        while True:
            found_active = False
            reverse_dir_order = False

            for dir in dir_order:
                lp.objective.direction = dir
                lp.optimize()

                sutil.check_solver_status(lp.status)

                if abs(lp.objective.value) > zero_cutoff:
                    remove_coef = {}
                    for i in range(n):
                        if not is_active[i] and abs(q_list[i].primal) > zero_cutoff:
                            is_active[i] = True
                            remove_coef[q_list[i]] = 0
                    found_active = True
                    lp.objective.set_linear_coefficients(remove_coef)
                    break

                reverse_dir_order = True

            if not found_active:
                if random_checks_left_count > 0:
                    set_random_weights()
                    continue
                break

            if reverse_dir_order:
                dir_order.reverse()

        candidate_reactions = [i for i in range(n) if is_active[i]]

    can_positive = [False] * n
    can_negative = [False] * n
    lp.objective = model.problem.Objective(Zero, direction="max")
    for i in candidate_reactions:
        for dir in ["min", "max"]:
            lp.objective.set_linear_coefficients({q_list[i]: 1.0 if dir == "max" else -1.0})
            
            lp.optimize()
            sutil.check_solver_status(lp.status)
            
            if lp.objective.value > zero_cutoff:
                if dir == "max":
                    can_positive[i] = True
                else:
                    can_negative[i] = True

        lp.objective.set_linear_coefficients({q_list[i]: 0.0})

    cyclic_reactions = [model.reactions[internal[i]].id for i in range(n) if can_positive[i] or can_negative[i]]
    cyclic_reactions_directions = [(can_positive[i], can_negative[i]) for i in range(n) if can_positive[i] or can_negative[i]]

    return cyclic_reactions, cyclic_reactions_directions
