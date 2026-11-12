"""Provide the building blocks of the loopless MILP formulations."""

from typing import TYPE_CHECKING, Dict, List

import numpy as np
from optlang.symbolics import Zero


if TYPE_CHECKING:
    from optlang.interface import Variable

    from cobra import Model, Reaction


def add_nullspace_energies(
    model: "Model",
    reactions: List["Reaction"],
    n_int: np.ndarray,
    zero_cutoff: float,
) -> List[Dict["Variable", float]]:
    """Add an energy variable G_i per reaction with N_int * G = 0.

    Returns the energy of each reaction as coefficients of a linear expression.
    """
    prob = model.problem
    energies = [prob.Variable(f"delta_g_{rxn.id}") for rxn in reactions]
    constraints = [
        prob.Constraint(Zero, lb=0, ub=0, name=f"nullspace_constraint_{i}")
        for i in range(len(n_int))
    ]
    model.add_cons_vars(energies + constraints)
    model.solver.update()
    for constraint, row in zip(constraints, n_int):
        constraint.set_linear_coefficients(
            {
                energy: coefficient
                for energy, coefficient in zip(energies, row)
                if abs(coefficient) > zero_cutoff
            }
        )
    return [{energy: 1.0} for energy in energies]


def add_potential_energies(
    model: "Model",
    s_int: np.ndarray,
    zero_cutoff: float,
) -> List[Dict["Variable", float]]:
    """Add a potential variable per metabolite, so that G = S_int^T * potentials.

    Returns the energy of each reaction as coefficients of a linear expression.
    """
    potentials = [
        model.problem.Variable(f"potential_{i}") for i in range(s_int.shape[0])
    ]
    model.add_cons_vars(potentials)
    return [
        {
            potential: coefficient
            for potential, coefficient in zip(potentials, column)
            if abs(coefficient) > zero_cutoff
        }
        for column in s_int.T
    ]


def add_indicators(
    model: "Model",
    reactions: List["Reaction"],
    energies: List[Dict["Variable", float]],
    max_bound: float,
) -> None:
    """Add one indicator per reaction that couples its direction to its energy."""
    prob = model.problem
    indicators = []
    to_add = []
    for rxn in reactions:
        # indicator variable a_i
        indicator = prob.Variable(f"indicator_{rxn.id}", type="binary")
        # lb_i * (1 - a_i) <= v_i <= ub_i * a_i
        on_off_forward = prob.Constraint(
            rxn.flux_expression - rxn.upper_bound * indicator,
            ub=0,
            name=f"on_off_forward_{rxn.id}",
        )
        on_off_reverse = prob.Constraint(
            rxn.flux_expression + rxn.lower_bound * indicator,
            lb=rxn.lower_bound,
            name=f"on_off_reverse_{rxn.id}",
        )
        indicators.append(indicator)
        to_add.extend([indicator, on_off_forward, on_off_reverse])
    model.add_cons_vars(to_add)

    # -(max_bound + 1) * a_i + 1 <= G_i <= -(max_bound + 1) * a_i + max_bound
    energy_ranges = [
        prob.Constraint(Zero, lb=1, ub=max_bound, name=f"delta_g_range_{rxn.id}")
        for rxn in reactions
    ]
    model.add_cons_vars(energy_ranges)
    model.solver.update()
    for energy_range, energy, indicator in zip(energy_ranges, energies, indicators):
        energy_range.set_linear_coefficients({**energy, indicator: max_bound + 1})


def add_directional_indicators(
    model: "Model",
    reactions: List["Reaction"],
    energies: List[Dict["Variable", float]],
    max_bound: float,
    flux_threshold: float,
) -> None:
    """Add one indicator per reaction direction, coupled to the energy.

    An active indicator requires a flux of at least `flux_threshold` in its
    direction. At most one indicator per reaction can be active.
    """
    prob = model.problem
    indicators = []
    to_add = []
    for rxn in reactions:
        # indicator variables a_i+ and a_i-
        indicator_maximum = prob.Variable(f"indicator_maximum_{rxn.id}", type="binary")
        indicator_minimum = prob.Variable(f"indicator_minimum_{rxn.id}", type="binary")
        single_direction = prob.Constraint(
            indicator_maximum + indicator_minimum,
            ub=1,
            name=f"single_nonzero_direction_{rxn.id}",
        )
        # flux_threshold * a_i+ <= v_i+ <= ub_i * a_i+
        on_off_maximum1 = prob.Constraint(
            rxn.forward_variable - rxn.upper_bound * indicator_maximum,
            ub=0,
            name=f"on_off_maximum1_{rxn.id}",
        )
        on_off_maximum2 = prob.Constraint(
            rxn.forward_variable - flux_threshold * indicator_maximum,
            lb=0,
            name=f"on_off_maximum2_{rxn.id}",
        )
        # flux_threshold * a_i- <= v_i- <= -lb_i * a_i-
        on_off_minimum1 = prob.Constraint(
            rxn.reverse_variable + rxn.lower_bound * indicator_minimum,
            ub=0,
            name=f"on_off_minimum1_{rxn.id}",
        )
        on_off_minimum2 = prob.Constraint(
            rxn.reverse_variable - flux_threshold * indicator_minimum,
            lb=0,
            name=f"on_off_minimum2_{rxn.id}",
        )
        indicators.append((indicator_maximum, indicator_minimum))
        to_add.extend(
            [
                indicator_maximum,
                indicator_minimum,
                single_direction,
                on_off_maximum1,
                on_off_maximum2,
                on_off_minimum1,
                on_off_minimum2,
            ]
        )
    model.add_cons_vars(to_add)

    energy_ranges = []
    for rxn in reactions:
        # G_i <= -(max_bound + 1) * a_i+ + max_bound
        energy_range_maximum = prob.Constraint(
            Zero, ub=max_bound, name=f"delta_g_range_maximum_{rxn.id}"
        )
        # G_i >= (max_bound + 1) * a_i- - max_bound
        energy_range_minimum = prob.Constraint(
            Zero, lb=-max_bound, name=f"delta_g_range_minimum_{rxn.id}"
        )
        energy_ranges.append((energy_range_maximum, energy_range_minimum))
    model.add_cons_vars([constraint for pair in energy_ranges for constraint in pair])
    model.solver.update()
    for (
        (energy_range_maximum, energy_range_minimum),
        energy,
        (
            indicator_maximum,
            indicator_minimum,
        ),
    ) in zip(energy_ranges, energies, indicators):
        energy_range_maximum.set_linear_coefficients(
            {**energy, indicator_maximum: max_bound + 1}
        )
        energy_range_minimum.set_linear_coefficients(
            {**energy, indicator_minimum: -(max_bound + 1)}
        )
