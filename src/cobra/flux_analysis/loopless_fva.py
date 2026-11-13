"""Provide the loopless parts of flux variability analysis."""

from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from warnings import warn

import optlang
from optlang.symbolics import Zero, add

from ..util import get_context
from ..util import solver as sutil
from .loopless import add_loopless


if TYPE_CHECKING:
    from cobra import Model


def _is_mass_balance(
    model: "Model",
    constraint: "optlang.interface.Constraint",
    coefficients: Dict["optlang.interface.Variable", float],
) -> bool:
    """Check whether a constraint is the unmodified mass balance of a metabolite.

    The name is not sufficient: users may have modified a metabolite
    constraint through the solver interface.
    """
    if constraint.name not in model.metabolites:
        return False
    metabolite = model.metabolites.get_by_id(constraint.name)
    expected = {}
    for reaction in metabolite.reactions:
        coefficient = reaction.get_coefficient(metabolite)
        expected[reaction.forward_variable] = coefficient
        expected[reaction.reverse_variable] = -coefficient
    return coefficients == expected


def requires_loopless_for_all(
    model: "Model",
    cyclic_reactions: List[str],
    cyclic_directions: List[Tuple[bool, bool]],
) -> bool:
    """Check whether non-cyclic directions need the loopless constraints too.

    Removing a cycle from a flux distribution decreases the variables of cyclic
    directions and keeps all others, which preserves the mass balances. The
    optimum of a non-cyclic direction is therefore the same with and without
    loopless constraints, unless decreasing these variables can violate another
    constraint of the model.
    """
    cyclic_variables = set()
    for reaction_id, directions in zip(cyclic_reactions, cyclic_directions):
        reaction = model.reactions.get_by_id(reaction_id)
        for variable, cyclic in zip(
            (reaction.reverse_variable, reaction.forward_variable), directions
        ):
            if not cyclic:
                continue
            # Decreasing the variable can violate its own lower bound or type.
            if variable.type != "continuous" or (
                variable.lb is not None and variable.lb > 0
            ):
                return True
            cyclic_variables.add(variable)

    if not cyclic_variables:
        return False

    for constraint in model.constraints:
        if constraint.indicator_variable is not None or not constraint.is_Linear:
            return True
        coefficients = constraint.get_linear_coefficients(constraint.variables)
        affected = cyclic_variables.intersection(coefficients)
        if not affected or _is_mass_balance(model, constraint, coefficients):
            continue
        # Decreasing a variable lowers the constraint expression if its
        # coefficient is positive and raises it otherwise.
        for variable in affected:
            coefficient = coefficients[variable]
            if (constraint.lb is not None and coefficient > 0) or (
                constraint.ub is not None and coefficient < 0
            ):
                return True

    return False


def _total_flux(model: "Model") -> "optlang.symbolics.Basic":
    """Return the sum of the forward and reverse variables of all reactions."""
    return add(
        [
            variable
            for reaction in model.reactions
            for variable in (reaction.forward_variable, reaction.reverse_variable)
        ]
    )


def create_loopless_model(
    model: "Model", method: str, cyclic_reactions: List[str]
) -> "Model":
    """Return a copy of the model with loopless constraints.

    Without cyclic reactions, no constraints are needed and the model itself
    is returned.
    """
    if not cyclic_reactions:
        return model
    loopless_model = model.copy()
    add_loopless(loopless_model, method=method, reactions=cyclic_reactions)
    return loopless_model


def add_loopless_objective_bounds(
    model: "Model",
    loopless_model: "Model",
    fraction_of_optimum: Optional[float],
    pfba_factor: Optional[float],
) -> None:
    """Bound the objective and the total flux of both models by loopless optima.

    Both optima are computed with `loopless_model`. The total flux is
    minimized after the objective has been bounded.
    """
    models = [model] if model is loopless_model else [model, loopless_model]
    if fraction_of_optimum is not None:
        optimum = loopless_model.slim_optimize(
            error_value=None,
            message="There is no optimal solution for the chosen loopless objective!",
        )
        bound = fraction_of_optimum * optimum
        for bounded_model in models:
            maximize = bounded_model.objective.direction == "max"
            bounded_model.add_cons_vars(
                [
                    bounded_model.problem.Constraint(
                        bounded_model.objective.expression,
                        lb=bound if maximize else None,
                        ub=None if maximize else bound,
                        name="fva_old_objective_constraint",
                    )
                ]
            )

    if pfba_factor is not None:
        if pfba_factor < 1.0:
            warn("The 'pfba_factor' should be larger or equal to 1.", UserWarning)
        with loopless_model:
            loopless_model.objective = loopless_model.problem.Objective(
                _total_flux(loopless_model), direction="min"
            )
            minimum_sum = loopless_model.slim_optimize(error_value=None)
        for bounded_model in models:
            bounded_model.add_cons_vars(
                [
                    bounded_model.problem.Constraint(
                        _total_flux(bounded_model),
                        ub=pfba_factor * minimum_sum,
                        name="flux_sum_constraint",
                    )
                ]
            )


def _set_mip_gap(model: "Model", mip_gap: float) -> None:
    """Set the MIP gap until the model context exits.

    Only CPLEX and Gurobi are supported, other solvers keep their gap.
    """
    solver = sutil.interface_to_str(model.problem)
    problem = model.solver.problem
    if solver == "cplex":
        parameter = problem.parameters.mip.tolerances.mipgap
        reset = partial(parameter.set, parameter.get())
        parameter.set(mip_gap)
    elif solver == "gurobi":
        reset = partial(problem.setParam, "MIPGap", problem.Params.MIPGap)
        problem.Params.MIPGap = mip_gap
    else:
        return
    context = get_context(model)
    if context:
        context(reset)


class ClippedFluxScreening:
    """Find loopless flux distributions in which many targets carry flux.

    A target is a reaction with a direction, "minimum" or "maximum". The model,
    which has to be in a context, gets directional loopless constraints and an
    objective that counts the targets with a flux of at least `flux_threshold`.
    Targets found by one solution are not searched for again.
    """

    def __init__(
        self,
        model: "Model",
        method: str,
        cyclic_reactions: List[str],
        targets: List[Tuple[str, str]],
        flux_threshold: float,
    ) -> None:
        """Add the constraints and the objective to the model."""
        # Any solution that contains a new target is sufficient.
        _set_mip_gap(model, 1.0)
        add_loopless(
            model,
            method=method,
            reactions=cyclic_reactions,
            flux_threshold=flux_threshold,
        )
        self._model = model
        self._cyclic_reactions = cyclic_reactions
        self._remaining = {
            (reaction_id, direction): model.variables[
                f"indicator_{direction}_{reaction_id}"
            ]
            for reaction_id, direction in targets
        }
        self._inactive: List[Tuple[str, str]] = []

        coefficients = {indicator: 1 for indicator in self._remaining.values()}
        model.objective = Zero
        model.objective.set_linear_coefficients(coefficients)
        model.objective.direction = "max"

        name = "loopless_fva_any_remaining_target"
        model.add_cons_vars([model.problem.Constraint(Zero, lb=1, name=name)])
        self._any_remaining = model.constraints[name]
        self._any_remaining.set_linear_coefficients(coefficients)

    def next_candidates(self) -> List[Tuple[str, str]]:
        """Solve the problem and return the targets that carry flux.

        The list is empty once none of the remaining targets can carry flux.
        """
        if not self._remaining:
            return []
        self._model.slim_optimize()
        if self._model.solver.status == optlang.interface.INFEASIBLE:
            return []
        sutil.check_solver_status(self._model.solver.status)

        variables = self._model.variables
        self._inactive = [
            (reaction_id, direction)
            for reaction_id in self._cyclic_reactions
            for direction in ("maximum", "minimum")
            if variables[f"indicator_{direction}_{reaction_id}"].primal < 0.5
        ]
        candidates = [
            target
            for target, indicator in self._remaining.items()
            if indicator.primal >= 0.5
        ]
        found = {self._remaining.pop(target): 0 for target in candidates}
        self._model.objective.set_linear_coefficients(found)
        self._any_remaining.set_linear_coefficients(found)
        return candidates

    def close_inactive_directions(self, model: "Model") -> None:
        """Close the directions without flux in the last solution.

        The given model then has no loops, even without loopless constraints.
        """
        for reaction_id, direction in self._inactive:
            reaction = model.reactions.get_by_id(reaction_id)
            if direction == "maximum":
                reaction.upper_bound = min(reaction.upper_bound, 0.0)
            else:
                reaction.lower_bound = max(reaction.lower_bound, 0.0)
