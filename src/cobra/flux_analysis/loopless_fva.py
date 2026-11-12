"""Provide the loopless parts of flux variability analysis."""

from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple
from warnings import warn

import optlang
from optlang.symbolics import Zero, add

from ..util import get_context
from ..util import solver as sutil
from .loopless import add_loopless


if TYPE_CHECKING:
    from cobra import Model


def requires_loopless_for_all(
    model: "Model",
    cyclic_reactions: List[str],
    cyclic_directions: List[Tuple[bool, bool]],
) -> bool:
    """Check whether cycle removal may violate the model's constraints.

    Removing a cycle decreases forward or reverse reaction variables while
    leaving other variables unchanged. Standard stoichiometric constraints are
    preserved. For other linear constraints, decreasing a variable must not
    decrease a lower-bounded expression or increase an upper-bounded one.

    This is a sufficient test for using LP for non-cyclic target directions,
    not a test of whether their optima actually change. Unsupported constraint
    types conservatively require loopless constraints for every target.
    """
    variables = model.variables
    cyclic_variables = set()
    for reaction_id, directions in zip(cyclic_reactions, cyclic_directions):
        reaction = model.reactions.get_by_id(reaction_id)
        for variable, cyclic in zip(
            (variables[reaction.reverse_id], variables[reaction.id]), directions
        ):
            if not cyclic:
                continue
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
        if not affected:
            continue

        # Recognize actual mass-balance rows, not just their names: users may
        # have modified a metabolite constraint through the solver interface.
        if constraint.name in model.metabolites:
            metabolite = model.metabolites.get_by_id(constraint.name)
            expected = {}
            for reaction in metabolite.reactions:
                coefficient = reaction.get_coefficient(metabolite)
                expected[variables[reaction.id]] = coefficient
                expected[variables[reaction.reverse_id]] = -coefficient
            if coefficients == expected:
                continue

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


def _add_loopless_objective_bounds(
    model: "Model",
    loopless_model: "Model",
    fraction_of_optimum: Optional[float],
    pfba_factor: Optional[float],
) -> None:
    """Compute loopless reference values and add the same bounds to both models.

    The original objective is constrained before minimizing the total flux.
    Only reaction variables enter the flux sum. Loopless auxiliary variables
    are excluded. If both arguments refer to the same model, add each bound
    only once.
    """
    models = [model] if model is loopless_model else [model, loopless_model]
    if fraction_of_optimum is not None:
        optimum = loopless_model.slim_optimize(
            error_value=None,
            message="There is no optimal solution for the chosen loopless objective!",
        )
        for target in models:
            bound = fraction_of_optimum * optimum
            maximize = target.objective.direction == "max"
            target.add_cons_vars(
                [
                    target.problem.Constraint(
                        target.objective.expression,
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
        for target in models:
            target.add_cons_vars(
                [
                    target.problem.Constraint(
                        _total_flux(target),
                        ub=pfba_factor * minimum_sum,
                        name="flux_sum_constraint",
                    )
                ]
            )


def create_loopless_model(
    model: "Model",
    method: str,
    cyclic_reactions: List[str],
    fraction_of_optimum: Optional[float],
    pfba_factor: Optional[float],
) -> "Model":
    """Return a copy of the model with loopless constraints.

    The objective and the total flux are bounded in both models. Without
    cyclic reactions, no constraints are needed and the model itself is
    returned.
    """
    loopless_model = model
    if cyclic_reactions:
        loopless_model = model.copy()
        add_loopless(loopless_model, method=method, reactions=cyclic_reactions)
    _add_loopless_objective_bounds(
        model, loopless_model, fraction_of_optimum, pfba_factor
    )
    if loopless_model is not model:
        loopless_model.objective = Zero
    return loopless_model


def _set_mip_gap(model: "Model", mip_gap: float) -> None:
    """Set the MIP gap of CPLEX or Gurobi until the model context exits."""
    if model.problem is getattr(optlang, "cplex_interface", None):
        parameter = model.solver.problem.parameters.mip.tolerances.mipgap
        reset = partial(parameter.set, parameter.get())
        parameter.set(mip_gap)
    elif model.problem is getattr(optlang, "gurobi_interface", None):
        reset = partial(
            model.solver.problem.setParam, "MIPGap", model.solver.problem.Params.MIPGap
        )
        model.solver.problem.Params.MIPGap = mip_gap
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

        name = "loopless_fva_at_least_one_remaining_candidate"
        model.add_cons_vars([model.problem.Constraint(Zero, lb=1, name=name)])
        self._any_remaining = model.constraints[name]
        self._any_remaining.set_linear_coefficients(coefficients)

    def next_candidates(self) -> List[Tuple[str, str]]:
        """Solve the problem and return the targets that carry flux."""
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

    def fix_inactive_directions(self, model: "Model") -> None:
        """Close the directions without flux in the last solution.

        All flux distributions of a model without loopless constraints are
        loopless once its cyclic reactions are restricted to these directions.
        """
        for reaction_id, direction in self._inactive:
            reaction = model.reactions.get_by_id(reaction_id)
            if direction == "maximum":
                reaction.upper_bound = 0.0
            else:
                reaction.lower_bound = 0.0
