"""Test functionalities of removing loops in model."""

from typing import Callable, Optional
from unittest.mock import DEFAULT

import pytest
from optlang.interface import INFEASIBLE, OPTIMAL

from cobra.core import Metabolite, Model, Reaction, get_solution
from cobra.flux_analysis.loopless import (
    add_loopless,
    loopless_fva_iter,
    loopless_solution,
)


@pytest.mark.parametrize("method", ["fastSNP", "potentials", "original"])
def test_loopless_benchmark_before(
    benchmark: Callable, ll_test_model: Model, method: str
) -> None:
    """Benchmark initial condition."""

    def _():
        with ll_test_model:
            add_loopless(ll_test_model, method=method)
            ll_test_model.optimize()

    benchmark(_)


def test_loopless_benchmark_after(benchmark: Callable, ll_test_model: Model) -> None:
    """Benchmark final condition."""
    benchmark(loopless_solution, ll_test_model)


def test_loopless_solution(ll_test_model: Model) -> None:
    """Test loopless_solution()."""
    opt_feasible = ll_test_model.slim_optimize()
    solution_feasible = loopless_solution(ll_test_model)
    ll_test_model.reactions.v3.lower_bound = 1
    opt_infeasible = ll_test_model.slim_optimize()
    solution_infeasible = loopless_solution(ll_test_model)
    assert solution_feasible.fluxes["v3"] == 0.0
    assert solution_feasible.objective_value == pytest.approx(opt_feasible)
    assert solution_infeasible.fluxes["v3"] == 1.0
    assert solution_infeasible.objective_value == pytest.approx(opt_infeasible)


def test_loopless_solution_fluxes(model: Model) -> None:
    """Test fluxes of loopless_solution()."""
    sol = model.optimize()
    ll_solution = loopless_solution(model, fluxes=sol.fluxes)
    assert len(ll_solution.fluxes) == len(model.reactions)
    assert ll_solution.objective_value == pytest.approx(sol.objective_value)


@pytest.mark.parametrize("reaction_id", ["EX_A", "entry"])
def test_loopless_fva_solution_removes_unrelated_cycle(
    ll_test_model: Model, reaction_id: str, mocker
) -> None:
    """Return the cleaned solution even when loop removal preserves the optimum."""
    model = ll_test_model
    source = Metabolite("source")
    model.reactions.EX_A.add_metabolites({model.metabolites.A: -1, source: 1})
    entry = Reaction("entry")
    entry.add_metabolites({source: -1, model.metabolites.A: 1})
    model.add_reactions([entry])
    model.reactions.EX_A.upper_bound = 10
    model.objective = model.reactions.get_by_id(reaction_id)

    # Supply a feasible optimal distribution with an extra cycle, independent
    # of which optimal basis the solver happens to choose.
    cyclic_solution = model.optimize()
    assert cyclic_solution.objective_value == pytest.approx(10.0)
    cyclic_solution.fluxes[["v1", "v2", "v3"]] = [11.0, 11.0, 1.0]
    mocker.patch(
        "cobra.flux_analysis.loopless.get_solution",
        side_effect=[cyclic_solution, DEFAULT],
        wraps=get_solution,
    )

    solution = loopless_fva_iter(model, model.reactions.get_by_id(reaction_id), True)
    assert solution.fluxes[reaction_id] == pytest.approx(10.0)
    assert solution.fluxes["v3"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("method", ["fastSNP", "potentials", "original"])
@pytest.mark.parametrize("flux_threshold", [None, 1e-2])
def test_add_loopless(
    ll_test_model: Model, method: str, flux_threshold: Optional[float]
) -> None:
    """Test add_loopless()."""
    add_loopless(ll_test_model, method=method, flux_threshold=flux_threshold)
    feasible_status = ll_test_model.optimize().status
    ll_test_model.reactions.v3.lower_bound = 1
    ll_test_model.slim_optimize()
    infeasible_status = ll_test_model.solver.status
    assert feasible_status == OPTIMAL
    assert infeasible_status == INFEASIBLE


def test_add_loopless_infinite_bounds(ll_test_model: Model) -> None:
    """Test add_loopless() with an infinite reaction bound."""
    ll_test_model.reactions.v1.upper_bound = float("inf")
    with pytest.raises(ValueError, match="clip_reaction_bounds"):
        add_loopless(ll_test_model)


def test_add_loopless_acyclic(ll_test_model: Model) -> None:
    """A model without internal cycles needs no loopless constraints."""
    ll_test_model.remove_reactions(["v3"])
    expected_objective = ll_test_model.slim_optimize()
    variables = list(ll_test_model.variables)
    constraints = list(ll_test_model.constraints)

    add_loopless(ll_test_model)

    assert ll_test_model.slim_optimize() == pytest.approx(expected_objective)
    assert list(ll_test_model.variables) == variables
    assert list(ll_test_model.constraints) == constraints


@pytest.mark.parametrize("method", ["fastSNP", "original"])
def test_add_loopless_boundary_only(ll_test_model: Model, method: str) -> None:
    """Boundary-only models cannot contain an internal cycle."""
    ll_test_model.remove_reactions(["v1", "v2", "v3"])
    expected_objective = ll_test_model.slim_optimize()

    add_loopless(ll_test_model, method=method)

    assert ll_test_model.slim_optimize() == pytest.approx(expected_objective)


@pytest.mark.parametrize("method", ["fastSNP", "original"])
@pytest.mark.parametrize("reactions", [[], ["EX_A", "DM_C"]])
def test_add_loopless_empty_internal_selection(
    ll_test_model: Model, method: str, reactions: list
) -> None:
    """Selecting no internal reactions leaves the feasible flux space intact."""
    ll_test_model.reactions.v3.lower_bound = 1
    expected_objective = ll_test_model.slim_optimize()
    variables = list(ll_test_model.variables)
    constraints = list(ll_test_model.constraints)

    add_loopless(ll_test_model, method=method, reactions=reactions)

    assert ll_test_model.slim_optimize() == pytest.approx(expected_objective)
    assert list(ll_test_model.variables) == variables
    assert list(ll_test_model.constraints) == constraints


def test_add_loopless_all_blocked(ll_test_model: Model) -> None:
    """Reactions fixed to zero have no feasible cyclic directions."""
    for reaction in ll_test_model.reactions:
        reaction.bounds = (0, 0)

    add_loopless(ll_test_model)

    assert ll_test_model.slim_optimize() == 0
