"""Test functionalities for blocked reaction analysis."""

from typing import List

import pytest

from cobra import Model
from cobra.exceptions import Infeasible
from cobra.flux_analysis import find_blocked_reactions


def test_find_blocked_reactions_solver_none(model: Model) -> None:
    """Test find_blocked_reactions() [no specific solver]."""
    result = find_blocked_reactions(model, model.reactions[40:46])
    assert result == ["FRUpts2"]


def test_find_blocked_reactions(model: Model, all_solvers: List[str]) -> None:
    """Test find_blocked_reactions()."""
    model.solver = all_solvers
    result = find_blocked_reactions(model, model.reactions[40:46])
    assert result == ["FRUpts2"]

    result = find_blocked_reactions(model, model.reactions[42:48])
    assert result == ["FRUpts2", "FUMt2_2"]

    result = find_blocked_reactions(model, model.reactions[30:50], open_exchanges=True)
    assert result == []


def test_find_blocked_reactions_order(model: Model) -> None:
    """Test find_blocked_reactions() returns the reactions in the given order."""
    reaction_ids = [reaction.id for reaction in model.reactions][::-1]
    result = find_blocked_reactions(model, reaction_ids)
    for blocked in (result, result.forward_blocked, result.reverse_blocked):
        assert blocked == [rxn_id for rxn_id in reaction_ids if rxn_id in blocked]


def test_find_blocked_reactions_without_clip(model: Model) -> None:
    """Test find_blocked_reactions() without clipping the fluxes."""
    result = find_blocked_reactions(model, abs_flux_clip=None)
    assert result == find_blocked_reactions(model)
    assert result.forward_blocked == find_blocked_reactions(model).forward_blocked


def test_find_blocked_reactions_clip_below_cutoff(model: Model) -> None:
    """Test find_blocked_reactions() with a clip smaller than the cutoff."""
    with pytest.raises(ValueError, match="abs_flux_clip"):
        find_blocked_reactions(model, zero_cutoff=1e-3, abs_flux_clip=1e-4)


def test_find_blocked_reactions_near_cutoff(ll_test_model: Model, caplog) -> None:
    """Test find_blocked_reactions() warns about flux bounds near the cutoff."""
    ll_test_model.reactions.v1.upper_bound = 2 * ll_test_model.tolerance
    find_blocked_reactions(ll_test_model, processes=1)
    assert "close to zero_cutoff" in caplog.text


def test_find_blocked_reactions_infeasible(model: Model) -> None:
    """Test find_blocked_reactions() with an infeasible model."""
    model.reactions.get_by_id("EX_glc__D_e").lower_bound = 0
    with pytest.raises(Infeasible):
        find_blocked_reactions(model)


def test_find_blocked_reactions_cycle_free_flux_unsupported(model: Model) -> None:
    """Test find_blocked_reactions() rejects approximate loopless FVA."""
    with pytest.raises(ValueError, match="cycleFreeFlux"):
        find_blocked_reactions(model, loopless="cycleFreeFlux")


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials"])
def test_find_blocked_reactions_with_loopless(
    model: Model, all_solvers: List[str], loopless: str
) -> None:
    """Test find_blocked_reactions() with loopless constraints."""
    model.solver = all_solvers
    regular_result = find_blocked_reactions(model, processes=1)
    loopless_result = find_blocked_reactions(model, loopless=loopless, processes=1)

    assert set(loopless_result) == set(regular_result)
    assert set(loopless_result.forward_blocked) == set(regular_result.forward_blocked)
    assert set(loopless_result.reverse_blocked) == set(regular_result.reverse_blocked)


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials"])
def test_find_blocked_reactions_loop_only(ll_test_model: Model, loopless: str) -> None:
    """A reaction that can only carry flux in a loop is blocked with `loopless`."""
    loopless_result = find_blocked_reactions(
        ll_test_model, loopless=loopless, processes=1
    )

    assert loopless_result == ["v3"]
    assert loopless_result.forward_blocked == ["v3"]
    assert set(loopless_result.reverse_blocked) == {"EX_A", "DM_C", "v1", "v2", "v3"}
