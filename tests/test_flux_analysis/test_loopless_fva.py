"""Test the loopless parts of flux variability analysis."""

from cobra import Model
from cobra.flux_analysis import find_cyclic_reactions
from cobra.flux_analysis.loopless_fva import (
    ClippedFluxScreening,
    requires_loopless_for_all,
)


def test_requires_loopless_for_all(ll_test_model: Model) -> None:
    """Only constraints that cycle removal can violate require the MILP."""
    cyclic_reactions, cyclic_directions = find_cyclic_reactions(ll_test_model)
    assert not requires_loopless_for_all(
        ll_test_model, cyclic_reactions, cyclic_directions
    )

    v1 = ll_test_model.reactions.v1
    minimum_flux = ll_test_model.problem.Constraint(v1.forward_variable, lb=1)
    ll_test_model.add_cons_vars([minimum_flux])
    assert requires_loopless_for_all(ll_test_model, cyclic_reactions, cyclic_directions)


def test_clipped_flux_screening(ll_test_model: Model) -> None:
    """Screening finds every direction that carries flux without a loop."""
    cyclic_reactions = find_cyclic_reactions(ll_test_model)[0]
    targets = [
        (rxn_id, direction)
        for rxn_id in ("v1", "v2", "v3")
        for direction in ("minimum", "maximum")
    ]
    validation_model = ll_test_model.copy()
    number_of_variables = len(ll_test_model.variables)
    found = []
    with ll_test_model:
        screening = ClippedFluxScreening(
            ll_test_model, "potentials", cyclic_reactions, targets, 0.1
        )
        candidates = screening.next_candidates()
        while candidates:
            found.extend(candidates)
            with validation_model:
                screening.close_inactive_directions(validation_model)
                assert validation_model.reactions.v3.bounds == (0.0, 0.0)
            candidates = screening.next_candidates()

    assert sorted(found) == [("v1", "maximum"), ("v2", "maximum")]
    assert len(ll_test_model.variables) == number_of_variables
