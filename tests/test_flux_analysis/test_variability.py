"""Test functionalities of Flux Variability Analysis."""

import logging
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import pytest

from cobra import Metabolite, Model, Reaction
from cobra.exceptions import Infeasible, OptimizationError
from cobra.flux_analysis.loopless import add_loopless
from cobra.flux_analysis.variability import (
    find_essential_genes,
    find_essential_reactions,
    flux_variability_analysis,
)
from cobra.util.array import create_stoichiometric_matrix


# FVA
def test_flux_variability_benchmark(
    large_model: Model, benchmark: Callable, all_solvers: List[str]
) -> None:
    """Benchmark FVA."""
    large_model.solver = all_solvers
    benchmark(
        flux_variability_analysis,
        large_model,
        reaction_list=large_model.reactions[1::100],
        processes=1,
    )


def test_pfba_flux_variability(
    model: Model,
    pfba_fva_results: pd.DataFrame,
    fva_results: pd.DataFrame,
    all_solvers: List[str],
) -> None:
    """Test FVA using pFBA."""
    model.solver = all_solvers
    with pytest.warns(UserWarning):
        flux_variability_analysis(
            model,
            pfba_factor=0.1,
            reaction_list=model.reactions[1::3],
            processes=1,
        )
    fva_out = flux_variability_analysis(
        model, pfba_factor=1.1, reaction_list=model.reactions, processes=1
    )
    fva_out.sort_index(inplace=True)
    assert np.allclose(fva_out, pfba_fva_results)
    abs_fva_out = fva_out.dropna().abs()
    abs_fva_results = fva_results.dropna().abs()
    comparison = np.isclose(abs_fva_out, abs_fva_results) | (
        abs_fva_out < abs_fva_results
    )
    assert comparison["minimum"].all()
    assert comparison["maximum"].all()


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials", "cycleFreeFlux"])
def test_loopless_pfba_fva(model: Model, loopless: str) -> None:
    """Test loopless FVA using pFBA."""
    loop_reactions = [model.reactions.get_by_id(rid) for rid in ("FRD7", "SUCDi")]
    fva_loopless = flux_variability_analysis(
        model,
        pfba_factor=1.1,
        reaction_list=loop_reactions,
        loopless=loopless,
        processes=1,
    )
    assert np.allclose(fva_loopless["maximum"], fva_loopless["minimum"])


def test_flux_variability(
    model: Model, fva_results: pd.DataFrame, all_solvers: List[str]
) -> None:
    """Test FVA."""
    model.solver = all_solvers
    fva_out = flux_variability_analysis(
        model, reaction_list=model.reactions, processes=1
    )
    fva_out.sort_index(inplace=True)
    assert np.allclose(fva_out, fva_results)


def test_flux_variability_directional_reactions(model: Model) -> None:
    """Test FVA with directional reactions."""
    full = flux_variability_analysis(
        model,
        reaction_list=["FRD7", "SUCDi", "FUM"],
        processes=1,
    )
    directional = flux_variability_analysis(
        model,
        reaction_list=[("FRD7", "max"), ("SUCDi", "min"), "FUM"],
        processes=1,
    )

    assert directional.at["FRD7", "maximum"] == pytest.approx(
        full.at["FRD7", "maximum"]
    )
    assert np.isnan(directional.at["FRD7", "minimum"])
    assert directional.at["SUCDi", "minimum"] == pytest.approx(
        full.at["SUCDi", "minimum"]
    )
    assert np.isnan(directional.at["SUCDi", "maximum"])
    assert directional.at["FUM", "minimum"] == pytest.approx(full.at["FUM", "minimum"])
    assert directional.at["FUM", "maximum"] == pytest.approx(full.at["FUM", "maximum"])


def test_flux_variability_invalid_direction(model: Model) -> None:
    """Test FVA with an unknown direction of a reaction."""
    with pytest.raises(ValueError, match="'min', 'max'"):
        flux_variability_analysis(model, reaction_list=[("FUM", "up")])


def test_flux_variability_abs_flux_clip_within_tolerance(model: Model) -> None:
    """Test FVA with a clip that is not above the tolerance."""
    with pytest.raises(ValueError, match="abs_flux_clip"):
        flux_variability_analysis(model, abs_flux_clip=model.tolerance)


@pytest.mark.parametrize("processes", [1, 2])
def test_flux_variability_abs_flux_clip(
    model: Model, all_solvers: List[str], processes: int
) -> None:
    """Test FVA with clipped absolute flux values."""
    if processes > 1 and "SKIP_MP" in os.environ:
        pytest.skip("unsafe for parallel execution")
    model.solver = all_solvers
    abs_flux_clip = 10.0

    fva_out = flux_variability_analysis(
        model,
        fraction_of_optimum=None,
        processes=1,
    )
    expected = fva_out.clip(lower=-abs_flux_clip, upper=abs_flux_clip)

    clipped_fva_out = flux_variability_analysis(
        model,
        fraction_of_optimum=None,
        abs_flux_clip=abs_flux_clip,
        processes=processes,
    )

    assert np.allclose(clipped_fva_out, expected)


def test_flux_variability_abs_flux_clip_keeps_solver(
    model: Model, all_solvers: List[str]
) -> None:
    """Clipped FVA leaves the solver able to optimize beyond the clip."""
    model.solver = all_solvers
    optimum = model.slim_optimize()
    flux_variability_analysis(
        model, fraction_of_optimum=None, abs_flux_clip=0.1, processes=1
    )
    assert model.slim_optimize() == pytest.approx(optimum)


def test_flux_variability_abs_flux_clip_iteration_limit(model: Model) -> None:
    """Clipped FVA does not take another abort of CPLEX for reaching the clip."""
    pytest.importorskip("cplex")
    model.solver = "cplex"
    model.solver.problem.parameters.simplex.limits.iterations.set(1)
    with pytest.raises(OptimizationError):
        flux_variability_analysis(
            model, fraction_of_optimum=None, abs_flux_clip=5.0, processes=1
        )


@pytest.mark.skipif("SKIP_MP" in os.environ, reason="unsafe for parallel execution")
def test_parallel_flux_variability(
    model: Model, fva_results: pd.DataFrame, all_solvers: List[str]
) -> None:
    """Test parallel FVA."""
    model.solver = all_solvers
    fva_out = flux_variability_analysis(model, processes=2)
    fva_out.sort_index(inplace=True)
    assert np.allclose(fva_out, fva_results)


# Loopless FVA
@pytest.mark.parametrize("internal_reaction", [False, True])
@pytest.mark.parametrize("all_fluxes", [False, True])
def test_flux_variability_loopless_acyclic(
    internal_reaction: bool, all_fluxes: bool
) -> None:
    """Loopless FVA preserves bounds for networks without internal cycles."""
    model = Model()
    model.solver = "glpk"
    a, b = Metabolite("A"), Metabolite("B")
    uptake, drain = Reaction("uptake"), Reaction("drain")
    uptake.add_metabolites({a: 1})
    drain.add_metabolites({b if internal_reaction else a: -1})
    uptake.bounds = drain.bounds = (0, 10)
    model.add_reactions([uptake, drain])
    if internal_reaction:
        internal = Reaction("internal")
        internal.add_metabolites({a: -1, b: 1})
        internal.bounds = (0, 10)
        model.add_reactions([internal])
    model.objective = drain

    result = flux_variability_analysis(
        model, loopless="fastSNP", all_fluxes=all_fluxes, processes=1
    )

    bounds = result[0] if all_fluxes else result
    assert np.allclose(bounds, 10)
    if all_fluxes:
        assert all(np.allclose(fluxes, 10) for fluxes in result[1].values())


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials", "cycleFreeFlux"])
def test_flux_variability_loopless_benchmark(
    model: Model, benchmark: Callable, all_solvers: List[str], loopless: str
) -> None:
    """Benchmark loopless FVA."""
    model.solver = all_solvers
    benchmark(
        flux_variability_analysis,
        model,
        loopless=loopless,
        reaction_list=model.reactions[1::50],
    )


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials", "cycleFreeFlux"])
def test_flux_variability_loopless(
    model: Model, all_solvers: List[str], loopless: str
) -> None:
    """Test loopless FVA."""
    model.solver = all_solvers
    loop_reactions = [model.reactions.get_by_id(rid) for rid in ("FRD7", "SUCDi")]
    fva_normal = flux_variability_analysis(model, reaction_list=loop_reactions)
    fva_loopless = flux_variability_analysis(
        model,
        reaction_list=loop_reactions,
        loopless=loopless,
    )

    assert not np.allclose(fva_normal["maximum"], fva_normal["minimum"])
    assert np.allclose(fva_loopless["maximum"], fva_loopless["minimum"])


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials", "cycleFreeFlux"])
def test_flux_variability_loopless_abs_flux_clip(
    model: Model, all_solvers: List[str], loopless: str
) -> None:
    """Test loopless FVA with clipped absolute flux values."""
    model.solver = all_solvers
    abs_flux_clip = 3.0

    loop_reactions = [model.reactions.get_by_id(rid) for rid in ("FRD7", "SUCDi")]
    fva_loopless = flux_variability_analysis(
        model,
        reaction_list=loop_reactions,
        loopless=loopless,
    )
    expected = fva_loopless.clip(lower=-abs_flux_clip, upper=abs_flux_clip)

    clipped_fva_out = flux_variability_analysis(
        model,
        reaction_list=loop_reactions,
        loopless=loopless,
        abs_flux_clip=abs_flux_clip,
    )

    assert np.allclose(clipped_fva_out, expected)


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials"])
def test_flux_variability_loopless_additional_constraint(
    ll_test_model: Model, loopless: str
) -> None:
    """A constraint that depends on a loop also limits reactions outside of it."""
    model = ll_test_model
    # Without loopless constraints, half of the flux through v1 and v2 can
    # leave through DM_C while the other half returns through v3.
    demand_requires_loop = model.problem.Constraint(
        model.reactions.DM_C.flux_expression - model.reactions.v3.flux_expression,
        ub=0,
    )
    model.add_cons_vars([demand_requires_loop])

    fva_out = flux_variability_analysis(
        model, ["DM_C"], loopless=loopless, fraction_of_optimum=None, processes=1
    )

    assert fva_out.at["DM_C", "maximum"] == pytest.approx(0.0, abs=1e-6)


def test_flux_variability_cycle_free_flux_abs_flux_clip(ll_test_model: Model) -> None:
    """Test clipped FVA with CycleFreeFlux for non-cyclic and cyclic reactions."""
    ll_test_model.reactions.v2 *= -1
    fva_out = flux_variability_analysis(
        ll_test_model, loopless="cycleFreeFlux", abs_flux_clip=0.1, processes=1
    )
    assert fva_out.at["DM_C", "maximum"] == pytest.approx(0.1)
    assert fva_out.at["v3", "maximum"] == pytest.approx(0.0, abs=1e-6)


# Internals (essentiality)
def test_fva_data_frame(model: Model) -> None:
    """Test DataFrame obtained from FVA."""
    df = flux_variability_analysis(model)
    assert np.all([df.columns.values == ["minimum", "maximum"]])


def test_fva_infeasible(model: Model) -> None:
    """Test FVA infeasibility."""
    infeasible_model = model.copy()
    infeasible_model.reactions.get_by_id("EX_glc__D_e").lower_bound = 0
    # ensure that an infeasible model does not run FVA
    with pytest.raises(Infeasible):
        flux_variability_analysis(infeasible_model)


def test_fva_minimization(model: Model) -> None:
    """Test minimization using FVA."""
    model.objective = model.reactions.EX_glc__D_e
    model.objective_direction = "min"
    solution = flux_variability_analysis(model, fraction_of_optimum=0.95)
    assert solution.at["EX_glc__D_e", "minimum"] == -10.0
    assert solution.at["EX_glc__D_e", "maximum"] == -9.5


def test_essential_genes(model: Model) -> None:
    """Test find_essential_genes()."""
    essential_genes = {
        "b2779",
        "b1779",
        "b0720",
        "b2416",
        "b2926",
        "b1136",
        "b2415",
    }
    observed_essential_genes = {g.id for g in find_essential_genes(model)}
    assert observed_essential_genes == essential_genes


def test_essential_reactions(model: Model) -> None:
    """Test find_essential_reactions()."""
    essential_reactions = {
        "GLNS",
        "Biomass_Ecoli_core",
        "PIt2r",
        "GAPD",
        "ACONTb",
        "EX_nh4_e",
        "ENO",
        "EX_h_e",
        "EX_glc__D_e",
        "ICDHyr",
        "CS",
        "NH4t",
        "GLCpts",
        "PGM",
        "EX_pi_e",
        "PGK",
        "RPI",
        "ACONTa",
    }
    observed_essential_reactions = {r.id for r in find_essential_reactions(model)}
    assert observed_essential_reactions == essential_reactions


@pytest.mark.parametrize("loopless", [None, "fastSNP", "potentials", "cycleFreeFlux"])
@pytest.mark.parametrize("abs_flux_clip", [None, 10.0])
@pytest.mark.parametrize("processes", [1, 2])
def test_fva_all_fluxes(
    model: Model,
    loopless: Optional[str],
    abs_flux_clip: Optional[float],
    processes: int,
) -> None:
    """`all_fluxes` keeps the distributions FVA already computes."""
    if processes > 1 and "SKIP_MP" in os.environ:
        pytest.skip("unsafe for parallel execution")
    reactions = model.reactions[:8]
    original_bounds = {r.id: r.bounds for r in model.reactions}
    original_objective = model.objective.expression
    original_variables = set(model.variables.keys())
    original_constraints = set(model.constraints.keys())
    bounds = flux_variability_analysis(
        model, reactions, loopless=loopless, abs_flux_clip=abs_flux_clip, processes=1
    )
    bounds_2, fluxes = flux_variability_analysis(
        model,
        reactions,
        loopless=loopless,
        abs_flux_clip=abs_flux_clip,
        processes=processes,
        all_fluxes=True,
    )

    # Asking for the fluxes must not change the bounds.
    assert np.allclose(bounds.values, bounds_2.values, atol=1e-6, equal_nan=True)

    assert set(fluxes) == {"minimum", "maximum"}
    for direction, frame in fluxes.items():
        assert set(frame.index) == {r.id for r in reactions}
        assert set(frame.columns) == {r.id for r in model.reactions}
        # The actual flux may exceed the clip, but clipping it must give
        # the reported bound. The full distribution must remain feasible.
        for rxn_id in frame.index:
            value = frame.at[rxn_id, rxn_id]
            if abs_flux_clip is not None:
                value = np.clip(value, -abs_flux_clip, abs_flux_clip)
            assert value == pytest.approx(bounds_2.at[rxn_id, direction], abs=1e-6)
        residual = create_stoichiometric_matrix(model).dot(
            frame[[r.id for r in model.reactions]].values.T
        )
        assert np.abs(residual).max() < 1e-6
        for rxn in model.reactions:
            assert (frame[rxn.id] >= rxn.lower_bound - 1e-6).all()
            assert (frame[rxn.id] <= rxn.upper_bound + 1e-6).all()

    assert {r.id: r.bounds for r in model.reactions} == original_bounds
    assert model.objective.expression == original_objective
    assert set(model.variables.keys()) == original_variables
    assert set(model.constraints.keys()) == original_constraints

    if loopless is None:
        return
    # Every returned distribution must be loopless, including those for
    # reactions that cannot themselves carry a loop -- those are optimized
    # without the loop constraints by default, which would leave loops in
    # their solution vectors.
    with model:
        add_loopless(model)
        reference = {r.id: r.bounds for r in model.reactions}
        for rxn_id in list(fluxes["maximum"].index)[:3]:
            row = fluxes["maximum"].loc[rxn_id]
            for rxn in model.reactions:
                rxn.bounds = (row[rxn.id] - 1e-6, row[rxn.id] + 1e-6)
            assert (
                model.slim_optimize(error_value=None) is not None
            ), f"flux distribution for {rxn_id} is not loopless"
            for rxn in model.reactions:
                rxn.bounds = reference[rxn.id]


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials"])
@pytest.mark.parametrize("processes", [1, 2])
def test_fva_all_fluxes_clipped_loopless(
    ll_test_model: Model, loopless: str, processes: int, caplog
) -> None:
    """Clipped loopless bounds come with loopless distributions."""
    if processes > 1 and "SKIP_MP" in os.environ:
        pytest.skip("unsafe for parallel execution")
    model = ll_test_model
    # Reverse v2 so that both a minimum and a maximum reach the clip.
    model.reactions.v2 *= -1
    requests = [("v2", "min"), ("v1", "max"), ("v3", "max"), ("EX_A", "min")]
    bounds, fluxes = flux_variability_analysis(
        model,
        requests,
        loopless=loopless,
        abs_flux_clip=10.0,
        processes=processes,
        all_fluxes=True,
    )

    assert list(fluxes["minimum"].index) == ["v2", "EX_A"]
    assert list(fluxes["maximum"].index) == ["v1", "v3"]
    assert np.isnan(bounds.at["v1", "minimum"])
    assert np.isnan(bounds.at["v2", "maximum"])
    assert bounds.at["v1", "maximum"] == pytest.approx(10.0)
    assert bounds.at["v2", "minimum"] == pytest.approx(-10.0)
    assert bounds.at["v3", "maximum"] == pytest.approx(0.0, abs=1e-6)

    # A failed screening falls back to single optimizations with a warning.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    for frame in fluxes.values():
        assert np.allclose(frame["v3"], 0.0, atol=1e-6)
        assert np.allclose(frame["EX_A"], 1000.0)
        assert np.allclose(frame["DM_C"], 1000.0)
        assert np.allclose(frame["v1"], 1000.0)
        assert np.allclose(frame["v2"], -1000.0)


@pytest.mark.parametrize("loopless", ["fastSNP", "potentials"])
def test_fva_all_fluxes_clipped_loopless_respects_bounds(
    ll_test_model: Model, loopless: str
) -> None:
    """Distributions of clipped reactions keep the forced flux of a cyclic reaction."""
    model = ll_test_model
    model.reactions.v1.lower_bound = 1.0
    parallel = Reaction("parallel")
    parallel.add_metabolites({model.metabolites.A: -1, model.metabolites.B: 1})
    model.add_reactions([parallel])

    _, fluxes = flux_variability_analysis(
        model,
        [("parallel", "max")],
        loopless=loopless,
        fraction_of_optimum=None,
        abs_flux_clip=0.5,
        processes=1,
        all_fluxes=True,
    )

    assert fluxes["maximum"].at["parallel", "v1"] >= 1.0 - model.tolerance


@pytest.mark.parametrize("reaction_list", [[], [("FUM", "max")]])
def test_fva_all_fluxes_empty_direction(model: Model, reaction_list: list) -> None:
    """Empty distributions retain reaction columns and no unrequested rows."""
    bounds, fluxes = flux_variability_analysis(
        model, reaction_list, processes=1, all_fluxes=True
    )
    assert fluxes["minimum"].empty
    assert list(fluxes["minimum"].columns) == [r.id for r in model.reactions]
    assert list(fluxes["maximum"].index) == list(bounds.index)
