"""Provide functions to remove thermodynamically infeasible loops."""

from typing import TYPE_CHECKING, Dict, List, Optional, Union
from warnings import warn

import numpy as np
from optlang.symbolics import Zero

from ..core import get_solution
from ..util import create_stoichiometric_matrix, nullspace
from .fast_snp import nullspace_fast_snp
from .find_cyclic_reactions import find_cyclic_reactions
from .helpers import normalize_cutoff
from .loopless_constraints import (
    add_directional_indicators,
    add_indicators,
    add_nullspace_energies,
    add_potential_energies,
)


if TYPE_CHECKING:
    from cobra import Model, Reaction, Solution


def _warn_about_integrality_tolerance(
    model: "Model", flux_threshold: float, max_bound: float
) -> None:
    """Warn if inactive indicators may allow a flux close to `flux_threshold`."""
    try:
        integrality_tolerance = model.solver.configuration.tolerances.integrality
    except AttributeError:
        integrality_tolerance = None

    if (
        integrality_tolerance is None
        or max_bound * integrality_tolerance >= flux_threshold * 0.5
    ):
        warn(
            f"The `flux_threshold` of {flux_threshold} is not well above the "
            f"product of the maximum bound ({max_bound}) and the integrality "
            f"tolerance of the solver ({integrality_tolerance}), so inactive "
            "indicators may not block fluxes reliably. Please see the notes "
            "of `add_loopless`.",
            UserWarning,
        )


def add_loopless(
    model: "Model",
    zero_cutoff: Optional[float] = None,
    method: str = "fastSNP",
    reactions: Optional[List[str]] = None,
    flux_threshold: Optional[float] = None,
) -> None:
    """Modify a model so all feasible flux distributions are loopless.

    It adds variables and constraints to a model which will disallow flux
    distributions with loops. This function *will* modify your model.

    If `method` is set to "original" or "fastSNP" the used formulation
    is described in [1]_. If `method` is set to "fastSNP", it uses a
    faster implementation based on the Fast-SNP algorithm [2]_.

    If `method` is set to "potentials", it uses metabolite potential
    variables instead of nullspace-based constraints.

    In most cases you probably want to use the much faster
    `loopless_solution`. May be used in cases where you want to add complex
    constraints and objectives (for instance quadratic objectives) to the
    model afterwards or use an approximation of Gibbs free energy directions
    in your model.

    Parameters
    ----------
    model : cobra.Model
        The model to which to add the constraints.
    zero_cutoff : positive float, optional
        Cutoff used for null space. Coefficients with an absolute value
        smaller than `zero_cutoff` are considered to be zero. The default
        uses the `model.tolerance` (default None).
    method : str, "original", "fastSNP", or "potentials", optional
        The method used to add loopless constraints. The "original" method
        uses the original nullspace formulation from [1]_, while "fastSNP"
        uses a faster nullspace implementation based on the FastSNP
        algorithm. The "potentials" method adds constraints based on
        metabolite potential variables. The "fastSNP" and "potentials"
        methods are much faster in most cases, with relative performance
        depending on the model and optimization problem.
    reactions : list of str, optional
        The list of reaction IDs to constrain. All cycles within these
        reactions will be removed. If `None`, all reactions will be constrained.
    flux_threshold : float, optional
        If provided, each constrained reaction gets one binary indicator
        per direction, ``indicator_maximum_<id>`` and
        ``indicator_minimum_<id>``, instead of a single one. An active
        indicator requires a flux of at least `flux_threshold` in its
        direction (default None).

    Notes
    -----
    An inactive indicator may still take a value up to the integrality
    tolerance of the solver, which permits a flux of that tolerance times the
    bound of the reaction. With `flux_threshold`, this product has to stay
    well below the threshold. If it does not, increase `flux_threshold`, lower
    the integrality tolerance, or reduce large bounds, for instance with
    :func:`cobra.manipulation.clip_reaction_bounds`.

    References
    ----------
    .. [1] Elimination of thermodynamically infeasible loops in steady-state
       metabolic models. Schellenberger J, Lewis NE, Palsson BO. Biophys J.
       2011 Feb 2;100(3):544-53. doi: 10.1016/j.bpj.2010.12.3707. Erratum
       in: Biophys J. 2011 Mar 2;100(5):1381.
    .. [2] Fast-SNP: a fast matrix pre-processing algorithm for efficient
       loopless flux optimization of metabolic models. Saa PA, Nielsen LK.
       Bioinformatics. 2016 Dec;32(24):3807–3814. doi: 10.1093/bioinformatics/btw555.
    """
    if method not in ["original", "fastSNP", "potentials"]:
        raise ValueError(f"unsupported method: {method}")

    zero_cutoff = normalize_cutoff(model, zero_cutoff)

    if reactions is None and method != "original":
        reactions = find_cyclic_reactions(model, zero_cutoff=zero_cutoff)[0]

    reactions_to_constrain = [
        i for i, r in enumerate(model.reactions) if not r.boundary
    ]
    if reactions is not None:
        reactions_set = set(reactions)
        reactions_to_constrain = [
            i
            for i, r in enumerate(model.reactions)
            if not r.boundary and r.id in reactions_set
        ]

    if not reactions_to_constrain:
        return

    s_int = create_stoichiometric_matrix(model)[:, np.array(reactions_to_constrain)]
    s_int = s_int[(np.abs(s_int) > zero_cutoff).any(axis=1)]
    constrained_reactions = [model.reactions[i] for i in reactions_to_constrain]

    max_bound = max(1000.0, max(max(abs(b) for b in r.bounds) for r in model.reactions))
    if np.isinf(max_bound):
        raise ValueError(
            "Loopless constraints require finite reaction bounds. Infinite "
            "bounds can be replaced with `cobra.manipulation.clip_reaction_bounds`."
        )

    if flux_threshold is not None:
        _warn_about_integrality_tolerance(model, flux_threshold, max_bound)

    if method == "potentials":
        energies = add_potential_energies(model, s_int, zero_cutoff)
    else:
        if method == "original":
            n_int = nullspace(s_int).T
        else:
            directions_int = np.sign([rxn.bounds for rxn in constrained_reactions])
            n_int = nullspace_fast_snp(
                model.problem, s_int, directions_int, zero_cutoff=zero_cutoff
            ).T
        if flux_threshold is None:
            # Reactions outside of the nullspace cannot be part of a loop.
            # With `flux_threshold` they still get their directional indicators.
            in_nullspace = (np.abs(n_int) > zero_cutoff).any(axis=0)
            n_int = n_int[:, in_nullspace]
            constrained_reactions = [
                rxn for rxn, keep in zip(constrained_reactions, in_nullspace) if keep
            ]
        energies = add_nullspace_energies(
            model, constrained_reactions, n_int, zero_cutoff
        )

    if flux_threshold is None:
        add_indicators(model, constrained_reactions, energies, max_bound)
    else:
        add_directional_indicators(
            model, constrained_reactions, energies, max_bound, flux_threshold
        )


def _add_cycle_free(model: "Model", fluxes: Dict[str, float]) -> None:
    """Add constraints for CycleFreeFlux.

    Parameters
    ----------
    model : cobra.Model
        The model to operate on.
    fluxes : dict of {str: float}
        A dictionary having keys as reaction IDs and values as their flux
        values.

    """
    model.objective = model.solver.interface.Objective(
        Zero, direction="min", sloppy=True
    )
    objective_vars = []
    for rxn in model.reactions:
        # The solver may violate the bounds within its tolerance.
        flux = min(max(fluxes[rxn.id], rxn.lower_bound), rxn.upper_bound)
        if rxn.boundary:
            rxn.bounds = (flux, flux)
            continue
        if flux >= 0:
            rxn.bounds = max(0, rxn.lower_bound), min(flux, rxn.upper_bound)
            objective_vars.append(rxn.forward_variable)
        else:
            rxn.bounds = max(flux, rxn.lower_bound), min(0, rxn.upper_bound)
            objective_vars.append(rxn.reverse_variable)

    model.objective.set_linear_coefficients({v: 1.0 for v in objective_vars})


def loopless_solution(
    model: "Model", fluxes: Optional[Dict[str, float]] = None
) -> "Solution":
    """Convert an existing solution to a loopless one.

    Removes as many loops as possible (see Notes).

    Uses the method from CycleFreeFlux [1]_ and is much faster than
    `add_loopless` and should therefore be the preferred option to get
    loopless flux distributions.

    Parameters
    ----------
    model : cobra.Model
        The model to which to add the constraints.
    fluxes : dict of {str, float}, optional
        A dictionary having keys as reaction IDs and values as their flux
        values. If not None will use the provided flux values to obtain a
        close loopless solution (default None).

    Returns
    -------
    cobra.Solution
        A solution object containing the fluxes with the least amount of
        loops possible or None if the optimization failed (usually happening
        if the flux distribution in `fluxes` is infeasible).

    Notes
    -----
    The returned flux solution has the following properties:

    - It contains the minimal number of loops possible and no loops at all
      if all flux bounds include zero and the objective is not in a cycle.
    - It has the same objective value as the original flux solution and assumes
      that the objective does not participate in a cycle
      (which is usually true since it consumes metabolites).
    - It has the same exact exchange fluxes as the previous solution.
    - All fluxes have the same sign (flow in the same direction) as the
      previous solution.

    When providing fluxes to the method, please note that those have to come from the
    exact same model that you provided, meaning that no bounds or coefficients have
    been changed, and the optimum has remained the same.

    References
    ----------
    .. [1] CycleFreeFlux: efficient removal of thermodynamically infeasible
       loops from flux distributions. Desouki AA, Jarre F, Gelius-Dietrich
       G, Lercher MJ. Bioinformatics. 2015 Jul 1;31(13):2159-65. doi:
       10.1093/bioinformatics/btv096.

    """
    # Need to reoptimize otherwise spurious solution artifacts can cause
    # all kinds of havoc
    # TODO: check solution status
    if fluxes is None:
        sol = model.optimize(objective_sense=None)
        fluxes = sol.fluxes
        opt = sol.objective_value
    else:
        opt = model.slim_optimize()

    with model:
        prob = model.problem
        # Fix the objective
        loopless_obj_constraint = prob.Constraint(
            model.objective.expression,
            lb=opt,
            name="loopless_obj_constraint",
        )
        model.add_cons_vars([loopless_obj_constraint])
        _add_cycle_free(model, fluxes)
        solution = model.optimize(objective_sense=None)
        solution.objective_value = loopless_obj_constraint.primal

    return solution


def loopless_fva_iter(
    model: "Model",
    reaction: "Reaction",
    solution: bool = False,
    zero_cutoff: Optional[float] = None,
) -> Union[float, Dict[str, float]]:
    """Plugin to get a loopless FVA solution from single FVA iteration.

    Assumes the following about `model` and `reaction`:
    1. The model objective is set to be `reaction`.
    2. The model has been optimized and contains the minimum/maximum flux
       for `reaction`.
    3. The model contains an auxiliary variable called "fva_old_objective"
       denoting the previous objective.

    Parameters
    ----------
    model : cobra.Model
        The model to be used.
    reaction : cobra.Reaction
        The reaction currently minimized/maximized.
    solution : bool, optional
        Whether to return the entire solution or only the minimum/maximum
        for `reaction` (default False).
    zero_cutoff : positive float, optional
        Cutoff used for loop removal. Fluxes with an absolute value smaller
        than `zero_cutoff` are considered to be zero. The default is to use
        `model.tolerance` (default None).

    Returns
    -------
    single float or dict of {str: float}
        Returns the minimized/maximized flux through `reaction` if
        `solution` is False. Otherwise, returns a loopless flux
        solution object containing the minimum/maximum flux for `reaction`.

    """
    zero_cutoff = normalize_cutoff(model, zero_cutoff)

    current = model.objective.value
    sol = get_solution(model)
    objective_dir = model.objective.direction

    # Handle a suddenly infeasible solution,
    # usually due to numerical instability
    if current is None:
        return None

    # Boundary reactions cannot be part of cycles, but their full solutions
    # can still contain cycles elsewhere in the model.
    if reaction.boundary and not solution:
        return current

    with model:
        _add_cycle_free(model, sol.fluxes)
        model.slim_optimize()

        # If the previous optimum is maintained in the loopless solution it was
        # loopless and we are done
        if abs(reaction.flux - current) < zero_cutoff:
            if solution:
                return get_solution(model)
            return current

        # If previous optimum was not in the loopless solution create a new
        # almost loopless solution containing only loops including the current
        # reaction. Than remove all of those loops.
        ll_sol = get_solution(model).fluxes
        reaction.bounds = (current, current)
        model.slim_optimize()
        almost_ll_sol = get_solution(model).fluxes

    with model:
        # find the reactions with loops using the current reaction and remove
        # the loops
        for rxn in model.reactions:
            rid = rxn.id
            if (abs(ll_sol[rid]) < zero_cutoff) and (
                abs(almost_ll_sol[rid]) > zero_cutoff
            ):
                rxn.bounds = max(0, rxn.lower_bound), min(0, rxn.upper_bound)

        if solution:
            best = model.optimize()
        else:
            model.slim_optimize()
            best = reaction.flux
    model.objective.direction = objective_dir
    return best
