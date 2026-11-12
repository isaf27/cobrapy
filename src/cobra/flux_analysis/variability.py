"""Provide variability based methods such as flux variability or gene essentiality."""

import logging
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from optlang.symbolics import Zero

from ..core import Configuration, get_solution
from ..util import ProcessPool
from ..util import solver as sutil
from .deletion import single_gene_deletion, single_reaction_deletion
from .find_cyclic_reactions import find_cyclic_reactions
from .loopless import loopless_fva_iter
from .loopless_fva import (
    ClippedFluxScreening,
    add_loopless_objective_bounds,
    create_loopless_model,
    requires_loopless_for_all,
)
from .parsimonious import add_pfba


if TYPE_CHECKING:
    from cobra import Gene, Model, Reaction


logger = logging.getLogger(__name__)
configuration = Configuration()


def _init_abs_flux_clip_target(abs_flux_clip: Optional[float], sense: str) -> None:
    """Initialize the global variable that is optimized instead of a flux.

    It is bounded by `abs_flux_clip` and, in `_fva_step`, by the flux of the
    optimized reaction, so the solver does not need to go beyond the clip.
    """
    global _abs_flux_clip_target
    global _abs_flux_clip_constraint
    _abs_flux_clip_target = None
    _abs_flux_clip_constraint = None

    if abs_flux_clip is None:
        return

    target_name = "_fva_abs_flux_clip_target"
    constraint_name = "_fva_abs_flux_clip_constraint"

    if target_name in _model.solver.variables:
        _abs_flux_clip_target = _model.variables[target_name]
    else:
        _abs_flux_clip_target = _model.problem.Variable(target_name)
        _model.add_cons_vars([_abs_flux_clip_target])

    if constraint_name in _model.solver.constraints:
        _abs_flux_clip_constraint = _model.constraints[constraint_name]
    else:
        _abs_flux_clip_constraint = _model.problem.Constraint(
            Zero,
            name=constraint_name,
        )
        _model.add_cons_vars([_abs_flux_clip_constraint])

    if sense == "max":
        _abs_flux_clip_target.lb = None
        _abs_flux_clip_target.ub = abs_flux_clip
        _abs_flux_clip_constraint.ub = 0
        _abs_flux_clip_constraint.lb = None
    else:
        _abs_flux_clip_target.lb = -abs_flux_clip
        _abs_flux_clip_target.ub = None
        _abs_flux_clip_constraint.lb = 0
        _abs_flux_clip_constraint.ub = None


def _init_worker(
    model: "Model",
    loopless: bool,
    sense: str,
    abs_flux_clip: Optional[float],
    all_fluxes: bool = False,
) -> None:
    """Initialize a global model object for multiprocessing.

    Parameters
    ----------
    model: cobra.Model
        The model to operate on.
    loopless: bool
        Whether to use loopless version.
    sense: {"max", "min"}
        Whether to maximise or minimise objective.
    abs_flux_clip: float, optional
        Clips optimal flux values by absolute value.
    all_fluxes: bool, optional
        Whether to keep the complete flux distribution for each optimum.

    """
    global _model
    global _loopless
    global _abs_flux_clip
    global _all_fluxes
    _model = model
    _model.solver.objective.direction = sense
    _loopless = loopless
    _abs_flux_clip = abs_flux_clip
    _all_fluxes = all_fluxes

    _init_abs_flux_clip_target(None if loopless else abs_flux_clip, sense)


def _set_objective_coefficient(rxn: "Reaction", coefficient: int) -> None:
    """Set the coefficient of the flux of a reaction in the global objective.

    With clipping, the objective is the clip target, which is tied to the flux.
    """
    if _abs_flux_clip_target is None:
        _model.solver.objective.set_linear_coefficients(
            {rxn.forward_variable: coefficient, rxn.reverse_variable: -coefficient}
        )
    else:
        _abs_flux_clip_constraint.set_linear_coefficients(
            {
                _abs_flux_clip_target: coefficient,
                rxn.forward_variable: -coefficient,
                rxn.reverse_variable: coefficient,
            }
        )
        _model.solver.objective.set_linear_coefficients(
            {_abs_flux_clip_target: coefficient}
        )


def _fva_step(reaction_id: str) -> Tuple[str, float, Optional[pd.Series]]:
    """Take a step for calculating FVA.

    Parameters
    ----------
    reaction_id: str
        The ID of the reaction.

    Returns
    -------
    tuple of (str, float, pandas.Series or None)
        The reaction ID, the possibly clipped flux value, and the optional
        complete flux distribution.

    """
    global _model
    global _loopless
    global _abs_flux_clip
    global _all_fluxes

    rxn = _model.reactions.get_by_id(reaction_id)

    # The previous objective assignment already triggers a reset
    # so directly update coefs here to not trigger redundant resets
    # in the history manager which can take longer than the actual
    # FVA for small models
    try:
        _set_objective_coefficient(rxn, 1)
        _model.slim_optimize()
        sutil.check_solver_status(_model.solver.status)
        fluxes = None
        if _loopless:
            if _all_fluxes:
                solution = loopless_fva_iter(_model, rxn, solution=True)
                value = None if solution is None else solution.fluxes[reaction_id]
                fluxes = None if solution is None else solution.fluxes
            else:
                value = loopless_fva_iter(_model, rxn)
        else:
            value = _model.solver.objective.value
            if _all_fluxes:
                fluxes = get_solution(_model).fluxes

        # handle infeasible case
        if value is None:
            value = float("nan")
            logger.warning(
                f"Could not get flux for reaction {rxn.id}, setting it to NaN. "
                "This is usually due to numerical instability."
            )
        elif _abs_flux_clip is not None:
            value = min(max(value, -_abs_flux_clip), _abs_flux_clip)
    finally:
        _set_objective_coefficient(rxn, 0)

    return reaction_id, value, fluxes


def _parse_reaction_requests(
    model: "Model",
    reaction_list: Optional[
        List[Union["Reaction", str, Tuple[Union["Reaction", str], str]]]
    ],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return the requested reaction IDs and the reactions requested per bound."""
    if reaction_list is None:
        reaction_ids = [r.id for r in model.reactions]
        return reaction_ids, {"minimum": reaction_ids, "maximum": reaction_ids}

    aliases = {"min": "minimum", "max": "maximum"}
    reaction_ids = []
    requested = {"minimum": set(), "maximum": set()}
    for entry in reaction_list:
        directions = ("minimum", "maximum")
        if isinstance(entry, tuple):
            entry, direction = entry
            direction = aliases.get(direction.lower(), direction.lower())
            if direction not in requested:
                raise ValueError(
                    "Directional reaction requests must use 'min', 'max', "
                    "'minimum' or 'maximum'."
                )
            directions = (direction,)
        reaction_id = model.reactions.get_by_any([entry])[0].id
        if not any(reaction_id in ids for ids in requested.values()):
            reaction_ids.append(reaction_id)
        for direction in directions:
            requested[direction].add(reaction_id)
    return reaction_ids, {
        direction: [rxn_id for rxn_id in reaction_ids if rxn_id in ids]
        for direction, ids in requested.items()
    }


def _split_cyclic_requests(
    requested: Dict[str, List[str]],
    cyclic_reactions: List[str],
    cyclic_directions: List[Tuple[bool, bool]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Split the requested bounds into those of non-cyclic and cyclic directions."""
    cyclic = {
        "minimum": {
            rxn_id
            for rxn_id, (negative, _) in zip(cyclic_reactions, cyclic_directions)
            if negative
        },
        "maximum": {
            rxn_id
            for rxn_id, (_, positive) in zip(cyclic_reactions, cyclic_directions)
            if positive
        },
    }
    non_cyclic_requests = {
        direction: [rxn_id for rxn_id in ids if rxn_id not in cyclic[direction]]
        for direction, ids in requested.items()
    }
    cyclic_requests = {
        direction: [rxn_id for rxn_id in ids if rxn_id in cyclic[direction]]
        for direction, ids in requested.items()
    }
    return non_cyclic_requests, cyclic_requests


def _add_objective_bounds(
    model: "Model", fraction_of_optimum: Optional[float], pfba_factor: Optional[float]
) -> None:
    """Constrain the objective and the total flux to stay close to their optima."""
    prob = model.problem
    if fraction_of_optimum is not None:
        # Safety check before setting up FVA.
        model.slim_optimize(
            error_value=None,
            message="There is no optimal solution for the chosen objective!",
        )
        # Add the previous objective as a variable to the model then set it to
        # zero. This also uses the fraction to create the lower/upper bound for
        # the old objective.
        # TODO: Use utility function here (fix_objective_as_constraint)?
        if model.solver.objective.direction == "max":
            fva_old_objective = prob.Variable(
                "fva_old_objective",
                lb=fraction_of_optimum * model.solver.objective.value,
            )
        else:
            fva_old_objective = prob.Variable(
                "fva_old_objective",
                ub=fraction_of_optimum * model.solver.objective.value,
            )
        fva_old_obj_constraint = prob.Constraint(
            model.solver.objective.expression - fva_old_objective,
            lb=0,
            ub=0,
            name="fva_old_objective_constraint",
        )
        model.add_cons_vars([fva_old_objective, fva_old_obj_constraint])

    if pfba_factor is not None:
        if pfba_factor < 1.0:
            warn(
                "The 'pfba_factor' should be larger or equal to 1.",
                UserWarning,
            )
        with model:
            add_pfba(model, fraction_of_optimum=0)
            ub = model.slim_optimize(error_value=None)
            flux_sum = prob.Variable("flux_sum", ub=pfba_factor * ub)
            flux_sum_constraint = prob.Constraint(
                model.solver.objective.expression - flux_sum,
                lb=0,
                ub=0,
                name="flux_sum_constraint",
            )
        model.add_cons_vars([flux_sum, flux_sum_constraint])


def _optimize_reactions(
    model: "Model",
    reaction_ids: List[str],
    sense: str,
    cycle_free_flux: bool,
    abs_flux_clip: Optional[float],
    processes: int,
    all_fluxes: bool,
) -> List[Tuple[str, float, Optional[pd.Series]]]:
    """Minimize or maximize the flux of each reaction, in parallel if requested."""
    if not reaction_ids:
        return []
    initargs = (model, cycle_free_flux, sense, abs_flux_clip, all_fluxes)
    processes = min(processes, len(reaction_ids))
    if processes > 1:
        # We create and destroy a new pool here in order to set the
        # objective direction for all reactions. This creates a
        # slight overhead but seems the most clean.
        chunk_size = len(reaction_ids) // processes
        with ProcessPool(
            processes, initializer=_init_worker, initargs=initargs
        ) as pool:
            return list(
                pool.imap_unordered(_fva_step, reaction_ids, chunksize=chunk_size)
            )
    _init_worker(*initargs)
    return list(map(_fva_step, reaction_ids))


def _confirm_clipped_bounds(
    model: "Model",
    candidates: List[Tuple[str, str]],
    abs_flux_clip: float,
    processes: int,
    all_fluxes: bool,
) -> Iterator[Tuple[str, str, Optional[pd.Series]]]:
    """Optimize the candidates and yield those whose bound reaches the clip."""
    for direction in ("minimum", "maximum"):
        rxn_ids = [rxn_id for rxn_id, d in candidates if d == direction]
        for rxn_id, bound, fluxes in _optimize_reactions(
            model,
            rxn_ids,
            sense=direction[:3],
            cycle_free_flux=False,
            abs_flux_clip=abs_flux_clip,
            processes=processes,
            all_fluxes=all_fluxes,
        ):
            if abs(bound) >= abs_flux_clip - model.tolerance:
                yield direction, rxn_id, fluxes


def _find_clipped_loopless_bounds(
    model: "Model",
    requested: Dict[str, List[str]],
    method: str,
    abs_flux_clip: float,
    cyclic_reactions: List[str],
    processes: int,
    all_fluxes: bool,
) -> Dict[str, Dict[str, Optional[pd.Series]]]:
    """Find requested bounds that reach `abs_flux_clip` without a loop.

    Each reaction found is mapped to a loopless flux distribution in which it
    reaches the clip if `all_fluxes` is set and to None otherwise. Bounds can
    be missed, so the remaining ones still have to be optimized.
    """
    targets = [
        (reaction_id, direction)
        for direction in ("minimum", "maximum")
        for reaction_id in requested[direction]
    ]
    logger.info(
        f"Screening {len(targets)} flux bounds for loopless fluxes "
        f"of at least {abs_flux_clip}."
    )

    result = {"minimum": {}, "maximum": {}}
    with model:
        # The screening only proposes candidates. They are confirmed by linear
        # programs on a copy without loopless constraints, in which the
        # directions without flux in the screening solution are closed.
        validation_model = model.copy()
        try:
            screening = ClippedFluxScreening(
                model, method, cyclic_reactions, targets, abs_flux_clip
            )
            while candidates := screening.next_candidates():
                with validation_model:
                    screening.close_inactive_directions(validation_model)
                    for direction, rxn_id, fluxes in _confirm_clipped_bounds(
                        validation_model,
                        candidates,
                        abs_flux_clip,
                        processes,
                        all_fluxes,
                    ):
                        result[direction][rxn_id] = fluxes
        except Exception:
            logger.warning(
                "Screening for clipped loopless flux bounds failed. "
                "The remaining bounds are optimized one by one.",
                exc_info=True,
            )

    logger.info(
        f"Found {len(result['minimum']) + len(result['maximum'])} flux bounds "
        f"that reach {abs_flux_clip}."
    )

    return result


def flux_variability_analysis(
    model: "Model",
    reaction_list: Optional[
        List[Union["Reaction", str, Tuple[Union["Reaction", str], str]]]
    ] = None,
    loopless: Union[Optional[str], bool] = None,
    fraction_of_optimum: Optional[float] = 1.0,
    pfba_factor: Optional[float] = None,
    abs_flux_clip: Optional[float] = None,
    processes: Optional[int] = None,
    all_fluxes: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """Determine the minimum and maximum flux value for each reaction.

    Parameters
    ----------
    model : cobra.Model
        The model for which to run the analysis. It will *not* be modified.
    reaction_list : list of cobra.Reaction, str or tuple, optional
        The reactions for which to obtain min/max fluxes. A tuple of a
        reaction and "minimum" or "maximum" ("min" or "max" for short)
        requests only that bound. If None will use all reactions in the
        model (default None).
    loopless : str, "potentials", "fastSNP" or "cycleFreeFlux", optional
        If this value is set, only loopless solutions will be returned.
        Boolean values are deprecated. The value selects the algorithm used
        to constrain the model to loopless solutions.
        Please also refer to the notes (default None).
    fraction_of_optimum : float, optional
        Must be <= 1.0. Requires that the objective value is at least the
        fraction times maximum objective value. A value of 0.85 for instance
        means that the objective has to be at least 85% of its maximum.
        If set to ``None``, the original objective is not constrained
        (default 1.0).
    pfba_factor : float, optional
        Add an additional constraint to the model that requires the total sum
        of absolute fluxes must not be larger than this value times the
        smallest possible sum of absolute fluxes, i.e., by setting the value
        to 1.1 the total sum of absolute fluxes must not be more than
        10% larger than the pFBA solution. Since the pFBA solution is the
        one that optimally minimizes the total flux sum, the `pfba_factor`
        should, if set, be larger than one. Setting this value may lead to
        more realistic predictions of the effective flux bounds
        (default None).
    abs_flux_clip : float, optional
        Clip the reported fluxes to the interval from ``-abs_flux_clip`` to
        ``abs_flux_clip``. This saves time when it only matters whether a
        flux can reach this value, for instance to find blocked reactions
        (default None).
    processes : int, optional
        The number of parallel processes to run. If not explicitly passed,
        will be set from the global configuration singleton (default None).
    all_fluxes : bool, optional
        Whether to also return the flux distribution found at each optimum
        (default False). FVA solves one optimization per reaction per direction
        and discards every solution but the objective value; setting this keeps
        them, which saves recomputing an FVA-sized batch of solves when the
        distributions themselves are wanted -- for example as a starting pool
        for sampling. With `loopless="fastSNP"` or `loopless="potentials"`, the
        loopless constraints are then used for all reactions so that every
        returned distribution is loopless, which can be slower. The
        distributions are not clipped: with `abs_flux_clip`, the flux of the
        optimized reaction can exceed the reported bound.

    Returns
    -------
    pandas.DataFrame or tuple of (pandas.DataFrame, dict of {str: pandas.DataFrame})
        A data frame with reaction identifiers as the index and two columns:
        - maximum: indicating the highest possible flux
        - minimum: indicating the lowest possible flux
        Directional reaction requests leave unrequested bounds as ``NaN``.
        If `all_fluxes` is True, a tuple whose second element is a dict
        with keys "maximum" and "minimum", each containing a DataFrame with
        the optimized reactions as rows/index and the obtained flux
        distributions for all reactions as columns. Unrequested directions
        have no corresponding row in these DataFrames.

    Notes
    -----
    This implements the fast version as described in [1]_. Please note that
    the flux distribution containing all minimal/maximal fluxes does not have
    to be a feasible solution for the model. Fluxes are minimized/maximized
    individually and a single minimal flux might require all others to be
    sub-optimal.

    Using the loopless option will lead to a significant increase in
    computation time (about a factor of 100 for large models).

    If `loopless` is set to "potentials" or "fastSNP", the optimal loopless
    flux bounds will be found by adding loopless constraints to the model.
    The "fastSNP" method uses the efficient Fast-SNP algorithm (see [2]_),
    while "potentials" uses metabolite potential variables. See
    :func:`add_loopless` for details of these loopless constraint
    formulations.

    With these two methods, the optimum for `fraction_of_optimum` and the
    minimal total flux for `pfba_factor` are loopless as well.

    If `loopless` is set to "cycleFreeFlux", the loops removal algorithm will be
    used (see [3]_). Note: this algorithm does not guarantee to find optimal bounds.

    References
    ----------
    .. [1] Computationally efficient flux variability analysis.
       Gudmundsson S, Thiele I.
       BMC Bioinformatics. 2010 Sep 29;11:489.
       doi: 10.1186/1471-2105-11-489, PMID: 20920235

    .. [2] Fast-SNP: a fast matrix pre-processing algorithm for efficient
       loopless flux optimization of metabolic models. Saa PA, Nielsen LK.
       Bioinformatics. 2016 Dec;32(24):3807–3814. doi: 10.1093/bioinformatics/btw555.

    .. [3] CycleFreeFlux: efficient removal of thermodynamically infeasible
       loops from flux distributions.
       Desouki AA, Jarre F, Gelius-Dietrich G, Lercher MJ.
       Bioinformatics. 2015 Jul 1;31(13):2159-65.
       doi: 10.1093/bioinformatics/btv096.
    """
    if loopless is not None and isinstance(loopless, bool):
        warn(
            "Passing a boolean value to the `loopless` argument is deprecated. "
            "Please pass either None, 'potentials', 'fastSNP' or 'cycleFreeFlux'.",
            DeprecationWarning,
            stacklevel=2,
        )
        loopless = "cycleFreeFlux" if loopless else None

    if loopless not in (None, "potentials", "fastSNP", "cycleFreeFlux"):
        raise ValueError(
            "The `loopless` argument must be either None, 'potentials', "
            "'fastSNP' or 'cycleFreeFlux'."
        )

    if abs_flux_clip is not None and abs_flux_clip <= model.tolerance:
        raise ValueError(
            "The `abs_flux_clip` argument must be bigger than `model.tolerance`."
        )

    reaction_ids, requested = _parse_reaction_requests(model, reaction_list)

    if processes is None:
        processes = configuration.processes

    fva_result = pd.DataFrame(
        np.nan, index=reaction_ids, columns=["minimum", "maximum"]
    )
    flux_rows: Dict[str, Dict[str, pd.Series]] = {"minimum": {}, "maximum": {}}

    cyclic_reactions, cyclic_directions = [], []
    if loopless is not None:
        cyclic_reactions, cyclic_directions = find_cyclic_reactions(model)
    # Only cyclic directions can differ between FVA and loopless FVA.
    plain_requests, cyclic_requests = _split_cyclic_requests(
        requested, cyclic_reactions, cyclic_directions
    )
    loopless_requests = cyclic_requests

    use_loopless_constraints = loopless in ("fastSNP", "potentials")
    with model:
        loopless_model = model
        if use_loopless_constraints:
            loopless_model = create_loopless_model(model, loopless, cyclic_reactions)
            add_loopless_objective_bounds(
                model, loopless_model, fraction_of_optimum, pfba_factor
            )
            loopless_model.objective = Zero
            if all_fluxes or requires_loopless_for_all(
                model, cyclic_reactions, cyclic_directions
            ):
                # Screening still covers the cyclic directions only.
                plain_requests = {"minimum": [], "maximum": []}
                loopless_requests = requested
        else:
            _add_objective_bounds(model, fraction_of_optimum, pfba_factor)

        model.objective = Zero  # This will trigger the reset as well

        if (
            use_loopless_constraints
            and abs_flux_clip is not None
            and any(cyclic_requests.values())
        ):
            # Bounds that reach the clip are cheaper to find by screening
            # than by one MILP each. Screening adds its own, directional
            # loopless constraints to the base model, which has the same
            # objective bounds as the loopless model.
            clipped = _find_clipped_loopless_bounds(
                model=model,
                requested=cyclic_requests,
                method=loopless,
                abs_flux_clip=abs_flux_clip,
                cyclic_reactions=cyclic_reactions,
                processes=processes,
                all_fluxes=all_fluxes,
            )
            for direction, clipped_fluxes in clipped.items():
                sign = -1 if direction == "minimum" else 1
                for rxn_id, fluxes in clipped_fluxes.items():
                    fva_result.at[rxn_id, direction] = sign * abs_flux_clip
                    if fluxes is not None:
                        flux_rows[direction][rxn_id] = fluxes
            loopless_requests = {
                direction: [
                    rxn_id for rxn_id in ids if rxn_id not in clipped[direction]
                ]
                for direction, ids in loopless_requests.items()
            }

        for is_loopless, requests in (
            (False, plain_requests),
            (True, loopless_requests),
        ):
            if not any(requests.values()):
                continue

            number_of_bounds = sum(len(ids) for ids in requests.values())
            clip_info = (
                f" (with [{-abs_flux_clip}, {abs_flux_clip}] clip)"
                if abs_flux_clip is not None
                else ""
            )
            logger.info(
                f"Performing {'loopless ' if is_loopless else ''}"
                "flux variability analysis for "
                f"{number_of_bounds} flux bounds{clip_info}."
            )

            cycle_free_flux = loopless == "cycleFreeFlux" and (
                is_loopless or all_fluxes
            )
            for direction in ("minimum", "maximum"):
                for rxn_id, value, fluxes in _optimize_reactions(
                    loopless_model if is_loopless else model,
                    requests[direction],
                    sense=direction[:3],
                    cycle_free_flux=cycle_free_flux,
                    abs_flux_clip=abs_flux_clip,
                    processes=processes,
                    all_fluxes=all_fluxes,
                ):
                    fva_result.at[rxn_id, direction] = value
                    if fluxes is not None:
                        flux_rows[direction][rxn_id] = fluxes

    if all_fluxes:
        return (
            fva_result,
            {
                direction: pd.DataFrame.from_dict(
                    rows, orient="index", columns=[r.id for r in model.reactions]
                ).reindex([rxn_id for rxn_id in reaction_ids if rxn_id in rows])
                for direction, rows in flux_rows.items()
            },
        )
    return fva_result


def find_essential_genes(
    model: "Model",
    threshold: Optional[float] = None,
    processes: Optional[int] = None,
) -> Set["Gene"]:
    """Return a set of essential genes.

    A gene is considered essential if restricting the flux of all reactions
    that depend on it to zero causes the objective, e.g., the growth rate,
    to also be zero, below the threshold, or infeasible.

    Parameters
    ----------
    model : cobra.Model
        The model to find the essential genes for.
    threshold : float, optional
        Minimal objective flux to be considered viable. By default this is
        1% of the maximal objective (default None).
    processes : int, optional
        The number of parallel processes to run. Can speed up the computations
        if the number of knockouts to perform is large. If not explicitly
        passed, it will be set from the global configuration singleton
        (default None).

    Returns
    -------
    set of cobra.Gene
        Set of essential genes.

    """
    if threshold is None:
        threshold = model.slim_optimize(error_value=None) * 1e-02
    deletions = single_gene_deletion(model, method="fba", processes=processes)
    essential = deletions.loc[
        deletions["growth"].isna() | (deletions["growth"] < threshold), :
    ].ids
    return {model.genes.get_by_id(g) for ids in essential for g in ids}


def find_essential_reactions(
    model: "Model",
    threshold: Optional[float] = None,
    processes: Optional[int] = None,
) -> Set["Reaction"]:
    """Return a set of essential reactions.

    A reaction is considered essential if restricting its flux to zero
    causes the objective, e.g., the growth rate, to also be zero, below the
    threshold, or infeasible.


    Parameters
    ----------
    model : cobra.Model
        The model to find the essential reactions for.
    threshold : float, optional
        Minimal objective flux to be considered viable. By default this is
        1% of the maximal objective (default None).
    processes : int, optional
        The number of parallel processes to run. Can speed up the computations
        if the number of knockouts to perform is large. If not explicitly
        passed, it will be set from the global configuration singleton
        (default None).

    Returns
    -------
    set of cobra.Reaction
        Set of essential reactions.

    """
    if threshold is None:
        threshold = model.slim_optimize(error_value=None) * 1e-02
    deletions = single_reaction_deletion(model, method="fba", processes=processes)
    essential = deletions.loc[
        deletions["growth"].isna() | (deletions["growth"] < threshold), :
    ].ids
    return {model.reactions.get_by_id(r) for ids in essential for r in ids}
