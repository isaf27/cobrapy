"""Provide variability based methods such as flux variability or gene essentiality."""

import logging
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from optlang.symbolics import Zero

from ..core import Configuration
from ..util import ProcessPool
from ..util import solver as sutil
from .deletion import single_gene_deletion, single_reaction_deletion
from .find_cyclic_reactions import find_cyclic_reactions
from .loopless import add_loopless, loopless_fva_iter
from .parsimonious import add_pfba


if TYPE_CHECKING:
    from cobra import Gene, Model, Reaction


logger = logging.getLogger(__name__)
configuration = Configuration()


def _init_abs_flux_clip_target(abs_flux_clip: Optional[float], sense: str) -> None:
    """Initialize global target variables for absolute flux clipping."""

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

    """
    global _model
    global _loopless
    global _abs_flux_clip
    _model = model
    _model.solver.objective.direction = sense
    _loopless = loopless
    _abs_flux_clip = abs_flux_clip

    _init_abs_flux_clip_target(None if loopless else abs_flux_clip, sense)


def _fva_step(reaction_id: str) -> Tuple[str, float]:
    """Take a step for calculating FVA.

    Parameters
    ----------
    reaction_id: str
        The ID of the reaction.

    Returns
    -------
    tuple of (str, float)
        The reaction ID with the flux value.

    """
    global _model
    global _loopless
    global _abs_flux_clip
    global _abs_flux_clip_target
    global _abs_flux_clip_constraint

    rxn = _model.reactions.get_by_id(reaction_id)

    # The previous objective assignment already triggers a reset
    # so directly update coefs here to not trigger redundant resets
    # in the history manager which can take longer than the actual
    # FVA for small models
    try:
        if _abs_flux_clip_target is None:
            _model.solver.objective.set_linear_coefficients(
                {rxn.forward_variable: 1, rxn.reverse_variable: -1}
            )
        else:
            _abs_flux_clip_constraint.set_linear_coefficients(
                {
                    _abs_flux_clip_target: 1,
                    rxn.forward_variable: -1,
                    rxn.reverse_variable: 1,
                }
            )
            _model.solver.objective.set_linear_coefficients({_abs_flux_clip_target: 1})

        _model.slim_optimize()
        sutil.check_solver_status(_model.solver.status)
        if _loopless:
            value = loopless_fva_iter(_model, rxn)
        else:
            value = _model.solver.objective.value

        # handle infeasible case
        if value is None:
            value = float("nan")
            logger.warning(
                f"Could not get flux for reaction {rxn.id}, setting it to NaN. "
                "This is usually due to numerical instability."
            )
        else:
            if _abs_flux_clip is not None:
                value = min(max(value, -_abs_flux_clip), _abs_flux_clip)
    finally:
        if _abs_flux_clip_target is None:
            _model.solver.objective.set_linear_coefficients(
                {rxn.forward_variable: 0, rxn.reverse_variable: 0}
            )
        else:
            _model.solver.objective.set_linear_coefficients({_abs_flux_clip_target: 0})
            _abs_flux_clip_constraint.set_linear_coefficients(
                {
                    _abs_flux_clip_target: 0,
                    rxn.forward_variable: 0,
                    rxn.reverse_variable: 0,
                }
            )

    return reaction_id, value


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
) -> pd.DataFrame:
    """Determine the minimum and maximum flux value for each reaction.

    Parameters
    ----------
    model : cobra.Model
        The model for which to run the analysis. It will *not* be modified.
    reaction_list : list of cobra.Reaction or str or tuple of (cobra.Reaction
        or str, str), optional
        The reactions for which to obtain flux bounds. Entries can be
        reactions, reaction IDs, or ``(reaction, direction)`` tuples. Direction
        can be ``"minimum"``, ``"maximum"``, ``"min"``, or ``"max"``. If a
        direction is given, only that bound is computed for the reaction. If
        ``None``, all reactions and both bounds are used (default None).
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
        Maximum absolute flux value reported by variability analysis. When set,
        maximum flux values are clipped to ``abs_flux_clip`` and minimum flux
        values are clipped to ``-abs_flux_clip``
        (default None).
    processes : int, optional
        The number of parallel processes to run. If not explicitly passed,
        will be set from the global configuration singleton (default None).

    Returns
    -------
    pandas.DataFrame
        A data frame with reaction identifiers as the index and two columns:
        - maximum: indicating the highest possible flux
        - minimum: indicating the lowest possible flux
        Directional reaction requests leave unrequested bounds as ``NaN``.

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

    if abs_flux_clip is not None and abs_flux_clip < 0:
        raise ValueError("The `abs_flux_clip` argument must be non-negative.")

    if reaction_list is None:
        reaction_ids = [r.id for r in model.reactions]
        requested_by_direction = {
            "minimum": set(reaction_ids),
            "maximum": set(reaction_ids),
        }
    else:
        requested_by_direction = {"minimum": set(), "maximum": set()}
        reaction_ids_set = set()

        def _add_reaction_request(
            reaction: Union["Reaction", str], directions: Tuple[str, ...]
        ) -> None:
            rxn = model.reactions.get_by_any([reaction])[0]
            reaction_ids_set.add(rxn.id)
            for direction in directions:
                requested_by_direction[direction].add(rxn.id)

        for reaction_entry in reaction_list:
            if isinstance(reaction_entry, tuple):
                reaction, direction = reaction_entry
                direction = direction.lower()
                if direction == "min":
                    direction = "minimum"
                elif direction == "max":
                    direction = "maximum"
                if direction not in ("minimum", "maximum"):
                    raise ValueError(
                        "Directional reaction requests must use 'min', 'max', "
                        "'minimum' or 'maximum'."
                    )
                _add_reaction_request(reaction, (direction,))
            else:
                _add_reaction_request(reaction_entry, ("minimum", "maximum"))

        reaction_ids = list(reaction_ids_set)

    if processes is None:
        processes = configuration.processes

    num_reactions = len(reaction_ids)
    processes = min(processes, num_reactions)

    fva_result = pd.DataFrame(
        {
            "minimum": np.full(num_reactions, np.nan, dtype=float),
            "maximum": np.full(num_reactions, np.nan, dtype=float),
        },
        index=reaction_ids,
    )

    reaction_ids_by_type = [
        {
            "minimum": [],
            "maximum": [],
        },
        {
            "minimum": [],
            "maximum": [],
        },
    ]
    if loopless is not None:
        cyclic_reactions, cyclic_directions = find_cyclic_reactions(model)
        cyclic_reaction_index = {r_id: i for i, r_id in enumerate(cyclic_reactions)}
        for r_id in reaction_ids:
            i = cyclic_reaction_index.get(r_id)
            for loc, dir in enumerate(("minimum", "maximum")):
                if r_id not in requested_by_direction[dir]:
                    continue
                if i is not None and cyclic_directions[i][loc]:
                    reaction_ids_by_type[1][dir].append(r_id)
                else:
                    reaction_ids_by_type[0][dir].append(r_id)
    else:
        for dir in ("minimum", "maximum"):
            reaction_ids_by_type[0][dir] = [
                r_id for r_id in reaction_ids if r_id in requested_by_direction[dir]
            ]

    prob = model.problem
    with model:
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

        model.objective = Zero  # This will trigger the reset as well
        for loopless_reactions, opt_rxn_ids in enumerate(reaction_ids_by_type):
            if len(opt_rxn_ids["minimum"]) == 0 and len(opt_rxn_ids["maximum"]) == 0:
                continue

            run_cycle_free_flux = bool(loopless_reactions)
            if loopless_reactions and loopless != "cycleFreeFlux":
                add_loopless(
                    model,
                    method=loopless,
                    reactions=cyclic_reactions,
                )
                run_cycle_free_flux = False

            for what in ("minimum", "maximum"):
                if len(opt_rxn_ids[what]) == 0:
                    continue

                cur_processes = min(processes, len(opt_rxn_ids[what]))
                if cur_processes > 1:
                    # We create and destroy a new pool here in order to set the
                    # objective direction for all reactions. This creates a
                    # slight overhead but seems the most clean.
                    chunk_size = len(opt_rxn_ids[what]) // cur_processes
                    with ProcessPool(
                        cur_processes,
                        initializer=_init_worker,
                        initargs=(
                            model,
                            run_cycle_free_flux,
                            what[:3],
                            abs_flux_clip,
                        ),
                    ) as pool:
                        for rxn_id, value in pool.imap_unordered(
                            _fva_step, opt_rxn_ids[what], chunksize=chunk_size
                        ):
                            fva_result.at[rxn_id, what] = value
                else:
                    _init_worker(model, run_cycle_free_flux, what[:3], abs_flux_clip)
                    for rxn_id, value in map(_fva_step, opt_rxn_ids[what]):
                        fva_result.at[rxn_id, what] = value

            logger.info(
                "Finished FVA for "
                f"{len(opt_rxn_ids['minimum']) + len(opt_rxn_ids['maximum'])} "
                f"reactions with loopless={loopless_reactions}."
            )

    return fva_result[["minimum", "maximum"]]


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
