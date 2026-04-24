"""Provides functions to find blocked reactions in a model."""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from optlang.symbolics import Zero

from ..core import Configuration, get_solution
from ..util import solver as sutil
from .find_cyclic_reactions import find_cyclic_reactions
from .helpers import normalize_cutoff
from .loopless import add_loopless
from .variability import flux_variability_analysis


if TYPE_CHECKING:
    from cobra import Model, Reaction


logger = logging.getLogger(__name__)
configuration = Configuration()


class BlockedReactionsResult(list[str]):
    """Result of blocked reactions analysis.

    Behaves as a list of fully blocked reaction identifiers for backward
    compatibility. Fully blocked reactions are blocked in both the forward
    and reverse directions. Direction-specific blocked reactions are also
    available through the ``forward_blocked`` and ``reverse_blocked``
    attributes.

    Attributes
    ----------
    forward_blocked : list of str
        Identifiers of reactions that cannot carry positive flux.
    reverse_blocked : list of str
        Identifiers of reactions that cannot carry negative flux.
    """

    def __init__(self, forward_blocked, reverse_blocked):
        """Initialize blocked reactions result."""
        super().__init__(list(set(forward_blocked) & set(reverse_blocked)))
        self.forward_blocked = forward_blocked
        self.reverse_blocked = reverse_blocked


def find_blocked_reactions(
    model: "Model",
    reaction_list: Optional[List[Union["Reaction", str]]] = None,
    loopless: Optional[str] = None,
    zero_cutoff: Optional[float] = None,
    open_exchanges: bool = False,
    processes: Optional[int] = None,
) -> "BlockedReactionsResult":
    """Find reactions that cannot carry any flux.

    The question whether or not a reaction is blocked is highly dependent
    on the current exchange reaction settings for a COBRA model. Hence an
    argument is provided to open all exchange reactions.

    Parameters
    ----------
    model : cobra.Model
        The model to analyze.
    reaction_list : list of cobra.Reaction or str, optional
        List of reactions to consider, the default includes all model
        reactions (default None).
    loopless : str, "fastSNP" or "potentials", optional
        If set, only loopless flux distributions are considered when checking
        whether reactions can carry flux. The value is passed to
        :func:`flux_variability_analysis` as its loopless method. Supported
        values are ``"fastSNP"`` and ``"potentials"`` (default None).
        See :func:`flux_variability_analysis` for more details.
    zero_cutoff : float, optional
        Flux value which is considered to effectively be zero. The default
        is set to use `model.tolerance` (default None).
    open_exchanges : bool, optional
        Whether or not to open all exchange reactions to very high flux
        ranges (default False).
    processes : int, optional
        The number of parallel processes to run. Can speed up the
        computations if the number of reactions is large. If not explicitly
        passed, it will be set from the global configuration singleton
        (default None).

    Returns
    -------
    BlockedReactionsResult
        A list of fully blocked reaction identifiers. Also has a
        ``forward_blocked`` attribute with reactions that cannot carry
        positive flux and a ``reverse_blocked`` attribute with reactions
        that cannot carry negative flux.

    Notes
    -----
    Sink and demand reactions are left untouched. Please modify them manually.

    """
    if loopless not in (None, "potentials", "fastSNP"):
        raise ValueError(
            "The `loopless` argument must be either None, 'potentials', 'fastSNP'."
        )

    zero_cutoff = normalize_cutoff(model, zero_cutoff)

    with model:
        max_bound = max(
            1000.0, max(max(abs(b) for b in r.bounds) for r in model.reactions)
        )

        if open_exchanges:
            for reaction in model.exchanges:
                reaction.bounds = (
                    min(reaction.lower_bound, -max_bound),
                    max(reaction.upper_bound, max_bound),
                )

        if reaction_list is None:
            reaction_list = model.reactions
        else:
            reaction_list = model.reactions.get_by_any(reaction_list)

        if loopless is None:
            # Limit the search space to reactions which have zero flux.
            # If the reactions already carry flux in this solution,
            # the active direction is known to be feasible and only
            # the opposite direction needs to be checked.
            model.slim_optimize()
            solution = get_solution(model, reactions=reaction_list)
            forward_unknown = solution.fluxes[
                solution.fluxes < zero_cutoff
            ].index.tolist()
            reverse_unknown = solution.fluxes[
                solution.fluxes > -zero_cutoff
            ].index.tolist()
            reaction_list = [
                (reaction_id, "maximum") for reaction_id in forward_unknown
            ] + [(reaction_id, "minimum") for reaction_id in reverse_unknown]

        # Run FVA to find reactions where both the minimal and maximal flux
        # are zero (below the cut off).
        flux_span = flux_variability_analysis(
            model,
            reaction_list=reaction_list,
            loopless=loopless,
            fraction_of_optimum=None,
            processes=processes,
        )

        forward_blocked = flux_span[
            flux_span["maximum"].notna() & (flux_span["maximum"] < zero_cutoff)
        ].index.tolist()
        reverse_blocked = flux_span[
            flux_span["minimum"].notna() & (flux_span["minimum"] > -zero_cutoff)
        ].index.tolist()

        return BlockedReactionsResult(forward_blocked, reverse_blocked)


def _prepare_cyclic_reactions_for_blocked(
    model: "Model",
    reaction_list: List[Union["Reaction", str]],
    zero_cutoff: float,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Filter reactions to cyclic directions for loopless blocked checks."""
    reaction_ids = [r if isinstance(r, str) else r.id for r in reaction_list]

    cyclic_reactions, cyclic_directions = find_cyclic_reactions(
        model,
        zero_cutoff=zero_cutoff,
    )
    cyclic_reaction_index = {r_id: i for i, r_id in enumerate(cyclic_reactions)}

    reactions_to_check = []
    for r_id in reaction_ids:
        if r_id in cyclic_reaction_index:
            c_idx = cyclic_reaction_index[r_id]
            for dir in ["minimum", "maximum"]:
                if cyclic_directions[c_idx][0 if dir == "minimum" else 1]:
                    reactions_to_check.append((r_id, dir))

    return reactions_to_check, cyclic_reactions


def _validate_blocked_reactions(
    model: "Model",
    validation_model: "Model",
    blocked_list: List[Tuple[str, str]],
    zero_cutoff: float,
    cyclic_reactions: List[str],
    processes: Optional[int],
) -> "BlockedReactionsResult":
    """Validate blocked candidates by fixing inactive cyclic directions."""
    for rid in cyclic_reactions:
        max_indicator = model.variables[f"indicator_maximum_{rid}"].primal
        min_indicator = model.variables[f"indicator_minimum_{rid}"].primal
        rxn = validation_model.reactions.get_by_id(rid)
        if max_indicator < 0.5:
            rxn.upper_bound = 0.0
        if min_indicator < 0.5:
            rxn.lower_bound = 0.0

    fluxes = flux_variability_analysis(
        validation_model,
        reaction_list=blocked_list,
        fraction_of_optimum=None,
        processes=processes,
    )

    blocked_by_direction = {
        "maximum": set(),
        "minimum": set(),
    }
    for rid, dir in blocked_list:
        if abs(fluxes.at[rid, dir]) < zero_cutoff:
            blocked_by_direction[dir].add(rid)

    for rid in cyclic_reactions:
        rxn = validation_model.reactions.get_by_id(rid)
        rxn.bounds = model.reactions.get_by_id(rid).bounds

    return BlockedReactionsResult(
        forward_blocked=list(blocked_by_direction["maximum"]),
        reverse_blocked=list(blocked_by_direction["minimum"]),
    )


def _find_blocked_reactions_loopless_directional(
    model: "Model",
    reaction_list: List[Tuple[str, str]],
    loopless: str,
    zero_cutoff: float,
    flux_threshold: float,
    cyclic_reactions: List[str],
    processes: Optional[int],
) -> "BlockedReactionsResult":
    """Find loopless blocked candidates with directional flux indicators.

    This function works efficiently because it finds many non-blocked
    reactions in one iteration by maximizing the number of active reactions.
    However, it may mark some non-blocked reactions as blocked (false
    positives) due to the nature of the optimization problem and numerical
    issues.
    """
    with model:
        validation_model = model.copy()

        add_loopless(
            model,
            zero_cutoff,
            method=loopless,
            reactions=cyclic_reactions,
            flux_threshold=flux_threshold,
        )

        candidates_by_direction = {
            "maximum": set(),
            "minimum": set(),
        }
        for rid, dir in reaction_list:
            candidates_by_direction[dir].add(rid)

        model.objective = Zero
        coefs = {
            model.variables[f"indicator_{dir}_{rid}"]: 1 for rid, dir in reaction_list
        }
        model.objective.set_linear_coefficients(coefs)
        model.objective.direction = "max"

        blocked_by_direction = {
            "maximum": set(),
            "minimum": set(),
        }

        while (
            len(candidates_by_direction["maximum"]) > 0
            or len(candidates_by_direction["minimum"]) > 0
        ):
            model.slim_optimize()
            sutil.check_solver_status(model.solver.status)

            remove_coef = {}
            blocked_reactions_to_validate = []
            for rid, dir in reaction_list:
                if rid in candidates_by_direction[dir]:
                    if model.variables[f"indicator_{dir}_{rid}"].primal >= 0.5:
                        candidates_by_direction[dir].remove(rid)
                        remove_coef[model.variables[f"indicator_{dir}_{rid}"]] = 0
                        blocked_reactions_to_validate.append((rid, dir))

            if len(remove_coef) == 0:
                break

            blocked_reactions = _validate_blocked_reactions(
                model=model,
                validation_model=validation_model,
                blocked_list=blocked_reactions_to_validate,
                zero_cutoff=zero_cutoff,
                cyclic_reactions=cyclic_reactions,
                processes=processes,
            )
            blocked_by_direction["maximum"].update(blocked_reactions.forward_blocked)
            blocked_by_direction["minimum"].update(blocked_reactions.reverse_blocked)

            model.objective.set_linear_coefficients(remove_coef)

        return BlockedReactionsResult(
            forward_blocked=list(
                candidates_by_direction["maximum"] | blocked_by_direction["maximum"]
            ),
            reverse_blocked=list(
                candidates_by_direction["minimum"] | blocked_by_direction["minimum"]
            ),
        )


def _find_blocked_reactions_loopless(
    model: "Model",
    reaction_list: List[Tuple[str, str]],
    loopless: str,
    zero_cutoff: float,
    processes: Optional[int],
    cyclic_reactions: List[str],
) -> "BlockedReactionsResult":
    """Find blocked reactions like :func:`find_blocked_reactions`.

    This helper accepts precomputed ``cyclic_reactions`` to avoid calculating
    them twice.
    """
    with model:
        add_loopless(
            model,
            zero_cutoff,
            method=loopless,
            reactions=cyclic_reactions,
        )

        flux_span = flux_variability_analysis(
            model,
            reaction_list=reaction_list,
            fraction_of_optimum=None,
            processes=processes,
        )

        forward_blocked = flux_span[
            flux_span["maximum"].notna() & (flux_span["maximum"] < zero_cutoff)
        ].index.tolist()
        reverse_blocked = flux_span[
            flux_span["minimum"].notna() & (flux_span["minimum"] > -zero_cutoff)
        ].index.tolist()

        return BlockedReactionsResult(forward_blocked, reverse_blocked)


def find_blocked_reactions_loopless(
    model: "Model",
    reaction_list: Optional[List[Union["Reaction", str]]] = None,
    loopless: str = "potentials",
    zero_cutoff: Optional[float] = None,
    open_exchanges: bool = False,
    processes: Optional[int] = None,
    flux_threshold: float = 1e-2,
) -> List["Reaction"]:
    """Find reactions that cannot carry flux in loopless flux distributions.

    This is a much faster alternative to calling
    :func:`find_blocked_reactions` with a loopless method. It first identifies
    reactions that can participate in cycles and then applies loopless
    constraints only where they can affect the blocked-reaction result.

    The question whether or not a reaction is blocked is highly dependent
    on the current exchange reaction settings for a COBRA model. Hence an
    argument is provided to open all exchange reactions.

    Parameters
    ----------
    model : cobra.Model
        The model to analyze.
    reaction_list : list of cobra.Reaction or str, optional
        List of reactions to consider, the default includes all model
        reactions (default None).
    loopless : str, "potentials" or "fastSNP"
        The loopless formulation passed to :func:`add_loopless`. The default
        uses metabolite potential variables (default "potentials").
    zero_cutoff : float, optional
        Flux value which is considered to effectively be zero. The default
        is set to use `model.tolerance` (default None).
    open_exchanges : bool, optional
        Whether or not to open all exchange reactions to very high flux
        ranges (default False).
    processes : int, optional
        The number of parallel processes to run. Can speed up the
        computations if the number of reactions is large. If not explicitly
        passed, it will be set from the global configuration singleton
        (default None).
    flux_threshold : float, optional
        Minimum flux required when directional loopless indicator variables
        are active. This is used to find blocked candidates efficiently before
        validating them with regular loopless FVA and ``zero_cutoff``
        (default 1e-2).

    Returns
    -------
    BlockedReactionsResult
        A list of fully blocked reaction identifiers. Also has a
        ``forward_blocked`` attribute with reactions that cannot carry
        positive flux and a ``reverse_blocked`` attribute with reactions
        that cannot carry negative flux.

    Notes
    -----
    Sink and demand reactions are left untouched. Please modify them manually.

    """
    if loopless not in ("potentials", "fastSNP"):
        raise ValueError(
            "The `loopless` argument must be either 'potentials' or 'fastSNP'."
        )

    zero_cutoff = normalize_cutoff(model, zero_cutoff)

    with model:
        max_bound = max(
            1000.0, max(max(abs(b) for b in r.bounds) for r in model.reactions)
        )

        if open_exchanges:
            for reaction in model.exchanges:
                reaction.bounds = (
                    min(reaction.lower_bound, -max_bound),
                    max(reaction.upper_bound, max_bound),
                )

        if reaction_list is None:
            reaction_list = model.reactions

        reactions_to_check, cyclic_reactions = _prepare_cyclic_reactions_for_blocked(
            model=model,
            reaction_list=reaction_list,
            zero_cutoff=zero_cutoff,
        )

        blocked_with_flux_threshold = _find_blocked_reactions_loopless_directional(
            model=model,
            reaction_list=reactions_to_check,
            loopless=loopless,
            zero_cutoff=zero_cutoff,
            flux_threshold=flux_threshold,
            cyclic_reactions=cyclic_reactions,
            processes=processes,
        )

        logger.info(
            "Found candidates for blocked reactions among cyclic reactions "
            "with loopless constraints: "
            f"{len(blocked_with_flux_threshold.forward_blocked)} (forward), "
            f"{len(blocked_with_flux_threshold.reverse_blocked)} (reverse)."
        )

        reactions_to_check = [
            (r_id, "maximum") for r_id in blocked_with_flux_threshold.forward_blocked
        ] + [(r_id, "minimum") for r_id in blocked_with_flux_threshold.reverse_blocked]

        blocked_with_loopless = _find_blocked_reactions_loopless(
            model=model,
            reaction_list=reactions_to_check,
            loopless=loopless,
            zero_cutoff=zero_cutoff,
            processes=processes,
            cyclic_reactions=cyclic_reactions,
        )

        logger.info(
            "Found blocked reactions among cyclic reactions with loopless constraints: "
            f"{len(blocked_with_loopless.forward_blocked)} (forward), "
            f"{len(blocked_with_loopless.reverse_blocked)} (reverse)."
        )

        blocked_without_constraints = find_blocked_reactions(
            model=model,
            reaction_list=reaction_list,
            zero_cutoff=zero_cutoff,
            open_exchanges=open_exchanges,
            processes=processes,
        )

        return BlockedReactionsResult(
            forward_blocked=blocked_without_constraints.forward_blocked
            + blocked_with_loopless.forward_blocked,
            reverse_blocked=blocked_without_constraints.reverse_blocked
            + blocked_with_loopless.reverse_blocked,
        )
