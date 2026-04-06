"""TODO"""

import logging
from typing import TYPE_CHECKING, List, Optional, Union, Tuple
from warnings import warn

from optlang.symbolics import Zero

from ..core import Configuration, get_solution
from ..util import solver as sutil
from .helpers import normalize_cutoff
from .variability import flux_variability_analysis
from .find_cyclic_reactions import find_cyclic_reactions
from .loopless import add_loopless


if TYPE_CHECKING:
    from cobra import Model, Reaction


logger = logging.getLogger(__name__)
configuration = Configuration()


class BlockedReactionsResult(list[str]):
    """Result of blocked reactions analysis.

    Behaves as a list of fully blocked reactions for backward compatibility,
    but also carries partially blocked reactions as an attribute.

    Attributes
    ----------
    forward_blocked : list of str
        Reaction identifiers that can't carry flux in the forward direction.
    reverse_blocked: list of str
        Reaction identifiers that can't carry flux in the reverse direction.
    """

    def __init__(self, forward_blocked, reverse_blocked):
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
        ``partially_blocked`` attribute with reactions that can only
        carry flux in one direction.

    Notes
    -----
    Sink and demand reactions are left untouched. Please modify them manually.

    """
    zero_cutoff = normalize_cutoff(model, zero_cutoff)

    with model:
        max_bound = max(1000.0, max(max(abs(b) for b in r.bounds) for r in model.reactions))

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
            forward_unknown = solution.fluxes[solution.fluxes < zero_cutoff].index.tolist()
            reverse_unknown = solution.fluxes[solution.fluxes > -zero_cutoff].index.tolist()
            reaction_list = [(reaction_id, "maximum") for reaction_id in forward_unknown] + \
                [(reaction_id, "minimum") for reaction_id in reverse_unknown]

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


def _build_reactions_to_check_with_loopless(
    model: "Model",
    reaction_list: List[Union["Reaction", str]],
    zero_cutoff: float,
) -> Tuple[List[Tuple[str, str]], List[str]]:
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
            for dir in ["rev", "fwd"]:
                if cyclic_directions[c_idx][0 if dir == "rev" else 1]:
                    reactions_to_check.append((r_id, dir))

    return reactions_to_check, cyclic_reactions


def _find_blocked_reactions_loopless_directional(
    model: "Model",
    reaction_list: List[Tuple[str, str]],
    loopless: str,
    zero_cutoff: float,
    flux_threshold: float,
    cyclic_reactions: List[str],
) -> "BlockedReactionsResult":
    with model:
        logger.info(
            f"search with flux_threshold: {len(reaction_list)} reactions to check."
        )

        add_loopless(
            model,
            zero_cutoff,
            method=loopless,
            reactions=cyclic_reactions,
            flux_threshold=flux_threshold,
        )

        # TODO: add LP-based check using found directions

        logger.info(
            f"loopless constraints added for {len(cyclic_reactions)} cyclic reactions."
        )

        blocked_by_direction = {
            "fwd": set(),
            "rev": set(),
        }
        for rid, dir in reaction_list:
            blocked_by_direction[dir].add(rid)

        model.objective = Zero
        coefs = {
            model.variables[f"indicator_{dir}_{rid}"]: 1
            for rid, dir in reaction_list
        }
        model.objective.set_linear_coefficients(coefs)
        model.objective.direction = "max"

        is_nonzero_num = 0
        while is_nonzero_num < len(reaction_list):
            model.slim_optimize()
            sutil.check_solver_status(model.solver.status)

            remove_coef = {}
            for rid, dir in reaction_list:
                if rid in blocked_by_direction[dir]:
                    rxn = model.reactions.get_by_id(rid)
                    if (dir == "fwd" and rxn.forward_variable.primal >= flux_threshold * 0.999) or \
                       (dir == "rev" and rxn.reverse_variable.primal >= flux_threshold * 0.999):
                        blocked_by_direction[dir].remove(rid)
                        remove_coef[model.variables[f"indicator_{dir}_{rid}"]] = 0

            is_nonzero_num += len(remove_coef)

            if len(remove_coef) == 0:
                logger.info(
                    f"search with flux_threshold: "
                    f"{len(reaction_list) - is_nonzero_num} are blocked."
                )
                break

            model.objective.set_linear_coefficients(remove_coef)
            
            logger.info(
                f"search with flux_threshold: found {len(remove_coef)} non-blocked reactions, "
                f"{len(reaction_list) - is_nonzero_num} remain to check."
            )

        return BlockedReactionsResult(
            forward_blocked=list(blocked_by_direction["fwd"]),
            reverse_blocked=list(blocked_by_direction["rev"]),
        )


def _find_blocked_reactions_loopless(
    model: "Model",
    reaction_list: List[Tuple[str, str]],
    loopless: str,
    zero_cutoff: float,
    processes: Optional[int],
    cyclic_reactions: List[str],
) -> "BlockedReactionsResult":
    with model:
        logger.info(
            f"FVA-based search: {len(reaction_list)} reactions to check."
        )

        add_loopless(
            model,
            zero_cutoff,
            method=loopless,
            reactions=cyclic_reactions,
        )
        
        flux_reactions_list = [
            (r_id, "max" if dir == "fwd" else "min")
            for r_id, dir in reaction_list
        ]

        flux_span = flux_variability_analysis(
            model,
            reaction_list=flux_reactions_list,
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
    loopless: str = 'potentials',
    zero_cutoff: Optional[float] = None,
    open_exchanges: bool = False,
    processes: Optional[int] = None,
    flux_threshold: float = 1e-2,
) -> List["Reaction"]:
    zero_cutoff = normalize_cutoff(model, zero_cutoff)

    with model:
        max_bound = max(1000.0, max(max(abs(b) for b in r.bounds) for r in model.reactions))

        if open_exchanges:
            for reaction in model.exchanges:
                reaction.bounds = (
                    min(reaction.lower_bound, -max_bound),
                    max(reaction.upper_bound, max_bound),
                )

        if reaction_list is None:
            reaction_list = model.reactions

        reactions_to_check, cyclic_reactions = _build_reactions_to_check_with_loopless(
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
        )

        logger.info(
            f"blocked among cyclic via directional: {len(blocked_with_flux_threshold.forward_blocked)}, {len(blocked_with_flux_threshold.reverse_blocked)}"
        )

        reactions_to_check = [(r_id, "fwd") for r_id in blocked_with_flux_threshold.forward_blocked] + \
            [(r_id, "rev") for r_id in blocked_with_flux_threshold.reverse_blocked]

        blocked_with_loopless = _find_blocked_reactions_loopless(
            model=model,
            reaction_list=reactions_to_check,
            loopless=loopless,
            zero_cutoff=zero_cutoff,
            processes=processes,
            cyclic_reactions=cyclic_reactions,
        )

        logger.info(
            f"blocked among cyclic: {len(blocked_with_loopless.forward_blocked)}, {len(blocked_with_loopless.reverse_blocked)}"
        )
        
        blocked_without_constraints = find_blocked_reactions(
            model=model,
            reaction_list=reaction_list,
            zero_cutoff=zero_cutoff,
            open_exchanges=open_exchanges,
            processes=processes,
        )

        return BlockedReactionsResult(
            forward_blocked=blocked_without_constraints.forward_blocked + blocked_with_loopless.forward_blocked,
            reverse_blocked=blocked_without_constraints.reverse_blocked + blocked_with_loopless.reverse_blocked
        )
