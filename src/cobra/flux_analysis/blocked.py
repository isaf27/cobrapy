"""TODO"""

import logging
from typing import TYPE_CHECKING, List, Optional, Union, Tuple
from warnings import warn

from optlang.symbolics import Zero

from ..core import Configuration, get_solution
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
        if loopless is None:
            # Limit the search space to reactions which have zero flux. If the
            # reactions already carry flux in this solution,
            # then they cannot be blocked.
            model.slim_optimize()
            solution = get_solution(model, reactions=reaction_list)
            reaction_list = solution.fluxes[
                solution.fluxes.abs() < zero_cutoff
            ].index.tolist()
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
            flux_span["maximum"] < zero_cutoff
        ].index.tolist()
        reverse_blocked = flux_span[
            flux_span["minimum"] > -zero_cutoff
        ].index.tolist()
        return BlockedReactionsResult(forward_blocked, reverse_blocked)


def find_blocked_reactions_loopless_fast(
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

        blocked = find_blocked_reactions(
            model,
            reaction_list=reaction_list,
            zero_cutoff=zero_cutoff,
            open_exchanges=open_exchanges,
            processes=processes,
        )

        if reaction_list is None:
            reaction_list = model.reactions

        reaction_to_index = {r.id: i for i, r in enumerate(reaction_list)}
        is_nonzero_fwd = [True] * len(reaction_list)
        is_nonzero_rev = [True] * len(reaction_list)
        for rid in blocked.forward_blocked:
            is_nonzero_fwd[reaction_to_index[rid]] = False
        for rid in blocked.reverse_blocked:
            is_nonzero_rev[reaction_to_index[rid]] = False

        reactions_to_check = []
        cyclic_reactions, cyclic_directions = find_cyclic_reactions(model)
        cyclic_reaction_index = {r_id: i for i, r_id in enumerate(cyclic_reactions)}
        for r in reaction_list:
            if r.id in cyclic_reaction_index:
                cid = cyclic_reaction_index[r.id]
                i = reaction_to_index[r.id]
                if cyclic_directions[cid][0] and is_nonzero_rev[i]:
                    is_nonzero_rev[i] = False
                    reactions_to_check.append((i, "rev"))
                if cyclic_directions[cid][1] and is_nonzero_fwd[i]:
                    is_nonzero_fwd[i] = False
                    reactions_to_check.append((i, "fwd"))

        with model:
            reactions_copy = [model.reactions.get_by_id(r.id) for r in reaction_list]

            add_loopless(
                model,
                zero_cutoff,
                method=loopless,
                reactions=cyclic_reactions,
                flux_threshold=flux_threshold,
            )
            
            model.objective = Zero
            coefs = {
                model.variables[f"indicator_{dir}_{reaction_list[i].id}"]: 1
                for i, dir in reactions_to_check
            }
            model.objective.set_linear_coefficients(coefs)
            model.objective.direction = "max"

            is_nonzero_num = 0
            while is_nonzero_num < len(reactions_to_check):
                model.slim_optimize()

                remove_coef = {}
                for i, dir in reactions_to_check:
                    rxn = reactions_copy[i]
                    if dir == "fwd":
                        if not is_nonzero_fwd[i] and rxn.forward_variable.primal >= flux_threshold * 0.999:
                            is_nonzero_fwd[i] = True
                            remove_coef[model.variables[f"indicator_{dir}_{rxn.id}"]] = 0
                    elif dir == "rev":
                        if not is_nonzero_rev[i] and rxn.reverse_variable.primal >= flux_threshold * 0.999:
                            is_nonzero_rev[i] = True
                            remove_coef[model.variables[f"indicator_{dir}_{rxn.id}"]] = 0

                is_nonzero_num += len(remove_coef)

                if len(remove_coef) == 0:
                    logger.info(
                        f"Finished pre-searching for blocked reactions. "
                        f"{len(reactions_to_check) - is_nonzero_num} reactions remain to check in post-searching."
                    )
                    break

                model.objective.set_linear_coefficients(remove_coef)
                
                logger.info(
                    f"Pre-searching: found {len(remove_coef)} non-blocked reactions, "
                    f"{len(reactions_to_check) - is_nonzero_num} remaining to check."
                )

        forward_blocked = [r.id for i, r in enumerate(reaction_list) if not is_nonzero_fwd[i]]
        reverse_blocked = [r.id for i, r in enumerate(reaction_list) if not is_nonzero_rev[i]]

        return BlockedReactionsResult(forward_blocked, reverse_blocked)
