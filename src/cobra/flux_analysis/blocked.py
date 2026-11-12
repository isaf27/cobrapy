"""Provide a function to find blocked reactions in a model."""

import logging
from typing import TYPE_CHECKING, List, Optional, Union

from ..core import Configuration, get_solution
from .helpers import normalize_cutoff
from .variability import flux_variability_analysis


if TYPE_CHECKING:
    from pandas import DataFrame

    from cobra import Model, Reaction


logger = logging.getLogger(__name__)
configuration = Configuration()


class BlockedReactionsResult(List[str]):
    """List of the identifiers of reactions that are blocked in both directions.

    Attributes
    ----------
    forward_blocked : list of str
        Identifiers of reactions that cannot carry positive flux.
    reverse_blocked : list of str
        Identifiers of reactions that cannot carry negative flux.
    """

    def __init__(self, forward_blocked: List[str], reverse_blocked: List[str]) -> None:
        """Initialize the list from the reactions blocked in each direction."""
        reverse_blocked_set = set(reverse_blocked)
        super().__init__(
            rxn_id for rxn_id in forward_blocked if rxn_id in reverse_blocked_set
        )
        self.forward_blocked = forward_blocked
        self.reverse_blocked = reverse_blocked


def _warn_near_zero_cutoff_fluxes(flux_span: "DataFrame", zero_cutoff: float) -> None:
    """Warn if FVA results are close to the blocked reaction cutoff."""
    lower_bound = 0.1 * zero_cutoff
    upper_bound = 10 * zero_cutoff

    closest_below = None
    closest_above = None
    for direction in ("minimum", "maximum"):
        for reaction_id, flux in flux_span[direction].dropna().items():
            dir_flux = flux * (1 if direction == "maximum" else -1)
            if dir_flux < lower_bound or dir_flux > upper_bound:
                continue
            flux_info = (dir_flux, reaction_id, direction, flux)
            if dir_flux < zero_cutoff:
                if closest_below is None or dir_flux > closest_below[0]:
                    closest_below = flux_info
            elif closest_above is None or dir_flux < closest_above[0]:
                closest_above = flux_info

    if closest_below is None and closest_above is None:
        return

    formatted_fluxes = []
    for label, flux_info in (("below", closest_below), ("above", closest_above)):
        if flux_info is None:
            continue
        _, reaction_id, direction, flux = flux_info
        formatted_fluxes.append(
            f"closest {label}: {reaction_id} ({direction}={flux:.3g})"
        )

    fluxes = ", ".join(formatted_fluxes)
    logger.warning(
        "Some reactions have flux bounds close to "
        f"zero_cutoff={zero_cutoff}, which may make "
        f"the blocked reaction result sensitive to numerical tolerance: {fluxes}.",
    )


def find_blocked_reactions(
    model: "Model",
    reaction_list: Optional[List[Union["Reaction", str]]] = None,
    loopless: Optional[str] = None,
    zero_cutoff: Optional[float] = None,
    open_exchanges: bool = False,
    abs_flux_clip: Optional[float] = 0.1,
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
        If set, a reaction also counts as blocked if it can only carry flux
        in a loop. The value selects the loopless method of
        :func:`flux_variability_analysis` (default None).
    zero_cutoff : float, optional
        Flux value which is considered to effectively be zero. The default
        is set to use `model.tolerance` (default None).
    open_exchanges : bool, optional
        Whether or not to open all exchange reactions to very high flux
        ranges (default False).
    abs_flux_clip : float, optional
        Fluxes are only optimized up to this absolute value, which must not
        be smaller than `zero_cutoff`, see
        :func:`flux_variability_analysis` (default 0.1).
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

    The result can be highly sensitive to ``zero_cutoff``. When exploring an
    appropriate cutoff, call :func:`flux_variability_analysis` directly once
    with the desired ``abs_flux_clip`` and analyze the returned flux bounds
    later.

    """
    if loopless not in (None, "potentials", "fastSNP"):
        raise ValueError(
            "The `loopless` argument must be either None, 'potentials', "
            f"or 'fastSNP', not {loopless!r}."
        )

    zero_cutoff = normalize_cutoff(model, zero_cutoff)

    if abs_flux_clip is not None and abs_flux_clip < zero_cutoff:
        raise ValueError(
            "The `abs_flux_clip` argument must not be smaller than `zero_cutoff`."
        )

    with model:
        if open_exchanges:
            max_bound = max(
                1000.0, max(max(abs(b) for b in r.bounds) for r in model.reactions)
            )
            for reaction in model.exchanges:
                reaction.bounds = (
                    min(reaction.lower_bound, -max_bound),
                    max(reaction.upper_bound, max_bound),
                )

        if reaction_list is None:
            reaction_list = model.reactions
        else:
            reaction_list = model.reactions.get_by_any(reaction_list)

        model.slim_optimize(
            error_value=None,
            message="There is no optimal solution for the chosen objective!",
        )
        if loopless is None:
            # A direction that clearly carries flux in this solution is not
            # blocked, so only the other directions need to be checked. With
            # `loopless`, the flux could be part of a loop and proves nothing.
            solution = get_solution(model, reactions=reaction_list)
            clear_flux = zero_cutoff if abs_flux_clip is None else 10 * abs_flux_clip
            forward_unknown = solution.fluxes[
                solution.fluxes < clear_flux
            ].index.tolist()
            reverse_unknown = solution.fluxes[
                solution.fluxes > -clear_flux
            ].index.tolist()
            reaction_list = [
                (reaction_id, "maximum") for reaction_id in forward_unknown
            ] + [(reaction_id, "minimum") for reaction_id in reverse_unknown]

        flux_span = flux_variability_analysis(
            model,
            reaction_list=reaction_list,
            loopless=loopless,
            fraction_of_optimum=None,
            abs_flux_clip=abs_flux_clip,
            processes=processes,
        )
        _warn_near_zero_cutoff_fluxes(flux_span, zero_cutoff)

        # A direction is blocked if its flux bound is below the cutoff. Bounds
        # that were not computed are NaN and compare as False.
        forward_blocked = flux_span.index[flux_span["maximum"] < zero_cutoff].tolist()
        reverse_blocked = flux_span.index[flux_span["minimum"] > -zero_cutoff].tolist()

        return BlockedReactionsResult(forward_blocked, reverse_blocked)
