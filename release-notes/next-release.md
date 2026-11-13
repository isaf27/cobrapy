# Release notes for cobrapy 0.33.0

## New features

- `spectra_cc` implements the SPECTRA consistency check, a sibling of `fastcc` for removing the blocked reactions of a model. It alternates a forward and a reverse LP, each driving as many of the still-unexplained reactions as possible towards a flux threshold at once, until that set stops shrinking. Unlike FASTCC it needs no reaction flipping: the forward LP's auxiliary variables are unbounded from below, so reactions that can only carry negative flux are simply picked up by the reverse LP instead. The two agree on the models in the test suite; on Recon3D and random sub-models of it they differ on a small number of reactions, and FVA confirms `spectra_cc`'s answer in every case. `spectra_cc` is also faster, by roughly 4-60x on the models measured, with the spread tracking how often `fastcc` falls back to its per-reaction search rather than tracking model size.
- `pfba` and `geometric_fba` gained a `raise_error` argument that is forwarded to the underlying solver-status check, so all FBA flavours can be invoked like `Model.optimize(raise_error=...)`. As with `Model.optimize`, a non-optimal status now emits a warning by default and only raises when `raise_error=True` (previously `pfba` always raised).
- `flux_variability_analysis` gained an `all_fluxes` argument. FVA solves one optimization per reaction per direction and keeps only the objective value; setting this to `True` also returns the flux distribution found at each optimum, as a dict mapping `"minimum"`/`"maximum"` to data frames indexed by the optimized reaction. This avoids re-running an FVA-sized batch of solves when the distributions themselves are wanted, for example as a starting pool for sampling. With `loopless="fastSNP"` or `loopless="potentials"` the loop constraints are applied to the whole model when this is set, so every returned distribution is loopless. The default return value is unchanged.
- `Dictlists` (`model.reactions`, `model.genes`, `model.metabolites`) are now generic containers and the types of their contents are now inferable by python type checkers.
- `add_loopless` gained `method="potentials"`, which formulates the loopless constraints with metabolite potentials instead of a nullspace basis, and a `flux_threshold` argument that adds one indicator per reaction direction. `flux_variability_analysis` accepts `loopless="potentials"`.
- `flux_variability_analysis` gained an `abs_flux_clip` argument that clips the reported bounds, which saves time in loopless FVA and, with CPLEX, in FVA without loopless constraints. It also accepts `fraction_of_optimum=None` to leave the objective unconstrained and `(reaction, direction)` tuples in `reaction_list` to compute only one bound of a reaction.
- `find_blocked_reactions` gained a `loopless` argument to find the reactions that cannot carry flux in any loopless flux distribution. It also gained an `abs_flux_clip` argument (default 0.1), up to which the flux bounds are determined. Its result still behaves as a list of the blocked reactions and additionally has the attributes `forward_blocked` and `reverse_blocked`.
- `cobra.manipulation.clip_reaction_bounds` clips all reaction bounds to a maximal absolute value.

## Fixes

- `add_loopless` now handles models and reaction selections without internal cycles, and `find_cyclic_reactions` returns empty results when there are no internal reactions.
- FASTCC now uses the feasible direction of negative-only irreversible reactions when constructing its consistency LP, preserving consistency under supported reaction reversal.
- type-annotations on resetable properties fixed.
- fixed the URL for the BIGG repository
- fixed the URL for the BioModels repository
- `write_sbml_model` no longer fails with a libSBML `TypeError` when reaction bounds or stoichiometries are numpy scalars.
- gene ids containing characters such as `@`, `+`, `*` or `#` are now parsed correctly in GPRs, so reactions using them can be copied (#1448)

## Other

- Downloads from BioModels and BIGG now get xfail markers as they are often blocked

## Deprecated features

## Backwards incompatible changes

- `find_blocked_reactions` moved from `cobra.flux_analysis.variability` to `cobra.flux_analysis.blocked`. Importing it from `cobra.flux_analysis` works as before.
- `flux_variability_analysis` with `loopless="fastSNP"` now computes the optimum for `fraction_of_optimum` and the minimal total flux for `pfba_factor` with loopless constraints, so its bounds can differ from earlier versions.
- The default `bound` of `find_cyclic_reactions` changed from 1e4 to 1e3.
- The new arguments `abs_flux_clip` of `flux_variability_analysis` and `loopless` of `find_blocked_reactions` are not the last ones, so the arguments after them (`processes`; `zero_cutoff`, `open_exchanges` and `processes`) have to be passed by keyword.
