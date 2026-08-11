# Release notes for cobrapy x.y.z

## New features

- `add_loopless` gained `method="potentials"`, which encodes the loop law
  through free metabolite potential variables (G = S_intᵀμ) instead of a
  computed null-space basis. Any internal cycle `n` satisfies `S_int n = 0`,
  so `nᵀG = 0` holds identically and no basis computation is needed; the
  encoded feasible flux space is identical to the existing methods. On
  genome-scale models the null-space step dominates construction (~98% of a
  17-minute build on Recon2), which `"potentials"` skips, building the same
  constraints in seconds. `flux_variability_analysis` accepts
  `loopless="potentials"` accordingly.
- `add_loopless` gained `flux_threshold`, which switches to a directional
  formulation with separate forward and reverse indicator variables, letting
  downstream analyses determine whether positive or negative loopless flux is
  feasible.
- `find_blocked_reactions` moved to the new `cobra.flux_analysis.blocked`
  module, returns a list-compatible result with direction-specific
  `forward_blocked` and `reverse_blocked` attributes, and supports loopless
  blocked-reaction detection with `loopless="potentials"` or
  `loopless="fastSNP"`.
- `flux_variability_analysis` gained `fraction_of_optimum=None` (disables the
  constraint on the original objective) and `abs_flux_clip` (clipped flux
  variability analysis, which is more efficient because bounds beyond the
  clipping value need not be proven).
- New helper `cobra.manipulation.clip_reaction_bounds` clips all reaction
  bounds to a finite absolute value before running numerically sensitive MILP
  analyses.

## Fixes

## Other

## Deprecated features

## Backwards incompatible changes
