# Release notes for cobrapy x.y.z

## New features

- Added `chrr` sampler for flux polytope sampling, which is guaranteed to yield uniformly distributed samples. Uses the optional dependency `hopsy`, installable with `pip install cobra[chrr]`.

## Fixes
- Rare race condition in cache directory creation from running seperate processes loading cobrapy on clean machine fixed. (https://github.com/opencobra/cobrapy/issues/1476)


## Other

## Deprecated features


## Backwards incompatible changes
