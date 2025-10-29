"""Test finding nullspace basis of matrix using Fast-SNP algorithm."""

from typing import List

import numpy as np

from cobra import Model
from cobra.flux_analysis.fast_snp import nullspace_fast_snp
from cobra.util import create_stoichiometric_matrix
from cobra.util import solver as sutil


def _validate_nullspace_basis(
    S: np.ndarray, directions: np.ndarray, basis: np.ndarray
) -> None:
    """Validate that the basis is a correct nullspace basis with given directions."""

    assert basis.shape[0] == S.shape[1]
    if basis.shape[1] == 0:
        return

    max_error = np.max(np.abs(S @ basis))
    assert max_error < 1e-6, "S @ basis is not zero"

    for v in basis.T:
        for i in range(len(v)):
            assert (
                directions[i, 0] == -1 or v[i] >= -1e-6
            ), f"Basis vector violates direction for index {i}"
            assert (
                directions[i, 1] == 1 or v[i] <= 1e-6
            ), f"Basis vector violates upper bound for index {i}"


def test_small_without_directions(all_solvers: List[str]) -> None:
    """Test nullspace_fast_snp on small example without directions."""

    S = np.array(
        [
            [1, -1, 0, 4, 2],
            [0, -1, 0, -1, -4],
            [3, 2, 1, 0, 2],
        ]
    )
    directions = np.array([[-1, 1]] * 5)

    basis = nullspace_fast_snp(sutil.solvers[all_solvers], S, directions)

    assert basis.shape == (5, 2)
    _validate_nullspace_basis(S, directions, basis)


def test_small_with_directions(all_solvers: List[str]) -> None:
    """Test nullspace_fast_snp on small example with directions."""

    S = np.array(
        [
            [1, -1, 0, 4, 2],
            [0, -1, 0, -1, -4],
            [3, 2, 1, 0, 2],
        ]
    )
    directions = np.array(
        [
            [0, 1],
            [0, 0],
            [-1, 0],
            [-1, 0],
            [-1, 1],
        ]
    )

    basis = nullspace_fast_snp(sutil.solvers[all_solvers], S, directions)

    assert basis.shape == (5, 1)
    _validate_nullspace_basis(S, directions, basis)


def test_random(all_solvers: List[str]) -> None:
    """Test nullspace_fast_snp on random matrices."""

    np.random.seed(42)
    for _ in range(20):
        m = np.random.randint(10, 15)
        n = np.random.randint(m + 1, m + 10)
        S = np.random.normal(size=(m, n))
        directions = np.array([[-1, 1]] * n)

        basis = nullspace_fast_snp(sutil.solvers[all_solvers], S, directions)
        _validate_nullspace_basis(S, directions, basis)


def test_stoichiometry_matrix(model: Model, all_solvers: List[str]) -> None:
    """Test nullspace_fast_snp on model's stoichiometric matrix and directions."""

    internal_reactions = [i for i, r in enumerate(model.reactions) if not r.boundary]
    S_int = create_stoichiometric_matrix(model)[:, np.array(internal_reactions)]

    no_directions = np.array([[-1, 1]] * S_int.shape[1])
    basis_without_directions = nullspace_fast_snp(
        sutil.solvers[all_solvers], S_int, no_directions
    )
    _validate_nullspace_basis(S_int, no_directions, basis_without_directions)

    directions = np.sign(
        np.array([model.reactions[i].bounds for i in internal_reactions])
    )
    basis_with_directions = nullspace_fast_snp(
        sutil.solvers[all_solvers], S_int, directions
    )
    _validate_nullspace_basis(S_int, directions, basis_with_directions)

    assert basis_with_directions.shape[1] <= basis_without_directions.shape[1]
