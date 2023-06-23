# def solve_diagonal(lower, diag, upper, b):
#     """Solve matrix Ax=b when A is diagonal"""
#     from scipy.sparse import linalg, diags
#     N = len(diag)
#     A = diags(
#         diagonals=[lower, diag, upper],
#         offsets=[-1, 0, 1],
#         shape=(N, N),
#         format='csr'
#     )
#     return linalg.spsolve(A, b)

import jax
from jax.experimental import sparse
from jax.scipy.sparse import linalg
import jax.numpy as jnp

def jax_matrix(lower, diag, upper):
    Matrix = jnp.zeros(len(diag) ** 2).reshape(len(diag), len(diag))
    rows, cols = jnp.diag_indices(len(diag))
    LR = rows[1:]
    LC = cols[:-1]
    UR = rows[:-1]
    UC = cols[1:]
    Matrix = Matrix.at[LR, LC].set(lower)
    Matrix = Matrix.at[UR, UC].set(upper)
    Matrix = Matrix.at[rows, cols].set(diag)
    Matrix_sp = sparse.BCOO.fromdense(Matrix)
    return Matrix_sp

def solve_diagonal(lower, diag, upper, b):
    return jax.scipy.sparse.linalg.cg(jax_matrix(lower,diag,upper),b)