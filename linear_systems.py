# linear_systems.py

import numpy as np
from scipy import linalg as la
from scipy import sparse
import time
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla


def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    B = A.copy()
    pivot_row = 0
    #loop through each column looking for the first nonzero entry
    for j in range(B.shape[1]):
        for i in range(B.shape[0]):
            if pivot_row == j and B[i][j] != 0 and i >= j:
                temp1 = B[j].copy()
                temp2 = B[i].copy()
                B[j] = temp2
                B[i] = temp1
                pivot_row += 1
            #reduce all terms below the pivot
            elif i > j and B[i][j] != 0:
                s = B[i][j]/B[j][j]
                B[i] = B[i] - s*B[j]
    return B


def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    #get the shapes
    m,n = A.shape[0],A.shape[1]
    U = A.copy()
    U = U.astype(float)
    L = np.eye(m)
    #loop through using formula in lab specs
    for j in range(n):
        for i in range(j+1,m):
            L[i][j] = U[i][j]/U[j][j]
            U[i][j:] = U[i][j:] - (L[i][j]*U[j][j:])
    return L,U



def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    L,U = lu(A)
    
    y = []
    y.append(b[0])
    n = len(b)
    #solve for y
    #loop through using formula (2.1)
    for k in range(1,n):
        y.append(b[k] - np.dot(L[k][:k],y))

    #solve for x
    x = []
    x.append(y[n-1]/U[n-1][n-1])
    #loop through using formula (2.2)
    for i in range(1,n):
        x.append((y[n-i-1] - np.dot(U[n-i-1][n-i:],x[::-1]))/(U[n-i-1][n-i-1]))
    return x[::-1]


def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    #make lists
    T1 = []
    T2 = []
    N1 = []
    N2 = []
    L = []
    #loop through input list
    for n in range(1,251):
        #generate matrix/vector
        A = np.random.random((n,n))
        b = np.random.random(n)
        #create domain list
        L.append(n)
        #time each function
        start1 = time.perf_counter()
        la.inv(A)@b
        t1 = time.perf_counter() - start1
        start2 = time.perf_counter()
        la.solve(A,b)
        t2 = time.perf_counter() - start2
        start3 = time.perf_counter()
        la.lu_solve(la.lu_factor(A),b)
        n1 = time.perf_counter() - start3
        B = la.lu_factor(A)
        start4 = time.perf_counter()
        la.lu_solve(B,b)
        n2 = time.perf_counter() - start4
        #create output list
        T1.append(t1)
        T2.append(t2)
        N1.append(n1)
        N2.append(n2)
    plt.title('Timing the functions')
    plt.plot() # Plot all curves on a base 2 log-log plot.
    plt.loglog(L, T1, 'b.-', base=2, lw=2)
    plt.loglog(L, T2, 'g.-', base=2, lw=2)
    plt.loglog(L, N1, 'r.-', base=2, lw=2)
    plt.loglog(L, N2, 'c.-', base=2, lw=2)
    plt.legend(['inv','solve','lu_factor','lu_solve'])
    plt.show()


def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    #create B using diagonals
    diagonals = [[1 for i in range(n-1)]]
    diagonals.append([-4 for j in range(n)])
    diagonals.append([1 for k in range(n-1)])
    offsets = [-1,0,1]
    B = sparse.diags(diagonals, offsets, shape=(n,n))
    #add them to a sparse matrix
    A = sparse.block_diag((B for i in range(n)))
    A.setdiag([1 for m in range(n**2-n)],-n)
    A.setdiag([1 for m in range(n**2-n)],n)
    return A


def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """

    #make lists
    T1 = []
    T2 = []
    L = []
    #loop through input list
    for n in [2**(k+1) for k in range(5)]:
        #generate matrix/vector
        A = prob5(n)
        b = np.random.random(n**2)
        #create domain list
        L.append(n)
        #time each function
        start1 = time.perf_counter()
        spla.spsolve(A.tocsr(),b)
        t1 = time.perf_counter() - start1
        start2 = time.perf_counter()
        la.solve(A.toarray(),b)
        t2 = time.perf_counter() - start2
        #create output list
        T1.append(t1)
        T2.append(t2)
        
    plt.title('Timing the functions')
    plt.plot() # Plot all curves on a base 2 log-log plot.
    plt.loglog(L, T1, 'b.-', base=2, lw=2)
    plt.loglog(L, T2, 'g.-', base=2, lw=2)
    plt.legend(['spsolve','solve'])
    plt.show()
