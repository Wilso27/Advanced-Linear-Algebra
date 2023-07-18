# qr_decomposition.py


import numpy as np
from scipy import linalg as la


def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m,n = A.shape
    Q = np.copy(A).astype(float)
    R = np.zeros((n,n)).astype(float)
    #loop through the psuedo code given in the lab
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]
        #loop through each column
        for j in range(i+1,n):
            R[i,j] = Q[:,j].T@Q[:,i]
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
    return Q,R


def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    #calculate the determinant of the triangular matrix
    return abs(np.prod(np.diag(la.qr(A)[1])))
    

def back_solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    P,L,U = la.lu(A)
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


def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    #do qr decomp
    Q,R = la.qr(A)
    y = Q.T@b
    #put it through back_solve
    return back_solve(R,y)


def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    #create Q and R matrices
    m,n = A.shape
    R = np.copy(A).astype(float)
    Q = np.identity(m)
    Q = Q.astype(float)
    #loop through each row
    for k in range(n):
        u = np.copy(R[k:,k]).astype(float)
        u[0] = u[0] + np.sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - 2*(np.outer(u,(u@R[k:,k:])))
        Q[k:,:] = Q[k:,:] - 2*np.outer(u,(u@Q[k:,:]))
    return Q.T,R


def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    m,n = A.shape
    H = np.copy(A).astype(float)
    Q = np.identity(m).astype(float)
    #loop through n-2 rows using hessenberg psuedo code
    for k in range(n-2):
        u = np.copy(H[k+1:,k]).astype(float)
        u[0] = u[0] + np.sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        #use array splicing to perform the operations
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,(u@H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2*np.outer((H[:,k+1:]@u),u)
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u,(u@Q[k+1:,:]))
    return H,Q.T
