# linear_transformations.py


from random import random
import numpy as np
from matplotlib import pyplot as plt
import time

# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    C = A.copy()
    B = np.array([[a,0],[0,b]])
    #apply the transformation
    for i in range(C.shape[1]):
        C[:,i] = B@C[:,i]        
    return C


def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    C = A.copy()
    B = np.array([[1,a],[b,1]])
    #apply the transformation
    for i in range(C.shape[1]):
        C[:,i] = B@C[:,i]        
    return C


def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    C = A.copy()
    B = np.array([[a**2-b**2,2*a*b],[2*a*b,b**2-a**2]])/(a**2+b**2)
    #apply the transformation
    for i in range(C.shape[1]):
        C[:,i] = B@C[:,i]        
    return C


def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    C = A.copy()
    #apply the transformation
    B = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])   
    return B@C


def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    #x values
    tlist = np.linspace(0,T,100)
    E = []
    M = []
    #create arrays for positions
    pe0 = np.array([x_e,0])
    pm0 = np.array([x_m,0]) - pe0
    #go through each x value and create a list
    for t in tlist:
        E.append(rotate(pe0,t*omega_e))
        M.append(rotate(pm0,t*omega_m))
    #make y values arrays
    EA = np.array(E)
    MA = np.array(M)
    #add them together for the moon position with respect to earth
    B = EA + MA
    #plot everything
    plt.gca().set_aspect("equal")
    plt.title('Solar System')
    plt.plot(EA[:,0],EA[:,1])
    plt.plot(B[:,0],B[:,1])

    plt.show()


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]


def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]


def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]


def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]


def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    T1 = []
    T2 = []
    L = []
    #Loop through a list of inputs
    for i in [1,2,4,8,16,32,64,128,256]:
        #generate matrices and vector
        A = random_matrix(i)
        B = random_matrix(i)
        x = random_vector(i)
        L.append(i)
        #time each function
        start1 = time.perf_counter()
        matrix_vector_product(A,x)
        t1 = time.perf_counter() - start1
        start2 = time.perf_counter()
        matrix_matrix_product(A,B)
        t2 = time.perf_counter() - start2
        T1.append(t1)
        T2.append(t2)
    ax1 = plt.subplot(121) # Plot curve.
    ax1.plot(L, T1, 'b.-', lw=2, ms=15, label="Matrix-Vector")
    plt.title("Matrix-Vector")
    ax2 = plt.subplot(122) # Plot curve.
    ax2.plot(L, T2, 'g.-', lw=2, ms=15, label="Matrix-Matrix")
    plt.title("Matrix-Matrix")
    plt.show()


def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    T1 = []
    T2 = []
    N1 = []
    N2 = []
    L = []
    #loop through input list
    for i in [1,2,4,8,16,32,64,128,256]:
        #generate matrix/vector
        A = random_matrix(i)
        B = random_matrix(i)
        x = random_vector(i)
        #create domain list
        L.append(i)
        #time each function
        start1 = time.perf_counter()
        matrix_vector_product(A,x)
        t1 = time.perf_counter() - start1
        start2 = time.perf_counter()
        matrix_matrix_product(A,B)
        t2 = time.perf_counter() - start2
        start3 = time.perf_counter()
        np.dot(A,x)
        n1 = time.perf_counter() - start3
        start4 = time.perf_counter()
        np.dot(A,B)
        n2 = time.perf_counter() - start4
        #create output list
        T1.append(t1)
        T2.append(t2)
        N1.append(n1)
        N2.append(n2)
    ax1 = plt.subplot(121) # Plot all curves on a regular lin-lin plot.
    ax1.plot(L, T1, 'b.-', lw=2, ms=15, label="Matrix-Vector")
    ax1.plot(L, T2, 'g.-', lw=2, ms=15, label="Matrix-Matrix")
    ax1.plot(L, N1, 'r.-', lw=2, ms=15, label="Numpy Matrix-Vector")
    ax1.plot(L, N2, 'c.-', lw=2, ms=15, label="Numpy Matrix-Matrix")
    ax1.legend(loc="upper left")
    ax2 = plt.subplot(122) # Plot all curves on a base 2 log-log plot.
    ax2.loglog(L, T1, 'b.-', base=2, lw=2)
    ax2.loglog(L, T2, 'g.-', base=2, lw=2)
    ax2.loglog(L, N1, 'r.-', base=2, lw=2)
    ax2.loglog(L, N2, 'c.-', base=2, lw=2)
    plt.show()


def testhorse():
    #test horse.npy
    data = np.load('horse.npy')
    A = rotate(data,np.pi/2)
    plt.plot(A[0], A[1], 'y,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.show()
