# lstsq_eigs.py


import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la


def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    #reduced qr
    Q,R = la.qr(A,mode='economic') 
    #least squares solution
    x = la.inv(R)@Q.T@b
    return x


def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    # load the data
    data = np.load("housing.npy")
    x_val = [i[0] for i in data]
    
    # create A and b
    b = np.vstack([j[1] for j in data])
    A = np.column_stack((np.vstack(x_val),np.ones((len(data),1))))
    ab = least_squares(A,b)
    X = np.arange(0,16,1)
    
    # creat line of best fit
    Y = [ab[0]*x + ab[1] for x in X]
    plt.scatter(x_val,b)
    plt.plot(X, Y, linewidth=3)
    plt.show()


def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    data = np.load("housing.npy")
    x_val, y_val = data[:, 0], data[:, 1]  #define the x and y values
    X = np.linspace(0,16,100)
    sp = [221,222,223,224]
    a = 0
    poly_label = ['Degree 3','Degree 6','Degree 9','Degree 12']
    for i in [3,6,9,12]: #loop through each degree
        #compute the polynomial 
        poly = np.poly1d(la.lstsq(np.vander(x_val,i+1),y_val)[0])
        #plot in subplots
        plt.subplot(sp[a])
        plt.title(poly_label[a])
        plt.tight_layout()
        plt.plot(X,poly(X))
        plt.scatter(x_val,y_val)
        plt.legend([poly_label[a],'Actual'])
        a += 1
    plt.scatter(x_val,y_val)
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")


def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    data = np.load("ellipse.npy")
    x,y = data[:,0],data[:,1]
    #create a matrix with appropriate x and y values from ellipse equation
    A = np.column_stack((x**2,x,x*y,y,y**2))
    #compute coefficients using least squares
    a,b,c,d,e = least_squares(A,np.ones(len(x)))
    plot_ellipse(a,b,c,d,e)
    plt.scatter(x,y)
    plt.legend(('Predicted','Actual'))
    plt.title('Ellipse Fit')
    plt.show()


def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = A.shape
    #random vector
    x = np.random.random(n)
    x = x/la.norm(x)
    xk = x
    #loop through to get x_N
    for k in range(N):
        xk_1 = A@xk
        #check if the vectors are close enough
        if la.norm(xk_1 - xk) < tol:
            break
        xk_1 = xk_1/la.norm(xk_1)
        xk = xk_1
    return xk_1@A@xk_1, xk_1


def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m,n = A.shape
    S = la.hessenberg(A)
    for k in range(N):
        Q,R = la.qr(S)
        S = R@Q
    eigs = []
    i = 0
    while i < n:
        if i == n-1 or abs(S[i+1,i]) < tol:
            eigs.append(S[i,i])
        else:
            a,b,c,d = S[i,i],S[i,i+1],S[i+1,i],S[i+1,i+1]
            lamb1 = ((a+d)+np.sqrt((a+d)**2-4*(a*d-b*c)))/2
            lamb2 = ((a+d)-np.sqrt((a+d)**2-4*(a*d-b*c)))/2
            eigs.append(lamb1)
            eigs.append(lamb2)
            i += 1
        i += 1
    return eigs
