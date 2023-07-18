"""The SVD and Image Compression."""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from imageio import imread


def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    A_H = A.T

    #Get eigenvalues and vectors
    eigvals,eigvecs = la.eigh(A_H@A)
    #get eigvec order
    indexing = np.argsort(eigvals)[::-1]
    #sort eigvals
    eigvals = np.array(sorted(eigvals,reverse=True))
    #get nonzero values
    eigvals = eigvals*(eigvals > tol)
    #get rid of zero eigvals
    r = np.count_nonzero(eigvals > 0)
    eigvals = eigvals[:r]
    #get singular values
    sigmas = np.sqrt(eigvals)
    #get indexing for eigenvectors
    eigvecs = eigvecs.T
    #sort the eigenvectors appropriately
    VT = eigvecs[indexing]
    V = VT.T
    #use a mask to check nonzero sigmas
    V_1 = V[:,:r]
    U_1 = (A@V_1)/sigmas
    # May have issues because it flips a few signs
    return U_1,(sigmas),V_1.T


def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    
    theta = np.linspace(0,2*np.pi,200) #create domain
    S = np.vstack((np.cos(theta),np.sin(theta))) #create S
    E = np.array(([1,0,0],[0,0,1]))
    U,sigma,VH = la.svd(A) #get svd
    sigma = np.diag(sigma) #make 1d array into 2x2

    #plot all transformations
    plt.subplot(221)
    plt.title('S,E')
    plt.axis('equal')
    plt.plot(S[0],S[1])
    plt.plot(E[0],E[1])

    plt.subplot(222)
    plt.title('VS,VE')
    plt.axis('equal')
    plt.plot((VH@S)[0],(VH@S)[1])
    plt.plot((VH@E)[0],(VH@E)[1])


    plt.subplot(223)
    plt.title('SigmaVS,SigmaVE')
    plt.axis('equal')
    plt.plot((sigma@VH@S)[0],(sigma@VH@S)[1])
    plt.plot((sigma@VH@E)[0],(sigma@VH@E)[1])

    plt.subplot(224)
    plt.title('USigmaVS,USigmaVE')
    plt.axis('equal')
    plt.plot((U@sigma@VH@S)[0],(U@sigma@VH@S)[1])
    plt.plot((U@sigma@VH@E)[0],(U@sigma@VH@E)[1])

    plt.tight_layout()
    plt.show()


def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    U,Sig,VH = la.svd(A)
    if s > len(Sig): #value error if s is too large
        raise ValueError('s is greater rank of A')
    #truncate the 3 matrices
    U_s = U[:,:s]
    Sig_s = np.diag(Sig[:s])
    VH_s = VH[:s,:]
    entries = U_s.size + s + VH_s.size #calculate number of entries
    return U_s@Sig_s@VH_s,entries #return A_s and entries


def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
      
    A_H = A.T

    #Get eigenvalues and vectors
    eigvals,eigvecs = la.eigh(A_H@A)
    #get eigvec order
    indexing = np.argsort(eigvals)[::-1]
    #sort eigvals
    eigvals = np.array(sorted(eigvals,reverse=True))
    #get values above err
    eigvals = eigvals*(eigvals >= err)
    #get rid of zero eigvals
    r = np.count_nonzero(eigvals > 0)
    eigvals = eigvals[:r]
    #get singular values
    sigmas = np.sqrt(eigvals)

    #check if err too small
    if err <= sigmas[-1]:
        raise ValueError('err is less than or equal to the smallest singular value')
    
    #get indexing for eigenvectors
    eigvecs = eigvecs.T
    #sort the eigenvectors appropriately
    VT = eigvecs[indexing]
    V = VT.T
    #use a mask to check nonzero sigmas
    V_1 = V[:,:r]
    U_1 = (A@V_1)/sigmas
    # May have issues because it flips a few signs
    entries = U_1.size + len(sigmas) + V_1.size #calculate number of entries
    return U_1,(sigmas),V_1.T,entries


def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    #read image and scale
    image = imread(filename)
    scaled = image/255
    map = None

    #check if gray or color
    if len(scaled.shape) == 3:#color
        #do rgb separately
        preR,R_entries = svd_approx(scaled[:,:,0],s)
        preG,G_entries = svd_approx(scaled[:,:,1],s)
        preB,B_entries = svd_approx(scaled[:,:,2],s)
        #calculate compressed entries
        comp_entries = R_entries + G_entries + B_entries
        R = np.clip(preR,0,1)
        G = np.clip(preG,0,1)
        B = np.clip(preB,0,1)
        comp = np.dstack((R,G,B))
    else: #gray
        comp,comp_entries = svd_approx(scaled,s)
        map = 'gray'
    
    entries_diff = image.size - comp_entries
    print(entries_diff)

    #plot as gray or color
    plt.imshow(comp,cmap=map)
    plt.axis('off')
    plt.suptitle('ye')

    plt.show()

    return 
