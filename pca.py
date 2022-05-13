# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the same order as their associated eigenvalues
def pca(data):

    cov = np.cov(data, rowvar = False)  # covariance matrix
    eig_val, eig_vec = np.linalg.eigh(cov)  # eigenvectors and eigenvalues

    sort = np.argsort(eig_val)[::-1]  # sorting
    sort_eig_val = eig_val[sort]
    sort_eig_vec = eig_vec[:, sort]

    return sort_eig_val, sort_eig_vec
