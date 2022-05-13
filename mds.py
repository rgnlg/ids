# input:   1) datamatrix as loaded by numpy.loadtxt('dataset.txt')
#	   2) an integer d specifying the number of dimensions for the output (most commonly used are 2 or 3)
# output:  1) an N x d numpy array containing the d coordinates of the N original datapoints projected onto the top d PCs
#
def mds(data, d):

    cov = np.cov(data, rowvar = False) # here I just use the same code I used in PCA function
    evals, evecs = np.linalg.eigh(cov)

    evals = evals[::-1]  # then I use the code for selecting dimensions from pca_lecture_handout
    evecs = evecs[:, ::-1]

    p_comp = evecs[:, :d]

    mat = data @ p_comp  # here we get the dot product

    return mat