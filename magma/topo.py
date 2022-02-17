import matplotlib.pyplot as plt
from pydec import abstract_simplicial_complex
from pydec.dec.rips_complex import rips_complex
import numpy as np
from ripser import ripser
from persim import plot_diagrams

def get_betti_numbers(data, eps, mink= 1, topk=1):
    """
    Returns the Betti numbers for a dataset, using the Vietoris-Rips complex.

    Params
    ------
    mink (int)
        Minimum betti number. If set to zero it will return the number
        of connected components.
    topk (int)
        Maximum betti number.
    """
    assert topk>0, "topk must be a positive integer."

    rc = rips_complex(data, eps) #Initialize Rips complex
    cmplx = rc.chain_complex() # Compute boundary operators.

    betti_numbers = []
    for i in range(mink,topk+1):
        try:
            bnp1 = cmplx[i+1].astype(float)  # edge boundary operator
            bn = cmplx[i].astype(float)  # face boundary operator

            bnp1_image_dim = np.linalg.matrix_rank(bnp1.A)
            bn_kernel_dim = bn.shape[1] - np.linalg.matrix_rank(bn.A)
            betti_n = bn_kernel_dim - bnp1_image_dim
            betti_numbers.append(betti_n)
        except:
            betti_numbers.append(None)

    return betti_numbers

def is_even(n):
    return not n%2

def get_noisy_flower(n_points=1000, n_petals=4, noise = 0.01,scaling=5):
    "Returns a point cloud for a flower with n petals with noise."
    n_petals /= 2

    # Make flower in polar coords
    rads = np.linspace(0, 2*np.pi, n_points)
    r = np.abs(np.cos(n_petals*rads))

    # Polar to rectangular
    x = r*np.cos(rads)
    y = r*np.sin(rads)

    data = np.vstack([x,y]).T
    data*=scaling

    gaussian_noise = np.random.normal(
    loc = 0, scale = noise, size = data.shape
    )
    noisy_flower = data + gaussian_noise
    return noisy_flower


def plot_flower_demo(n_points=300, n_petals=3):
    flowers = get_noisy_flower(n_points, n_petals=3, noise =0.2)
    n_betti = get_betti_numbers(flowers,eps=.5)[0]


    print(f"Found {n_betti} cycles.")
    plt.figure()
    plt.scatter(*flowers.T, c =np.arange(n_points))

    dgms = ripser(flowers)['dgms']
    plt.figure()
    plot_diagrams(dgms, show=True)


def drawLineColored(X, C):
    for i in range(X.shape[0]-1):
        plt.plot(X[i:i+2, 0], X[i:i+2, 1], c=C[i, :], lineWidth = 3)

def plotCocycle2D(D, X, cocycle, thresh):
    """
    Given a 2D point cloud X, display a cocycle projected
    onto edges under a given threshold "thresh"

    Params
    ------
    D (np.array)
        Distance matrix used during computation, if used a subsample
        of data.

        Can be retrieved from res['dperm2all']
        after calling res = ripser(x).

    X (np.array)
        Dataset.

    cocycle ()

    thresh (float)
        Distance threshold.

    """
    #Plot all edges under the threshold
    N = X.shape[0]
    t = np.linspace(0, 1, 10)
    c = plt.get_cmap('Greys')
    C = c(np.array(np.round(np.linspace(0, 255, len(t))), dtype=np.int32))
    C = C[:, 0:3]

    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                Y = np.zeros((len(t), 2))
                Y[:, 0] = X[i, 0] + t*(X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t*(X[j, 1] - X[i, 1])
                drawLineColored(Y, C)
    #Plot cocycle projected to edges under the chosen threshold
    for k in range(cocycle.shape[0]):
        [i, j, val] = cocycle[k, :]
        if D[i, j] <= thresh:
            [i, j] = [min(i, j), max(i, j)]
            a = 0.5*(X[i, :] + X[j, :])
            plt.text(a[0], a[1], '%g'%val, color='b')
    #Plot vertex labels
    for i in range(N):
        plt.text(X[i, 0], X[i, 1], '%i'%i, color='r')
    plt.axis('equal')


def get_lifetimes(diagram):
    "Lifetime is death time - birth time."
    return diagram[:, 1]-diagram[:, 0]



def demo_cycles():
    x = get_noisy_flower(100, n_petals=3, noise =0.1)
    result = ripser(x, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    dgm1 = diagrams[1]
    idx = np.argmax(dgm1[:, 1] - dgm1[:, 0])

    # plot persistence diag
    plt.figure()
    plot_diagrams(diagrams, show = False)
    plt.scatter(dgm1[idx, 0], dgm1[idx, 1], 20, 'k', 'x')
    plt.title("Max 1D birth = %.3g, death = %.3g"%(dgm1[idx, 0], dgm1[idx, 1]))
    plt.show()

    # plot cocycle
    cocycle = cocycles[1][idx]
    thresh = dgm1[idx, 0] #Project cocycle onto edges that have lengths less than or equal to the birth time
    plt.figure()
    plotCocycle2D(D, x, cocycle, thresh)
    plt.title("1-Form Thresh=%g"%thresh)
    plt.show()
