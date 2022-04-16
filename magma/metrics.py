# ### Metrics classif
#     - [>]  accuracy (torch)
#     - [>]  topk_acc (torch)
#     - [>]  confusion_matrix
# ### Metric clustering
#     - [>]  purity
#     - [>]  entropy
#     - [>]  nmi
#     - [>]  nmi
#     - [>]  topk (numpy)

import numpy as np
import numba
import torch
from typing import Optional, Sequence, Tuple, Union
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
from tqdm import tqdm 
from sklearn.metrics import jaccard_score
from joblib import Parallel, delayed
import multiprocessing as mp

def generalized_distance_matrix(X,Y):
    """
    Returns the distances between all datapoints from X and Y,
    considering each datapoint to be a row vector in each matrix individually.
    """
    n_x,k_x = X.shape
    n_y,k_y = Y.shape

    assert k_x == k_y, 'Number of cols of data X is %d and of Y is %d.'%(k_x, k_y) # dimensionality of vector spaces must be equal

    diag_x = np.zeros((n_x, 1))
    diag_y = np.zeros((1, n_y))

    for i in range(n_x):
        diag_x[i] = np.dot(X[i], X[i])

    for j in range(n_y):
        diag_y[0, j] = np.dot(Y[j], Y[j])

    g1 = diag_x@np.ones((1,n_y))
    g2 = np.ones((n_x,1))@diag_y
    D = g1 + g2 - 2*X@Y.T

    # Ensure diagonal is zero for case where computing D(X,X)
    if n_x == n_y:
        di = np.diag_indices(n_x)
        # Set all values along diagonal to zero
        D[di] = 0

    # Check for small negative values
    mask = D < 0
    if D[mask].size > 0:
        tol_neg = 1e-5
        ix_neg = np.where(mask)
        reset_vals = D[ix_neg]*-1 # flip sign
        assert np.all(reset_vals < tol_neg), "There are negative values in the distance matrix."
        D[ix_neg] =reset_vals

    return np.sqrt(D)

def generalized_distance_matrix_torch(X,Y):
    """
    Returns the distances between all datapoints from X and Y,
    considering each datapoint to be a row vector in each matrix individually.

    Notes
    -----
    Let $X \in R^{n_x \times k}, Y \in R^{n_y \times k}$, the distance between all
    points in X to all points in Y is:

    $D = \mathrm{diag}(XX^T) \mathbf{1}^T_{n_y} + \mathbf{1}_{n_x} \mathrm{diag}(YY^T)^T -2 XY^T$

    where $D_{(i,j)} = || x_i - x_j ||^2$ , i.e. the squared euclidean distance.
    """
    n_x,k_x = X.shape
    n_y,k_y = Y.shape
    dev = X.device

    assert k_x == k_y, 'Number of cols of data X is %d and of Y is %d'%(k_x, k_y) # dimensionality of vector spaces must be equal


    diag_x = torch.zeros((n_x, 1)).to(dev)
    diag_y = torch.zeros((1, n_y)).to(dev)

    for i in range(n_x):
        diag_x[i] = torch.dot(X[i], X[i])

    for j in range(n_y):
        diag_y[0, j] = torch.dot(Y[j], Y[j])

    g1 = diag_x @ torch.ones((1, n_y), device=dev)
    g2 = torch.ones((n_x, 1), device=dev) @ diag_y

    D = g1 + g2 - 2*X@Y.T

    return torch.sqrt(D)

def accuracy(y_pred:torch.Tensor, y_true:torch.Tensor):
    "Returns the accuracy (fraction) between predicted and true labels."

    # if y_pred.shape[1] == 1:
    #     y_pred = y_pred.flatten()

    acc = torch.eq(y_true, y_pred).sum().item() / y_true.shape[0]
    return acc

def topk_acc(y_pred:torch.Tensor, y_true:torch.Tensor, k = 5):
    """
    Returns topk accuracy from multiclass classification.
    Expect that `y_pred` as logits of size (y_true.shape[0], classes).
    """
    # Get indices of top k predictions along axis 1
    top_k_ixs = y_pred.topk(k = k, dim = 1).indices
    acc = torch.eq(y_true.view(-1,1), top_k_ixs).sum().item() / y_true.shape[0]
    return acc

def confusion_matrix(pred_labels, true_labels):
    """
    Returns a confusion matrix from a multiclass classification
    set of labels. Expects labels to be integers between (0, n_classes).

    Params
    ------
    pred_labels (array-like):
        List of labels as predicted by a classification algorithm.

    true_labels (array-like):
        List of ground truth labels.

    Returns
    -------
    conf_mat (array-like):
        Confusion matrix.
    """

    n_labels = int(max(np.max(pred_labels), np.max(true_labels)) + 1)


    conf_mat = np.zeros(shape = (n_labels, n_labels))

    for (i, j) in zip(pred_labels, true_labels):
        conf_mat[i,j] +=1

    return conf_mat


def purity(cm:np.array)->float:
    """
    Returns clustering purity given a confusion matrix.

    Params
    ------
    cm (array)
        Confusion matrix. Expects to have row indices
        indicating predicted values, and columns indicating
        ground truth.

    Returns
    -------
    purity(float)
        Clustering purity. Bounded between [0,1].
    """
    #Indices of maximum values of ground truth labels
    max_ixs_gt = cm.argmax(axis = 1)

    # Dummy array to get the values per row
    row_indexer = np.arange(cm.shape[0])
    sum_max_values = cm[row_indexer, max_ixs_gt].sum()
    purity=sum_max_values/cm.sum()
    return purity


def element_wise_entropy(px):
    """
    Returns a numpy array with element wise entropy calculated as -p_i*log_2(p_i).

    Params
    ------
    px (np.array)
        Array of individual probabilities, i.e. a probability vector or distribution.

    Returns
    -------
    entropy (np.array)
        Array of element-wise entropies.
    """
    if isinstance(px, list):
        px = np.array(px)

    # Make a copy of input array
    entropy = px.copy()

    # Get indices of nonzero probability values
    nz = np.nonzero(entropy)

    # Compute -pi*log_2(p_i) element-wise
    entropy[nz] *= - np.log2(entropy[nz])

    return entropy


def entropy(ps):
    "Returns the entropy of a probability distribution `ps`."
    # Get nonzero indices
    nz = np.nonzero(ps)

    # Compute entropy for nonzero indices
    entropy = np.sum(-ps[nz]*np.log2(ps[nz]))

    return entropy

def kl_div_naive(p,q): 
    kl = sum(p*np.log2(p/q))
    return kl

def kl_div(p:np.array,q:np.array)-> float:
    """
    Returns Kullback-Leibler divergence given two probability distros.
    """
    nz_p = np.nonzero(p)
    nz_q = np.nonzero(q)

    a = np.sum(p[nz_p]*np.log2(p[nz_p]))
    b = np.sum(p[nz_q]*np.log2(q[nz_q]))
    kl = a - b
    return kl 

def JSD(p:np.array,q:np.array)->float: #jsd
    "Returns Jensen-Shannon div given two prob distros"
    m = 0.5*(p+q)
    jensen_shannon_div = 0.5*kl_div(p,m) + 0.5*kl_div(q,m)
    return jensen_shannon_div


def kld_torch(p:torch.tensor,q:torch.tensor)-> float:
    """
    Returns Kullback-Leibler divergence given two probability distros.
    """
    nz_p = torch.nonzero(p)
    nz_q = torch.nonzero(q)

    a = torch.sum(p[nz_p]*torch.log2(p[nz_p]))
    b = torch.sum(p[nz_q]*torch.log2(q[nz_q]))
    kl = a - b
    return kl 

def JSD_torch(p:torch.tensor,q:torch.tensor)-> float:
    "Returns Jensen-Shannon div given two prob distros"
    m = 0.5*(p+q)
    jensen_shannon_div = 0.5*kld_torch(p,m) + 0.5*kld_torch(q,m)
    return jensen_shannon_div


def nmi_from_labels(pred_labels, true_labels):
	"""
	Returns the normalized mutual information (NMI) of two clustering
	or classification labels X and Y as:

	NMI(X,Y) = (H(X) + H(Y)) / H(X,Y)

	The intuition is that a high NMI corresponds to a dependence between
	the labels, i.e. a good classification or clustering performance.

	Params
	------
	pred_labels : arrays to compute the NMI from.

	Returns
	-------
	nmi (float):
		Normalized mutual information NMI(X,Y).

	Notes
	-----
	The NMI is bounded between 1 and 2. That is,

	* If X = Y, H(X) = H(Y) = H(X,Y) = I(X,Y), thus NMI = 2

	* If $| X \cap Y | = 0$, then H(X, Y) = H(X) + H(Y), I(X,Y) = 0,
	  and NMI = 1

	  FMI = NMI -1.

	  There is another parametrization defined as:
	  NMI(X,Y) = I(X;Y) / ((H(X) + H(Y))/2)
	  From : https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
	"""

	# Compute joint probability
	p_xy = confusion_matrix(pred_labels, true_labels)

	p_xy /= np.sum(p_xy)

	# Marginal distribution of the predicted labels
	p_x = p_xy.sum(axis = 1)

	# Marg. dist. of true_labels
	p_y = p_xy.sum(axis = 0)

	h_x = entropy(p_x)
	h_y = entropy(p_y)

	# Sum across both axes
	h_xy = np.sum(element_wise_entropy(p_xy))

	nmi = (h_x + h_y) / h_xy

	return nmi

def get_clus_metrics(y_pred, y_true):
    """
    Returns 3 clustering metrics.
    Designed to work for cases where ground truth is known
    or a comparison between sets is amenable.
    """
    nmi = nmi_from_labels(
        y_true, y_pred
    )

    ari = adjusted_rand_score(
        y_true, y_pred
    )

    cm = confusion_matrix(y_pred, y_true)
    pur = purity(cm)

    print('NMI: %.2f'%nmi)
    print('Adjusted Rand Index: %.2f'%ari)
    print('Purity: %.2f'%pur)

    return nmi, ari, pur


def topk(arr, k = 2, axis = 1):
	"""
	Returns the top-k indices over an axis.
	Numpy analog of torch.topk(tensor,k,axis).indices

	Example
	-------
	foo = np.arange(12).reshape(4, 3)
	print(foo)
	>>> array([[ 0,  1,  2],
		       [ 3,  4,  5],
		       [ 6,  7,  8],
		       [ 9, 10, 11]])

	top_ixs = sc.topk(foo, k = 2, axis = 0)
	print(top_ixs)
	>> array([[3, 3, 3],
       		  [2, 2, 2]])

	print(foo[top_ixs[:, 0], 0])
	>>> array([9, 6])

	top_ixs_row = sc.topk(foo, k = 2, axis = 1)
	print(foo[1, top_ixs_row[1, :]])
	>>> array([5, 4])

	"""

	ixs = np.argpartition(arr, -k, axis = axis)

	if axis == 0:
		return np.flipud(ixs[-k:, :])
	elif axis == 1:
		return np.fliplr(ixs[:, -k:])
	else:
		print('Function available for 2D arrays only.')
		return None


def multilabel_accuracy(y_true, y_pred):
    """
    Average ratio of intersection over union of \hat{y} and y.
    """
    tmp = 0
    for i in range(y_true.shape[0]):
        tmp+=np.sum(np.logical_and(y_true[i], y_pred[i]))/np.sum(np.logical_or(y_true[i], y_pred[i]))
    return tmp / y_true.shape[0]


@numba.njit
def hamming_dist(y_true, y_pred):
    "Returns the average hamming distance for multilabel classification."
    running_sum=0

    n_samples = y_true.shape[0]
    for i in range(n_samples):
        running_sum += np.sum(y_true[i] != y_pred[i])

    hamming_dist = running_sum / n_samples
    return hamming_dist

@numba.njit
def precision(y_true, y_pred):
    """
    Average ratio of cardinality of intersection between y_i and \hat{y_i}
    over cardinality of predicted labels \hat{y_i}.
    """
    running_sum = 0
    n_samples = y_true.shape[0]
    for i in range(n_samples):
        if np.sum(y_pred[i]) == 0:
            continue
        else:
            running_sum += np.logical_and(y_true[i], y_pred[i]).sum() / np.sum(y_pred[i])
    recall = running_sum / n_samples
    return recall


@numba.njit
def recall(y_true, y_pred):
    """
    Average ratio of cardinality of intersection between y_i and \hat{y_i}
    over cardinality of true labels y_i.
    """
    running_sum = 0
    n_samples = y_true.shape[0]

    for i in range(n_samples):
        if np.sum(y_true[i]) == 0:
            continue
        else:
            running_sum += np.logical_and(y_true[i], y_pred[i]).sum() / np.sum(y_true[i])
    recall = running_sum / n_samples

    return recall


def affinity(data,sigma):
    n, _ = data.shape[0]
    D = pairwise_distances(data)
    A = np.exp(-D**2/sigma**2) - np.eye(n)

    return A

def affinity_norm(data, k):
    """
    Params
    ------
    k : k-th nearest neighbor

    """
    tree=KDTree(data, leaf_size=2)
    distances, indices = tree.query(data, k = k)
    k_dist= distances[:, -1]
    densities = 1/k_dist
    D_scaled = D**2 * densities * densities.reshape(-1,1)
    A_scaled = np.exp(-D_scaled)

    return A_scaled



def _jaccard(idx_query, idx_pred, targets_arr): 
    """
    Computes Jaccard score between binary vectors.
    """
    y = targets_arr[idx_query]
    y_hat = targets_arr[idx_pred]
    
    return jaccard_score(y, y_hat)

def run_jaccard_score_calc( 
    cosine_arr, 
    targets_arr,
    df_combos,
    name_to_ix_target,
    topk=30,
    axis = 1,
    n_cores=-1
    )->np.array:
    """
    Returns the average jaccard score for top predictions.

    Params
    ------
    cosine_arr (np.array)
        Cosine similarity array. It has shape (mols,cells). 
        Note : in most cases mols = structure + cell type combination.

    targets_arr (np.array)
        Binary matrix where the i-th row contains the targets of the i-th molecule.
    
    df_combos (pd.DataFrame)
        Dataframe with `n_combos` rows. Each combination is a (molecule, celltype) pair.

    name_to_ix_target (dict)
        Mapping going from molecule name to idx in `target_arr` matrix.

    topk(int, default =30)
        kNearest neighbors to look for. 

    axis (int, default = 1)
        Axis along which you can find cells. If axis == 0, cells are along rows.
    
    n_cores (int, default = -1)
        Number of cores to run in parallel. If set to -1 it detects max number of cores.

    """
    n_cores = mp.cpu_count() if n_cores == -1 else n_cores

    n_cells = cosine_arr.shape[axis]
    
    if axis == 0: 
        cosine_arr = cosine_arr.T

    jacc_scores = np.zeros(n_cells)
    for i in tqdm(range(n_cells)):

        # Get idx of query mol (perturbing the i-th cell)
        idx_query = name_to_ix_target[i]

        # Get the top indices of the closest molecule-cell pairs
        ix_top_combos = np.argsort(cosine_arr[:, i])[::-1][:topk]
        pred_mols = df_combos.iloc[ix_top_combos].drug_name.values

        # get ids from target from molnames
        pred_ids = [name_to_ix_target[x] for x in pred_mols]
        
        jaccard_scores = Parallel(n_cores)(
            delayed(_jaccard)(idx_query, idx_pred, targets_arr) for idx_pred in pred_ids
        )

        avg_jaccard_score = np.mean(jaccard_scores)
        jacc_scores[i] = avg_jaccard_score
    
    return jacc_scores