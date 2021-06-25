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
import torch
from typing import Optional, Sequence, Tuple, Union

def generalized_distance_matrix(X,Y):
    """
    Returns the distances between all datapoints from X and Y,
    considering each datapoint to be a row vector in each matrix individually.
    """
    n_x,k_x = X.shape
    n_y,k_y = Y.shape

    assert k_x == k_y # dimensionality of vector spaces must be equal

    diag_x = np.zeros((n_x, 1))
    diag_y = np.zeros((1, n_y))

    for i in range(n_x):
        diag_x[i] = np.dot(X[i], X[i])

    for j in range(n_y):
        diag_y[0, j] = np.dot(Y[j], Y[j])

    D = diag_x@np.ones((1,n_y)) + np.ones((n_x,1))@diag_y - 2*X@Y.T

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

    assert k_x == k_y


    diag_x = torch.zeros((n_x, 1))
    diag_y = torch.zeros((1, n_y))

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
    nmi = metrics.normalized_mutual_info_score(
        y_true, y_pred
    )

    ari = metrics.adjusted_rand_score(
        y_true, y_pred
    )

    cm = sc.confusion_matrix(y_pred, y_true)
    pur = sc.purity(cm)

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
