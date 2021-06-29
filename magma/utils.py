# ### utils
#     - [>]  trainers
#     - [>]  initialize
#     - [>]  torch_adata
#     - [>]  try_gpu
#     - [>]  cells: get_count_stats, log_norm, cv_filter
from .metrics import accuracy
from .metrics import topk_acc
from .metrics import generalized_distance_matrix
from .metrics import generalized_distance_matrix_torch

from typing import Optional, Sequence, Tuple, Union

import scipy.io as sio
import scipy.stats as st
from scipy import sparse

import numpy as np
import pandas as pd
import anndata as ad

import toolz as tz
import tqdm
import os
import collections
from sklearn import metrics
from sklearn.utils import sparsefuncs
from joblib import Parallel, delayed

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader

import torch_geometric
from torch_geometric.data import Batch

from .chemspace import get_drug_batch
#from rdkit.Chem.Draw import rdMolDraw2D
#from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
#from rdkit import DataStructs

def train_supervised_gcn(
    model:nn.Module,
    data:torch_geometric.data.Data,
    loss_fn,
    optimizer,
    multiclass = False,
    n_out = 1
)->Tuple[float, float]:
    """
    Single fwd-bwd pass on GraphConvNet model.
    Returns loss and accuracy.
    """

    y_true = torch.tensor(data.y, dtype = torch.long)

    optimizer.zero_grad()
    y_pred = model(data)

    if multiclass:
        loss = loss_fn(y_pred, y_true)
        y_hat = y_pred.argmax(dim = 1)
        acc = accuracy(y_hat, y_true)

    else:
        loss = loss_fn(
            y_pred.float(),
            y_true.reshape(-1, n_out).float()
        )

        acc = accuracy(y_pred, y_true)


    loss.backward()
    optimizer.step()

    return loss, acc

def val_supervised_gcn(
    model, data, loss_fn, multiclass = False, n_out = 1
)-> float:


    y_pred = model(data)
    #y_true = torch.from_numpy(np.array(data.y, dtype=np.int16)) #, device = device)
    y_true = torch.tensor(data.y, dtype = torch.long)

    if multiclass:
        loss = loss_fn(y_pred, y_true)
        y_hat = y_pred.argmax(dim = 1)
        acc = accuracy(y_hat, y_true)
    else:
        loss = loss = loss_fn(
            y_pred.float(),
            y_true.reshape(-1, n_out).float()
        )
        acc = accuracy(y_pred, y_true)

    return loss.mean(), acc


def supervised_trainer_gcn(
    n_epochs:int,
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    multiclass= False,
    n_classes = 1,
    logs_per_epoch = 5,
    model_dir:str = None,
    model_name:str = None,
    early_stopping_tol:float = 0.3,
)-> Tuple[list, np.ndarray, np.ndarray]:
    """
    Wrapper function to train a GNN, returns train and val loss, and val accuracy.
    Currently designed for classification problems.

    Params
    ------
    n_epochs (int)
        Number of forward-backward passes through all the training data.

    train_loader, val_loader
        torch_geometric.data.Dataloaders of training and validation set.
        The validation set is used for estimating model convergence.

    model (nn.Module)
        Supervised neural net model.

    criterion (torch.nn.modules.loss object)
        Loss function.

    optimizer (torch.optim object)
        Optimizer, e.g. Adam or RMSProp.

    multiclass (bool, default = False)
        Whether the model is a softmax classification model.

    n_classes (int, default = 1)
        Dimensionality of output dimension.

    model_dir (str, default = None)
        Path to store trained models.
        If set to None it will not store the model's weights.

    model_name (str, default = None)
        Filename of the model to be stored. If set to None and `model_dir` is specified,
        the model will be stored as `model.pt`.

    early_stopping_tol (float, default = 0.1)
        Tolerance to stop the training.
        It is used as the fractional increase in the validation loss
        in order to stop the training. I.e. in pseudocode:

        Stop if val_loss[i] > (1+early_stopping_tol)*val_loss[i-1]

        The higher the value the more tolerant to run for the number of epochs.
        If the value is small the traning loop can be too sensitive to small
        increases in the validation loss.
    """

    batch_size = train_loader.batch_size
    print_every = np.floor(train_loader.dataset.__len__() / batch_size / logs_per_epoch) # minibatches

    train_loss_vector = [] # to store training loss
    val_loss_vector = np.empty(shape = n_epochs)
    val_acc_vector = np.empty(shape = n_epochs)

    cuda = torch.cuda.is_available()

    if cuda:
        device = try_gpu()
        torch.cuda.set_device(device)
        model = model.to(device)

    for epoch in np.arange(n_epochs):

        running_loss = 0

        # TRAINING LOOP
        for ix, data in tqdm.tqdm(enumerate(train_loader)):

            #input_tensor = data.view(batch_size, -1).float()

            if cuda:
                data.edge_attr = data.edge_attr.cuda()
                data.edge_index = data.edge_index.cuda()
                data.x = data.x.cuda()
                data.y = torch.tensor(data.y, device = device)
                data.ptr  = data.ptr.cuda()
                data.batch = data.batch.cuda()


            train_loss, train_acc = train_supervised_gcn(
                model,
                data, # graph and label in data object
                criterion,
                optimizer,
                multiclass=multiclass,
                n_out =n_classes
                )

            running_loss += train_loss.item()

            # Print loss
            if ix % print_every == print_every -1 :

                # Print average loss
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, ix+1, running_loss / print_every))

                train_loss_vector.append(running_loss / print_every)

                # Reinitialize loss
                running_loss = 0.0

        # VALIDATION LOOP
        with torch.no_grad():
            validation_loss = []
            val_accuracy = []

            for i, data in enumerate(tqdm.tqdm(val_loader)):

                if cuda:
                    data.edge_attr = data.edge_attr.cuda()
                    data.edge_index = data.edge_index.cuda()
                    data.x = data.x.cuda()
                    data.y = torch.tensor(data.y, device = device)
                    data.ptr  = data.ptr.cuda()
                    data.batch = data.batch.cuda()


                val_loss, val_acc = val_supervised_gcn(
                    model, data, criterion, multiclass, n_classes
                    )

                validation_loss.append(val_loss)
                val_accuracy.append(val_acc)

            mean_val_loss = torch.tensor(validation_loss).mean()
            mean_accuracy = torch.tensor(val_accuracy).mean()

            val_loss_vector[epoch] = mean_val_loss
            val_acc_vector[epoch] = mean_accuracy

            print('Val. loss %.3f'% mean_val_loss)
            print('Val. acc %.3f'% (mean_accuracy*100))

        # EARLY STOPPING LOOP
        if epoch > 0:
            if val_loss_vector[epoch] > (1+early_stopping_tol)*val_loss_vector[epoch-1]:
                print('Finished by early stopping at epoch %d'%(epoch))
                return train_loss_vector, val_loss_vector, val_acc_vector

        # SAVE MODEL
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            if model_name is not None:
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, model_name + '_' + str(epoch) + '.pt')
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, 'model' + '_' + str(epoch) + '.pt')
                )

    print('Finished training')

    return train_loss_vector, val_loss_vector, val_acc_vector


def train_supervised(
    model,
    input_tensor,
    y_true,
    loss_fn,
    optimizer,
    multiclass =False,
    n_out = 1,
    ):
    """
    Wrapper function to make forward and backward pass with minibatch
    using a supervised model (classification or regression).

    Params
    ------
    n_out (int, default = 1)
        Dimensionality of output dimension. Leave as 1 for multiclass,
        i.e. the output is a probability distribution over classes (e.g. MNIST).
    """

    # Zero out grads
    model.zero_grad()
    y_pred = model(input_tensor)

    #Note that if it's a multiclass classification (i.e. the output is a
    # probability distribution over classes) the loss_fn
    # nn.NLLLoss(y_pred, y_true) uses as input y_pred.size = (n_batch, n_classes)
    # and y_true.size = (n_batch), that's why it doesn't get reshaped.

    if multiclass:
        loss = loss_fn(y_pred, y_true)
        y_hat = y_pred.argmax(dim = 1)
        acc = accuracy(y_hat, y_true)

    else: # Backprop error
        loss = loss_fn(y_pred, y_true.view(-1, n_out).float())
        try:
            acc = accuracy(y_pred, y_true.view(-1, n_out).float())
        except:
            acc = None

    loss.backward()
    # Update weights
    optimizer.step()

    return loss, acc

def validation_supervised(model, input_tensor, y_true, loss_fn, multiclass =False, n_classes= 1):
    """
    Returns average loss for an input batch of data with a supervised model.
    If running on multiclass mode, it also returns the accuracy.
    """

    y_pred = model(input_tensor.float())
    if multiclass:
        loss = loss_fn(y_pred, y_true)

        y_hat = y_pred.argmax(dim = 1)
        acc = accuracy(y_hat, y_true)
    else:
        loss = loss_fn(y_pred, y_true.view(-1, n_classes).float())
        try:
            acc = accuracy(y_pred, y_true.view(-1, n_out).float())
        except:
            acc = None

    return loss.mean().item(), acc

def supervised_trainer(
    n_epochs:int,
    train_loader:DataLoader,
    val_loader:DataLoader,
    model:nn.Module,
    criterion,
    optimizer,
    multiclass:bool = False,
    n_classes:int = 1,
    logs_per_epoch:int = 5,
    train_fn:callable = train_supervised,
    model_dir:str = None,
    model_name:str = None,
    early_stopping_tol:float = 0.2,
    **kwargs
    ):
    """
    Wrapper function to train a supervised model for n_epochs.
    Currently designed for classification and regression.
    Returns train loss, validation loss and accuracy.

    Params
    ------
    n_epochs (int)
        Number of forward-backward passes through all the training data.

    train_loader, val_loader
        Torch dataloaders of training and validation set. The validation set
        is used for estimating model convergence.

    model (nn.Module)
        Supervised neural net model.

    criterion (torch.nn.modules.loss object)
        Loss function.

    optimizer (torch.optim object)
        Optimizer, e.g. Adam or RMSProp.

    multiclass (bool, default = False)
        Whether the model is a softmax classification model.

    n_classes (int, default = 1)
        Dimensionality of output dimension. Leave as 1 for multiclass,
        i.e. the output is a probability distribution over classes (e.g. MNIST).

    model_dir (str, default = None)
        Path to store trained models. If set to None it will not store the model weights.

    model_name (str, default = None)
        Filename of the model to be stored. If set to None and `model_dir` is specified,
        the model will be stored as `model.pt`

    early_stopping_tol (float, default = 0.1)
        Tolerance to stop the training.
        It is used as the fractional increase in the validation loss
        in order to stop the training. I.e. in pseudocode:

        Stop if val_loss[i] > (1+early_stopping_tol)*val_loss[i-1]

        The higher the value the more tolerant to run for the number of epochs.
        If the value is small the traning loop can be too sensitive to small
        increases in the validation loss.

    **kwargs
        All kwargs go to the train_fn and val_fn functions.

    Returns
    -------
    train_loss_vector(array-like)
        List with loss at every minibatch, of size (minibatch*n_epochs).

    val_loss_vector(array-like)
        Numpy array with validation loss for every epoch.
    """

    batch_size = train_loader.batch_size
    print_every = np.floor(train_loader.dataset.__len__() / batch_size / logs_per_epoch) # minibatches

    train_loss_vector = [] # to store training loss
    val_loss_vector = np.empty(shape = n_epochs)
    val_acc_vector = np.empty(shape = n_epochs)

    cuda = torch.cuda.is_available()

    if cuda:
        device = try_gpu()
        torch.cuda.set_device(device)
        model = model.to(device)

    for epoch in np.arange(n_epochs):

        running_loss = 0

        # TRAINING LOOP
        for ix, (data, y_true) in enumerate(tqdm.tqdm(train_loader)):

            input_tensor = data.view(batch_size, -1).float()

            if cuda:
                input_tensor = input_tensor.cuda(device = device)
                y_true = y_true.cuda(device = device)

            train_loss, train_acc = train_fn(
                model,
                input_tensor,
                y_true,
                criterion,
                optimizer,
                multiclass=multiclass,
                n_out =n_classes,
                **kwargs
                )

            running_loss += train_loss.item()

            # Print loss
            if ix % print_every == print_every -1 :

                # Print average loss
                print('[%d, %6d] loss: %.3f' %
                      (epoch + 1, ix+1, running_loss / print_every))

                train_loss_vector.append(running_loss / print_every)

                # Reinitialize loss
                running_loss = 0.0

        # VALIDATION LOOP
        with torch.no_grad():
            validation_loss = []
            validation_accuracy = []

            for i, (data, y_true) in enumerate(tqdm.tqdm(val_loader)):

                input_tensor = data.view(batch_size, -1).float()

                if cuda:
                    input_tensor = input_tensor.cuda(device = device)
                    y_true = y_true.cuda(device = device)

                val_loss, val_acc = validation_supervised(
                    model, input_tensor, y_true, criterion, multiclass, n_classes
                    )

                validation_loss.append(val_loss)
                validation_accuracy.append(val_acc)

            mean_val_loss = torch.tensor(validation_loss).mean().item()
            mean_val_acc = torch.tensor(validation_accuracy).mean().item()

            val_loss_vector[epoch] = mean_val_loss
            val_acc_vector[epoch] = mean_val_acc

            print('Val. loss %.3f'% mean_val_loss)
            print('Val. accuracy %.3f'% (mean_val_acc*100))


        # EARLY STOPPING LOOP
        if epoch > 0:
            if val_loss_vector[epoch] > (1+early_stopping_tol)*val_loss_vector[epoch-1]:
                print('Finished by early stopping at epoch %d'%(epoch))
                return train_loss_vector, val_loss_vector

        # SAVE MODEL
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            if model_name is not None:
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, model_name + '_' + str(epoch) + '.pt')
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, 'model' + '_' + str(epoch) + '.pt')
                )


    print('Finished training')

    return train_loss_vector, val_loss_vector, val_acc_vector


def print_loss_in_loop(epoch, idx_batch, running_loss, print_every, message='loss'):
    print_msg = '[%d, %5d] ' + message + ' : %.3f'
    print(print_msg%\
          (epodch + 1, idx_batch+1, running_loss / print_every))

def supervised_model_predict(
    model:nn.Module,
    data_loader,
    criterion,
    n_points = None,
    n_feats= None,
    multiclass=False,
    n_outputs =1,
    score = True
    ):
    """
    Analog to model.predict_proba() from sklearn. Returns a prediction vector given a torch dataloder
    and model. It is designed for working with basic supervised models like binary or multilabel
    classification, and regression.

    Params
    ------

    model (torch.nn.model)
        Trained supervised model.

    data_loader

    n_points (int)
        Number of instances (rows) in the dataset. If not provided, the function will
        try to extract it from the dataloader.

    n_feats (int)
        Input dimensions for the model / number of columns in the dataset. If not provided,
        the function will try to extract it from the dataloader.

    n_outputs (int, default = 1)
        Number of outputs of the model. Defaults to 1 dim output, for regression or
        binary classification.

    Returns
    -------
    y_pred (np.array)
        Array with raw predictions from a forward pass of the model.

    """
    if n_points == None and n_feats == None:
        try:
            n_points, n_feats = data_loader.dataset.data.shape
        except:
            print('Need to supply number of datapoints and features in input data.')

    batch_size = data_loader.batch_size

    cuda = torch.cuda.is_available()
    device = try_gpu()

    model = model.to(device)

    # Initialize predictions array
    y_pred = torch.zeros(n_points, n_outputs)

    if score:
        cum_sum_loss = 0
        cum_sum_acc = 0

    with torch.no_grad():

        for ix, (x, y) in tqdm.tqdm(enumerate(data_loader)):

            if cuda:
                x= x.cuda()
            if cuda and score:
                y =y.cuda()

            # Reshape input for feeding to model
            x = x.view(-1, n_feats)

            outputs = model(x.float())

            y_pred[ix * batch_size : ix * batch_size + batch_size, :] = outputs

            if score:
                if multiclass:
                    if cuda:
                        mean_loss = criterion(outputs, y).mean().cpu().detach().numpy()
                    else:
                        mean_loss = criterion(outputs, y).mean().detach().numpy()

                    acc = accuracy(y, outputs.argmax(axis = 1))#.item()

                else:
                    if cuda:
                        mean_loss = criterion(outputs, y.view(-1, n_outputs).float()).mean().cpu().detach().numpy()

                    else:
                        mean_loss = criterion(outputs, y.view(-1, n_outputs).float()).mean().detach().numpy()

                    acc = accuracy(y.view(-1, n_outputs), outputs.argmax(axis = 1))#.mean().item()

                cum_sum_loss+= mean_loss
                cum_sum_acc +=acc

                moving_avg_acc = cum_sum_acc / (ix+1)
                moving_avg_loss = cum_sum_loss / (ix + 1)



        if score:
            print("Mean accuracy: %.2f" %moving_avg_acc)
            print("Mean validation loss: %.2f"%moving_avg_loss)

    return y_pred.detach().numpy()



def get_positive_negative_indices_batch(
        y_true:torch.Tensor, index_dict:dict, cuda:bool = None
    )->Tuple[np.array, np.array, np.array]:
    """
    Returns indices for positive and negative anchors,
    to use in metric learning using hinge triplet loss,
    given labels (y_true) for multiclass classification.

    Params
    ------
    y_true(torch.Tensor)
        Labels from sample codes.

    cuda (bool, default = None)
        Whether cuda is available for use.

    Returns
    -------
    positive_anchor_ixs, negative_anchor_ixs, perm_labels

    """

    max_index = max(index_dict.keys())

    # Get cuda status
    if cuda is None:
        cuda = torch.cuda.is_available()

    # Send labels to cpu
    if cuda:
        y_true = y_true.cpu()

    # Make labels from torch.tensor -> numpy array
    labels = y_true.numpy()

    # Shuffle labels
    perm_labels= np.random.permutation(labels)

    # Check if any of shuffled labels didn't change
    ix_eq = (labels == perm_labels)

    # Get the indices where those unchanged labels reside
    ix_to_flip = np.nonzero(ix_eq)[0]

    # Enter loop if the permuted labels
    # and the original labels coincide in an entry
    if len(ix_to_flip) >= 1:
        # If any of the labels to flip is the last code
        # subtract as adding would result in error

        label_flip_max_code = np.any(perm_labels[ix_to_flip] == max_index)

        label_flip_min_code = np.any(perm_labels[ix_to_flip] == 0)

        # Unlikely case where the batch contains both
        # the first and last index
        # this will cause the hinge loss to be the margin
        if label_flip_max_code and label_flip_min_code:
            print('At least one label in the positive and negative is the same')
            pass


        elif label_flip_max_code and not label_flip_min_code:
            perm_labels[ix_to_flip] = perm_labels[ix_to_flip] - 1

        # Fall back to add an index
        else :
            perm_labels[ix_to_flip] = perm_labels[ix_to_flip] + 1


    # Check that all labels are different
    #assert np.all(labels != perm_labels)


    # Get anchor indices for samples
    positive_anchor_ixs = [np.random.choice(index_dict[l], size = 1)[0] for l in labels]

    negative_anchor_ixs = [np.random.choice(index_dict[l], size = 1)[0] for l in perm_labels]

    return positive_anchor_ixs, negative_anchor_ixs, perm_labels


class JointEmbeddingTrainer:
    """
    Class for training the joint embedding model.
    """
    def __init__(
        self,
        model,
        adata,
        batch_size,
        train_loader,
        val_loader,
        index_dict_train:dict,
        index_dict_test:dict,
        name_to_mol:dict,
        ix_to_name:dict,
        lr:float = 1e-5,
        n_epochs:int = 20,
        metric_learning:bool = True,
        contrastive_learning:bool = True,
        p_norm_metric:int = 2,
        margin:float = 3.,
        model_name:str = None,
        model_dir:str = None,
        ):
        """
        """
        device = try_gpu()
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.adata = adata
        self.train_loader, self.val_loader = train_loader, val_loader

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.to(device)

        self.n_epochs = n_epochs
        self.index_dict_train = index_dict_train
        self.index_dict_test = index_dict_test
        self.name_to_mol = name_to_mol
        self.ix_to_name = ix_to_name

        self.hinge_loss = nn.TripletMarginLoss(margin=margin, p=p_norm_metric)
        self.criterion = nn.NLLLoss()
        self.ordering_labels = torch.arange(batch_size).to(device)

        self.contrastive = contrastive_learning #bool
        self.metric = metric_learning #bool

        self.n_train_batches = len(train_loader.dataset) // batch_size
        self.n_test_batches = len(val_loader.dataset) // batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

        if self.contrastive == False and self.metric ==False:
            raise AssertionError(
                'Either one or both of contrastive learning and metric learning have to be active.'
            )


    def contrastive_learning_loop(self, mol_embedding, cell_embedding):
        """Returns contrastive learning loss and cross-retrieval accuracy for a minibatch."""
        # Make batch of molecular graphs
        cell_embedding_norm = cell_embedding / cell_embedding.norm(dim= -1, keepdim = True)
        mol_embedding_norm = mol_embedding / mol_embedding.norm(dim= -1, keepdim = True)

        # Extract learnt scalar
        logit_scale = self.model.logit_scale.exp()

        # Get cosine similarities
        # returns tensor of shape (mols, cells)
        logits = logit_scale * mol_embedding_norm @ cell_embedding_norm.t()

        # Get classification predictions across axes
        y_pred_mols = F.log_softmax(logits, dim = 1)
        y_pred_cells = F.log_softmax(logits, dim = 0)

        # Calculate accuracies
        mol_acc = accuracy(y_pred_mols.argmax(axis =1), self.ordering_labels)
        cell_acc = accuracy(y_pred_cells.argmax(axis = 0), self.ordering_labels)

        acc = (cell_acc + mol_acc)/ 2

        # Compute contrastive learning loss
        loss_mols = self.criterion(y_pred_mols, ordering_labels)
        loss_cells = self.criterion(y_pred_cells, ordering_labels)

        cl_loss = (loss_mols + loss_cells)/2

        return cl_loss, acc

    def metric_learning_loop(self, y_true, cell_embedding, mol_embedding):
        """Returns average hinge loss from cells2mols and mols2cells for a minibatch."""
        pos_cell_ixs, neg_cell_ixs, perm_y_labels = get_positive_negative_indices_batch(
            y_true, self.index_dict_train, cuda = self.cuda
        )

        # Get positive and negative anchors for cells
        positive_anchors_cells = torch.from_numpy(self.adata[pos_cell_ixs].X.A)
        negative_anchors_cells = torch.from_numpy(self.adata[neg_cell_ixs].X.A)

        if self.cuda:
            positive_anchors_cells = positive_anchors_cells.cuda()
            negative_anchors_cells = negative_anchors_cells.cuda()

        # Get negative anchors for molecules
        permuted_molecule_batch = Batch.from_data_list(
            get_drug_batch(
                torch.from_numpy(perm_y_labels),
                self.name_to_mol,
                self.ix_to_name,
                cuda = self.cuda
            )
        )

        # Compute embeddings
        positive_cell_embeddings = self.model.encode_cell(positive_anchors_cells)
        negative_cell_embeddings = self.model.encode_cell(negative_anchors_cells)
        permuted_molecule_embeddings = self.model.encode_molecule(permuted_molecule_batch)

        # Compute metric learning loss
        # (anchor, positive, negative)
        hinge_cells_anchor = self.hinge_loss(
            cell_embedding, mol_embedding, permuted_molecule_embeddings
        )

        hinge_mols_anchor = self.hinge_loss(
            mol_embedding, positive_cell_embeddings, negative_cell_embeddings
        )

        metric_learning_loss = (hinge_cells_anchor + hinge_mols_anchor)/2

        return metric_learning_loss


    def train_step(self, input_tensor, y_true):
        "A single training step for a minibatch."

        self.model.zero_grad()

        if self.cuda:
            input_tensor = input_tensor.cuda()
            y_true = y_true.cuda()

        # Make batch of molecular graphs
        molecule_batch = Batch.from_data_list(
            get_drug_batch(
                y_true,
                self.name_to_mol,
                self.ix_to_name,
                cuda = self.cuda
            )
        )

        # Compute cell and molecule embeddings
        cell_embedding = self.model.encode_cell(input_tensor.view(self.batch_size, -1).float())
        mol_embedding = self.model.encode_molecule(molecule_batch)

        if self.contrastive:
            cl_loss, train_acc = self.contrastive_learning_loop(mol_embedding, cell_embedding)

            if not metric:
                cl_loss.backward()
                self.optimizer.step()

                results_dict = {
                    'train_loss': {
                        'contrastive_loss': cl_loss.item(),
                        'metric_learning_loss': None
                    },
                    'train_acc': train_acc
                }

                return results_dict

        if self.metric:
            met_loss = self.metric_learning_loop(y_true, cell_embedding, mol_embedding)

            if not self.contrastive:
                met_loss.backward()
                self.optimizer.step()

                results_dict = {
                    'train_loss': {'contrastive_loss': None,'metric_learning_loss': met_loss.item()},
                    'train_acc': None
                }

                return results_dict


        #if self.contrastive and self.metric:
        # both contrastive and metric learning active
        loss = cl_loss + metric_learning_loss

        loss.backward()
        self.optimizer.step()
        results_dict = {
            "train_loss": {
                "contrastive_loss": cl_loss.item(),
                "metric_learning_loss": metric_learning_loss.item(),
            },
            "train_acc": train_acc,
        }

        return results_dict

    @torch.no_grad()
    def val_step(self, input_tensor, y_true):
        """
        """
        #self.model.eval()

        if self.cuda:
            input_tensor = input_tensor.cuda()
            y_true = y_true.cuda()

        # Make batch of molecular graphs
        molecule_batch = Batch.from_data_list(get_drug_batch(y_true, cuda = self.cuda))

        # Compute cell and molecule embeddings
        cell_embedding = self.model.encode_cell(input_tensor.view(self.batch_size, -1).float())
        mol_embedding = self.model.encode_molecule(molecule_batch)

        if self.contrastive:
            cl_loss, test_acc = self.contrastive_learning_loop(mol_embedding, cell_embedding)

            if not metric:
                results_dict = {
                    'test_loss': {'contrastive_loss': cl_loss.item(),'metric_learning_loss': None},
                    'test_acc': test_acc
                }
                return results_dict

        if self.metric:
            met_loss = self.metric_learning_loop(y_true, cell_embedding, mol_embedding)
            if not self.contrastive:
                results_dict = {
                    'test_loss': {'contrastive_loss': None,'metric_learning_loss': met_loss.item()},
                    'test_acc': None
                }
                return results_dict

        #if self.contrastive and self.metric:
        # else: both contrastive and metric learning active
        loss = cl_loss + metric_learning_loss

        results_dict = {
            "test_loss": {
                "contrastive_loss": cl_loss.item(),
                "metric_learning_loss": met_loss.item(),
            },
            "test_acc": test_acc,
        }

        return results_dict

    def train(self)-> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Trains the joint embedding model for n_epochs.
        Returns the train and validation loss and accuracy as dataframes.
        """
        df_train_loss, df_train_acc = pd.DataFrame(), pd.DataFrame()
        df_test_loss, df_test_acc = pd.DataFrame(), pd.DataFrame()

        for epoch in np.arange(self.n_epochs):

            self.model.train()
            for ix, (input_tensor, y_true) in tqdm.tqdm(enumerate(self.train_loader)):

                # Train step
                results_dict_train = self.train_step(input_tensor, y_true)
                df_train_loss = df_train_loss.append(results_dict_train['train_loss'], ignore_index = True)
                df_train_acc = df_train_acc.append(
                    {'train_acc': results_dict_train['train_acc']}, ignore_index = True
                )

                mean_cl = df_train_loss.contrastive_loss.mean()
                mean_ml = df_train_loss.metric_learning_loss.mean()
                mean_acc = df_train_acc.train_acc.mean()
                print('Epoch %d \n'%(epoch+1))
                print('--------------------')
                print('Train contrastive loss: %.3f '%(mean_cl if mean_cl is not np.nan else 0.0))
                print('Train metric learning loss: %.3f '%(mean_ml if mean_ml is not np.nan else 0.0))
                print('Train accuracy: %.3f'%(mean_acc*100 if mean_acc is not np.nan else 0.0))
                print('\n')

            self.model.eval()
            for ix, (input_tensor, y_true) in tqdm.tqdm(enumerate(self.val_loader)):

                # Val step
                results_dict_test = self.val_step(input_tensor, y_true)
                df_test_loss = df_test_loss.append(results_dict_test['test_loss'], ignore_index = True)
                df_test_acc = df_train_acc.append(
                    {'test_acc': results_dict_test['test_acc']}, ignore_index = True
                )

                mean_cl_ = df_test_loss.contrastive_loss.mean()
                mean_ml_ = df_test_loss.metric_learning_loss.mean()
                mean_acc_ = df_test_acc.test_acc.mean()

                print('Val contrastive loss: %.3f '%(mean_cl_ if mean_cl_ is not np.nan else 0.0))
                print('Val metric learning loss: %.3f '%(mean_ml_ if mean_ml_ is not np.nan else 0.0))
                print('Validation accuracy: %.3f'%(mean_acc_*100 if mean_acc_ is not np.nan else 0.0))
                print('\n')

            # SAVE MODEL
            if self.model_dir is not None:
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)

                if self.model_name is not None:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.model_dir, self.model_name + '_' + str(epoch) + '.pt')
                    )
                else:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.model_dir, 'model' + '_' + str(epoch) + '.pt')
                    )

        # Summarize results
        df_train_logs = pd.concat([df_train_loss, df_train_acc], axis = 1)
        df_test_logs = pd.concat([df_test_loss, df_test_acc], axis = 1)

        epoch_indicator_train = np.concatenate(
            [np.repeat(epoch, self.n_train_batches) for epoch in np.arange(1, n_epochs+1)]
        )

        epoch_indicator_test = np.concatenate(
            [np.repeat(epoch, self.n_test_batches) for epoch in np.arange(1, n_epochs +1)]
        )

        df_train_logs['epoch'] = epoch_indicator_train
        df_test_logs['epoch'] = epoch_indicator_test

        # Set logs as attributes
        self.train_logs = df_train_logs
        self.test_logs = df_test_logs

        df_train_agg = df_train_logs.groupby('epoch').mean()
        df_test_agg = df_test_logs.groupby('epoch').mean()

        self.best_model_ix = df_test_agg.test_acc.argmax()

        return df_train_agg, df_test_agg


def train_vae(
	model:nn.Module,
	input_tensor,
	optimizer)->torch.tensor:
    """
    Forward-backward pass of a VAE model.
    """
    model.zero_grad()
    reconstructed, mu, log_var = model(input_tensor)
    loss = model.loss(reconstructed, input_tensor, mu, log_var)

    # Backprop error
    loss.backward()
    # Update weights
    optimizer.step()

    return loss


def validate_vae(
	model:nn.Module,
	input_tensor,
	optimizer
    )->torch.tensor:

    reconstructed, mu, log_var = model(input_tensor)
    loss = model.loss(reconstructed, input_tensor, mu, log_var)
    return loss.mean()

def vae_trainer(
    n_epochs:int,
    train_loader,
    val_loader,
    model:nn.Module,
    optimizer,
    conditional_gen = False,
    logs_per_epoch = 5):

    """
    Wrapper function to train a VAE model for n_epochs.

    Params
    ------
	n_epochs(int)
		Number of epochs to run the model.

	train_loader()
		Dataloader for training set.

	val_loader()
		Dataloader for validation set.

	model(nn.Module)
		VAE model.


    Returns
    -------
    train_loss_vector
    val_loss_vector
    """

    batch_size = train_loader.batch_size
    print_every = np.floor(
        train_loader.dataset.__len__() / batch_size / logs_per_epoch
        )

    train_loss_vector = []
    val_loss_vector = np.empty(shape = n_epochs)

    cuda = torch.cuda.is_available()

    if cuda:
        device = try_gpu()
        torch.cuda.set_device(device)
        model = model.to(device)

    for epoch in np.arange(n_epochs):

        running_loss = 0.0

        # TRAINING LOOP

        for ix, data in enumerate(tqdm.tqdm(train_loader)):

            # Reshape minibatch
            input_tensor = data.view(batch_size, -1).float()

            if cuda:
                input_tensor = input_tensor.cuda(device = device)

            train_loss = train_vae(model, input_tensor, optimizer, batch_size)

            running_loss +=train_loss.item()

            # Print loss
            if ix % print_every == print_every -1 : # ix starts at 0
                print('[%d, %5d] VAE loss : %.3f' %
                    (epoch + 1, ix +1, running_loss / print_every)
                    )

                train_loss_vector.append(running_loss / print_every)

                # Restart loss
                running_loss = 0.0

        # VALIDATION LOOP
        for ix, data in enumerate(tqdm.tqdm(val_loader)):
            validation_loss = []

            # Reshape minibatch
            input_tensor = data.view(batch_size, -1).float()

            if cuda:
                input_tensor = input_tensor.cuda(device = device)

            val_loss = validate_vae(model, input_tensor, optimizer)

            validation_loss.append(val_loss)

        mean_val_loss = torch.tensor(validation_loss).mean()
        val_loss_vector[epoch] = mean_val_loss

        print('Val. loss %.3f'% mean_val_loss)

    print('Finished training.')

    return train_loss_vector, val_loss_vector




def try_gpu(i=0):
    """
    Return gpu(i) if exists, otherwise return cpu().

    Extracted from https://github.com/d2l-ai/d2l-en/blob/master/d2l/torch.py
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def initialize_network_weights(
    net:nn.Module, method = 'kaiming', seed = 4
    )-> nn.Module:
    """
    Initialize fully connected and convolutional layers' weights
    using the Kaiming (He) or Xavier method.
    This method is recommended for ReLU / SELU based activations.
    """

    torch.manual_seed(seed)

    if method == 'kaiming':
        for module in net.modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(module.weight)
                try:
                    nn.init.uniform_(module.bias)
                except:
                    pass

            elif isinstance(module, (nn.GRU, nn.LSTM)):
                for name, param in module.named_parameters():
                    if 'bias' in name :
                        nn.init.uniform_(param)
                    elif  'weight' in name:
                        nn.init.kaiming_uniform_(param)
                    else:
                        pass

            else:
                pass


    elif method == 'xavier':
        for module in net.modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                try:
                    nn.init.uniform_(module.bias)
                except:
                    pass

            elif isinstance(module, (nn.GRU, nn.LSTM)):
                for name, param in module.named_parameters():
                    if 'bias' in name :
                        nn.init.uniform_(param)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    else:
                        pass

            else:
                pass

    elif method == 'xavier_normal':
        for module in net.modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(module.weight)
                try:
                    nn.init.uniform_(module.bias)
                except:
                    pass

            elif isinstance(module, (nn.GRU, nn.LSTM)):
                for name, param in module.named_parameters():
                    if 'bias' in name :
                        nn.init.uniform_(param)
                    elif  'weight' in name:
                        nn.init.xavier_normal_(param)
                    else:
                        pass

            else:
                pass


    else:
        raiseNameError('Method not found. Only valid for `kaiming` or `xavier` initialization.')

    return net



class adata_torch_dataset(Dataset):
    "Convert an adata to a torch.Dataset"
    def __init__(
        self, data= None, transform = None, supervised = False,
        target_col = None, g_cols = None, multilabel = False)->torch.tensor:
        """
        Base class for a single cell dataset in .h5ad, i.e. AnnData format
        This object enables building models in pytorch.
        It currently supports unsupervised (matrix factorization / autoencoder)
        and general supervised (classification/regression) models.

        Note: the ToTensor() transform can end up normalizing count matrices.
        See more on: https://pytorch.org/docs/0.2.0/_modules/torchvision/transforms.html#ToTensor

        Params
        ------
        data (ad.AnnData)
            AnnDataset containing the count matrix in the data.X object.

        transform (torchvision.transforms, default= None)
            A torchvision.transforms-type transformation, e.g. ToTensor()

        supervised (bool, default = False)
            Indicator variable for supervised models.

        target_col (string/array-like, default = None)
            If running a supervised model, target_col should be a column
            or set of columns in the adata.obs dataframe.
            When running a binary or multiclass classifier, the labels
            should be in a single column in a int64 format.
            I repeat, even if running a multiclass classifier, do not specify
            the columns as one-hot encoded. The one-hot encoded vector
            will be specified in the classifier model. The reason is that,
            nn.CrossEntropyLoss() and the more numerically stable nn.NLLLoss()
            takes the true labels as input in integer form (e.g. 1,2,3),
            not in one-hot encoded version (e.g. [1, 0, 0], [0, 1, 0], [0, 0, 1]).

            When running a multilabel classifier (multiple categorical columns,
            e.g ´cell_type´ and `behavior`), specify the columns as a **list**.

            In this case, we will use the nn.BCELoss() using the one-hot encoded
            labels. This is akin to a multi-output classification.

        g_cols(list, default = None)
            List of columns in an auxiliary variable for conditional generation.

        multilabel (bool, default = False)
            Indicator variable to specify a multilabel classifier dataset.

        Returns
        -------
        data_point(torch.tensor)
            A single datapoint (row) of the dataset in torch.tensor format.

        target(torch.tensor)
            If running supervised model, the "y" or target label to be predicted.
        """

        self.data = data # This is the h5ad / AnnData

        self.supervised = supervised
        self.target_col = target_col
        self.transform = transform

        from scipy import sparse
        # Indicator of data being in sparse matrix format.
        self.sparse = sparse.isspmatrix(data.X)

        self.multilabel = multilabel
        self.g_cols = g_cols

        if self.multilabel:
            from sklearn.preprocessing import OneHotEncoder
            # Initialize one hot encoder
            enc = OneHotEncoder(sparse = False)
            self.one_hot_encoder = enc

            n_categories = len(self.target_col)

            # Extract target data
            y_data = self.data.obs[self.target_col].values.astype(str).reshape(-1, n_categories)

            # Build one hot encoder
            self.one_hot_encoder.fit(y_data)

            # Get one-hot matrix and save as attribute
            self.multilabel_codes = self.one_hot_encoder.transform(y_data)

    def __len__(self):
        return self.data.n_obs

    def __getitem__(self, ix):

        if type(ix) == torch.Tensor:
            ix = ix.tolist()

        # Get a single row of dataset and convert to numpy array if needed
        if self.sparse:
            data_point = self.data[ix, :].X.A.astype(np.float64)

        else:
            data_point = self.data[ix, :].X.astype(np.float64)

        # if self.conv:
        #     image = image.reshape(1, self.res, self.res)

        if self.transform is not None:
            data_point = self.transform(data_point)

        # Get all columns for multilabel classification codes
        if self.supervised and self.multilabel:
            target = self.multilabel_codes[ix, :]
            #target = self.transform(target)
            return data_point, target


        # Softmax-classification plus conditional generator
        elif self.supervised and self.g_cols is not None:
            target = self.data.obs.iloc[ix][self.target_col]

            # Extract vector of for conditional generation
            g_vars = self.data.obs.iloc[ix][self.g_cols].values.astype(np.float32)
            return data_point, target, torch.from_numpy(g_vars)#.view(1,1,-1)

        # Get categorical labels for multiclass or binary classification
        # or single column for regression (haven't implemented multioutput reg.)
        elif self.supervised:
            target  = self.data.obs.iloc[ix][self.target_col]
            #target = self.transform(target)
            return data_point, target

        # Fallback to unsupervised case.
        else:
            return data_point

    def codes_to_cat_labels(self, one_hot_labels):
        """
        Returns categorical classes from labels in one-hot format.

        Params
        ------
        one_hot_labels (array-like)
            Labels of (a potentially new or predicted) dataset
            in one-hot-encoded format.

        Returns
        -------
        cat_labels(array-like, or list of array-like)
            Categorical labels of the one-hot encoded input.

        """

        cat_labels = self.one_hot_encoder.inverse_transform(one_hot_labels)

        return cat_labels



# Make curried to allow kwarg calls on tz.pipe()
@tz.curry
def get_count_stats(
	adata,
	mt_prefix = None,
	ribo_prefix = None
    )-> ad.AnnData:

	"""
	Returns an AnnData with extra columns in its `obs` object
	for the number of counts per cell `n_counts` (and log10 (counts) ),
	abd the number of expressed genes in each cell `n_genes`.
	Additionally it can get the fraction of mitochondrial and ribosomal
	genes if prefixes are provided.

	TODO: Add filtering functionality

	Params
	------
	adata (ad.AnnData)
		Input dataset in AnnData format. It should contain a count matrix
		(cells x genes) as the `.X` object in the AnnData.

	mt_prefix (str, default = 'MT-'):
		Prefix to match mitochondrial genes.
		For human the prefix is `MT-` and for the mouse is `mt-`.

	ribo_prefix(default=None)
		For human the prefixes are ('RPS', 'RPL').

	Returns
	-------
	adata (ad.AnnData)
		AnnData with columns in the `.obs` dataframe corresponding to
		count stats.
	"""

	if not sparse.isspmatrix_csr(adata.X):
		adata.X = sparse.csr_matrix(adata.X)

	# Number of transcripts per cell
	adata.obs['n_counts'] = adata.X.sum(axis = 1)
	adata.obs['log_counts'] = np.log10(adata.obs.n_counts)

	# Number of genes with more than one count
	adata.obs['n_genes'] = (adata.X > 0).sum(axis = 1)

	# Get mitochondrial and ribosomal genes
	if mt_prefix is not None:
		# Use string methods from pandas to make bool array
		mito_genes = adata.var.gene_name.str.startswith(mt_prefix)

		if mito_genes.sum()> 1:

			# Compute the fraction of mitochondrial genes
			adata.obs["frac_mito"] = adata[:, mito_genes].X.A.sum(axis =1) / adata.obs.n_counts

	if ribo_prefix is not None:

		if isinstance(ribo_prefix, (list, tuple)):
			# Initialize bool array
			ribo_genes = np.zeros(adata.n_vars, dtype = bool)

			# Loop through each prefix and flip to True
			# where we get a match.
			for prefix in ribo_prefix:
				ribo_genes_tmp = adata.var.gene_name.str.startswith(prefix)
				ribo_genes +=ribo_genes_tmp

			if ribo_genes.sum()> 1:
				adata.obs["frac_ribo"] = adata[:, ribo_genes].X.A.sum(axis =1) / adata.obs.n_counts

	return adata


# Curry to be able to add arguments in a tz.pipe
@tz.curry
def lognorm_cells(
	adata_,
	scaling_factor = 1e4,
	log = True)-> ad.AnnData:

	"""
	Cell count normalization as in scanpy.pp.normalize_total.
	Expects count matrix in sparse.csr_matrix format.

	Each gene's expression value in a given cell is given by :

	g_i = \mathrm{ln} ( \frac{g_i \times \beta }{\sum g_i} + 1 )

	where β is the scaling factor.

	Params
	------
	adata_ (ad.AnnData):
		Count matrix with cell and gene annotations.

	scaling_factor(float, default = 1e4)
		Factor to scale gene counts to represent the counts in
		the cell. If scaling_factor =1e6, the values will
		represent counts per million.

	log (bool, default = True)
		Optional argument to allow for returning the scaled cells
		without normalizing.

	Returns
	-------
	adata (ad.AnnData):
		Anndata with normalized and log transformed count matrix.
	"""

	# Make a copy because normalization is done in-place
	adata = adata_.copy()

	if not sparse.isspmatrix_csr(adata.X):
		adata.X = sparse.csr_matrix(adata.X)

	# Get total counts per cell from `obs` df
	if 'n_counts' in adata.obs.columns:
		counts = adata.obs.n_counts.values

	else:
		counts = adata.X.sum(axis = 1).flatten()

	# Convert to numpy matrix to array to be able to flatten
	scaled_counts = np.array(counts).flatten() / scaling_factor

	# Efficient normalization in-place for sparse matrix
	sparsefuncs.inplace_csr_row_scale(adata.X, 1/scaled_counts)

	# Call the log1p() method on the csr_matrix
	if log:

		adata.X = adata.X.log1p()

	return adata

# Curry to enable adding arguments in a tz.pipe()
@tz.curry
def cv_filter(
	adata,
	min_mean = 0.025,
	min_cv= 1,
	return_highly_variable = False)-> ad.AnnData:

	"""
	Performs the Coefficient of Variation filtering according
	to the Poisson / Binomial counting statistics. The model assumes
	the coefficient of variation per gene is given by :

	\mathrm{log} (CV) \approx - \frac{1}{2}\mathrm{log} (\mu) + \epsilon


	The values will be computed assuming a normalized and
	log-scaled count matrix.

	Params
	------
	min_mean (float, default = 0.025).
		Lower bound cutoff for the mean of the gene feature.

	min_cv (float, default = None)
		Lower bound for the coefficient of variation of the
		gene feature. Recommended value 1.

	return_highly_variable(bool, default = True)
		Whether to return an AnnData with the columns corresponding
		to only the highly variable genes.
		Note: even when running with `return_highly_variable=False`
		the function will return genes only with nonzero mean and
		nonzero variance, i.e. it will discard those genes.

	Returns
	-------
	adata_filt (ad.AnnData)
		AnnData with coeffifient of variation stats on the `var`
		dataframe.
	"""

	# Calculate mean and variance across cells
	mean, var = sparsefuncs.mean_variance_axis(adata.X, axis = 0)

	# Check if there are nonzero values for the mean or variance
	ix_nonzero = list(set(np.nonzero(mean)[0]).intersection(set(np.nonzero(var)[0])))

	if len(ix_nonzero) > 0:
		# Use numpy-like filtering to select only genes with nonzero entries
		adata = adata[:, ix_nonzero].copy()

		# Recompute mean and variance of genes across cells
		mean, var = sparsefuncs.mean_variance_axis(adata.X, axis = 0)

		# Get nonzero mean indices
		nz = np.nonzero(mean)

		# Check that there are only nonzero mean values
		assert adata.n_vars == nz[0].shape[0]
	else:
		print ('Only zero mean or variance values for the genes in the count matrix.')
		return None

	std_dev = np.sqrt(var)

	# Element-wise coefficient of variation
	cv = std_dev / mean
	log_cv = np.log(cv)
	log_mean = np.log(mean)

	df_gene_stats = pd.DataFrame(
	    np.vstack([mean, log_mean, var, cv, log_cv]).T,
	    columns=["mean", "log_mean", "var", "cv", "log_cv"],
	    index = adata.var.index
	)

	new_adata_var = pd.concat(
	    [adata.var, df_gene_stats],
	    axis = 1
	)

	adata.var = new_adata_var

	slope, intercept, r, pval, stderr = st.linregress(log_mean, log_cv)
	poisson_prediction_cv = slope*log_mean + intercept

	# Binary array of highly variable genes
	gene_sel = log_cv > poisson_prediction_cv

	adata.var['highly_variable'] = gene_sel.astype(int)

	if min_mean and min_cv is not None:
		adata_filt = adata[:,((adata.var.highly_variable == True)&\
								(adata.var['mean'] > min_mean)&\
								(adata.var['cv'] > min_cv))].copy()
	else:
		adata_filt = adata[:, adata.var.highly_variable == True].copy()

	if return_highly_variable:
		return adata_filt

	else:
		return adata

@tz.curry
def sample_to_name(sample_id, eliminate_parens = False, eliminate_hcl = True):
    """
    Returns processed version of sample id.

    The best way to match is to try to match annotations in lowercase.
    """

    if 'ethylisothiourea sulfate' in sample_id:
        return 'Methylisothiourea sulfate'

    # Eliminate "_CD3" overhang
    s = sample_id.split('_CD3')[0]
    # Trim spaces
    s = s.strip()
    if eliminate_parens:
        s = s.split('(')[0]
        # Trim spaces
        s = s.strip()

    # Remove HCl overhang
    if eliminate_hcl:
        s = s.split(' HCl')[0]
        s = s.strip()

    return s

def get_ix_nondup(labels):
    """
    Returns a binary array given a set of categorical labels.
    Used for filtering out duplicated labels in a minibatch when using
    the n-way cross entropy ranking loss (online ranking) for joint embedding
    training.

    Example
    -------
    x = np.random.randint(0, 10, 10)
    x
    >>> array([4, 8, 8, 7, 9, 8, 8, 3, 8, 9])

    maskr = get_ix_nondup(x)
    maskr
    >>> array([ True,  True, False,  True,  True, False, False,  True, False,
       False])

    x[maskr]
    >>> array([4, 8, 7, 9, 3])

    """

    if isinstance(labels, torch.Tensor):
        is_tensor = True
        dev = labels.device

        if labels.device.type == 'cuda':
            labels = labels.cpu()
        if labels.requires_grad:
            labels = labels.detach()
        labels = labels.numpy()

    else:
        is_tensor =False

    # Gen binary array of non-duplicated labels
    mask = ~pd.Series(labels).duplicated().values

    # Check that no duplicated values remain
    assert len(np.nonzero(pd.Series(labels[mask]).duplicated().values)[0]) == 0

    if is_tensor:
        mask = (torch.from_numpy(mask)).to(dev)
    return mask


def get_acc_df_cell2mol(df_cells, name_to_target, name_to_class, k=1):
    """
    Returns a top-k accuracy dataframe per sample.

    Params
    ------
    df_cells (pd.DataFrame)
        Cell dataframe (from adata or df_embedding) that contains the top-k accuracy.

    """
    pred_df = (
        df_cells.groupby(["drug_name", "top" + str(k) + "_accuracy"]).size().unstack().fillna(0)
    )

    pred_arr = pred_df.values / pred_df.values.sum(axis=1).reshape(-1, 1) * 100

    perc_pred_df = pd.DataFrame(pred_arr, index=pred_df.index, columns=pred_df.columns)

    accuracy_df = (
        perc_pred_df.sort_values(by=0, ascending=True)[1].to_frame().reset_index()
    )

    accuracy_df.rename(columns = {1:'top@'+ str(k) + '_accuracy'}, inplace = True)

    accuracy_df['target'] = accuracy_df['drug_name'].map(name_to_target)
    accuracy_df['drug_class'] = accuracy_df['drug_name'].map(name_to_class)
    accuracy_df['sample_class'] = accuracy_df['drug_name'] + '_' + accuracy_df['drug_class'].str.lower()

    return accuracy_df



def freedman_diaconis_rule(arr):
	"""
	Calculates the number of bins for a histogram using the Freedman-Diaconis Rule.

	Modified from https://github.com/justinbois/bebi103/blob/master/bebi103/viz.py

	"""
	h = 2* (np.percentile(arr, q=75) - np.percentile(arr, q = 25))/ np.cbrt(len(arr))

	if h == 0.0:
		n_bins = 3
	else:
		n_bins = int(np.ceil(arr.max() - arr.min()) / h)

	return n_bins

def l1_norm(arr1, arr2):
	'''
	Compute the L1-norm between two histograms.
	It uses the Freedman-Diaconis criterion to determine the number of bins.

	It will be positive if the mean(arr2) > mean(arr1) following the convention
	from PopAlign.

	Modified from https://github.com/thomsonlab/popalign/blob/master/popalign/popalign.py

	Parameters
	----------
	arr1 (array-like)
		Distribution of gene for population 1.
	arr2 (array-like)
		Distribution of gene for population 2.

	Returns
	-------
	l1_score(float)
		L1 norm between normalized histograms of gene distributions.

	Example
	-------
	import numpy as np
	from sc_utils import sc

	x = np.random.normal(loc = 0, size = 100)
	y = np.random.normal(loc = 3, size = 100)

	sc.l1_norm(x, y)
	>>>1.46
	'''

	if len(arr1) == len(arr2):
		nbins = freedman_diaconis_rule(arr1)

	else:
		nbins_1 = freedman_diaconis_rule(arr1)
		nbins_2 = freedman_diaconis_rule(arr2)

		nbins = int((nbins_1 + nbins_2)/2)


	max1, max2 = np.max(arr1), np.max(arr2) # get max values from the two subpopulations
	max_ = max(max1,max2) # get max value to define histogram range
	if max_ == 0:
		return 0
	else:
		b1, be1 = np.histogram(arr1, bins=nbins, range=(0,max_)) # compute histogram bars
		b2, be2 = np.histogram(arr2, bins=nbins, range=(0,max_)) # compute histogram bars
		b1 = b1/len(arr1) # scale bin values
		b2 = b2/len(arr2) # scale bin values
		if arr1.mean()>=arr2.mean(): # sign l1-norm value based on mean difference
			l1_score = -np.linalg.norm(b1-b2, ord=1)
			return l1_score
		else:
			l1_score = np.linalg.norm(b1-b2, ord=1)
			return l1_score

def ecdf(x)->(np.array, np.array):
    '''
    Returns ECDF of a 1-D array.

    Params
    ------

    x(array or list)
        Input array, distribution of a random variable.

    Returns
    -------
    x_sorted : sorted x array.
    ecdf : array containing the ECDF of x.
    '''
    n = len (x)
    x_sorted = np.sort(x)
    ecdf = np.linspace(0, 1, len(x_sorted))
    return x_sorted, ecdf

def get_stats(distro_x, distro_y):
    """
    Returns statistics from testing that `distro_x` takes larger values that `distro_y`.

    Returns
    -------
    ks, pval_ks, l1_score
    """
    # For a given value of the data, ECDF of sample 1 takes values less than sample 2
    ks, pval_ks = st.ks_2samp(distro_x, distro_y, alternative="less")

    # Positive if mean(distro_x) > mean(distro_y)
    l1_score = l1_norm(distro_y, distro_x)

    return ks, pval_ks, l1_score


# def get_ix_drug(drugbank, drug_name, verbose = False)->np.ndarray:
#     """Returns index of molecule in drugbank."""
#     try:
#         ix_ = drugbank[drugbank['drug_name'] ==drug_name].index.values[0]
#
#     except :
#         ix_ = drugbank[drugbank['drug_name'].str.contains(drug_name)].index.values[0]
#     if verbose:
#         print('Getting drugbank index for :%s'%drugbank.iloc[ix_]['drug_name'] )
#     return ix_
#
# def get_ix_cells(adata, drug_name, verbose = False)->np.ndarray:
#     """Returns index of cells perturbed by `drug_name` in adata"""
#     try:
#         ix_cells = adata[adata.obs['drug_name']==drug_name].obs.index.values
#     except:
#         ix_cells = adata[adata.obs['drug_name'].str.contains(drug_name)].obs.index.values
#     if verbose :
#         print('Getting adata cell indices for :%s'%adata[ix_cells[0]].obs['drug_name'].values[0] )
#     return ix_cells
#
#
# def get_cosine_distribution_drug(drugbank, adata, query_drug_name, perturb_drug_name, cosine_arr, verbose = False):
#     """
#     Returns the cosine similarity distribution for the cells perturbed with
#     `perturb_drug_name` (indexed in adata), and a molecule `query_drug_name` (indexed in drugbank).
#     If `query_drug_name` and `perturb_drug_name` are the same, it returns the
#     cosine similarity of the given molecule against the cells perturbed by it.
#
#     Note: Expects cosine_arr to be of shape (n_mols, n_cells)
#
#     Params
#     ------
#     query_drug_name (str)
#         Name of the drug to query against.
#
#     perturb_drug_name (str)
#         Name of the drug that perturbed the cells to retrieve.
#
#     Returns
#     -------
#     cosine_similarity_distribution
#
#     Note:Expects cosine_arr to be shape (mols, cells)
#     """
#     ix_drug = get_ix_drug(drugbank, query_drug_name, verbose)
#     ix_cells = get_ix_cells(adata, perturb_drug_name, verbose)
#     cosine_similarity_distribution = cosine_arr[ix_drug, ix_cells]
#
#     return cosine_similarity_distribution
#
#
# def get_cosine_drug_one_vs_all(
#     drugbank,
#     adata,
#     drug_name,
#     cosine_arr,
#     verbose = False
# ):
#     """
#     Returns the cosine similarity distribution of a molecule with cells perturbed by it,
#     and the cos. sim. dist. of the molecule with cells coming from other samples.
#
#     Expects cosine_arr to be of shape (n_mols, n_cells)
#     """
#     n_mols, n_cells = cosine_arr.shape
#     ix_drug, ix_cells = get_ix_drug(drugbank, drug_name), get_ix_cells(adata, drug_name)
#
#     # Get cosine similarity distribution of a drug with itself
#     cosine_cells_drug = cosine_arr[ix_drug, ix_cells]
#
#     # Get the indices of all perturbed with other molecules but `drug_name`
#     other_cells_ix = np.array(list(set(np.arange(n_cells)) - set(ix_cells)))
#     cosine_others = cosine_arr[ix_drug, other_cells_ix]
#     return cosine_cells_drug, cosine_others
#
#
# def get_cosine_distribution_df(
#     drug_name,
#     drugbank,
#     adata,
#     cosine_arr,
#     drugbank_to_selleck,
#     n_top = 10000,
#     cols_viz = ['sample_class', 'target', 'drug_class', 'pn', 'drug_name'],
#     filter_by = 'sample_class',
#     n_cells_filter = 5,
#     anti = False,
#     return_acc_only = False
# )->pd.DataFrame:
#     """
#     Returns an annotated dataframe of the cells with highest cosine similarity to
#     a query molecule `drug_name`.
#
#     Notes: Assumes an adata and drugbank (dataframe) exist and that their indices
#     have been reset.
#
#     Params
#     ------
#     drug_name (str)
#     n_top (int, default= 10000,)
#         Number of cells with highest similarity to retrieve.
#
#     cols_viz (list, default= ['sample_class', 'target', 'drug_class', 'pn'], )
#         Which columns to use for visualization. Cols have to be in adata.
#
#     filter_by (str, default= 'sample_class')
#         Column to filter noise cells.
#
#     n_cells_filter (int, default = 5,)
#         Lower bound threshold above to which filter noise cells, i.e.
#         if a sample has less than `n_cells_filter` in the top cells,
#         that sample won't be in the final visualization.
#
#     anti (bool = False)
#         Whether to reverse order, get cells with lowest cosine similarity.
#
#     Returns
#     -------
#     df_viz
#     """
#     ix_ = get_ix_drug(drugbank, drug_name, verbose = False)
#
#     name_of_drug = drugbank.iloc[ix_]['drug_name']
#     name_of_drug = drugbank_to_selleck[name_of_drug]
#     print('Returning predictions for %s'%name_of_drug)
#
#     # Reverse order : get cells with lowest cosine sim
#     if anti:
#         ix_top_cells = np.argsort(cosine_arr[ix_])[:n_top]
#     else:
#         ix_top_cells = np.argsort(cosine_arr[ix_])[::-1][:n_top]
#
#     # Make a dataframe containing the cosine similarities and cols_viz
#     df_viz = adata[ix_top_cells].obs[cols_viz]
#
#     #try:
#     sample_val_counts = df_viz.drug_name.value_counts()
#     if name_of_drug in sample_val_counts.index.values:
#         n_correct = sample_val_counts[name_of_drug]
#
#         acc = n_correct / sample_val_counts.sum() * 100
#         print('Accuracy: %.3f'%acc)
#         if return_acc_only:
#             return acc
#         else:
#             pass
#     else:
#         print('Accuracy: 0')
#         if return_acc_only:
#             return 0
#     #except:
#    #     pass
#
#     df_viz['cosine_similarity'] = cosine_arr[ix_][ix_top_cells]
#     val_counts = df_viz[filter_by].value_counts()
#     samples_in = val_counts[val_counts > n_cells_filter].index.values
#
#     return df_viz[df_viz[filter_by].isin(samples_in)]
#




class EvaluateCrossRetrieval:
    """
    Base class to evaluate cross modality-retrieval a joint embedding
    of cells and molecules.

    It is designed for evaluation in a test set, comprised of a tuple
    (test molecules, test cells). Nevertheless, one can pass the full datasets
    and still leverage the functionalities.
    """
    def __init__(
        self,
        df_drugs,
        adata,
        model,
        model_type = 'nn',
        dataset = 'thomsonlab',
        drugs_col_name = 'sample_id'
    ):
        """
        Params
        ------
        df_drugs (pd.DataFrame)
            Annotated version of the drugs in the test set.
            It must ideally have the following in its columns:
            ['drug_class', 'target', 'drug_name']

        adata(pd.DataFrame)
            Test adata containing count matrix as .X and projected cells in its
            `obs.` dataframe.

        model (nn.Module)
            Joint embedding model. to-do: extend functionality for CCA or other models.

        model_type(str, default = 'nn')
            Running neural net or CCA model.

        dataset(str, default = 'thomsonlab')
            Sets the formmatting options for a specific dataset.

        drugs_col_name(str, default = 'sample_id')
            If there's a specific column name for the name of drugs in the df_drugs dataset.

        """

        self.cuda = torch.cuda.is_available()

        # Format column names
        if 'drug_name' not in adata.obs.columns:
            if dataset == 'thomsonlab':
                adata.obs['drug_name'] = adata.obs['sample_id'].apply(
                    lambda x: sample_to_name(str(x), eliminate_parens = True, eliminate_hcl = False)
                ).str.lower()

            elif dataset == 'sciplex':
                adata.obs['drug_name'] = adata.obs['product_name'].apply(
                    lambda x: sample_to_name(str(x), eliminate_parens = True, eliminate_hcl = False)
                ).str.lower()

        if 'drug_name' not in df_drugs:
            df_drugs['drug_name'] = df_drugs[drugs_col_name].apply(
                lambda x: sample_to_name(str(x), eliminate_parens = True, eliminate_hcl = False)
            ).str.lower()

        df_drugs_test = df_drugs[df_drugs.drug_name.isin(adata.obs.drug_name.unique())]

        if dataset == 'sciplex':
            df_drugs_test.drop_duplicates(subset = ['drug_name'], inplace = True)

        # Check drugs in both datasets coincide
        #assert len(set(adata.obs.drug_name.unique()) - set(df_drugs_test.drug_name.unique())) == 0, 'Drugs in both datasets do not coincide'

        # We will use numpy-indexing so reset them
        adata.obs.reset_index(drop = True, inplace = True)
        df_drugs_test.reset_index(drop = True, inplace = True)

        # Assign an index to each drug.
        codes, unique_drugs = np.arange(len(df_drugs_test)), df_drugs_test.drug_name.values #pd.factorize(df_drugs_test['drug_name'])

        # Make sure we only have unique drugs
        assert len(unique_drugs) == df_drugs_test.drug_name.unique().shape[0]

        df_drugs_test['sample_code'] = codes

        self.drugbank = df_drugs_test
        self.adata = adata
        self.model = model

        self.ix_to_name = dict(zip(codes, unique_drugs))
        #dict(df_drugs_test[['sample_code', 'drug_name']].values)
        self.name_to_ix = dict(zip(unique_drugs, codes))
        #{val:key for key,val in self.ix_to_name.items()}

        self.sample_counts = adata.obs.drug_name.value_counts()
        self.sample_counts_idx = {
            self.name_to_ix[sample]: self.sample_counts[sample] \
            for sample in self.sample_counts.keys()
        }

        self.test_drugs = self.drugbank.drug_name.values

        # Make drug target and drug class annotation dictionaries
        if dataset == 'sciplex':
            self.name_to_target = dict(adata.obs[['drug_name', 'target']].values)
        else:# thomsonlab
            self.drugbank.rename(columns = {'Target': 'target'}, inplace = True)
            self.name_to_target = dict(df_drugs_test[['drug_name', 'target']].values)
            self.name_to_class = dict(df_drugs_test[['drug_name','drug_class']].values)

        self.test_drugs_ixs = [self.name_to_ix[drug] for drug in self.test_drugs]

        # For each cell, get its perturbation's index
        self.ix_samples_cell = np.array(
            [self.name_to_ix[drug] for drug in self.adata.obs['drug_name'].values]
        )

        # Assign some colormaps for plotting
        self.colormaps = {
            'drug_class': 'Blues_r',
            'within_class_acc': 'Blues_r',
            'Target': 'Oranges_r', 'target': 'Oranges_r',
            'pn': 'Greens_r',
            'pathway': 'Purples_r'
            }

    def eval_pipeline(
        self,
        plot = False,
        mode = 'cosine',
        project_mols = True,
        n_cores = 2
        ):
        """
        Runs all evaluation
        """

        self.compute_cosine_arr(
            return_ = False, project_mols = project_mols, n_dims = 64
        )

        # Saves mol2cell accuracies in self.m2c_acc and in self.drugbank
        self.eval_mol2cell_accuracy(mode= mode, return_ = False)

        # Saves results in self.df_c2m for top5 accuracy
        self.eval_cell2mol_accuracy()
        self.get_acc_df_cell2mol()

        # Run KS tests
        self.run_ks_one_vs_all(n_cores)

        # Run mol2cell above mean
        #self.

        # Plot results !
        if plot:
            pass

    def eval_summary(self):
        pass

    def get_ix_drug(self, drug_name):
        return self.name_to_ix.get(drug_name, 'None')

    def get_ix_cells(self, drug_name, verbose = False):
        try:
            ix_cells = self.adata[self.adata.obs['drug_name']==drug_name].obs.index.values
        except:
            ix_cells = self.adata[self.adata.obs['drug_name'].str.contains(drug_name)].obs.index.values
        if verbose :
            print('Getting adata cell indices for :%s'%adata[ix_cells[0]].obs['drug_name'].values[0] )
        return ix_cells.astype(int)

    def compute_mol_from_smiles(self):
        self.drugbank['mol'] = self.drugbank.SMILES.apply(
            Chem.MolFromSmiles
        )

        self.name_to_mol = dict(self.drugbank[['drug_name', 'mol']].values)

    @torch.no_grad()
    def project_molecules(self):
        """
        Computes molecule embeddings in self.drugbank df.
        """
        #Get Rdkit mols in place
        self.compute_mol_from_smiles()

        labels_tensor = torch.arange(len(self.drugbank))

        drugs_tensor = get_drug_batch(
            labels_tensor,
            self.name_to_mol,
            self.ix_to_name,
            cuda = self.cuda
        )

        self.model.eval()

        mol_embedding = self.model.molecule_encoder.project(
            Batch.from_data_list(drugs_tensor)
        )

        if self.cuda: # bring to CPU
            mol_embedding=mol_embedding.cpu().numpy()
        else:
            mol_embedding = mol_embedding.numpy()

        return mol_embedding

    def compute_cosine_arr(self, return_ = False, project_mols = False, n_dims = 64):
        """
        Computes cosine array. It stores an output array
        """
        if project_mols:
            mol_embedding = self.project_molecules()
        else:
            mol_embedding = self.drugbank[['dim_' + str(i) for i in range(1,n_dims +1)]].values

        cell_embedding = self.adata.obs[['dim_' + str(i) for i in range(1, n_dims+1)]].values
        # Normalize to make row vectors
        mol_embedding_norm  = mol_embedding / np.linalg.norm(mol_embedding, axis = 1).reshape(-1,1)
        cell_embedding_norm = cell_embedding / np.linalg.norm(cell_embedding, axis = 1).reshape(-1,1)

        # Compute cosine similarity, shape (molecules, cells)
        cosine_arr = np.matmul(mol_embedding_norm, cell_embedding_norm.T)
        #print('Shape of cosine similarity array: {0}'.format(cosine_arr.shape))
        self.cosine_arr = cosine_arr

        if return_:
            return cosine_arr

    def compute_dist_arr(self):
        #self.D
        #self.top_ixs_l2
        pass

    def get_top_ixs(self, data_type = 'mols', mode = 'cosine', top_k = 15):
        "Returns the top indices from a cosine similarity or L2 distance matrix."
        axis = 1 if data_type == 'mols' else 0
        #print(axis)
        largest = True if mode == 'cosine' else 0

        if data_type == 'mols':
            top_k = self.sample_counts.max()
        if mode == 'cosine':
            top_ixs = (
                torch.from_numpy(self.cosine_arr)
                .topk(k=top_k, largest=largest, dim=axis)
                .indices.numpy()
            )

        else: # distance matrix
            top_ixs = (
                torch.from_numpy(self.D)
                .topk(k=top_k, largest=largest, dim=axis)
                .indices.numpy()
            )

        return top_ixs


    def get_cosine_drug_one_vs_all(self, drug_name)->Tuple[np.ndarray, np.ndarray]:
        """
        Returns the cosine similarity distributions of a drug with cells perturbed by it,
        and all other cells coming from other samples.
        """
        n_mols, n_cells = self.cosine_arr.shape
        ix_drug, ix_cells = self.get_ix_drug(drug_name), self.get_ix_cells(drug_name)

        # Get cosine similarity distribution of a drug with itself
        cosine_cells_drug = self.cosine_arr[ix_drug, ix_cells]

        # Get the indices of all perturbed with other molecules but `drug_name`'s
        other_cells_ix = np.array(list(set(np.arange(n_cells)) - set(ix_cells)))
        cosine_others = self.cosine_arr[ix_drug, other_cells_ix]
        return cosine_cells_drug, cosine_others



    def eval_mol2cell_accuracy(self, mode= 'cosine', return_ = False):

        top_ixs_mols = self.get_top_ixs(data_type = 'mols', mode = mode)

        # Initialize molecule accuracies list
        accs = []
        for i, drug_ix in enumerate(self.test_drugs_ixs):

            # Get the drug indices for each of the top cells given molecule query
            top_ix_mol = self.ix_samples_cell[top_ixs_mols[i]]

            # Get only the top n indices, for n the number of cells sampled in experiment
            top_ix_mol_normalized = top_ix_mol[: int(self.sample_counts_idx[drug_ix])]

            # Acc : fraction of correct cells
            acc = np.sum(top_ix_mol_normalized == drug_ix) / (self.sample_counts_idx[drug_ix]) * 100

            accs.append(acc)

        self.m2c_acc = accs
        self.drugbank['accuracy'] = accs

        if return_:
            return accs

    def eval_cell2mol_accuracy(self, mode = 'cosine'):
        top_ixs_cells = self.get_top_ixs(data_type = 'cells', mode = mode).T

        acc_indicator = np.zeros((self.adata.n_obs, 5))

        if isinstance(self.test_drugs_ixs, list):
            self.test_drugs_ixs = np.array(self.test_drugs_ixs)

        for i, sample_ix in tqdm.tqdm(enumerate(self.ix_samples_cell)):

            # Get top 1, top3, top5, 10, and 15 accuracy
            acc_indicator[i, 0] = 1 if sample_ix == self.test_drugs_ixs[top_ixs_cells[i, 0]] else 0
            acc_indicator[i, 1] = 1 if sample_ix in self.test_drugs_ixs[top_ixs_cells[i, :3]] else 0
            acc_indicator[i, 2] = 1 if sample_ix in self.test_drugs_ixs[top_ixs_cells[i, :5]] else 0
            acc_indicator[i, 3] = 1 if sample_ix in self.test_drugs_ixs[top_ixs_cells[i, :10]] else 0
            acc_indicator[i, 4] = 1 if sample_ix in self.test_drugs_ixs[top_ixs_cells[i, :15]] else 0


        self.c2m_global_acc = acc_indicator.sum(axis = 0)/ self.adata.n_obs *100

        ks = [1, 3, 5, 10, 15]
        df_acc = pd.DataFrame(acc_indicator, columns = ['top' + str(i) + '_accuracy' for i in ks])

        self.adata.obs = pd.concat([self.adata.obs, df_acc.set_index(self.adata.obs.index)], axis = 1)


    def eval_m2c_mean(self, drug_name, mode = 'mean'):
        """
        Computes the fraction of cells that have cosine similarity w.r.t. to its own molecule
        higher than the mean of the distribution across all cells.
        """
        drug_ix = self.get_ix_drug(drug_name)

        # Get cosine distribution for drug and all others
        cosine_distro_drug, cosine_distro_others = self.get_cosine_drug_one_vs_all(drug_name)

        if mode=='mean':
            central_measure = np.concatenate([ cosine_distro_drug, cosine_distro_others]).mean()
        elif mode=='median':
            central_measure = np.median(np.concatenate([ cosine_distro_drug, cosine_distro_others]))
        else:
            raise NotImplementedError('%s is not implemented'%mode)

        n_significant = (cosine_distro_drug > central_measure).sum()
        percent_significant = n_significant / len(cosine_distro_drug) * 100

        return percent_significant

    def eval_m2c_mean_all(self, mode = 'mean', n_cores = 4, return_ = False):
        "Evaluate above-mean accuracy for all drugs."

        acc_arr = Parallel(n_jobs = n_cores)(
            delayed(self.eval_m2c_mean)(drug, mode)
            for drug in tqdm.tqdm(
                    self.test_drugs, position = 0, leave = True
                )
        )

        self.drugbank['acc_mean'] = acc_arr

        if return_:
            return acc_arr

    def run_ks_test(self, drug_name):
        "Returns statistics of running one vs all test for a given drug."
        own, others = self.get_cosine_drug_one_vs_all(drug_name)
        ks, pval_ks, l1_score = get_stats(own, others)
        return ks, pval_ks, l1_score

    def run_ks_one_vs_all(
        self,
        n_cores = 4,
        stat_metric = 'ks_pval',
        thresh_stat = 1e-4,
        return_ = False
        ):
        """
        Returns results from testing the mol2cell cosine similarity distributions of a drug
        with cells perturbed by it, and other cells.

        Params
        ------
        n_cores (int, default = 4)
            Number of processors to use for the parallellization.

        stat_metric (str, default = 'ks_pval')

        cols (list)

        Notes
        -----
        The rationale is that the cosine similarity between a given drug and cells pertrubed by it
        should be higher than to all other cells if the model has learnt a meaningful relationship.

        Runs on parallel using joblib.
        """

        results = Parallel(n_jobs = n_cores)(
            delayed(self.run_ks_test)(drug) for drug in tqdm.tqdm(
                    self.test_drugs, position = 0, leave = True
                )
        )

        self.df_stat_tests = pd.DataFrame(
            results, columns = ['ks_score', 'ks_pval', 'l1_score']
        )

        self.drugbank = pd.concat([self.drugbank, self.df_stat_tests], axis = 1)

        if "pval" in stat_metric:
            top_drug_df = self.drugbank[self.drugbank[stat_metric] < thresh_stat].sort_values(
                by=stat_metric, ascending=True
            )

        # use score
        elif "score" in stat_metric:
            top_drug_df = df_drugs_test_[df_drugs_test_[stat_metric] > thresh_stat].sort_values(
                by=stat_metric, ascending=False
            )

        else:
            raise AssertionError('metric should be either pvalue or score.')

        self.top_drugs_ks = top_drug_df

        if return_:
            return top_drug_df

    def get_cosine_distribution_df(
        self,
        drug_name,
        n_top = None,
        cols_viz = ['drug_name', 'target', 'drug_class'],
        return_acc_only = False,
        filter_by = 'drug_name',
        n_cells_filter = 10,
        anti = False
        ):
        """
        Returns a dataframe of the cells closest to a molecule, grouped by sample.

        Params
        ------
        filter_by (str, default = 'drug_name')
            Column to filter out spurious high similarity.
        """

        if n_top is None:
            n_top = self.adata[self.adata.obs.drug_name == drug_name].n_obs

        ix_ = self.get_ix_drug(drug_name)

        #name_of_drug = self.name_to_ix[ix_] #drugbank.iloc[ix_]['drug_name']

        #name_of_drug = drugbank_to_selleck[name_of_drug]
        #print('Returning predictions for %s'%name_of_drug)

        # Reverse order : get cells with lowest cosine sim
        if anti:
            ix_top_cells = np.argsort(self.cosine_arr[ix_])[:n_top]
        else:
            ix_top_cells = np.argsort(self.cosine_arr[ix_])[::-1][:n_top]

        # Make a dataframe containing the cosine similarities and cols_viz
        df_viz = self.adata[ix_top_cells].obs[cols_viz]
        df_viz['drug_name'] = df_viz['drug_name'].astype(str)

        #try:
        sample_val_counts = df_viz.drug_name.value_counts()
        if drug_name in sample_val_counts.index.values:
            n_correct = sample_val_counts[drug_name]
            acc = n_correct / sample_val_counts.sum() * 100
            print('Accuracy: %.3f'%acc)
            if return_acc_only:
                return acc
            else:
                pass
        else:
            print('Accuracy: 0')
            if return_acc_only:
                return 0
        #except:
       #     pass

        df_viz['cosine_similarity'] = self.cosine_arr[ix_][ix_top_cells]
        val_counts = df_viz[filter_by].value_counts()
        samples_in = val_counts[val_counts > n_cells_filter].index.values
        return df_viz[df_viz[filter_by].isin(samples_in)]


    def plot_top_mol2cell(self):
        "Assumes `eval_mol2cell_accuracy` has been executed."
        try:
            self.drugbank['name_class'] = self.drugbank['name']+ ['_'] + self.drugbank['drug_class']
        except:
            pass

        fig = plt.figure(figsize =(1, 4))
        try:
            sns.heatmap(
                self.drugbank.sort_values(by = 'accuracy', ascending = False).head(15).set_index('name_class')['accuracy'].to_frame(),
                cmap = 'mako_r', annot = True, label = 'accuracy (%)', vmin = 0, #vmax = 35
                       )

        except:
            sns.heatmap(
                self.drugbank.sort_values(by = 'accuracy', ascending = False).head(15).set_index('drug_name')['accuracy'].to_frame(),
                cmap = 'mako_r', annot = True, label = 'accuracy (%)', vmin = 0, #vmax = 35
                       )

    def plot_boxplot_m2c(self, plot = 'accuracy', cat = 'drug_class', filt_by = 0):
        #plot = "accuracy"
        #by = "drug_class"

        fig = plt.figure(figsize=(3, 4))

        sns.boxplot(
            data=self.drugbank[self.drugbank[plot] > filt_by].sort_values(
                by=[plot, cat], ascending=False
            ),
            x=plot,
            y=cat,
            color="lightgrey",  # alpha = 0.4
        )

        sns.stripplot(
            data=self.drugbank[self.drugbank[plot] > 1].sort_values(
                by=[plot, cat], ascending=False
            ),
            x=plot,
            y=cat,
            palette=self.colormaps[cat],
        )

        return fig

    def plot_ks(self, drug_name, export = None, path_figs= None, model_name= ''):
        "To run after executing `run_ks_one_vs_all`"
        data = self.drugbank[self.drugbank.drug_name == drug_name]

        #drug_ = data['drug_name']
        #print(drug_)
        drug = drug_name.split()[0]
        #print(drug)
        #acc = data['accuracy']
        #within_class_acc = data['within_class_acc']
        ks, pval, l1_score, acc = data[['ks_score', 'ks_pval', 'l1_score', 'accuracy']].squeeze()

        try:
            pval = np.log10(pval)
        except:
            pval = 0

        own, others = self.get_cosine_drug_one_vs_all(drug_name)

        sorted_drug, ecdf_drug = ecdf(own)
        sorted_other, ecdf_other = ecdf(others)

        plt.figure(figsize = (3.5, 1.7))
        plt.plot(sorted_drug, ecdf_drug, label = drug + ' cells', color = 'dodgerblue')
        plt.plot(sorted_other, ecdf_other, label = 'cells from other samples', color = 'lightgrey')
        plt.legend(
            #title = 'KS: %.2f, pval: %.3f, l1: %.2f'%(ks, pval, l1_score),
            bbox_to_anchor = (1.04, 0), loc = 'lower left'
                  )

        plt.title('One-vs-rest test KS: %.2f, KS pval: 1x10^ %.1f, l1: %.2f \n acc: %.1f'%(
            ks, pval, l1_score, acc
        ),)
        plt.xlabel(r'$\mathrm{cos} \theta$ to %s mol.'%drug)
        plt.ylabel('ECDF')

        if export:
            plt.savefig(
                os.path.join(path_figs, drug + '_ks_test_%s.png'%model_name), bbox_inches = 'tight', dpi = 230
            );

    def plot_ks_all(self, export = True, path_figs= '../figs', model_name = ''):
        """
        Plots ECDFs of cosine similarity distributions of correct drug vs all others.
        Considers only top drugs. Assumes `run_ks()` has been called already.

        """
        # Assert if self.top_drug_ks exists.

        for i, drug in self.test_drugs:
            plot_ks(drug, export, path_figs, model_name)

    def get_acc_df_cell2mol(self, k=5, return_ = True):

        """
        Returns a top-k accuracy dataframe per sample.

        Params
        ------
        df_cells (pd.DataFrame)
            Cell dataframe (from adata or df_embedding) that contains the top-k accuracy.

        """

        df_cells = self.adata.obs

        #name_to_target = None, name_to_class= None,

        pred_df = (
            df_cells.groupby(["drug_name", "top" + str(k) + "_accuracy"]).size().unstack().fillna(0)
        )

        pred_arr = pred_df.values / pred_df.values.sum(axis=1).reshape(-1, 1) * 100

        perc_pred_df = pd.DataFrame(pred_arr, index=pred_df.index, columns=pred_df.columns)

        accuracy_df = (
            perc_pred_df.sort_values(by=0, ascending=True)[1].to_frame().reset_index()
        )

        accuracy_df.rename(columns = {1:'top'+ str(k) + '_accuracy'}, inplace = True)

        try:
            if self.name_to_target is not None:
                accuracy_df['target'] = accuracy_df['drug_name'].map(name_to_target)
                #accuracy_df['name_target'] = accuracy_df['drug_name'] + '_' + accuracy_df['target'].str.lower()
        except:
            pass
        try:
            if self.name_to_class is not None:
                accuracy_df['drug_class'] = accuracy_df['drug_name'].map(name_to_class)
                accuracy_df['sample_class'] = accuracy_df['drug_name'] + '_' + accuracy_df['drug_class'].str.lower()
        except:
            pass

        self.c2m_acc = accuracy_df
        if return_:
            return accuracy_df
