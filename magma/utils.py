# ### utils
#     - [>]  trainers
#     - [>]  initialize
#     - [>]  torch_adata
#     - [>]  accuracy
#     - [>]  topk_acc
#     - [>]  confusion_matrix
#     - [>]  try_gpu
#     - [>]  cells: get_count_stats, log_norm, cv_filter
#     - [>]  viz: set_plotting_style


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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch_geometric


def train_supervised_gcn(
    model:nn.Module,
    data:torch_geometric.data.Data,
    loss_fn,
    optimizer,
    multiclass = False,
    n_out = 1
)->torch.tensor:
    """Single fwd-bwd pass on GraphConvNet model."""

    y_true = torch.tensor(data.y, dtype = torch.long)
        #np.array(data.y, dtype=np.int16), dtype =torch.Long
        #) #, device = data.device)
    #y_true = data.y

    optimizer.zero_grad()
    y_pred = model(data)

    if multiclass:
        loss = loss_fn(y_pred, y_true)
    else:
        loss = loss_fn(
            y_pred.float(),
            y_true.reshape(-1, n_out).float()
        )

    loss.backward()
    optimizer.step()

    return loss

def val_supervised_gcn(
    model, data, loss_fn, multiclass = False, n_out = 1
)-> float:


    y_pred = model(data)
    #y_true = torch.from_numpy(np.array(data.y, dtype=np.int16)) #, device = device)
    y_true = torch.tensor(data.y, dtype = torch.long)

    if multiclass:
        loss = loss_fn(y_pred, y_true)
    else:
        loss = loss = loss_fn(
            y_pred.float(),
            y_true.reshape(-1, n_out).float()
        )

    return loss.mean()


def supervised_trainer_gcn(
    n_epochs,
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    multiclass= False,
    n_classes = 1,
    train_prints_per_epoch = 5
):

    batch_size = train_loader.batch_size
    print_every = np.floor(train_loader.dataset.__len__() / batch_size / train_prints_per_epoch) # minibatches

    train_loss_vector = [] # to store training loss
    val_loss_vector = np.empty(shape = n_epochs)

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
                data= data.cuda(device=device)
                #input_tensor = input_tensor.cuda(device = device)
                #y_true = y_true.cuda(device = device)

            train_loss = train_supervised_gcn(
                model,
                data,
                #y_true,
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

            for i, data in enumerate(tqdm.tqdm(val_loader)):
                #input_tensor = data.view(batch_size, -1).float()

                if cuda:
                    data=data.cuda(device=device)
                    #input_tensor = input_tensor.cuda(device = device)
                    #y_true = y_true.cuda(device = device)

                val_loss = val_supervised_gcn(
                    model, data, criterion, multiclass, n_classes
                    )

                validation_loss.append(val_loss)

            mean_val_loss = torch.tensor(validation_loss).mean()
            val_loss_vector[epoch] = mean_val_loss

            print('Val. loss %.3f'% mean_val_loss)

    print('Finished training')

    return train_loss_vector, val_loss_vector


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
    Helper function to make forward and backward pass with minibatch
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

    else: # Backprop error
        loss = loss_fn(y_pred, y_true.view(-1, n_out).float())

    loss.backward()
    # Update weights
    optimizer.step()

    return loss

def validation_supervised(model, input_tensor, y_true, loss_fn, multiclass =False, n_classes= 1):
    """
    Returns average loss for an input batch of data with a supervised model.
    If running on multiclass mode, it also returns the accuracy.
    """

    y_pred = model(input_tensor.float())
    if multiclass:
        loss = loss_fn(y_pred, y_true)
        #acc = accuracy(y_true, y_pred)
    else:
        loss = loss_fn(y_pred, y_true.view(-1, n_classes).float())

    return loss.mean().item()

def supervised_trainer(
    n_epochs:int,
    train_loader:DataLoader,
    val_loader:DataLoader,
    model:nn.Module,
    criterion,
    optimizer,
    multiclass:bool = False,
    n_classes:int = 1,
    train_prints_per_epoch:int = 5,
    train_fn:callable = train_supervised,
    model_dir:str = None,
    model_name:str = None,
    early_stopping_tol:float = 0.2,
    **kwargs
    ):
    """
    Wrapper function to train a supervised model for n_epochs.
    Currently designed for classification and regression.
    Notes: Not yet suited for segmentation problems.

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
    print_every = np.floor(train_loader.dataset.__len__() / batch_size / train_prints_per_epoch) # minibatches

    train_loss_vector = [] # to store training loss
    val_loss_vector = np.empty(shape = n_epochs)

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

            train_loss = train_fn(
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

            for i, (data, y_true) in enumerate(tqdm.tqdm(val_loader)):

                input_tensor = data.view(batch_size, -1).float()

                if cuda:
                    input_tensor = input_tensor.cuda(device = device)
                    y_true = y_true.cuda(device = device)

                val_loss = validation_supervised(
                    model, input_tensor, y_true, criterion, multiclass, n_classes
                    )

                validation_loss.append(val_loss)

            mean_val_loss = torch.tensor(validation_loss).mean().item()
            val_loss_vector[epoch] = mean_val_loss

            print('Val. loss %.3f'% mean_val_loss)


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
                torch.save(model.state_dict(), model_dir + model_name + '_' + str(epoch) + '.pt')
            else:
                torch.save(model.state_dict(), model_dir + 'model' + '_' + str(epoch) + '.pt')


    print('Finished training')

    return train_loss_vector, val_loss_vector


def print_loss_in_loop(ep, ix, running_loss, print_every, message='loss'):
    print_msg = '[%d, %5d] ' + message + ' : %.3f'
    print(print_msg%\
          (ep + 1, ix+1, running_loss / print_every))

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



def train_vae(
	model:nn.Module,
	input_tensor,
	optimizer)->torch.tensor:
    """
    Forward-backward pass of a VAE model.
    """

    # Zero-out grads
    model.zero_grad()

    # Make forward computation
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
	optimizer)->torch.tensor:

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
    train_prints_per_epoch = 5):

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
        train_loader.dataset.__len__() / batch_size / train_prints_per_epoch
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

def accuracy(y_pred, y_true):
    "Returns the accuracy between predicted and true labels."
    acc = torch.eq(y_true, y_pred).sum().item() / y_true.shape[0]
    return acc

def topk_acc(y_pred, y_true, k = 5):
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
	ribo_prefix = None)-> ad.AnnData:

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

	# Check that slope is approx -1/2
	print(f'The slope of the model is {np.round(slope,3)}.')

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
        if labels.requires_grad:
            labels = labels.detach()
        labels = labels.numpy()

    # Gen binary array of non-duplicated labels
    mask = ~pd.Series(x).duplicated().values

    # Check that no duplicated values remain
    assert len(np.nonzero(pd.Series(labels[mask]).duplicated().values)[0]) == 0

    return mask
