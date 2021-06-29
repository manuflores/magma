### models
# - [>]  layers
# - [>]  gcn
# - [>]  gat
# - [>]  supervised_model
# - [>]  vae
# - [>]  JointEmbedding
from typing import Optional, Sequence, Tuple, Union

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
import torch.linalg as LA

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, IterableDataset, DataLoader

import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

import copy

class BnLinear(nn.Module):
	"""Linear layer with batch normalization."""
	def __init__(self, input_dim, output_dim, kwargs = None):
	    super(BnLinear, self).__init__()

	    if kwargs is not None:
	        self.linear = nn.Linear(input_dim, output_dim, **kwargs)
	    else:
	        self.linear = nn.Linear(input_dim, output_dim)

	    self.bn = nn.BatchNorm1d(output_dim)

	def forward(self, x):
	    x = self.linear(x)
	    x = self.bn(x)

	    return x

class BnGraphConvLayer(GCNConv):
	"""Graph Conv Layer with batch normalization."""
	def __init__(self, in_channels, out_channels, **kwargs):
	    super(GCNConv, self).__init__()

	    self.graph_conv = GCNConv(in_channels, out_channels)
	    self.bn = nn.BatchNorm1d(out_channels)
	    self.in_channels = in_channels
	    self.out_channels = out_channels

	def forward(self, x, edge_index):
	    x = self.graph_conv(x, edge_index)
	    x = self.bn(x)

	    return x

class BnGATConv(nn.Module):
	"""Graph Attention layer with Batch normalization."""
	def __init__(self, in_channels, out_channels, kwargs={}):
	    """Batchnorm Graph Attention Conv layer. """
	    super(BnGATConv, self).__init__()


	    self.graph_conv = GATConv(
	        in_channels,
	        out_channels,
	        **kwargs
	    )

	    self.bn = nn.BatchNorm1d(out_channels)
	    self.in_channels = in_channels
	    self.out_channels = out_channels

	def forward(self, x, edge_index):
	    x = self.graph_conv(x, edge_index)
	    x = self.bn(x)

	    return x


class GNNBase(nn.Module):
    """Base class for graph neural networks (GNNs)."""
    def __init__(self):
        super(GNNBase, self).__init__()
        self.conv_encoder = None
        self.linear_layers = None
        self.final_layer = None
        self.multiple_linear = None
        self.pooling = None
        self.model_type = None
        self.activation_func_conv = None
        self.activation_func_linear = None

    def activations_hook(self, grad):
        "Registers gradients for an intermediate tensor."
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def forward(self, data:torch_geometric.data.Data):

        x, edge_index = data.x, data.edge_index

        for conv_layer in self.conv_encoder:
            x = conv_layer(x, edge_index)
            x = self.activation_func_conv(x)

        if self.pooling == 'mean':
            x = global_mean_pool(x, data.batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, data.batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, data.batch)
        else:
            raise ValueError('Pooling method not specified or implented.')


        # no non-linear activations in linear layers
        # (only linear transformations)
        if self.multiple_linear:
            for dense_layer in self.linear_layers:
                x = dense_layer(x)
                if self.activation_func_linear is not None:
                    x = self.activation_func_linear(x)
                x = F.dropout(x, p=0.3, training=self.training)

        x = self.final_layer(x)

        if self.model_type == 'regression':
            return x

        elif self.model_type == 'multiclass':
            x = F.log_softmax(x, dim =1)
            return x
        elif self.model_type == 'binary':
            return F.sigmoid(x)
        elif self.model_type == 'multilabel':
            return F.sigmoid(x)

        else:
            raise ValueError(
                'model_type needs to be one of: ["regression", "multiclass", "binary", "multilabel"]'
            )

    def project(self, data, pool = True, reg_hook_input = False, reg_hook_conv = False):
        """
        Projects data up to last hidden layer for visualization.

        Params
        -------
        data (torch_geometric.data.data.Data)
            A graph in torch_geometric format. Composed of node_features `x`,
            edge_indices, and edge_features.

        pool(bool, default =True)
            Optional kwarg, if set to True gets graph embeddings
            from node embeddings.

		reg_hook_input (bool, default= False)
			Whether to record the gradients since for the input graph.
			This is helpful to compute gradinput graph attribution method.

		reg_hook_conv (bool, default = False)
			Whether to record the grads after the last conv layer.
			This method is helpful for gradCAM.

        """

        x, edge_index = data.x, data.edge_index

		# For gradinput
        if reg_hook_input:
            h = x.register_hook(self.activations_hook)

		# Forward pass through conv layers
        for conv_layer in self.conv_encoder:
            x = conv_layer(x, edge_index)
            x = self.activation_func_conv(x)
            x = torch.tanh(x)

        if reg_hook_conv:
            h = x.register_hook(self.activations_hook)

        # Get graph embedding
        if pool:
            if self.pooling == 'mean':
                x = global_mean_pool(x, data.batch)
            elif self.pooling == 'add':
                x = global_add_pool(x, data.batch)
            elif self.pooling == 'max':
                x = global_max_pool(x, data.batch)

        # Return node embeddings
        else:
            return x

        # Project to last layer
        if self.multiple_linear:
            for dense_layer in self.linear_layers:
                x = dense_layer(x)
                if self.activation_func_linear is not None:
                	x = self.activation_func_linear(x)


        return x

    @torch.no_grad()
    def project_to_latent_space(self, data_loader, n_feats, latent_dim):
        """
        Returns a generator to project dataset into latent space,
        i.e. last hidden layer.

        Params
        ------
        data_loader (torch.DataLoader)
            DataLoader which handles the batches and parallelization.

        n_feats (int)
            Number of dimensions of original dataset.

        latent_dim (int)
            Number of dimensions of layer to project onto.

        Returns (yields)
        -------
        encoded_sample (array-like generator)
            Generator of a single encoded data point in a numpy array format.
        """

        # Set no_grad mode to avoid updating computational graph.
        # with torch.no_grad()

        cuda = torch.cuda.is_available()

        for ix, batch_x in enumerate(tqdm.tqdm(data_loader)):

            if cuda:
                batch_x = batch_x.cuda()
                batch_x_preds = self.project(batch_x).cpu().detach().numpy()

            else:
                batch_x_preds = self.project(batch_x).detach().numpy()

            for x in batch_x_preds:
                encoded_sample = x.reshape(latent_dim)
                yield encoded_sample


class GraphConvNetwork(GNNBase):
    "A graph neural network model based with Graph Convolutional Blocks."
    def __init__(
        self,
        dims_conv,
        dims_lin,
        model_type = 'multiclass',
        pooling = 'mean',
		act_func_conv = torch.tanh,
		act_func_linear = torch.tanh
        ):

        """
        A graph neural network model based with Graph Convolutional Blocks.

        Params
        ------
        dims_conv(list)
            List of input-output dimensions of the convolutional layers.

        dims_lin (list)
            List of dimensions of fully-connected layers, i.e. MLPs.
            It expects the first dim to be the size of the last conv layer,
            and the last dim to be that of the number of classes to predict.

            Thus if provided a list, the length has to be more than three:
            (last_conv_layer_dim, (intermediate_dims), output_dim).

        pooling (str, default = 'mean')
            Method of pooling for going from node embeddings to graph embeddings.
            Only 'mean', 'sum' or 'max' implemented.

        model_type (str, default = `multiclass`)
            Describes the type of supervised model to use.
            Defaults to multiclass i.e. softmax classification.
            Needs to be one of ["regression", "multiclass", "binary", "multilabel"]

        """
        super(GraphConvNetwork, self).__init__()

        conv_layers = [
            BnGraphConvLayer(dims_conv[i - 1], dims_conv[i])
            for i in range(1, len(dims_conv))
        ]

        self.conv_encoder = nn.ModuleList(conv_layers)

		# Multiple FC layer mode
        if isinstance(dims_lin, list):
            linear_layers = [
                BnLinear(
					dims_lin[i - 1], dims_lin[i],
					#{'bias': False}
					)
                for i in range(1, len(dims_lin) - 1)
            ]

            self.output_dim = dims_lin[-1]
            self.linear_layers = nn.ModuleList(linear_layers)
            self.final_layer = BnLinear(
				dims_lin[-2], self.output_dim,
				#{'bias': False}
				)
            self.multiple_linear = True

        # Single output FC layer mode
        elif isinstance(dims_lin, int):
            self.output_dim = dims_lin
            self.linear_layers= None
            self.final_layer = BnLinear(
                dims_conv[-1], self.output_dim, #{'bias': False}
            )

            self.multiple_linear = False
        else:
            raise TypeError('`dims_lin` was expecting a list or int.')


        self.pooling = pooling
        self.model_type = model_type
        self.activation_func_conv = act_func_conv
        self.activation_func_linear = act_func_linear


class GraphAttentionNetwork(GNNBase):
    """
    Graph Convolutional Net with attention mechanism for supervised model tasks.
    Contains functionality to get node and graph embeddings.
    """
    def __init__(
        self,
        dims_conv,
        dims_lin,
        pooling = 'mean',
        model_type = 'multiclass',
		act_func_conv = torch.tanh,
		act_func_linear = torch.tanh,
        attention_layer_kwargs = {}
    ):
        """
        Graph Convolutional Net with attention mechanism for supervised model tasks.
        It is designed such that there is only a single fully connected (FC / linear)
        layer after the convolutional layers to make node attribution methods for
        interpretability.

        However, it allows to add multiple FC layers.

        Notes: The default of `model_type` is `softmax` i.e. multiclass classification.

        Params
        ------
        dims_conv (list)
            List of input-output dimensions of the convolutional layers,

        dims_lin (int or list)
            List of dimensions of fully-connected layers, i.e. MLPs.
            It expects the first dim to be the size of the last conv layer,
            and the last dim to be that of the number of classes to predict.

            Thus if provided a list, the length has to be more than three:
            (last_conv_layer_dim, (intermediate_dims), output_dim).

        pooling (str, default = 'mean')
            Method of pooling for going from node embeddings to graph embeddings.
            Only 'mean', 'sum' or 'max' implemented.

        attention_layer_kws (dict, default = None)
            Keyword arguments for GATConv layer.
            Defaults are the following (initialized inside function):
            'heads' : 3, # number of attention heads
            'concat': False, # average attention heads, not concat.
            'bias' : False, # force a linear transformation (not affine)
            'add_self_loops': False # don't allow node self-attention

        model_type (str, default = `softmax`)
            Describes the type of supervised model to use.
            Needs to be one of ["regression", "softmax", "binary", "multilabel"]

        """
        super().__init__()

        _att_layer_kwargs = {
            'heads' : 3,
            'concat': False,
            'bias' : False,
            'add_self_loops': False
        }

        #if attention_layer_kwargs is not None:
        for key, val in attention_layer_kwargs.items():
            _att_layer_kwargs[key] = val

        # Attention heads
        self.heads = _att_layer_kwargs['heads']

        # Initialize Graph Conv layers
        conv_layers = [
            BnGATConv(
                dims_conv[i - 1],
                dims_conv[i],
                _att_layer_kwargs
            ) for i in range(1, len(dims_conv))
        ]

        self.conv_encoder = nn.ModuleList(conv_layers)

        # Multiple FC layer mode
        if isinstance(dims_lin, list):
            linear_layers = [
                BnLinear(
					dims_lin[i - 1], dims_lin[i],# {'bias': False}
				)
                for i in range(1, len(dims_lin) - 1)
            ]

            self.output_dim = dims_lin[-1]
            self.linear_layers = nn.ModuleList(linear_layers)
            self.final_layer = BnLinear(
				dims_lin[-2],
				self.output_dim,
				#{'bias': False}
			)

            self.multiple_linear = True

        # Single output FC layer mode
        elif isinstance(dims_lin, int):
            self.output_dim = dims_lin
            self.linear_layers= None
            self.final_layer = BnLinear(
                dims_conv[-1], self.output_dim, #{'bias': False}
            )

            self.multiple_linear = False
        else:
            raise TypeError('`dims_lin` was expecting a list or int.')

        self.pooling = pooling
        self.model_type = model_type
        self.activation_func_conv = act_func_conv
        self.activation_func_linear = act_func_linear



class supervised_model(nn.Module):
    """
    Deep multi-layer perceptron (MLP) for classification and regression.
    It is built with Linear layers that use Batch Normalization
    and tanh as activation functions. The model type is defined
    using the `model` argument.

    Params
    ------
    dims (list):
        Dimensions of the MLP. First element is the input dimension,
        final element is the output dimension, intermediate numbers
        are the dimension of the hidden layers.

    model (str, default = 'regression'):
        Type of supervised model. Options are
        'regression': For MLP regression.
        'multiclass': For multiclass classification (single categorical variable).
        'binary': For binary classification.
        'multilabel': For multilabel classification, i.e when multiple
        categorical columns are to be predicted.

        Notes: 'multiclass' uses F.log_softmax as activation
        layer. Use nn.NLLLoss() as loss function.

    dropout (bool, default = True)

    """

    def __init__(self, dims, model = 'regression', dropout = True):
        super(supervised_model, self).__init__()

        self.output_dim = dims[-1]

        # Start range from 1 so that dims[i-1] = dims[0]
        linear_layers = [BnLinear(dims[i-1], dims[i]) for i in range(1, len(dims[:-1]))]

        self.fc_layers = nn.ModuleList(linear_layers)

        self.final_layer = BnLinear(dims[-2], self.output_dim)

        self.model = model
        self.dropout=dropout
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.cuda = torch.cuda.is_available()

    def project(self, x):
        "Projects data up to last hidden layer for visualization."

        for fc_layer in self.fc_layers[:-1]:
            x = fc_layer(x)
            x = self.tanh(x)

        x = self.fc_layers[-1](x)

        return x

    def project_to_latent_space(self, data_loader, n_feats, latent_dim):
        """
        Returns a generator to project dataset into latent space,
        i.e. last hidden layer.

        Params
        ------
        data_loader (torch.DataLoader)
            DataLoader which handles the batches and parallelization.

        n_feats (int)
            Number of dimensions of original dataset.

        latent_dim (int)
            Number of dimensions of layer to project onto.

        Returns (yields)
        -------
        encoded_sample (array-like generator)
            Generator of a single encoded data point in a numpy array format.
        """

        # Set no_grad mode to avoid updating computational graph.
        #with torch.no_grad()

        cuda = torch.cuda.is_available()

        # Iterate through all of the batches in the DataLoader
        for batch_x, targets in tqdm.tqdm(data_loader):

            #
            if cuda:
                batch_x = batch_x.cuda() #, targets.cuda()

            # Reshape to eliminate batch dimension
            batch_x = batch_x.view(-1, n_feats)

            # Project into latent space and convert tensor to numpy array
            if cuda:
                batch_x_preds = self.project(batch_x.float()).cpu().detach().numpy()
            else:
                batch_x_preds = self.project(batch_x.float()).detach().numpy()

            # For each sample decoded yield the line reshaped
            # to only a single array of size (latent_dim)
            for x in batch_x_preds:
                encoded_sample = x.reshape(latent_dim)

                yield encoded_sample


    def forward(self, x):

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = self.tanh(x)

        # Pass through final linear layer
        if self.model == 'regression':
            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            return x

        elif self.model == 'multiclass':
            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            x = F.log_softmax(x, dim = 1)

            return x

        elif self.model == 'binary':
            if self.dropout:
                x = F.dropout(x, p = 0.3)
            x = self.final_layer(x)
            x = F.sigmoid(x)

            return x

        elif self.model == 'multilabel':
            if self.dropout:
                x = F.dropout(x)
            x = self.final_layer(x)
            x = F.sigmoid(x)

            return x

        else:
            print(self.model, ' is not a valid model type.')


class VariationalAutoencoder(nn.Module):
    """
    Variational autoencoder (VAE) model with linear layers.
    """
    def __init__(self, dims, beta = 1, recon_loss = 'MSE'):
        """
        Variational autoencoder (VAE) module with single hidden layer.
        Contains functions to reconstruct a dataset and
        project it into a latent space.

        Note: set `beta` = 0 to recover a classic autoencoder (no KL divergence).

        Params
        ------
        dims (array-like):
            Dimensions of the networks given by the number of neurons
            of the form [input_dim, hidden_dim_1, ..., hidden_dim_n, latent_dim],
            where `input_dim` is the number of features in the dataset.

            Note: The encoder and decoder will have a symmetric architecture.

        recon_loss (str, defult = 'MSE')
            Reconstruction loss. Avaiable loss functions are
            binary cross entropy ('BCE') and mean squared error ('MSE').

            We have empirically found that using BCE on minmax normalized
            data gives good results.

        """

        super(VariationalAutoencoder, self).__init__()


        self.input_dim = dims[0]
        #self.output_dim = dims[0]
        self.embedding_dim = dims[-1]

        # ENCODER

        # Start range from 1 so that dims[i-1] = dims[0]
        hidden_layers_encoder = [
            BnLinear(dims[i-1], dims[i]) for i in range(1, len(dims[:-1]))
        ]

        self.encoder_hidden = nn.ModuleList(hidden_layers_encoder)

        # Stochastic layers
        self.mu = BnLinear(dims[-2], self.embedding_dim)
        self.logvar = BnLinear(dims[-2], self.embedding_dim)


        # DECODER
        dims_dec = dims[::-1]

        hidden_layers_decoder = [
            BnLinear(dims_dec[i-1], dims_dec[i]) for i in range(1, len(dims_dec[:-1]))
        ]

        self.decoder_hidden = nn.ModuleList(hidden_layers_decoder)

        self.reconstruction_layer = BnLinear(dims_dec[-2], self.input_dim)


        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        #self.tanh= nn.Tanh()

        self.beta = beta

        if recon_loss == 'BCE':

            self.reconstruction_loss = nn.BCELoss()

        elif recon_loss == 'MSE':
            self.reconstruction_loss = nn.MSELoss()
        else:
            print('Recon loss not implemented, choose one of ["MSE", "BCE"]]')

    def encoder(self, x):
        """
        Encode a batch of samples and return posterior parameters
        mu and logvar for each point.

        Attempts to generate probability distribution P(z|x)
        from the data by fitting a variation distribution Q_φ(z|x).
        Returns the two parameters of the distributon (µ, log σ²).

        """
        for fc_layer in self.encoder_hidden:
            x = fc_layer(x)
            x = torch.tanh(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

    def decoder(self, z):
        """
        Decodes a batch of latent variables.

        Generative network. Generates samples from the original data
        distribution P(x) by approximating p_θ(x|z). It uses a Sigmoid activation
        for the output layer, so input data must be normalized between 0 and 1
        (e.g. min-max normalized).
        """
        for fc_layer in self.decoder_hidden:
            z = fc_layer(z)
            z = torch.tanh(z)

        x = self.reconstruction_layer(z)
        #x = self.sigmoid(x)

        return x

    def reparam(self, mu, logvar):
        """
        Reparametrization trick to sample z values.
        This is a stochastic procedure, and returns
        the mode during evaluation.
        """

        if self.training:

            std = logvar.mul(0.5).exp_()

            epsilon = Variable(torch.randn(mu.size()), requires_grad = False)

            if mu.is_cuda:
                epsilon = epsilon.cuda()

            #z = ϵ * σ + µ
            z = epsilon.mul(std).add_(mu)

            return z

        else:
            return mu

    def forward(self, x):
        """
        Forward pass through Encoder-Decoder.
        """

        mu, logvar = self.encoder(x.view(-1, self.input_dim))
        z = self.reparam(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def loss(self, reconstruction, x, mu, logvar):

        """
        Variationa loss, i.e. evidence lower bound (ELBO)
        It uses closed form of KL divergence between two Gaussians.

        Params
        ------
        x (torch.tensor)
            Minibatch of input data.

        mu (torch.tensor)
            Output of mean of a stochastic gaussian layer.
            Used to compute KL-divergence.

        logvar(torch.tensor)
            Output of log(σ^2) of a stochastic gaussian layer.
            Used to compute KL-divergence.


        Returns
        -------
        variational_loss(torch.tensor)
            Sum of KL divergence plus reconstruction loss.

        """

        recon_loss = self.reconstruction_loss(
            reconstruction, x.view(-1, self.input_dim)
            )

        # Gaussian - Std. Gaussian KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #  Normalize by same number of elems as in reconstruction
        KLD /= x.view(-1, self.input_dim).data.shape[0] * self.input_dim

        variational_loss = recon_loss + self.beta*KLD

        return variational_loss

    def get_z(self, x):

        """
        Encode a batch of data points into their latent representation z.
        """

        mu, logvar = self.encoder(x.view(-1, self.input_dim))

        z = self.reparam(mu, logvar)

        return z

    def project_data_into_latent_cell(self, data_loader):
        """
        Generator function to project dataset into latent space.

        Params
        ------
        data_loader (torch.DataLoader)
            DataLoader which handles the batches and parallelization.

        n_feats (int)
            Number of dimensions of original dataset.

        model (nn.Module)
            Neural network model to be used for inference.

        Returns (yields)
        -------
        encoded_sample (array-like generator)
            Generator single encoded sample in a numpy array format.
        """

        cuda = torch.cuda.is_available()

        # Iterate through all of the batches in the DataLoader
        for batch_x in tqdm.tqdm(data_loader):

            if cuda:
                batch_x = batch_x.cuda()

            # Reshape to eliminate batch dimension
            batch_x = batch_x.view(-1, self.input_dim)

            # Project into latent space and convert tensor to numpy array
            if cuda:
                batch_x_preds = self.get_z(batch_x.float()).cpu().detach().numpy()
            else:
                batch_x_preds = self.get_z(batch_x.float()).detach().numpy()

            # For each sample decoded yield the line reshaped
            # to only a single array of size (latent_dim)
            for x in batch_x_preds:
                encoded_sample = x.reshape(self.embedding_dim)

                yield encoded_sample


    def get_gaussian_samples(self, x, n_samples = 10):
        """
        Generates n_samples from the posterior distribution
        p(z|x) ~ Normal (µ (x), diag(σ(x))) from a single data point x.
        This function is designed to be used after training.
        """
        x = x.view(-1, self.input_size)

        mu, log_var = self.encoder(x)

        var = log_var.exp_()

        cov_mat = torch.diag(var.flatten())

        gaussian = td.MultivariateNormal(mu, cov_mat)

        gaussian_samples = gaussian.sample((n_samples,))

        return gaussian_samples

class JointEmbedding(nn.Module):
    """
    Joint embedding using contrastive learning training.
    """
    def __init__(self, mol_encoder, cell_encoder):
        super(JointEmbedding, self).__init__()

        mol_encoder = copy.deepcopy(mol_encoder)
        cell_encoder = copy.deepcopy(cell_encoder)

        self.molecule_encoder = mol_encoder
        self.cell_encoder = cell_encoder

        # Learn temperature parameter
        self.logit_scale =nn.Parameter(torch.rand(1)*4)

    def encode_molecule(self, molecule_batch):
        molecule_embedding = self.molecule_encoder.project(
            molecule_batch
        )
        return molecule_embedding

    def encode_cell(self, cell_batch):
        cell_embedding = self.cell_encoder.project(cell_batch)
        return cell_embedding

    def forward(self, molecules, cells):
        cell_embedding = self.encode_cell(cells)
        mol_embedding = self.encode_molecule(molecules)

        # Normalize embeddings : make unit vectors
        cell_embedding = cell_embedding / cell_embedding.norm(dim= -1, keepdim = True)
        mol_embedding = mol_embedding / mol_embedding.norm(dim= -1, keepdim = True)

        # Get cosine similarities
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * mol_embedding@cell_embedding.t()

        return logits


def cov_mat(X:torch.Tensor, Y:torch.Tensor = None)->torch.Tensor:
    """
    Returns covariance or cross-covariance matrix. Assumes mean centered data.
    Note: Always returns a cross-covariance matrix with larger rows than columns.

    Params
    ------
	X (torch.Tensor)
		Dataset with mean centered columns.
	Y (torch.Tensor)
		Second dataset to calculate cross-covariance matrix.


    Returns
    -------
	cov(torch.Tensor)
		Covariance or cross-covariance matrix respectively.
    """
    N, k = X.shape

    #Compute cross-covariance if Y present
    if Y is not None:
        print('Running cross-covariance matrix mode.')
        N_, d = Y.shape

        # Check matching dimensions in first axis
        assert N== N_

        #Always return longer rows
        if k>=d:
            cov = (1/N) * X.T@Y
        else:
            cov = (1/N)*Y.T@X
        return cov

    cov = (1/N) * X.T@X

    return cov

def cca_torch(
	Xa:torch.Tensor, Xb:torch.Tensor, n_components:int = None
	)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns canonical correlations (rhos) and weight matrices (Wa, Wb)
	corresponding to canonical weights using CCA.

	Note: Function assumes that Xa has the same or a smaller dimensionality than Xb,
	and that data is mean centered.

	Also assumes torch.linalg is imported as `LA`.

	Params
	------
	Xa, Xb (torch.Tensor)
		Paired datasets. It is assumed that they are column-centered, i.e. that
		the mean of each column is equal to one.

	n_components

	Returns
	-------
	rhos (torch.Tensor)
		Canonical correlations between Za^i and Zb^i.

	Wa (torch.Tensor)
		Canonical weights for dataset Xa. Multiply to get embeddings Za = Xa@Wa

	Wb (torch.Tensor)
		Canonical weights for dataset Xb.
    """

    device = Xa.device

    N, dim_a = Xa.shape
    n, dim_b = Xb.shape

    assert N==n, "Xa and Xb should have the same number of datapoints (rows)."

    assert dim_a<=dim_b, "Xa should have a smaller number of columns that Xb."

    r = 1e-4 # add regularization for invertibility

    # Calculate covariance matrices
    Caa = cov_mat(Xa) + r*torch.eye(dim_a, device = device)
    Cbb = cov_mat(Xb) + r*torch.eye(dim_b, device = device)
    Cba = cov_mat(Xb, Xa)
    Cab = Cba.T

    cross_corr = LA.inv(Cbb)@Cba@LA.inv(Caa)@Cab

    # Eigenvectors are the canonical weights for matrix Xb
    eigvals, Wb = torch.eig(cross_corr, eigenvectors = True)

    # Canonical correlations
    rhos = torch.sqrt(eigvals)[:, 0]

    # Canonical weights for matrix Xa
    Wa = torch.zeros((dim_a,dim_a), device = device)

    # Get canonical weights for Wa
    for i in range(dim_a):
        alpha = LA.inv(Caa) @ Cab @ Wb[:, i].reshape(-1,1)
        Wa[:, i] = alpha.flatten()

    # Return top-n eigenvectors
    if n_components is not None:
        Wa = Wa[:, :n_components]
        Wb = Wb[:, :n_components]

    return rhos, Wa, Wb
