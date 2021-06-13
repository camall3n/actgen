import torch

from .. import nnutils

class InverseModel(nnutils.Network):
    """
    Description:
        Predicts the action distribution given a (state, next_state) pair.

    Parameters:
        - n_actions : Int
            The number of actions or action dimensions
        - params : dict
            Dictionary of hyperparameters
    """
    def __init__(self, n_actions, params, discrete=True):
        super(InverseModel, self).__init__()
        self.discrete = discrete
        self.body = torch.nn.Sequential(
            torch.nn.Linear(params['latent_dim'] * 2, params['inv_layer_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['inv_layer_size'], params['inv_layer_size']),
            torch.nn.ReLU(),
        )
        if self.discrete:
            self.log_pr_linear = torch.nn.Linear(params['inv_layer_size'], n_actions)
            self.cross_entropy = torch.nn.CrossEntropyLoss()
        else:
            self.mean_linear = torch.nn.Linear(params['inv_layer_size'], n_actions)
            self.log_std_linear = torch.nn.Linear(params['inv_layer_size'], n_actions)
            self.log_sig_min = params['inv_log_std_min']
            self.log_sig_max = params['inv_log_std_max']

    def forward(self, z0, z1):
        """
        Compute the action distribution for two subsequent states.

        If the action space is discrete, the output will be log probabilities,
        (It's the output of a linear layer without a softmax, so effectively 
        it is the log of probabilities )
        otherwise it will be the mean and std deviation of a multivariate
        Gaussian distribution with diagonal covariance matrix.

        Typically z0, z1 will be latent embeddings (e.g. the output of an encoder).
        """
        context = torch.cat((z0, z1), -1)
        shared_vector = self.body(context)

        if self.discrete:
            return self.log_pr_linear(shared_vector)
        else:
            mean = self.mean_linear(shared_vector)
            log_std = self.log_std_linear(shared_vector)
            log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
            std = log_std.exp()
            return mean, std

    def compute_loss(self, z0, a, z1):
        """
        Compute cross-entropy loss for a batch of states, actions, and next-states.

        Typically z0, z1 will be latent embeddings (e.g. the output of an encoder).
        Actions should have dtype torch.long if the action space is discrete,
        otherwise they should have dtype torch.float32.
        """
        if self.discrete:
            log_pr_actions = self.forward(z0, z1)
            l_inverse = self.cross_entropy(input=log_pr_actions, target=a)
        else:
            mean, std = self.forward(z0, z1)
            cov = torch.diag_embed(std, dim1=1, dim2=2)
            normal = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
            log_pr_action = normal.log_prob(a)
            l_inverse = -1 * log_pr_action.mean(dim=0)

        return l_inverse
