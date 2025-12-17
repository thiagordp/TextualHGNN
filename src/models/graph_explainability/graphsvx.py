import torch
import torch.nn as nn
import numpy as np
import scipy.special
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from tqdm import tqdm
from copy import deepcopy


class GraphSVX(nn.Module):
    """
    Adapted GraphSVX for Graph Classification on DiffPool.

    Changes from original:
    1. Works with Raw Logits (removes .exp() calls).
    2. Operates on Dense Adjacency (compatible with DiffPool).
    3. Self-contained (no external dependencies).
    """

    def __init__(self, model, device, num_samples=100, reg_type='lasso'):
        super().__init__()
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.reg_type = reg_type
        self.model.eval()

    def shapley_kernel(self, s, M):
        """
        Computes the Shapley Kernel weights.
        s: size of coalition (number of included nodes).
        M: total number of nodes.
        """

        weights = []
        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                weights.append(1000.0)  # Enforce high weight for full/empty
            else:
                # Combinatorial weight
                try:
                    w = (M - 1) / (scipy.special.binom(M, a) * a * (M - a))
                except:
                    w = 1e-6  # Numerical stability fallback
                weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def mask_generation(self, M, num_samples):
        """
        Generates random binary masks (coalitions).

        Args:
            M: Number of features/nodes
            num_samples: Number of samples to generate
        Returns:
        """
        # 1. Always include Full (1s) and Empty (0s)
        z_ = torch.randint(0, 2, (num_samples, M)).float()

        # Enforce Full and Empty samples at the start
        z_[0, :] = 1.0
        z_[1, :] = 0.0

        # Calculate kernel weights
        s = z_.sum(dim=1)
        # A weight vector used to train the Weighted Linear Regression.
        # The regression will prioritize fitting the model accurately on the Full, Empty, and "Almost Full/Empty" samples.
        weights = self.shapley_kernel(s, M)
        return z_, weights

    def get_predictions(self, x_dense, adj_dense, mask_dense, z_):
        """
        Computes f(z) for all masks.
        Args:
            x_dense: input feature matrix
            adj_dense: adjecency matrix
            mask_dense: Masks
            z_: ??
        Returns:
            Predictions.
        """
        num_samples = z_.shape[0]
        num_nodes = x_dense.size(1)

        predictions = []

        # Mean feature value for "background" (Missingness)
        # Using Zero baseline is standard for sparsity, or mean of current graph
        bg_val = 0.0

        for i in range(num_samples):
            # z_[i] is a binary mask of shape [num_nodes]
            mask_vec = z_[i].to(self.device)

            # 1. Perturb Features: x * mask
            # x_dense: [1, N, F]
            x_perturbed = x_dense * mask_vec.view(1, num_nodes, 1)

            # 2. Perturb Structure: Zero out rows/cols of removed nodes
            # This is crucial for DiffPool to stop "seeing" the connections
            adj_perturbed = adj_dense.clone()

            # Create a 2D mask (outer product) to keep only edges between KEPT nodes
            # mask_2d[i, j] = 1 only if node i AND node j are kept
            mask2d = torch.outer(mask_vec, mask_vec).unsqueeze(0)  # [1, N, N]
            adj_perturbed = adj_perturbed * mask2d

            # 3. Forward Pass
            with torch.no_grad():
                # DiffPoolWrapper returns logits
                out = self.model(x_perturbed, adj_perturbed, mask_dense)
                if isinstance(out, tuple): out = out[0]

                # We store the Raw Logit for the predicted class
                # (We don't need the class here, just the vector to regress on)
                predictions.append(out.cpu().numpy())

        return np.vstack(predictions)  # [num_samples, num_classes]

    def explain(self, x, adj, target_class, top_k=5):
        """
        Main explanation loop.
        Args:
            x: feature matrix [1, N, F]
            adj: adjacency matrix [1, N, N]
            target_class: True Class
            top_k: Number of top k features/nodes

        Returns:
            Top k indices.
        """

        num_nodes = x.size(1)

        # 1. Generate Coalitions
        z_, weights = self.mask_generation(num_nodes, self.num_samples)

        # 2. Get Model Outputs (Logits)
        # Returns [samples, classes]
        y_pred = self.get_predictions(x, adj, None, z_)

        # 3. Select Target Class Logits
        y_target = y_pred[:, target_class]

        # 4. Fit Weighted Linear Regression (Surrogate)
        # We try Lasso for sparsity, fallback to Ridge if it fails
        Z_np = z_.numpy()
        W_np = weights.numpy()

        try:
            if self.reg_type == 'lasso':
                reg = Lasso(alpha=0.01, fit_intercept=True)
            else:
                reg = Ridge(alpha=0.1, fit_intercept=True)

            reg.fit(Z_np, y_target, sample_weight=W_np)
            shapley_values = reg.coef_
        except Exception as e:
            # Fallback to simple Linear Regression
            reg = LinearRegression()
            reg.fit(Z_np, y_target, sample_weight=W_np)
            shapley_values = reg.coef_

        # 5. Extract Top-K Nodes
        # Sort by contribution (Highest coefficient = Most important)
        top_k_indices = np.argsort(shapley_values)[::-1][:top_k].tolist()

        return top_k_indices
