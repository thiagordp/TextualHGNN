import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Any
import logging
import inspect

import tqdm


class GraphFidelityEvaluator:
    """
    Unified Fidelity Evaluator for Hierarchical (DiffPool) and Flat (GCN/GAT) GNNs.

    Principles:
    1. Level 0 (Input): Uniform Feature Masking.
       - DiffPool: Mask features + Force override_S (Frozen Topology).
       - GCN/GAT: Mask features + Static Topology (Frozen by definition).
    2. Level > 0 (Latent): Feature Masking + Frozen Topology (DiffPool only).
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

        # Detect if model is DiffPool-like (returns tuple, accepts override_S)
        self.is_diffpool = self._check_is_diffpool()

    def _get_probs(self, output: torch.Tensor) -> torch.Tensor:
        """
        Robustly converts model output to Probabilities [0, 1].
        Works for both Logits (raw scores) and LogSoftmax.
        """
        # If output is already probabilities (unlikely for GNNs, but safety check)
        if output.min() >= 0 and output.max() <= 1.0 and torch.abs(output.sum(dim=1) - 1.0).mean() < 1e-4:
            return output

        # Apply Softmax to get probabilities
        # Softmax(Logits) -> Probs
        # Softmax(LogSoftmax) -> Probs (Shift Invariance property)
        return F.softmax(output, dim=1)

    def _check_is_diffpool(self) -> bool:
        """Introspects model forward signature to check for 'override_S' argument."""
        signature = inspect.signature(self.model.forward)
        return 'override_S' in signature.parameters

    def _run_forward(self,
                     x: torch.Tensor,
                     adj: Optional[torch.Tensor] = None,
                     mask: Optional[torch.Tensor] = None,
                     edge_index: Optional[torch.Tensor] = None,
                     batch: Optional[torch.Tensor] = None,
                     override_S: Optional[List[torch.Tensor]] = None,
                     debug: bool = False) -> Tuple[torch.Tensor, Any]:
        """
        Polymorphic forward pass handling both Dense (DiffPool) and Sparse (GCN) inputs.

        Returns:
            (log_probs, aux_outputs)
            - aux_outputs is s_matrices for DiffPool, None for GCN.
        """
        # 1. DiffPool (Dense) Path
        if self.is_diffpool:
            if adj is None:
                raise ValueError("DiffPool model requires 'adj' input.")

            # DiffPool signature: forward(x, adj, mask=None, debug=False, override_S=None)
            out = self.model(x, adj, mask, debug=debug, override_S=override_S)

            # Unpack DiffPool Tuple
            # Assumes format: (log_probs, ..., (s01, s12)) OR (log_probs, ..., (s01, s12), None)
            if isinstance(out, tuple):
                log_probs = out[0]
                # We assume the last element contains the S matrices (based on previous DiffPoolMinCut)
                # If debug=True, it might return more complex nested tuples.
                # Let's robustly grab the last item or specific debug item.
                aux_data = out[-1] if debug else None
                # Note: In non-debug mode, DiffPool typically returns (log_probs, l_loss, e_loss)
                # We only need S matrices which are usually ONLY available in debug mode or we must cache them.
                # *Adjustment*: get_reference uses debug=True to get S. Ablation steps use debug=False (fast).
            else:
                log_probs = out
                aux_data = None

            return log_probs, aux_data

        # 2. GCN/GAT (Sparse) Path
        else:
            if edge_index is None:
                # Fallback: Try to convert Dense adj to Sparse if edge_index missing?
                # For now, assume caller provides correct inputs.
                pass

                # Standard GNN signature: forward(x, edge_index, batch)
            # Some might take (x, adj) if they are DenseGCN but not DiffPool.
            # We try standard PyG args first.
            if edge_index is not None:
                out = self.model(x, edge_index, batch)
            else:
                # Fallback for DenseGCN/GraphAtt without DiffPool logic
                out = self.model(x, adj, mask)

            # Handle Models that return Tuples (logits, loss, ...)
            if isinstance(out, tuple):
                return out[0], None

            # Standard GNNs usually return just log_probs (or logits)
            return out, None

    def get_reference_prediction(self,
                                 x: torch.Tensor,
                                 adj: Optional[torch.Tensor] = None,
                                 mask: Optional[torch.Tensor] = None,
                                 edge_index: Optional[torch.Tensor] = None,
                                 batch: Optional[torch.Tensor] = None) -> Tuple[float, Any, int]:
        """
        Runs baseline. Returns (prob, s_matrices_or_None, pred_class).
        """
        with torch.no_grad():
            # For DiffPool, we need debug=True to extract S matrices for later freezing
            # For GCN, debug arg might cause error if not supported, so we check self.is_diffpool

            logits, aux_data = self._run_forward(
                x, adj, mask, edge_index, batch,
                debug=self.is_diffpool  # Only use debug mode for DiffPool
            )

            # Use Softmax instead of Exp
            probs = self._get_probs(logits)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class].item()

            # Handle S-matrix extraction for DiffPool debug output
            s_matrices = None
            if self.is_diffpool and aux_data is not None:
                # In your DiffPoolMinCut debug return:
                # (log_softmax, ..., (emb_l1, adj_l1), (emb_l2, adj_l2), cluster_logits, (s01, s12))
                # The last item is (s01, s12).
                # If 'aux_data' is just the last item tuple:
                if isinstance(aux_data, tuple) and len(aux_data) >= 1:
                    s_matrices = aux_data  # Assign (s01, s12)

        return pred_prob, s_matrices, pred_class

    def fidelity_level_0(self,
                         x: torch.Tensor,
                         explanation_nodes: List[int],
                         target_class: int,
                         baseline_prob: float,
                         # Optional args for model types
                         adj: Optional[torch.Tensor] = None,
                         mask: Optional[torch.Tensor] = None,
                         edge_index: Optional[torch.Tensor] = None,
                         batch: Optional[torch.Tensor] = None,
                         s_matrices: Optional[Any] = None) -> Tuple[float, float]:
        """
        Calculates Fidelity at Level 0 (Input) for ANY Graph Model.
        """
        num_nodes = x.size(1) if x.dim() > 2 else x.size(0)  # Handle [B, N, F] or [N, F]

        # --- Fidelity+ (Necessity) ---
        x_plus = x.clone()

        # Mask features
        # Support both [B, N, F] (Dense) and [N, F] (Sparse)
        if x_plus.dim() == 3:  # Dense [Batch, Node, Feat]
            for node_idx in explanation_nodes:
                if node_idx < x_plus.size(1):
                    x_plus[:, node_idx, :] = 0.0
        else:  # Sparse [Node, Feat]
            x_plus[explanation_nodes, :] = 0.0

        with torch.no_grad():
            # log_probs, _ = self._run_forward(
            #     x_plus, adj, mask, edge_index, batch,
            #     override_S=s_matrices  # Only used if is_diffpool=True
            # )
            # prob_plus = torch.exp(log_probs)[0, target_class].item()
            # --- FIX: Use Softmax ---
            logits_plus, _ = self._run_forward(
                x_plus, adj, mask, edge_index, batch, override_S=s_matrices
            )
            probs_plus = self._get_probs(logits_plus)
            prob_plus = probs_plus[0, target_class].item()

        fid_plus = baseline_prob - prob_plus

        # --- Fidelity- (Sufficiency) ---
        x_minus = x.clone()

        # Create Keep Mask
        if x_minus.dim() == 3:
            keep_mask = torch.zeros(x_minus.size(1), dtype=torch.bool, device=self.device)
            for node_idx in explanation_nodes:
                if node_idx < x_minus.size(1):
                    keep_mask[node_idx] = True
            x_minus[:, ~keep_mask, :] = 0.0
        else:
            keep_mask = torch.zeros(x_minus.size(0), dtype=torch.bool, device=self.device)
            keep_mask[explanation_nodes] = True
            x_minus[~keep_mask, :] = 0.0

        with torch.no_grad():
            # log_probs, _ = self._run_forward(
            #     x_minus, adj, mask, edge_index, batch,
            #     override_S=s_matrices
            # )
            # prob_minus = torch.exp(log_probs)[0, target_class].item()

            logits_minus, _ = self._run_forward(
                x_minus, adj, mask, edge_index, batch, override_S=s_matrices
            )
            # --- FIX: Use Softmax ---
            probs_minus = self._get_probs(logits_minus)
            prob_minus = probs_minus[0, target_class].item()

        fid_minus = baseline_prob - prob_minus

        return fid_plus, fid_minus

    # Fidelity_level_L remains strictly for DiffPool (checks self.is_diffpool) ...
    def fidelity_level_L(self,
                         x: torch.Tensor,
                         adj: torch.Tensor,
                         mask: torch.Tensor,
                         explanation_clusters: List[int],
                         level: int,
                         s_matrices: List[torch.Tensor],
                         target_class: int,
                         baseline_prob: float) -> Tuple[float, float]:

        if not self.is_diffpool:
            raise ValueError("Level L Fidelity is only defined for DiffPool models.")

        # Determine target module
        target_module = None
        if level == 1:
            target_module = self.model.gnn2_embed
        elif level == 2:
            target_module = self.model.lin

        # Fidelity+ Hook
        def hook_plus(module, args):
            input_x = args[0].clone()
            for c_idx in explanation_clusters:
                if c_idx < input_x.size(1):
                    input_x[:, c_idx, :] = 0.0
            return (input_x, *args[1:])

        handle = target_module.register_forward_pre_hook(hook_plus)
        try:
            # with torch.no_grad():
            #     out = self.model(x, adj, mask, override_S=s_matrices)
            #     prob_plus = torch.exp(out[0])[0, target_class].item()
            with torch.no_grad():
                out = self.model(x, adj, mask, override_S=s_matrices)
                logits = out[0] if isinstance(out, tuple) else out
                # --- Use Softmax ---
                prob_plus = self._get_probs(logits)[0, target_class].item()
        finally:
            handle.remove()

        fid_plus = baseline_prob - prob_plus

        # Fidelity- Hook
        def hook_minus(module, args):
            input_x = args[0].clone()
            keep_mask = torch.zeros(input_x.size(1), dtype=torch.bool, device=input_x.device)
            for c_idx in explanation_clusters:
                if c_idx < input_x.size(1):
                    keep_mask[c_idx] = True
            input_x[:, ~keep_mask, :] = 0.0
            return (input_x, *args[1:])

        handle = target_module.register_forward_pre_hook(hook_minus)
        try:
            # with torch.no_grad():
            #     out = self.model(x, adj, mask, override_S=s_matrices)
            #     prob_minus = torch.exp(out[0])[0, target_class].item()
            with torch.no_grad():
                out = self.model(x, adj, mask, override_S=s_matrices)
                logits = out[0] if isinstance(out, tuple) else out
                # --- Use Softmax ---
                prob_minus = self._get_probs(logits)[0, target_class].item()
        finally:
            handle.remove()

        fid_minus = baseline_prob - prob_minus

        return fid_plus, fid_minus


def calculate_fidelity(model: torch.nn.Module,
                       x: torch.Tensor,
                       explanation: List[int],
                       level: int = 0,
                       # Flexible args
                       adj: Optional[torch.Tensor] = None,
                       mask: Optional[torch.Tensor] = None,
                       edge_index: Optional[torch.Tensor] = None,
                       batch: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """
    Unified entry point for Fidelity Calculation.
    Supports DiffPool (Dense) and GCN/GAT (Sparse).
    """
    evaluator = GraphFidelityEvaluator(model, x.device)

    # 1. Get Baseline
    baseline_prob, s_matrices, target_class = evaluator.get_reference_prediction(
        x, adj, mask, edge_index, batch
    )

    # 2. Dispatch
    if level == 0:
        return evaluator.fidelity_level_0(
            x=x,
            explanation_nodes=explanation,
            target_class=target_class,
            baseline_prob=baseline_prob,
            adj=adj, mask=mask, edge_index=edge_index, batch=batch,
            s_matrices=s_matrices
        )
    else:
        # Requires DiffPool inputs
        if adj is None or s_matrices is None:
            raise ValueError(f"Level {level} requires DiffPool inputs (adj) and internals (s_matrices).")

        return evaluator.fidelity_level_L(
            x, adj, mask, explanation, level, s_matrices, target_class, baseline_prob
        )


# TODO: Understand the whole code file and debug the function below.
def jaccard_distance(list_a: List[int], list_b: List[int]) -> float:
    """
    Computes Jaccard Distance between two sets of indices.
    Formula: 1 - (|A n B| / |A u B|)
    Returns 0.0 if sets are identical, 1.0 if disjoint.
    """
    set_a = set(list_a)
    set_b = set(list_b)

    if not set_a and not set_b:
        return 0.0  # Both empty, consider stable

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    if union == 0:
        return 1.0  # Should not happen if check above passes

    return 1.0 - (intersection / union)


def calculate_stability(model: torch.nn.Module,
                        explainer_func: Any,
                        x: torch.Tensor,
                        target_class: int = None,
                        # Flexible args
                        adj: Optional[torch.Tensor] = None,
                        mask: Optional[torch.Tensor] = None,
                        edge_index: Optional[torch.Tensor] = None,
                        batch: Optional[torch.Tensor] = None,
                        noise_std: float = 0.01,
                        n_samples: int = 10,
                        **explainer_kwargs) -> float:
    """
    Calculates Graph Explanation Stability (Instability) [Agarwal et al. 2023].
    Measures how much the explanation changes under small feature perturbations.

    Args:
        model: The GNN model.
        explainer_func: A callable function that takes (model, x, adj/edge_index, ...)
                        and returns a List[int] of important node indices.
        x: Input node features.
        target_class: The class predicted by the model on the original input.
                      If None, it is computed automatically.
        noise_std: Standard deviation of Gaussian noise added to features.
        n_samples: Number of perturbation samples to average over.
        **explainer_kwargs: Additional arguments to pass to the explainer_func.

    Returns:
        Average Jaccard Distance (Float). Lower means more stable.
    """
    evaluator = GraphFidelityEvaluator(model, x.device)

    # 1. Get Reference (Original) Prediction and Explanation
    orig_prob, s_mats, pred_class = evaluator.get_reference_prediction(
        x, adj, mask, edge_index, batch
    )

    # Use provided target_class or the model's prediction
    target_label = target_class if target_class is not None else pred_class

    # Generate Original Explanation
    # Note: We assume explainer_func signature matches standard call
    explanation_orig = explainer_func(
        model=model,
        x=x,
        adj=adj,
        edge_index=edge_index,
        target_class=target_label,
        **explainer_kwargs
    )

    distances = []
    valid_samples = 0

    for _ in range(n_samples):
        # 2. Perturb Features: x' = x + N(0, std)
        noise = torch.randn_like(x) * noise_std
        x_perturbed = x + noise

        # 3. Check Prediction Consistency constraint
        # "The model's prediction must not change between original and perturbed graph"
        with torch.no_grad():
            log_probs_p, _ = evaluator._run_forward(
                x_perturbed, adj, mask, edge_index, batch,
                override_S=None  # <--- CHANGED: Let the model re-cluster dynamically
            )
            pred_class_p = torch.argmax(torch.exp(log_probs_p), dim=1).item()

        if pred_class_p == target_label:
            # 4. Generate Perturbed Explanation
            explanation_perturbed = explainer_func(
                model=model,
                x=x_perturbed,
                adj=adj,
                edge_index=edge_index,
                target_class=target_label,
                **explainer_kwargs
            )

            # 5. Compute Distance
            dist = jaccard_distance(explanation_orig, explanation_perturbed)
            distances.append(dist)
            valid_samples += 1

    # If no perturbations resulted in the same class, stability is undefined (or we can return 0.0/1.0)
    # Returning None or 0.0 with a warning is safer.
    if valid_samples == 0:
        logging.warning("Stability: No perturbed samples maintained the original prediction class.")
        return 0.0

    return sum(distances) / valid_samples


def calculate_stability_adapted(model, explainer_func, x, adj, mask, target_class,
                                edge_index=None, n_samples=5, noise_std=0.01, **kwargs):
    """
    Calculates Stability allowing DiffPool to re-cluster (override_S=None).
    """
    evaluator = GraphFidelityEvaluator(model, x.device)

    expl_orig = explainer_func(
        model,
        x,
        adj,
        edge_index,
        target_class,
        **kwargs
    )

    distances = []
    valid_count = 0

    for _ in tqdm.tqdm(range(n_samples), desc="Stability: Perturbing samples"):
        noise = torch.randn_like(x) * noise_std
        x_perturbed = x + noise

        with torch.no_grad():
            # log_probs_p, _ = evaluator._run_forward(
            #     x_perturbed, adj, mask, override_S=None
            # )
            # pred_p = torch.argmax(torch.exp(log_probs_p), dim=1).item()
            logits_p, _ = evaluator._run_forward(
                x_perturbed, adj, mask, edge_index, override_S=None
            )
            # Use Softmax here too!
            probs_p = evaluator._get_probs(logits_p)
            pred_class_p = torch.argmax(probs_p, dim=1).item()

        if pred_class_p == target_class:
            expl_p = explainer_func(
                model, x_perturbed, adj, edge_index, target_class, **kwargs
            )
            distances.append(jaccard_distance(expl_orig, expl_p))
            valid_count += 1

    return sum(distances) / valid_count if valid_count > 0 else 0.0


def calculate_sparsity(explanation: Union[List[int], torch.Tensor], num_elements: int) -> float:
    """
        Calculates the Sparsity of an explanation.
        Formula: 1 - (|Explanation| / |Total_Elements|)

        Args:
            explanation: A list of indices or a binary mask tensor identifying important nodes/features.
            num_elements: The total number of nodes/features in the original graph.

        Returns:
            float: Sparsity score (0.0 to 1.0). Higher is usually better (more concise).
        """

    # 1. Determine size of explanation
    if isinstance(explanation, list):
        # Case: List of indices (e.g., from SubgraphX)
        expl_size = len(set(explanation))
    elif isinstance(explanation, torch.Tensor):
        if explanation.dtype == torch.bool:
            # Case: Boolean Mask
            expl_size = explanation.sum().item()
        elif explanation.numel() == num_elements:
            # Case: Soft Mask (floats) - usually thresholded, but here we assume binary input
            # or we treat it as "amount of information"
            expl_size = (explanation > 0).sum().item()
        else:
            # Case: Tensor of indices
            expl_size = explanation.numel()
    else:
        raise ValueError(f"Unsupported explanation type: {type(explanation)}")

    # 2. Safety Check
    if num_elements == 0:
        return 0.0

    # 3. Calculate Ratio
    ratio = expl_size / num_elements

    # Clamp ratio to [0, 1] to handle edge cases where |M| > N (shouldn't happen but good for safety)
    ratio = max(0.0, min(1.0, ratio))

    return 1.0 - ratio
