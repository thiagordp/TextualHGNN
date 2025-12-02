import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Any
import logging
import inspect


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

            log_probs, aux_data = self._run_forward(
                x, adj, mask, edge_index, batch,
                debug=self.is_diffpool  # Only use debug mode for DiffPool
            )

            # Handle S-matrix extraction for DiffPool debug output
            s_matrices = None
            if self.is_diffpool and aux_data is not None:
                # In your DiffPoolMinCut debug return:
                # (log_softmax, ..., (emb_l1, adj_l1), (emb_l2, adj_l2), cluster_logits, (s01, s12))
                # The last item is (s01, s12).
                # If 'aux_data' is just the last item tuple:
                if isinstance(aux_data, tuple) and len(aux_data) >= 1:
                    s_matrices = aux_data  # Assign (s01, s12)

            probs = torch.exp(log_probs)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class].item()

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
            log_probs, _ = self._run_forward(
                x_plus, adj, mask, edge_index, batch,
                override_S=s_matrices  # Only used if is_diffpool=True
            )
            prob_plus = torch.exp(log_probs)[0, target_class].item()

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
            log_probs, _ = self._run_forward(
                x_minus, adj, mask, edge_index, batch,
                override_S=s_matrices
            )
            prob_minus = torch.exp(log_probs)[0, target_class].item()

        fid_minus = baseline_prob - prob_minus

        return fid_plus, fid_minus

    # ... fidelity_level_L remains strictly for DiffPool (checks self.is_diffpool) ...
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

        # ... (Existing DiffPool implementation from previous step) ...
        # [Copy-Paste the hook-based logic here]
        # For brevity, I will reference the logic from the previous turn
        # as it remains unchanged for the DiffPool branch.

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
            with torch.no_grad():
                out = self.model(x, adj, mask, override_S=s_matrices)
                prob_plus = torch.exp(out[0])[0, target_class].item()
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
            with torch.no_grad():
                out = self.model(x, adj, mask, override_S=s_matrices)
                prob_minus = torch.exp(out[0])[0, target_class].item()
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