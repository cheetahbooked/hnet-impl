from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from hrm.common import trunc_normal_init_
from hrm.layers import rms_norm, SwiGLU, CastedLinear
from hrm.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    # State vectors: [B, D]
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float

    rms_norm_eps: float = 1e-5
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        # Self Attention removed for vector input
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Applying MLP residual block structure (Residual + MLP + Post Norm)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O components (embedding, lm_head, puzzle logic, positional encoding) removed
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states (D dimensional)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input_vector: torch.Tensor):
        # Input is expected to be vector [B, D]. Cast to forward dtype.
        return input_vector.to(self.forward_dtype)

    def empty_carry(self, batch_size: int):
        # Carry states are [B, D]
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        # States are [B, D], reset_flag view needs to be [B, 1]
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # Input encoding: expect "vector_input"
        input_embeddings = self._input_embeddings(batch["vector_input"])

        # Forward iterations (No sequential/positional info needed)
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings)
        z_H = self.H_level(z_H, z_L)

        # Output is the final vector state z_H
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        final_vector = z_H

        # Q head operates directly on state vector [B, D] -> [B, 2]
        q_logits = self.q_head(z_H).to(torch.float32)
        
        return new_carry, final_vector, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    # Removed puzzle_emb property

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        # Assume input vector is stored under "vector_input"
        main_input = batch["vector_input"]
        batch_size = main_input.shape[0]
        device = main_input.device

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {}
        for k, v in carry.current_data.items():
            # Calculate view for broadcasting based on current tensor rank
            mask_view = (-1,) + (1,) * (batch[k].ndim - 1)
            new_current_data[k] = torch.where(carry.halted.view(mask_view), batch[k], v)

        # Forward inner model
        # Returns (new_inner_carry, final_vector, (q_halt, q_continue))
        new_inner_carry, final_vector, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "final_vector": final_vector,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
