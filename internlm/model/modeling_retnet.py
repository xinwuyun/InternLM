# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import copy
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm
    
from internlm.core.context import IS_TENSOR_PARALLEL, ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.initialize.initialize_tensor import normal_, scaled_init_method_normal
from internlm.model.embedding import Embedding1D
from internlm.model.utils import gather_forward_split_backward, try_import_RMSNorm
from internlm.solver.pipeline_utils import partition_uniform
from internlm.utils.checkpoint import activation_checkpoint
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER
from internlm.model.ssm.h3 import H3
MODEL_TYPE = "retnet"

logger = get_logger(__file__)
RMSNorm = try_import_RMSNorm()

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x
    
def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError

def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module

def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn

class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)

class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1
        
class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        value_factor=2,
        gate_fn="swish",
    ):
        super().__init__()
        self.args = args
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim * self.factor, bias=True))
        self.g_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim * self.factor, bias=True))
        
        self.out_proj = MultiwayWrapper(args, nn.Linear(embed_dim * self.factor, embed_dim, bias=True))

        self.group_norm = MultiwayWrapper(args, LayerNorm(self.head_dim, eps=args.layernorm_eps, elementwise_affine=False))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(
        self,
        qr, kr, v,
        decay,
        incremental_state
    ):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (1 - 1 / scale).view(self.num_heads, 1, 1) + kv / scale.view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output
    
    def chunk_recurrent_forward(
        self,
        qr, kr, v,
        inner_mask
    ):
        mask, cross_decay, inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)
        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t # bsz * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat, v) # bsz * num_heads * num_value_heads * chunk_len * head_dim
        
        # reduce kv in one chunk
        kv = kr_t @ (v * mask[:, -1, :, None])
        kv = kv.view(bsz, num_chunks, self.num_heads, self.key_dim, self.head_dim)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, self.head_dim).to(v)
        
        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).clamp(min=1)

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        cross_output = (qr * inner_decay) @ kv_recurrent
        output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        return output
    
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(
        self,
        x,
        rel_pos,
        chunkwise_recurrent=False,
        incremental_state=None
    ):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)
        
        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output

class RetNetConfig(object):
    def __init__(self, **kwargs):
        self.dtype = kwargs.pop("dtype", torch.float32)
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.decoder_retention_heads = kwargs.pop("num_retention_heads", 3)
        self.decoder_ffn_embed_dim = int(self.hidden_size * kwargs.pop("mlp_ratio", 4.0))
        self.num_layers = kwargs.pop("num_layers", 12)
        self.dropout = kwargs.pop("drop_rate", 0.0)
        self.residual_in_fp32 = kwargs.pop("residual_in_fp32", False)
        
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        # moe not supported！
        self.moe_freq = kwargs.pop("moe_freq", 0)
        self.moe_top1_expert = kwargs.pop("moe_top1_expert", False)
        self.moe_expert_count = kwargs.pop("moe_expert_count", 0)
        self.moe_gating_use_fp32 = kwargs.pop("moe_gating_use_fp32", True)
        self.moe_eval_capacity_token_fraction = kwargs.pop(
            "moe_eval_capacity_token_fraction", 0.25
        )
        self.moe_second_expert_policy = kwargs.pop("moe_second_expert_policy", "random")
        self.moe_normalize_gate_prob_before_dropping = kwargs.pop(
            "moe_normalize_gate_prob_before_dropping", False
        )
        self.use_xmoe = kwargs.pop("use_xmoe", False)
        
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.multiway = kwargs.pop("multiway", False)
        self.share_decoder_input_output_embed = kwargs.pop(
            "share_decoder_input_output_embed", False
        )
        self.max_target_positions = kwargs.pop("max_target_positions", 1024)
        self.no_output_layer = kwargs.pop("no_output_layer", False)
        self.layernorm_eps = kwargs.pop("layer_norm_epsilon", 1e-5)
        # Blockwise
        self.chunkwise_recurrent = kwargs.pop("chunkwise_recurrent", False)
        self.recurrent_chunk_size = kwargs.pop("recurrent_chunk_size", 512)
        # Text
        self.vocab_size = kwargs.pop("vocab_size", -1)
        # Fairscale
        self.checkpoint_activations = kwargs.pop("checkpoint_activations", False)
        self.fsdp = kwargs.pop("fsdp", False)
        self.ddp_rank = kwargs.pop("ddp_rank", 0)
        self.xpos_rel_pos = kwargs.pop("xpos_rel_pos", False)
        self.xpos_scale_base = kwargs.pop("xpos_scale_base", 512)

        if self.deepnorm:
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.decoder_normalize_before = True
            self.deepnorm = False
        if self.use_xmoe:
            self.moe_normalize_gate_prob_before_dropping = True
            self.moe_second_expert_policy = "random"
            assert self.moe_freq > 0 and self.moe_expert_count > 0

    def override(self, args):
        for hp in self.__dict__.keys():
            if args.pop(hp, None) is not None:
                self.__dict__[hp] = args.pop(hp, None)
    
class RetNetRelPos(nn.Module):
    def __init__(self, args):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, args.hidden_size // args.decoder_retention_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(args.decoder_retention_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = args.recurrent_chunk_size
        
    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[:, None, None]
            inner_decay = inner_decay[:, :, None] / (scale / scale[:, -1, None])
            retention_rel_pos = ((sin, cos), (mask, cross_decay, inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class DecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        depth,
        is_moe_layer=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.hidden_size
        self.dropout_module = torch.nn.Dropout(args.dropout)
        self.residual_in_fp32 = args.residual_in_fp32
        
        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, gpc.config.model.num_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.retention = self.build_retention(self.embed_dim, args)

        self.normalize_before = args.decoder_normalize_before

        self.retention_layer_norm = LayerNorm(self.embed_dim, eps=args.layernorm_eps)

        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.decoder_ffn_embed_dim

        self.ffn = self.build_ffn(
            self.embed_dim,
            self.args,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, eps=args.layernorm_eps)

        if args.deepnorm:
            self.alpha = math.pow(2.0 * gpc.config.model.num_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
            args.subln,
        )

    def build_retention(self, embed_dim, args):
        return MultiScaleRetention(
            args,
            embed_dim,
            args.decoder_retention_heads,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x
    
    def forward(
        self,
        x,
        incremental_state=None,
        chunkwise_recurrent=False,
        retention_rel_pos=None,
    ):
        residual = x
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        if self.normalize_before:
            with torch.autocast(device_type="cuda", dtype=torch.float):
                x = self.retention_layer_norm(x)
        
        x = self.retention(
            x.float(),
            incremental_state=incremental_state,
            rel_pos=retention_rel_pos,
            chunkwise_recurrent=chunkwise_recurrent,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            with torch.autocast(device_type="cuda", dtype=torch.float):
                x = self.retention_layer_norm(x)

        residual = x
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
            
        if self.normalize_before:
            with torch.autocast(device_type="cuda", dtype=torch.float):
                x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x.to(self.ffn.fc1.weight.dtype))
            l_aux = None
        else:
            x, l_aux = self.moe_layer(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            with torch.autocast(device_type="cuda", dtype=torch.float):
                x = self.final_layer_norm(x)

        return x

class RetNetDecoder(nn.Module):
    def __init__(
        self,
        args,
        first=True,
        last=True,
        start_layer_idx=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.first = first
        self.last = last
        self.args = args

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.hidden_size
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if self.first:
            self.embed_tokens = Embedding1D(args.vocab_size, self.embed_dim, dtype=args.dtype)
            if args.layernorm_embedding:
                self.layernorm_embedding = LayerNorm(embed_dim, eps=args.layernorm_eps)
            else:
                self.layernorm_embedding = None
        
        if self.last:
            if args.decoder_normalize_before:
                self.layer_norm = LayerNorm(embed_dim, eps=args.layernorm_eps)
            else:
                self.layer_norm = None
            self.output_projection = self.build_output_projection(args)
            
        self.layers = nn.ModuleList([])
        for i in range(args.num_layers):
            self.layers.append(
                self.build_decoder_layer(
                    args,
                    depth=i + start_layer_idx,
                )
            )

        self.num_layers = len(self.layers)

        

        self.retnet_rel_pos = RetNetRelPos(args)
        self.chunkwise_recurrent = args.chunkwise_recurrent
        self.recurrent_chunk_size = args.recurrent_chunk_size
        

        if args.deepnorm:
            init_scale = math.pow(8.0 * gpc.config.model.num_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if args.subln:
            init_scale = math.sqrt(math.log(gpc.config.model.num_layers * 2))
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.mul_(init_scale)

    def build_output_projection(
        self,
        args,
    ):
        # default False
        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
                dtype=args.dtype
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.hidden_size, args.vocab_size, bias=False, dtype=args.dtype
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.hidden_size**-0.5
            )
        return output_projection

    def build_decoder_layer(
        self, args, depth
    ):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=False,
        )
        # if args.checkpoint_activations:
        #     layer = checkpoint_wrapper(layer)
        # if args.fsdp:
        #     layer = wrap(layer)
        return layer

    def forward_embedding(
        self,
        tokens,
        token_embedding=None,
        incremental_state=None,
    ):
        if incremental_state is not None and not self.is_first_step(incremental_state):
            tokens = tokens[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)
    
    def forward(
        self,
        x=None,
        input_ids=None,
        incremental_state=None,
        features_only=False,
        return_all_hiddens=False,
        token_embeddings=None,
        **kwargs
    ):
        
        # embed tokens
        if self.first:
            x, _ = self.forward_embedding(
                input_ids, token_embeddings, incremental_state
            )
            
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        if self.chunkwise_recurrent and x.size(1) % self.recurrent_chunk_size != 0:
            padding_len = self.recurrent_chunk_size - x.size(1) % self.recurrent_chunk_size
            slen = x.size(1) + padding_len
            x = F.pad(x, (0, 0, 0, padding_len))
        else:
            slen = x.size(1)
        # relative position
        is_first_step = self.is_first_step(incremental_state)
        retention_rel_pos = self.retnet_rel_pos(slen, incremental_state is not None and not is_first_step, chunkwise_recurrent=self.chunkwise_recurrent)
        # decoder layers
        # inner_states = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                if idx not in incremental_state:
                    incremental_state[idx] = {}
            
            x = layer(
                x,
                incremental_state[idx] if incremental_state is not None else None,
                retention_rel_pos=retention_rel_pos,
                chunkwise_recurrent=self.chunkwise_recurrent,
            )
            # inner_states.append(x)
        
        if self.last:
            if self.chunkwise_recurrent and x.size(1) % self.recurrent_chunk_size != 0:
                x = x[:, :x.size(1), :]

            if self.layer_norm is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    x = self.layer_norm(x)

            if not features_only:
                x = self.output_layer(x.to(dtype=self.output_projection.weight.dtype))

        return x

    def output_layer(self, features):
        return self.output_projection(features)

def _build_retnet_config(**kwargs):
    # create a base retnet config
    # args = _build_retnet_base_config(RetNetConfig())
    # args.override(kwargs)
    args = RetNetConfig(**kwargs)
    return args

def _build_generic_model_1d(num_layers, num_chunks, device=torch.device("cuda"), **kwargs):
    """
    build generic model 1d

    Args:
        num_layers (int): The number of layer.
        num_chunks (int): The number of partitions in pipeline parallel.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]
    if gpc.is_rank_for_log():
        logger.info(f"The layer sharding is {all_parts}.")

    models = []

    if kwargs["checkpoint"] is True:
        kwargs["checkpoint_fraction"] = 1.0
    else:
        kwargs["checkpoint_fraction"] = 0

    for start, end in parts:
        kwargs["num_layers"] = end - start
        config = _build_retnet_config(**kwargs)
        if gpc.is_rank_for_log():
            logger.info(config.__dict__)
        chunk = RetNetDecoder(config, 
                              first=start == 0, 
                              last=end == num_layers and len(all_parts[-1]) != 0,
                              start_layer_idx=start
                              ).to(device)

        models.append(chunk)
    torch.distributed.barrier()
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model

@MODEL_INITIALIZER.register_module(module_name=MODEL_TYPE)
def build_retnet_with_cfg(
    num_chunks=1,
    checkpoint=False,
    dtype=torch.float,
    embed_split_hidden=False,
    num_layers=48,
    hidden_size=2048,
    vocab_size=50304,
    embed_grad_scale=1,
    parallel_output=False,
    num_retention_heads=32,
    mlp_ratio=4.0,
    residual_in_fp32=False,
    norm_type="layernorm",
    drop_rate=0,
    apply_post_layer_norm=False,  # pylint: disable=W0613
    layer_norm_epsilon=1e-5,
    is_reward=False,
    dropout_selective_checkpoint=True,
    use_scaled_init: bool = True,
    use_swiglu: bool = True,
    use_flash_attn: bool = True,
    sequence_parallel: bool = False,  # pylint: disable=W0613
):
    """
    Builde model with config

    Args:
        num_chunks (int): The number of partitions in pipeline parallel. 1 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. False by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        embed_split_hidden (bool): Split the embedding layer in the hidden state dimention or vocabulary dimention.
                                    False by default.
        num_layers (int): The number of layer. 48 by default.
        hidden_size (int): The size of hidden state. 2048 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        num_attention_heads (int): The number of attention head. 32 by default.
        mlp_ratio (int): The ratio of MLP layers. 4.0 by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default. It cannot be used temporarily
                                 because this parameter requires inconsistent data types to be passed between pipelines,
                                 which requires significant modifications to internlm.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        drop_rate (float): The dropout rate of input hidden state. 0 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        apply_post_layer_norm (bool): Whether to apply post layer norm. False by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        is_reward (bool): Whether to use reward model. False by default.
        dropout_selective_checkpoint (bool): It can only be enabled when checkpoint is disabled. True by default.
        use_scaled_init (bool): Whether to use scaled init. True by default.
        use_swiglu (bool): Whether to use swiglu. True by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.

    """
    
    cfg = dict(
        hidden_size=hidden_size,
        num_retention_heads=num_retention_heads,
        checkpoint=checkpoint,
        dtype=dtype,
        embed_split_hidden=embed_split_hidden,
        vocab_size=vocab_size,
        embed_grad_scale=embed_grad_scale,
        parallel_output=parallel_output,
        mlp_ratio=mlp_ratio,
        residual_in_fp32=residual_in_fp32,
        norm_type=norm_type,
        drop_rate=drop_rate,
        # attn_drop_rate=attn_drop_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        is_reward=is_reward,
        dropout_selective_checkpoint=dropout_selective_checkpoint,
        use_scaled_init=use_scaled_init,
        use_swiglu=use_swiglu,
        use_flash_attn=use_flash_attn
    )

    return _build_generic_model_1d(num_layers=num_layers, num_chunks=num_chunks, **cfg)
