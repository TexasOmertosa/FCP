"""Floating-point settings for the selected FCP experiments."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class Numerics:
    bf16: bool = False
    matmul_tf32: bool = False
    cudnn_tf32: bool = False
    mha_fastpath: bool = True
    attention: str = 'auto'
    project_all_queries: bool = False
    device_frequencies: bool = False
    separate_attention_projections: bool = False

    def metadata(self, device):
        return {**asdict(self if device.type == 'cuda' else Numerics()),
                'device': device.type}

    @contextmanager
    def backend(self, device, upper=None):
        if device.type != 'cuda':
            yield
            return
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.mha.get_fastpath_enabled())
        fields = () if upper is None else (upper.A, upper.B)
        blocks = [block for field in fields for block in field.blocks]
        projections = [field.project_all_queries for field in fields]
        buffers = [field.frequencies for field in fields]
        previous = [(block.attention_backends, block.separate_attention_projections) for block in blocks]
        if self.attention == 'efficient-self-flash-cross' and not blocks:
            raise ValueError('Layer-specific attention requires the FCP model.')
        layer_backends = ((SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION)
                          if self.attention == 'efficient-self-flash-cross' else (None, None))
        attention = {'math': SDPBackend.MATH, 'flash': SDPBackend.FLASH_ATTENTION}.get(self.attention)
        try:
            if self.device_frequencies:
                frequencies = torch.exp(-math.log(10000) * torch.arange(64, device=device) / 63)
                for field in fields:
                    field.frequencies = frequencies
            for field in fields:
                field.project_all_queries = self.project_all_queries
            for block in blocks:
                block.attention_backends = layer_backends
                block.separate_attention_projections = self.separate_attention_projections
            torch.backends.cuda.matmul.allow_tf32 = self.matmul_tf32
            torch.backends.cudnn.allow_tf32 = self.cudnn_tf32
            torch.backends.mha.set_fastpath_enabled(self.mha_fastpath)
            with sdpa_kernel(attention) if attention is not None else nullcontext():
                yield
        finally:
            for field, projection, buffer in zip(fields, projections, buffers):
                field.project_all_queries = projection
                field.frequencies = buffer
            for block, (backends, separate) in zip(blocks, previous):
                block.attention_backends = backends
                block.separate_attention_projections = separate
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[:2]
            torch.backends.mha.set_fastpath_enabled(flags[2])

    def autocast(self, device):
        return torch.autocast(device.type, dtype=torch.bfloat16,
                              enabled=self.bf16 and device.type == 'cuda')

    @contextmanager
    def forward(self, device, upper=None):
        with self.backend(device, upper), self.autocast(device):
            yield


FP32 = Numerics()
MATH_FP32 = Numerics(mha_fastpath=False, attention='math')
MATH_BF16 = Numerics(bf16=True, mha_fastpath=False, attention='math')
PUSHT_DEPLOYMENT = Numerics(bf16=True, cudnn_tf32=True, attention='efficient-self-flash-cross',
                            project_all_queries=True, device_frequencies=True)
FOURROOM_UPPER = Numerics(bf16=True, matmul_tf32=True, cudnn_tf32=True, attention='flash',
                         project_all_queries=True, device_frequencies=True,
                         separate_attention_projections=True)
LIBERO_TRAIN = replace(MATH_BF16, project_all_queries=True, separate_attention_projections=True)
ANTMAZE_UPPER = Numerics(bf16=True, matmul_tf32=True, cudnn_tf32=True, attention='math',
                        project_all_queries=True, device_frequencies=True,
                        separate_attention_projections=True)

UPPER_TRAIN = {
    'pusht': Numerics(bf16=True, matmul_tf32=True, cudnn_tf32=True),
    'cube': PUSHT_DEPLOYMENT,
    'scene2': LIBERO_TRAIN,
    'caddy': LIBERO_TRAIN,
    'antmaze': replace(ANTMAZE_UPPER, mha_fastpath=False),
    'fourroom': FOURROOM_UPPER,
}
UPPER_DEPLOYMENT = {
    'pusht': PUSHT_DEPLOYMENT,
    'cube': Numerics(project_all_queries=True, device_frequencies=True),
    'scene2': replace(LIBERO_TRAIN, bf16=False, cudnn_tf32=True),
    'caddy': LIBERO_TRAIN,
    'antmaze': ANTMAZE_UPPER,
    'fourroom': FOURROOM_UPPER,
}
UPPER_PRIOR_CACHE = {**UPPER_DEPLOYMENT, 'scene2': LIBERO_TRAIN}

PRIOR_TRAIN = {
    'pusht': MATH_BF16, 'cube': FP32, 'scene2': MATH_BF16, 'caddy': MATH_BF16,
    'antmaze': MATH_FP32, 'fourroom': MATH_FP32,
}
PRIOR_VAL = {**PRIOR_TRAIN, 'scene2': MATH_FP32, 'caddy': MATH_FP32}
