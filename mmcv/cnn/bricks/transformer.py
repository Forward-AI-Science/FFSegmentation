"""Pure-PyTorch transformer bricks replacing mmcv.cnn.bricks.transformer."""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .drop import build_dropout


class MultiheadAttention(nn.Module):
    """Multi-head attention wrapper compatible with mmcv API."""

    def __init__(self, embed_dims: int, num_heads: int, attn_drop: float = 0.,
                 proj_drop: float = 0., dropout_layer: Optional[dict] = None,
                 init_cfg: Optional[dict] = None, batch_first: bool = False,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=attn_drop,
            batch_first=batch_first)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_pos: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if identity is None:
            identity = query
        if key is None:
            key = query
        if value is None:
            value = key
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out, _ = self.attn(query, key, value,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return identity + self.dropout_layer(self.proj_drop(out))


class FFN(nn.Module):
    """Feed-forward network with residual."""

    def __init__(self, embed_dims: int = 256, feedforward_channels: int = 1024,
                 num_fcs: int = 2, act_cfg: dict = dict(type='ReLU', inplace=True),
                 ffn_drop: float = 0., dropout_layer: Optional[dict] = None,
                 add_identity: bool = True, init_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        assert num_fcs >= 2
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = self._build_activation(act_cfg)
        self.add_identity = add_identity

        layers = []
        in_channels = embed_dims
        for i in range(num_fcs - 1):
            layers.extend([
                nn.Linear(in_channels, feedforward_channels),
                self.activate,
                nn.Dropout(ffn_drop),
            ])
            in_channels = feedforward_channels
        layers.append(nn.Linear(in_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    @staticmethod
    def _build_activation(cfg: dict) -> nn.Module:
        from mmcv.cnn.bricks.activation import build_activation_layer
        return build_activation_layer(cfg)

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        out = self.layers(x)
        out = self.dropout_layer(out)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class BaseTransformerLayer(nn.Module):
    """A basic transformer layer (encoder or decoder)."""

    def __init__(self, attn_cfgs: Union[dict, List[dict]],
                 ffn_cfgs: Union[dict, List[dict]] = dict(
                     type='FFN', embed_dims=256, feedforward_channels=1024,
                     num_fcs=2, ffn_drop=0., act_cfg=dict(type='ReLU', inplace=True)),
                 operation_order: Optional[Tuple[str, ...]] = None,
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: Optional[dict] = None,
                 batch_first: bool = False, **kwargs):
        super().__init__()
        self.batch_first = batch_first

        if isinstance(attn_cfgs, dict):
            attn_cfgs = [attn_cfgs]
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [ffn_cfgs]

        if operation_order is None:
            num_attn = len(attn_cfgs)
            operation_order = tuple(['self_attn'] * num_attn + ['norm'] * num_attn + ['ffn', 'norm'])

        self.operation_order = operation_order
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
        num_ffn = operation_order.count('ffn')
        num_norm = operation_order.count('norm')

        self.attentions = nn.ModuleList()
        for i, acfg in enumerate(attn_cfgs):
            acfg = acfg.copy()
            acfg.pop('type', None)
            self.attentions.append(MultiheadAttention(batch_first=batch_first, **acfg))

        self.ffns = nn.ModuleList()
        for fcfg in ffn_cfgs:
            fcfg = fcfg.copy()
            fcfg.pop('type', None)
            self.ffns.append(FFN(**fcfg))

        from mmcv.cnn.bricks.norm import build_norm_layer
        embed_dims = attn_cfgs[0].get('embed_dims', 256)
        self.norms = nn.ModuleList()
        for _ in range(num_norm):
            _, norm = build_norm_layer(norm_cfg, embed_dims)
            self.norms.append(norm)

    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_pos: Optional[torch.Tensor] = None,
                attn_masks: Optional[List[torch.Tensor]] = None,
                query_key_padding_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None] * (len(self.attentions))
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [attn_masks] * len(self.attentions)

        for layer in self.operation_order:
            if layer in ('self_attn', 'cross_attn'):
                temp_key = temp_value = query if layer == 'self_attn' else key
                query = self.attentions[attn_index](
                    query, temp_key, temp_value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos, key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=(query_key_padding_mask if layer == 'self_attn'
                                      else key_padding_mask))
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity=identity)
                ffn_index += 1
                identity = query

        return query

    @property
    def pre_norm(self) -> bool:
        return self.operation_order[0] == 'norm'


def build_transformer_layer(cfg: dict) -> nn.Module:
    """Build a transformer layer (or any module) from config dict."""
    if cfg is None:
        return nn.Identity()
    cfg = cfg.copy()
    layer_type = cfg.pop('type', 'BaseTransformerLayer')
    from mmengine.registry import MODELS
    cls = MODELS.get(layer_type)
    if cls is not None:
        return cls(**cfg)
    # Fallback: try as a plain class name
    cls_map = {
        'BaseTransformerLayer': BaseTransformerLayer,
        'FFN': FFN,
        'MultiheadAttention': MultiheadAttention,
    }
    cls = cls_map.get(layer_type)
    if cls is not None:
        return cls(**cfg)
    raise KeyError(f'Unknown transformer layer type: {layer_type}')


def build_transformer_layer_sequence(cfg: dict) -> nn.Module:
    """Build a sequence of transformer layers."""
    if cfg is None:
        return nn.Identity()
    cfg = cfg.copy()
    num_layers = cfg.pop('num_layers', 1)
    layer_cfg = cfg.pop('layer_cfg', cfg)
    layers = nn.ModuleList([build_transformer_layer(layer_cfg.copy())
                            for _ in range(num_layers)])
    return layers


__all__ = ['MultiheadAttention', 'FFN', 'BaseTransformerLayer',
           'build_transformer_layer', 'build_transformer_layer_sequence']
