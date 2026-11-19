from __future__ import annotations

import os

import torch
import torch.nn as nn

from utils.backbone_utils import DREAM_BACKBONE_FAMILY, inspect_pretrained_backbone
from utils.modeling_llada import LLaDAConfig, LLaDAModel


FULL_CLS_ARCH = "full_cls"
READOUT_CLS_ARCH = "readout_cls"
AUTO_POOLING_MODE = "auto"
MEAN_VISIBLE_POOLING = "mean_visible"
MAX_VISIBLE_POOLING = "max_visible"
LAST_VISIBLE_POOLING = "last_visible"
READOUT_POOLING_MODES = {
    READOUT_CLS_ARCH,
    MEAN_VISIBLE_POOLING,
    MAX_VISIBLE_POOLING,
    LAST_VISIBLE_POOLING,
}
VALUE_MODEL_POOLING_CHOICES = (
    AUTO_POOLING_MODE,
    FULL_CLS_ARCH,
    READOUT_CLS_ARCH,
    MEAN_VISIBLE_POOLING,
    MAX_VISIBLE_POOLING,
    LAST_VISIBLE_POOLING,
)


def ensure_finite_(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x.masked_fill_(torch.isinf(x) & (x > 0), 1e4)
        x.masked_fill_(torch.isinf(x) & (x < 0), -1e4)
        x.masked_fill_(torch.isnan(x), 0.0)
    return x


class ValueModel(nn.Module):
    """Value model with two selectable CLS architectures.

    `full_cls`
    - Current exact CLS-token model.
    - `[CLS]` is part of the 27-layer backbone.

    `readout_cls`
    - Cheap approximation requested by the user.
    - Content tokens are encoded once with no CLS token in the backbone.
    - A learned CLS query reads the final token features with a masked cross-attention head.
    - Counterfactual scoring can then reuse cached content features and only change the CLS mask.
    """

    def __init__(
        self,
        config: LLaDAConfig,
        n_layers: int = 27,
        mask_id: int = 126336,
        architecture: str = FULL_CLS_ARCH,
        readout_heads: int = 8,
        pooling_mode: str = AUTO_POOLING_MODE,
        readout_num_queries: int = 1,
        readout_aggregation: str = "mean",
    ):
        super().__init__()
        self.config = config
        self.mask_id = int(mask_id)
        self.architecture = str(architecture)
        self.pooling_mode = self._resolve_pooling_mode(pooling_mode)
        self.readout_num_queries = max(int(readout_num_queries), 1)
        self.readout_aggregation = str(readout_aggregation)
        self.supports_cached_counterfactual = self.architecture == READOUT_CLS_ARCH

        self.llada = LLaDAModel(config)
        self.llada.transformer.blocks = self.llada.transformer.blocks[:n_layers]

        self.time_embed = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )

        if self.architecture == FULL_CLS_ARCH:
            self.cls_embedding = nn.Parameter(torch.empty(1, 1, config.d_model))
            nn.init.normal_(self.cls_embedding, mean=0.0, std=0.02)
        elif self.architecture == READOUT_CLS_ARCH:
            if self.pooling_mode == READOUT_CLS_ARCH:
                if self.readout_aggregation not in {"mean", "max", "concat"}:
                    raise ValueError(
                        f"Unsupported readout_aggregation={self.readout_aggregation!r}. "
                        "Expected one of: mean, max, concat."
                    )
                if config.d_model % int(readout_heads) != 0:
                    raise ValueError(
                        f"readout_heads={readout_heads} must divide d_model={config.d_model}"
                    )
                self.cls_queries = nn.Parameter(torch.empty(1, self.readout_num_queries, config.d_model))
                nn.init.normal_(self.cls_queries, mean=0.0, std=0.02)
                self.readout_attn = nn.MultiheadAttention(
                    embed_dim=config.d_model,
                    num_heads=int(readout_heads),
                    dropout=0.0,
                    batch_first=True,
                )
                self.readout_norm = nn.LayerNorm(config.d_model)
                self.readout_ff = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.SiLU(),
                    nn.Linear(config.d_model, config.d_model),
                )
                self.readout_ff_norm = nn.LayerNorm(config.d_model)
        else:
            raise ValueError(
                f"Unknown value model architecture: {self.architecture}. "
                f"Expected one of: {FULL_CLS_ARCH}, {READOUT_CLS_ARCH}."
            )

        regression_input_dim = config.d_model
        if self.architecture == READOUT_CLS_ARCH and self.pooling_mode == READOUT_CLS_ARCH:
            if self.readout_aggregation == "concat":
                regression_input_dim = config.d_model * self.readout_num_queries

        self.regression_head = nn.Sequential(
            nn.Linear(regression_input_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def _resolve_pooling_mode(self, pooling_mode: str) -> str:
        mode = str(pooling_mode or AUTO_POOLING_MODE)
        if mode not in VALUE_MODEL_POOLING_CHOICES:
            raise ValueError(
                f"Unknown pooling_mode: {pooling_mode!r}. "
                f"Expected one of: {', '.join(VALUE_MODEL_POOLING_CHOICES)}."
            )
        if self.architecture == FULL_CLS_ARCH:
            if mode in {AUTO_POOLING_MODE, FULL_CLS_ARCH}:
                return FULL_CLS_ARCH
            raise ValueError("pooling_mode must be 'auto' or 'full_cls' when architecture='full_cls'")
        if mode == AUTO_POOLING_MODE:
            return READOUT_CLS_ARCH
        if mode == FULL_CLS_ARCH:
            raise ValueError("pooling_mode='full_cls' is only valid when architecture='full_cls'")
        if mode not in READOUT_POOLING_MODES:
            raise ValueError(
                f"pooling_mode={mode!r} is not valid for architecture='readout_cls'."
            )
        return mode

    def _default_cls_attend_to(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cls_attend_to = input_ids.ne(self.mask_id)
        if attention_mask is not None:
            cls_attend_to = cls_attend_to & attention_mask.bool()
        return cls_attend_to

    def _normalize_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.long)
        return attention_mask.to(device=input_ids.device, dtype=torch.long)

    def _compute_time_query(self, t: torch.Tensor, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t.to(torch.float32)).to(device=device, dtype=dtype).unsqueeze(1)
        return t_emb

    def _embed_content_tokens(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        x = self.llada.transformer.wte(input_ids)
        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)
        x = self.llada.transformer.emb_drop(x)
        return x

    def _content_attention_bias(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        attention_bias = torch.zeros((batch_size, 1, seq_len, seq_len), device=attention_mask.device, dtype=dtype)
        key_visibility = attention_mask.to(dtype=dtype)[:, None, None, :]
        attention_bias = attention_bias + ((1.0 - key_visibility) * -1e4)
        ensure_finite_(attention_bias)
        return attention_bias

    def _forward_backbone(self, x: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        if hasattr(self.llada.transformer, "blocks"):
            for block in self.llada.transformer.blocks:
                x, _ = block(x, attention_bias=attention_bias)
        elif hasattr(self.llada.transformer, "block_groups"):
            for block_group in self.llada.transformer.block_groups:
                x, _ = block_group(x, attention_bias=attention_bias)
        x = self.llada.transformer.ln_f(x)
        return x

    def _prepare_full_cls_inputs(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cls_attend_to: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_x = self.llada.transformer.wte(input_ids)
        if self.config.input_emb_norm:
            token_x = token_x * (self.config.d_model**0.5)
        token_x = self.llada.transformer.emb_drop(token_x)
        cls_x = self.cls_embedding.expand(batch_size, -1, -1).to(device=device, dtype=token_x.dtype)
        x = torch.cat([cls_x, token_x], dim=1)

        attention_mask = self._normalize_attention_mask(input_ids, attention_mask)
        if cls_attend_to is None:
            cls_attend_to = self._default_cls_attend_to(input_ids, attention_mask)
        else:
            if cls_attend_to.shape != input_ids.shape:
                raise ValueError(
                    "cls_attend_to must match input_ids shape. "
                    f"Got {tuple(cls_attend_to.shape)} vs {tuple(input_ids.shape)}."
                )
            cls_attend_to = cls_attend_to.to(device=device).bool() & attention_mask.bool()

        cls_mask = torch.ones((batch_size, 1), device=device, dtype=torch.long)
        full_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        full_seq_len = seq_len + 1
        attention_bias = torch.zeros(
            (batch_size, 1, full_seq_len, full_seq_len),
            device=device,
            dtype=x.dtype,
        )
        key_visibility = full_attention_mask.to(dtype=x.dtype)[:, None, None, :]
        attention_bias = attention_bias + ((1.0 - key_visibility) * -1e4)

        cls_visibility = torch.cat(
            [
                torch.ones((batch_size, 1), device=device, dtype=torch.bool),
                cls_attend_to.to(device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        attention_bias[:, :, 0:1, :] = (1.0 - cls_visibility.to(dtype=x.dtype)[:, None, None, :]) * -1e4
        ensure_finite_(attention_bias)
        return x, attention_bias

    def encode_tokens(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode content tokens once. Only meaningful for `readout_cls`."""
        attention_mask = self._normalize_attention_mask(input_ids, attention_mask)
        x = self._embed_content_tokens(input_ids, t)
        attention_bias = self._content_attention_bias(attention_mask, x.dtype)
        return self._forward_backbone(x, attention_bias)

    def _resolve_visible_mask(
        self,
        attention_mask: torch.Tensor,
        cls_attend_to: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cls_attend_to is None:
            if input_ids is None:
                raise ValueError("input_ids is required when cls_attend_to is not provided")
            cls_attend_to = self._default_cls_attend_to(input_ids, attention_mask)
        else:
            if cls_attend_to.shape != attention_mask.shape:
                raise ValueError(
                    "cls_attend_to must match attention_mask shape. "
                    f"Got {tuple(cls_attend_to.shape)} vs {tuple(attention_mask.shape)}."
                )
            cls_attend_to = cls_attend_to.to(attention_mask.device).bool()
        visible = cls_attend_to.bool() & attention_mask.bool()
        has_any_visible = visible.any(dim=1, keepdim=True)
        return torch.where(has_any_visible, visible, attention_mask.bool())

    def _pool_visible_features(
        self,
        token_features: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        if token_features.dim() != 3:
            raise ValueError(f"token_features must have shape (B, L, H), got {tuple(token_features.shape)}")
        if visible.shape != token_features.shape[:2]:
            raise ValueError(
                "visible mask must match token_features batch/length. "
                f"Got {tuple(visible.shape)} vs {tuple(token_features.shape[:2])}."
            )

        if self.pooling_mode == MEAN_VISIBLE_POOLING:
            weights = visible.unsqueeze(-1).to(token_features.dtype)
            pooled = (token_features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return pooled

        if self.pooling_mode == MAX_VISIBLE_POOLING:
            masked = token_features.masked_fill(~visible.unsqueeze(-1), -float("inf"))
            pooled = masked.max(dim=1).values
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            return pooled

        if self.pooling_mode == LAST_VISIBLE_POOLING:
            positions = torch.arange(
                token_features.shape[1],
                device=token_features.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(token_features.shape[0], -1)
            last_pos = torch.where(
                visible,
                positions,
                torch.full_like(positions, -1),
            ).max(dim=1).values.clamp_min(0)
            row_idx = torch.arange(token_features.shape[0], device=token_features.device, dtype=torch.long)
            return token_features[row_idx, last_pos]

        raise RuntimeError(f"_pool_visible_features called with unsupported pooling_mode={self.pooling_mode!r}")

    def score_encoded(
        self,
        encoded_tokens: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_attend_to: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score one or more CLS visibility patterns from cached content features."""
        if self.architecture != READOUT_CLS_ARCH:
            raise RuntimeError("score_encoded is only available for readout_cls architecture")

        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must have shape (B, L), got {tuple(attention_mask.shape)}")

        batch_encoded, seq_len, hidden_dim = encoded_tokens.shape
        if batch_encoded != attention_mask.shape[0]:
            if batch_encoded == 1:
                encoded_tokens = encoded_tokens.expand(attention_mask.shape[0], -1, -1)
                batch_encoded = encoded_tokens.shape[0]
            else:
                raise ValueError(
                    "encoded_tokens batch must match attention_mask batch. "
                    f"Got {batch_encoded} vs {attention_mask.shape[0]}."
                )

        if input_ids is not None and input_ids.shape[0] != batch_encoded:
            if input_ids.shape[0] == 1:
                input_ids = input_ids.expand(batch_encoded, -1)
            else:
                raise ValueError(
                    "input_ids batch must match encoded_tokens batch. "
                    f"Got {input_ids.shape[0]} vs {batch_encoded}."
                )

        query_batch = batch_encoded
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.shape[0] != query_batch:
            if t.shape[0] == 1:
                t = t.expand(query_batch, -1)
            else:
                raise ValueError(f"t batch must match encoded batch. Got {t.shape[0]} vs {query_batch}.")

        token_features = encoded_tokens.to(torch.float32)
        visible = self._resolve_visible_mask(
            attention_mask=attention_mask,
            cls_attend_to=cls_attend_to,
            input_ids=input_ids,
        )

        if self.pooling_mode == READOUT_CLS_ARCH:
            query = self.cls_queries.expand(query_batch, -1, -1).to(device=token_features.device, dtype=torch.float32)
            query = query + self._compute_time_query(t, query_batch, torch.float32, token_features.device)
            key_padding_mask = ~visible
            attn_out, _ = self.readout_attn(
                query=query,
                key=token_features,
                value=token_features,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            cls_hidden = self.readout_norm(query + attn_out)
            cls_hidden = self.readout_ff_norm(cls_hidden + self.readout_ff(cls_hidden))
            cls_hidden = cls_hidden.to(torch.float32)
            if self.readout_aggregation == "mean":
                summary = cls_hidden.mean(dim=1)
            elif self.readout_aggregation == "max":
                summary = cls_hidden.max(dim=1).values
            elif self.readout_aggregation == "concat":
                summary = cls_hidden.reshape(query_batch, -1)
            else:
                raise RuntimeError(f"Unsupported readout_aggregation={self.readout_aggregation!r}")
            return self.regression_head(summary)

        pooled = self._pool_visible_features(token_features, visible)
        return self.regression_head(pooled.to(torch.float32))

    def score_masked_views(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_attend_to: torch.Tensor,
        max_views_per_chunk: int | None = None,
    ) -> torch.Tensor:
        """Score multiple Base/Drop/Hold views for the same token sequence."""
        if cls_attend_to.dim() != 2:
            raise ValueError(f"cls_attend_to must have shape (N, L), got {tuple(cls_attend_to.shape)}")

        repeat_n = int(cls_attend_to.shape[0])
        if max_views_per_chunk is None:
            max_views_per_chunk = int(os.getenv("DIFFU_CF_VIEW_CHUNK", "32"))
        max_views_per_chunk = max(int(max_views_per_chunk), 1)

        if self.architecture == READOUT_CLS_ARCH:
            encoded = self.encode_tokens(input_ids=input_ids, t=t, attention_mask=attention_mask)
            if repeat_n <= max_views_per_chunk:
                att = attention_mask
                if att.shape[0] == 1 and repeat_n > 1:
                    att = att.expand(repeat_n, -1)
                return self.score_encoded(
                    encoded_tokens=encoded,
                    t=t,
                    attention_mask=att,
                    cls_attend_to=cls_attend_to,
                    input_ids=input_ids,
                ).squeeze(-1)

            scores = []
            for start in range(0, repeat_n, max_views_per_chunk):
                end = min(start + max_views_per_chunk, repeat_n)
                view_chunk = cls_attend_to[start:end]
                att_chunk = attention_mask
                if att_chunk.shape[0] == 1 and view_chunk.shape[0] > 1:
                    att_chunk = att_chunk.expand(view_chunk.shape[0], -1)
                scores.append(
                    self.score_encoded(
                        encoded_tokens=encoded,
                        t=t,
                        attention_mask=att_chunk,
                        cls_attend_to=view_chunk,
                        input_ids=input_ids,
                    ).squeeze(-1)
                )
            return torch.cat(scores, dim=0)

        if input_ids.shape[0] != 1:
            raise ValueError("score_masked_views expects a single-row input for full_cls architecture")
        t_batch = t if t.shape[0] == repeat_n else t.expand(repeat_n, -1)
        repeated_input = input_ids.expand(repeat_n, -1)
        repeated_mask = attention_mask.expand(repeat_n, -1)
        return self(
            repeated_input,
            t_batch,
            attention_mask=repeated_mask,
            cls_attend_to=cls_attend_to,
        ).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cls_attend_to: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.architecture == FULL_CLS_ARCH:
            x, attention_bias = self._prepare_full_cls_inputs(
                input_ids=input_ids,
                t=t,
                attention_mask=attention_mask,
                cls_attend_to=cls_attend_to,
            )
            x = self._forward_backbone(x, attention_bias)
            cls_hidden = x[:, 0, :]
            cls_hidden = cls_hidden + self._compute_time_query(
                t,
                cls_hidden.shape[0],
                cls_hidden.dtype,
                cls_hidden.device,
            ).squeeze(1)
            return self.regression_head(cls_hidden.to(torch.float32))

        attention_mask = self._normalize_attention_mask(input_ids, attention_mask)
        encoded = self.encode_tokens(input_ids=input_ids, t=t, attention_mask=attention_mask)
        return self.score_encoded(
            encoded_tokens=encoded,
            t=t,
            attention_mask=attention_mask,
            cls_attend_to=cls_attend_to,
            input_ids=input_ids,
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        n_layers: int = 27,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float16,
        mask_id: int = 126336,
        architecture: str = FULL_CLS_ARCH,
        readout_heads: int = 8,
        pooling_mode: str = AUTO_POOLING_MODE,
        readout_num_queries: int = 1,
        readout_aggregation: str = "mean",
    ):
        spec = inspect_pretrained_backbone(path)
        if spec.family == DREAM_BACKBONE_FAMILY:
            from models_dream.value_model import ValueModel as DreamValueModel

            return DreamValueModel.from_pretrained(
                path=path,
                n_layers=n_layers,
                device=device,
                torch_dtype=torch_dtype,
                mask_id=mask_id,
                architecture=architecture,
                readout_heads=readout_heads,
                pooling_mode=pooling_mode,
                readout_num_queries=readout_num_queries,
                readout_aggregation=readout_aggregation,
            )

        from utils.modeling_llada import LLaDAModelLM

        llada_lm = LLaDAModelLM.from_pretrained(path, torch_dtype=torch_dtype).to(device)
        config = llada_lm.model.config
        val_model = cls(
            config,
            n_layers=n_layers,
            mask_id=mask_id,
            architecture=architecture,
            readout_heads=readout_heads,
            pooling_mode=pooling_mode,
            readout_num_queries=readout_num_queries,
            readout_aggregation=readout_aggregation,
        ).to(device)

        val_model.llada.to(torch_dtype)
        val_model.time_embed.to(torch.float32)
        val_model.regression_head.to(torch.float32)
        if hasattr(val_model, "cls_embedding"):
            val_model.cls_embedding.data = val_model.cls_embedding.data.to(torch.float32)
        if hasattr(val_model, "cls_queries"):
            val_model.cls_queries.data = val_model.cls_queries.data.to(torch.float32)
        if hasattr(val_model, "readout_attn"):
            val_model.readout_attn.to(torch.float32)
            val_model.readout_norm.to(torch.float32)
            val_model.readout_ff.to(torch.float32)
            val_model.readout_ff_norm.to(torch.float32)

        val_model.llada.load_state_dict(llada_lm.model.state_dict(), strict=False)
        return val_model


CLSValueModel = ValueModel
