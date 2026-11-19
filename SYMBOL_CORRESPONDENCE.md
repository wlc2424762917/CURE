# Paper-to-Code Symbol Correspondence

This table maps the main notation used in the CURE paper to the implementation in this repository.

| Paper symbol | Meaning | Implementation | Location |
|---|---|---|---|
| `x_t` | Partially denoised generation state | `x`, `proposed_x`, `proposed_row` | `utils/counterfactual_safe_steering.py` |
| `alpha_t` | Normalized denoising progress | `t_batch`, computed from the remaining mask ratio | `utils/counterfactual_safe_steering.py` |
| `V*(x_t, alpha_t; Omega)` | Safety value of a state under a visible-token set | `val_model(...)` and `scores` | `utils/counterfactual_safe_steering.py`, `models/value_model.py` |
| `U_theta` | Unsafe value | `1.0 - base_safe` | `utils/counterfactual_safe_steering.py` |
| `Omega_t` | Visible token positions used by the value readout | `cls_attend_to` | `models/value_model.py` |
| `Omega_(t,-i)` | View with candidate token `i` removed | Drop-one row in `cls_attend_batch` | `utils/counterfactual_safe_steering.py` |
| `Omega_(t,i)` | View retaining only candidate token `i` | Hold-one row in `cls_attend_batch` | `utils/counterfactual_safe_steering.py` |
| `Delta^nec_t(i)` | Necessity contribution of token `i` | `necessity_unsafe` | `utils/counterfactual_safe_steering.py` |
| `Delta^suf_t(i)` | Sufficiency contribution of token `i` | `sufficiency_unsafe` | `utils/counterfactual_safe_steering.py` |
| `lambda_nec`, `lambda_suf` | Necessity and sufficiency weights | `cfg.necessity_weight`, `cfg.sufficiency_weight` | `utils/counterfactual_safe_steering.py` |
| `rho_(t,i)` | Counterfactual unsafe score for token `i` | `unsafe_score` | `utils/counterfactual_safe_steering.py` |
| `p_(t,i)` | Proposal probability/confidence | `token_probs` | `utils/counterfactual_safe_steering.py` |
| `lambda_risk`, `lambda_unc` | Risk and uncertainty weights | `cfg.unsafe_weight`, `cfg.uncertainty_weight` | Configuration files |
| `eta_(t,i)` | Final re-mask score | `remask_score` | `utils/counterfactual_safe_steering.py` |
| `tau` | Re-mask threshold | `cfg.remask_threshold` and optional threshold schedule | Configuration files |
| `B` | Maximum number of re-masked positions | `max_remask` | `utils/counterfactual_safe_steering.py` |
| `R_t` | Positions re-masked at step `t` | `remasked_positions` | `utils/counterfactual_safe_steering.py` |
| `A_t` | Audited token positions | New positions plus uncertain historical positions | `utils/counterfactual_safe_steering.py` |
| `T` | Number of denoising steps | `cfg.steps` | Configuration files |
| `L_r` | Response generation length | `cfg.gen_length` or the editable response span | Configuration files and evaluation scripts |
| `e_t` | Time embedding | `self.time_embed(t)` | `models/value_model.py` |
| `q_safe` | Learned safety readout query | `self.cls_queries` | `models/value_model.py` |
| `V` | Predicted safety value in `[0, 1]` | `self.regression_head` | `models/value_model.py` |
| `L_reg` | Value regression objective | Smooth L1 loss | `scripts/train_value_model.py` |
| `L_margin` | Pairwise value-ordering objective | `batch_pairwise_ranking_loss` | `scripts/train_value_model.py` |
| suppression | Prevent immediate re-selection of re-masked tokens | `suppressed_token_ids`, `suppressed_active` | Steering modules |

## Architecture names

| Code name | Description |
|---|---|
| `full_cls` | Adds a learned CLS token to the value-model backbone and encodes it jointly with content tokens. |
| `readout_cls` | Encodes content tokens once, then uses masked cross-attention from learned readout queries. This supports cached counterfactual views. |
| `mean_visible` | Pools encoded features over visible tokens with a mean. |
| `max_visible` | Pools encoded features over visible tokens with a max. |
| `last_visible` | Uses the final visible token feature. |
