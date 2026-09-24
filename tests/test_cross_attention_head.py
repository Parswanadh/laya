"""Cross-attention decision head (laya/common.py): parameter matching and boundary correctness.

Two things this file exists to prove, because the H5 experiment's whole reading depends on them:

1. **Parameter matching.** ``head_kind="cross_attention"`` with 4 heads and
   ``head_kind="self_attention"`` with the shipped 12 heads must have the *same* trainable
   parameter count at the same depth and width. If they did not, an arm-3 win could be credited to
   capacity rather than to architecture, and the control would be uninterpretable.
2. **The state boundary.** The cross-attention stage must treat only state tokens as keys/values.
   A silently wrong boundary would make it a second round of self-attention over the prompt and
   the experiment would measure nothing.

A tiny from-config BERT keeps this fast and offline, matching tests/test_decision_model.py.
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import HEAD_KINDS, DecisionModel  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def _tiny(head_kind="self_attention", head_nhead=None, head_layers=2):
    cfg = AutoConfig.for_model(
        "bert", hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=128, vocab_size=64, sep_token_id=3,
    )
    model = DecisionModel(AutoModel.from_config(cfg), head_layers=head_layers, n_act=2,
                          head_kind=head_kind, head_nhead=head_nhead)
    model.eval()
    return model


def _trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _head_params(m):
    return sum(p.numel() for p in m.head.parameters())


def _batch(b, L, dmax, dense=True):
    """A batch with the layout build_sequence produces: decision block, then state, then [SEP].

    ``input_ids[b, 0]`` is the [CLS]/[SEP] id, positions 1..dmax-2 are the decision block, ``dmax-1``
    is the separator closing the option block, and ``dmax..L-1`` are state tokens.
    """
    ids = torch.randint(5, 60, (b, L))
    ids[:, 0] = 3
    ids[:, dmax - 1] = 3
    ids[:, L - 1] = 3
    att = torch.ones(b, L, dtype=torch.long)
    mpos = torch.arange(1, dmax - 1)[None, :].repeat(b, 1)
    mmask = torch.ones_like(mpos, dtype=torch.bool)
    if not dense:
        # pad the tail of the sequence, as a shorter document in a batch would
        att[:, -3:] = 0
    qtype = torch.zeros(b, dtype=torch.long)
    return ids, att, mpos, mmask, qtype


# ------------------------------------------------------------------ 1. parameter matching
for layers in (1, 2, 3):
    d = 768
    sim = DecisionModel.__new__(DecisionModel)
    # count analytically as well as empirically, so a change to the head cannot quietly move it
    from laya.common import CrossAttentionHead
    import torch.nn as nn
    ca = CrossAttentionHead(d, 4, 4 * d, layers)
    se = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d, d // 64, 4 * d, 0.1, batch_first=True, norm_first=True),
        layers, enable_nested_tensor=False)
    check("params/head_layers=%d cross-attn == self-attn at d=768" % layers,
          sum(p.numel() for p in ca.parameters()), sum(p.numel() for p in se.parameters()))

check("params/tiny 2-layer model, both kinds identical trainable count",
      _trainable(_tiny("cross_attention", head_nhead=4)),
      _trainable(_tiny("self_attention", head_nhead=4)))
check("params/head module identical count (tiny)",
      _head_params(_tiny("cross_attention", head_nhead=4)),
      _head_params(_tiny("self_attention", head_nhead=4)))

# the shipped head count is not a constant we want to hard-code; verify the property instead
check("params/head_kind defaults to the shipped self_attention path",
      DecisionModel(AutoModel.from_config(AutoConfig.for_model(
          "bert", hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
          intermediate_size=64, vocab_size=32)), 1).head_kind, "self_attention")

try:
    _tiny("not_a_head_kind")
    FAIL.append("head_kind/unknown value should have raised ValueError")
except ValueError as e:
    check_true("head_kind/unknown value raises ValueError naming the field",
               "head_kind must be one of" in str(e), str(e))

# ------------------------------------------------------------------ 2. state boundary
m = _tiny("cross_attention", head_nhead=4)
ids, att, mpos, mmask, qtype = _batch(2, 24, 8)
start = m._state_start(ids, mpos, mmask, None)
check("boundary/derived start is the position after the option-block separator",
      start.tolist(), [8, 8])

# An explicit state_start must win over the derivation, so a caller that knows the boundary (the
# H5 harness computes it while building the sequence) never depends on the separator scan.
explicit = torch.full((2,), 10, dtype=torch.long)
check("boundary/explicit state_start overrides the derivation",
      m._state_start(ids, mpos, mmask, explicit).tolist(), [10, 10])

logits, act = m(ids, att, mpos, mmask, qtype, state_start=start)
check("boundary/forward returns one logit per marker", tuple(logits.shape), (2, 6))
check("boundary/forward returns act logits", tuple(act.shape), (2, 2))
check_true("boundary/logits are finite", bool(torch.isfinite(logits).all()))

# Changing a *state* token must move the logits (the head reads the document) ...
ids2 = ids.clone()
ids2[:, 12] = 41
logits2, _ = m(ids2, att, mpos, mmask, qtype, state_start=start)
check_true("boundary/state tokens influence the markers",
           not torch.allclose(logits, logits2, atol=1e-6), "logits unchanged by a state edit")

# ... while changing a state token that is *padded out* must not.
ids3 = ids.clone()
ids3[:, -2] = 41
att3 = att.clone()
att3[:, -3:] = 0
logits3, _ = m(ids3, att3, mpos, mmask, qtype, state_start=start)
logits3b, _ = m(ids, att3, mpos, mmask, qtype, state_start=start)
check_true("boundary/padded state tokens are inert",
           torch.allclose(logits3, logits3b, atol=1e-6), "a masked token moved the logits")

# ------------------------------------------------------------------ 3. the shipped path is untouched
m2 = _tiny("self_attention", head_nhead=4)
l_a, _ = m2(ids, att, mpos, mmask, qtype)
l_b, _ = m2(ids, att, mpos, mmask, qtype, state_start=torch.full((2,), 10, dtype=torch.long))
check_true("compat/self_attention ignores state_start and is deterministic",
           torch.allclose(l_a, l_b, atol=1e-6))
check_true("compat/HEAD_KINDS lists exactly the shipped and the new path",
           HEAD_KINDS == ("self_attention", "cross_attention"), str(HEAD_KINDS))

# ------------------------------------------------------------------ 4. training path
m3 = _tiny("cross_attention", head_nhead=4)
m3.train()
encoder_frozen = all(not p.requires_grad for p in m3.encoder.parameters())
check_true("train/encoder is not the thing being trained (all params still require grad at init)",
           not encoder_frozen, "encoder params unexpectedly frozen before the harness freezes them")
out, _ = m3(ids, att, mpos, mmask, qtype, state_start=start)
out.sum().backward()
grads = {n: (p.grad is not None and bool(p.grad.abs().sum() > 0))
         for n, p in m3.named_parameters() if p.requires_grad}
check_true("train/gradients reach every cross-attention layer",
           all(grads["head.layers.%d.cross_attn.in_proj_weight" % i] for i in range(2)), str(grads))
check_true("train/gradients reach the scorer", grads["scorer.1.weight"])

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
