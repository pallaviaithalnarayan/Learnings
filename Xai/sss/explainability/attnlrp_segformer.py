import torch
import torch.nn.functional as F
from contextlib import contextmanager

# ---------- helpers ----------
def upsample_like(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

@contextmanager
def eval_nograd(model):
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)

# ---------- tracer to capture attention ----------
class AttnTracer:
    """
    Registers forward hooks to capture softmaxed attention matrices from all MHSA blocks.
    Works if each attention module exposes the post-softmax tensor at some submodule
    (often `attn_drop` or right after softmax).
    """
    def __init__(self, encoder):
        self.encoder = encoder
        self.attn = []     # list of [B, heads, Nq, Nk] from shallow -> deep
        self.handles = []

    def _attach(self, mod):
        # find attention modules inside each block; adjust attribute names for your codebase
        for name, m in mod.named_modules():
            if "attn" in name and hasattr(m, "attn_drop"):
                # hook after the softmax
                self.handles.append(
                    m.attn_drop.register_forward_hook(self._save_attn)
                )

    def _save_attn(self, module, inputs, output):
        # output expected to be probs: [B, heads, N, N]
        self.attn.append(output.detach())

    def __enter__(self):
        # Walk all encoder stages/blocks
        for stage in self.encoder:                # e.g., encoder[0..3]
            for block in getattr(stage, "blocks"): # adjust if your name differs
                self._attach(block)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

# ---------- AttnLRP-lite propagator ----------
class AttnLRPExplainer:
    """
    Attention-aware relevance for SegFormer.
    Assumptions:
    - model(image, return_features=True) returns:
        logits: [B, C, H, W]
        aux: {
            'last_feature': [B, Cdec, H, W],      # decoder output before classifier
            'decoder_feats': [f1,f2,f3,f4],       # per-stage decoded features (spatial)
            'encoder_tokens': [t1,t2,t3,t4],      # per-stage tokens [B, Ni, Ci], optional
            'spatial_shapes': [(H1,W1),..,(H4,W4)]
        }
    - You may need to add return_features=True path in your code.
    - We use attention probs to route relevance across tokens (query->key).
    """
    def __init__(self, model):
        self.model = model

    def _seed_from_logits(self, logits, cls_idx=None, pixel=None):
        B, C, H, W = logits.shape
        R = torch.zeros_like(logits)
        if cls_idx is None:
            # seed the predicted class per-pixel
            cls_map = logits.argmax(dim=1)  # [B,H,W]
        else:
            cls_map = torch.full((B, H, W), cls_idx, device=logits.device, dtype=torch.long)
        for b in range(B):
            if pixel is None:
                # seed whole map for selected class at once (uniform 1)
                R[b, cls_map[b]] = 1.0
            else:
                h, w = pixel
                R[b, cls_map[b, h, w], h, w] = 1.0
        return R

    def _back_head_zplus(self, R_out, decode_head):
        """
        Simple z+ through 1x1 classifier: distribute class relevance back to decoder channels.
        """
        classifier = decode_head.classifier  # Conv2d(Cdec, C, k=1). Adjust name if needed.
        W = classifier.weight.clamp(min=0)  # [C, Cdec, 1, 1]
        # conv_transpose routes class relevance back to features
        R_feat = F.conv_transpose2d(R_out, W, bias=None, stride=1)
        return R_feat  # [B, Cdec, H, W]

    def _split_to_stages(self, R_last, decode_head, decoder_feats):
        """
        Distribute relevance from the fused decoder feature back to each branch.
        Approximate by proportional splitting via positive weights of fusion 1x1s.
        If decode head concatenates per-stage features + 1x1, adapt accordingly.
        """
        # Fallback: if not sure, equally split to each branch resized to its spatial size.
        n = len(decoder_feats)
        R_stages = []
        for i, f in enumerate(decoder_feats):
            R_i = upsample_like(R_last, f) / n
            R_stages.append(R_i)  # [B, Ci, Hi, Wi]
        return R_stages

    def _spatial_to_tokens(self, feat, H, W):
        # [B,C,H,W] -> [B, N, C] with N=H*W
        B, C, Hs, Ws = feat.shape
        assert Hs == H and Ws == W
        x = feat.flatten(2).transpose(1, 2).contiguous()
        return x  # [B, N, C]

    def _tokens_to_spatial_sum(self, R_tok, H, W):
        # [B, N, C] -> [B, 1, H, W] (sum over channels)
        B, N, C = R_tok.shape
        x = R_tok.sum(dim=2).view(B, 1, H, W)
        return x

    def run(self, model, image, decode_head, encoder, attn_list, aux, pixel=None, cls_idx=None):
        """
        attn_list: list of attention matrices [layer0, layer1, ...] captured in forward order.
        We traverse from deepest to shallowest and route relevance via attn (query->key).
        """
        logits = aux['logits']      # stash logits in aux before calling run
        R_out = self._seed_from_logits(logits, cls_idx=cls_idx, pixel=pixel)

        # Head: class relevance -> fused decoder feature
        R_last = self._back_head_zplus(R_out, decode_head)  # [B, Cdec, H, W]

        # Split to per-stage decoded features (approx)
        dec_feats = aux['decoder_feats']           # [f1..f4]
        spatial_shapes = aux['spatial_shapes']     # [(H1,W1)..(H4,W4)]
        R_stage = self._split_to_stages(R_last, decode_head, dec_feats)

        # Map per-stage spatial relevance to tokens
        R_tokens_by_stage = []
        for (R_i, (Hi, Wi)) in zip(R_stage, spatial_shapes):
            R_tokens_by_stage.append(self._spatial_to_tokens(R_i, Hi, Wi))  # [B, Ni, Ci]

        # Attention-aware backprop through encoder stages (deep->shallow)
        # We don't know exact block counts per stage from your code; assume attn_list is aligned deepest-last.
        # For a first pass, we simply "rollout" relevance with attn: R_keys = A^T * R_queries per head, averaged across heads.
        # This is AttnLRP-lite: it respects attention flow & residuals approximately.

        # Concatenate all stages' tokens into one sequence per stage if needed.
        # Here we just apply per-stage attention stacks if you keep them grouped.

        # Flatten all attentions (deepest first)
        for A in reversed(attn_list):
            # A: [B, heads, N, N]
            R_tok_list = []
            for R_tok in R_tokens_by_stage:
                B, N, C = R_tok.shape
                # aggregate channel relevance per token, distribute via attention
                Rq = R_tok.mean(dim=2, keepdim=True)  # [B, N, 1]
                Rq = Rq.permute(0, 2, 1)              # [B, 1, N]
                A_mean = A.mean(dim=1)                # [B, N, N]
                # keys relevance = A^T * queries relevance
                Rk = torch.bmm(A_mean.transpose(1, 2), Rq)  # [B, N, 1]
                # expand back to channels equally
                Rk = Rk.permute(0, 2, 1).expand_as(R_tok)
                R_tok_list.append(Rk)
            R_tokens_by_stage = R_tok_list

        # Map tokens back to spatial and upsample/sum to input size
        H_in, W_in = image.shape[-2:]
        R_img = torch.zeros((image.size(0), 1, H_in, W_in), device=image.device)
        for R_tok, (Hi, Wi) in zip(R_tokens_by_stage, spatial_shapes):
            R_sp = self._tokens_to_spatial_sum(R_tok, Hi, Wi)      # [B,1,Hi,Wi]
            R_img = R_img + upsample_like(R_sp, image)

        # Normalize 0..1 for viz
        R_img = (R_img - R_img.min()) / (R_img.max() - R_img.min() + 1e-6)
        return R_img  # [B,1,H,W]
