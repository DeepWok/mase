"""
Calibration-aware per-matmul rotation search (MASE module-level transform).

Given a base quantization config (with all online Hadamard rotations OFF),
this pass:

1. Runs the standard quantize_module_transform_pass once. The base config is
   patched so that every matched attention block uses the *_mx{int,fp}_rotate
   class but with all per-stage rotate flags set to False — this gives the
   same numerics as plain mx{int,fp} while making per-trial flag-flipping cheap.
2. Computes baseline perplexity on a calibration loader.
3. For each supported matmul type per decoder layer
   (q_proj / k_proj / v_proj / o_proj / qk_matmul / av_matmul /
    up_proj / gate_proj / down_proj),
   toggles online Hadamard rotation ON for that type uniformly across all
   layers, computes perplexity, then toggles back.
4. Greedy decision: any matmul type whose single-on perplexity beats the
   baseline is added to the winning set.
5. Re-toggles ALL winners ON to leave the model in the final searched state,
   measures combined perplexity, and writes a JSON summary to
   ``output_json`` (when provided).

Toggling is done in place — linear types are physically swapped between
plain MX{Int,FP} and their Rotate counterpart (sharing the underlying weight
Parameter via ``from_linear``), and attention stages flip the
``qk_use_rotate`` / ``av_use_rotate`` / ``kv_cache_use_rotate`` instance
attrs on the rotate attention class (Qwen3AttentionMXIntRotate /
LlamaAttentionMXIntRotate / LlamaAttentionMXFPRotate).

Coverage today includes MXInt and MXFP attention rotation for fused Qwen3-MoE.
Its fused expert tensors are deliberately outside rotation search; only the
four attention projections and three attention stages are eligible. Adding a
new architecture requires (a) implementing
``<Arch>AttentionMX{Int,FP}Rotate`` with the per-stage flag attrs and
(b) appending it to ``_ROTATE_ATTENTION_CLASSES`` below. Linear-side wiring
is shared and needs no changes.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path

import torch
import tqdm

logger = logging.getLogger(__name__)

from chop.nn.quantized.modules.linear import (
    LinearMXFP,
    LinearMXInt,
    RotateMXFPLinear,
    RotateMXIntLinear,
)
from chop.nn.quantized.modules.phase_context import force_runtime_phase
from chop.nn.quantized.modules.qwen3.attention import Qwen3AttentionMXIntRotate
from chop.nn.quantized.modules.qwen3_moe.attention import (
    Qwen3MoeAttentionMXFPRotate,
    Qwen3MoeAttentionMXIntRotate,
)
from chop.nn.quantized.modules.llama.attention import (
    LlamaAttentionMXIntRotate,
    LlamaAttentionMXFPRotate,
)

from .quantize import quantize_module_transform_pass


# ---------------------------------------------------------------------------
# Format-agnostic registries
# ---------------------------------------------------------------------------
# (plain class, rotate counterpart) pairs — the linear toggle uses both
# directions of this map. A single search can mix MXInt and MXFP linears
# because each module's own type picks the right pair.
_LINEAR_PLAIN_TO_ROTATE = {
    LinearMXInt: RotateMXIntLinear,
    LinearMXFP: RotateMXFPLinear,
}
_LINEAR_ROTATE_TO_PLAIN = {v: k for k, v in _LINEAR_PLAIN_TO_ROTATE.items()}

# All rotate attention classes the search can flip per-stage flags on.
_ROTATE_ATTENTION_CLASSES = (
    Qwen3AttentionMXIntRotate,
    Qwen3MoeAttentionMXFPRotate,
    Qwen3MoeAttentionMXIntRotate,
    LlamaAttentionMXIntRotate,
    LlamaAttentionMXFPRotate,
)

# Map base format name → rotate class name used in the patched q_config.
_NAME_TO_ROTATE_NAME = {
    "mxint": "mxint_rotate",
    "mxfp":  "mxfp_rotate",
}


# Order matters only for the readability of the report.
LINEAR_MATMUL_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
)
ATTENTION_MATMUL_TYPES = ("qk_matmul", "av_matmul", "kv_cache")
ALL_MATMUL_TYPES = LINEAR_MATMUL_TYPES + ATTENTION_MATMUL_TYPES
_FUSED_EXPERT_MATMUL_TYPES = frozenset(
    {"up_proj", "gate_proj", "down_proj"}
)

# Map attention stage name -> instance attr on the rotate attention classes.
_ATTN_STAGE_TO_FLAG = {
    "qk_matmul": "qk_use_rotate",
    "av_matmul": "av_use_rotate",
    "kv_cache": "kv_cache_use_rotate",
}


def _has_fused_qwen3_moe_experts(network: torch.nn.Module) -> bool:
    return any(
        hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and type(module).__name__.startswith("Qwen3MoeExperts")
        for module in network.modules()
    )


def _resolve_rotation_scope(
    network: torch.nn.Module,
    requested: tuple[str, ...],
    *,
    explicitly_requested: bool,
) -> tuple[tuple[str, ...], dict]:
    if not _has_fused_qwen3_moe_experts(network):
        return requested, {
            "architecture": "generic",
            "excluded_matmul_types": [],
        }
    unsupported = tuple(
        matmul for matmul in requested if matmul in _FUSED_EXPERT_MATMUL_TYPES
    )
    if unsupported and explicitly_requested:
        raise ValueError(
            "selective rotation does not support fused Qwen3-MoE expert "
            f"tensors: {unsupported}; search attention projections/stages only"
        )
    eligible = tuple(
        matmul for matmul in requested if matmul not in _FUSED_EXPERT_MATMUL_TYPES
    )
    return eligible, {
        "architecture": "qwen3_moe_fused",
        "eligible_matmul_types": list(eligible),
        "excluded_matmul_types": sorted(_FUSED_EXPERT_MATMUL_TYPES),
        "excluded_reason": "fused expert tensors have no rotation lowerer",
    }


def _patch_base_args_for_rotate_class(base_args: dict) -> dict:
    """Force every attention selector to use the rotate class with all stage
    flags OFF — gives baseline (non-rotated) numerics while leaving the
    per-stage toggle hooks in place for the search.

    Handles both ``name == "mxint"`` and ``name == "mxfp"`` selectors,
    bumping each to its corresponding ``*_rotate`` registry name. Stage
    blocks are looked up both at the config top level (legacy flat configs)
    and inside ``prefill`` / ``decode`` phase buckets (phase-split configs).
    """
    args = copy.deepcopy(base_args)

    def _stage_scopes(cfg: dict):
        """Yield every dict that may hold qk/av/kv stage blocks."""
        yield cfg
        for bucket in ("prefill", "decode"):
            if isinstance(cfg.get(bucket), dict):
                yield cfg[bucket]

    # The selectors live as top-level keys (everything except control keys).
    for key, val in args.items():
        if key in ("by", "gptq", "token_collector"):
            continue
        if not isinstance(val, dict) or "config" not in val:
            continue
        cfg = val["config"]
        if not isinstance(cfg, dict):
            continue
        name = cfg.get("name")
        # Heuristic: an attention selector has nested matmul/cache blocks
        # (at top level or inside a phase bucket).
        has_attn_substructure = any(
            isinstance(scope.get(stage), dict)
            for scope in _stage_scopes(cfg)
            for stage in ("qk_matmul", "av_matmul", "kv_cache")
        )
        if not has_attn_substructure:
            continue
        if name in _NAME_TO_ROTATE_NAME:
            cfg["name"] = _NAME_TO_ROTATE_NAME[name]
        # Force all three stages OFF in the baseline regardless of whether
        # the user pre-set anything — the search drives them.
        for scope in _stage_scopes(cfg):
            for stage in ("qk_matmul", "av_matmul", "kv_cache"):
                stage_cfg = scope.get(stage)
                if isinstance(stage_cfg, dict):
                    stage_cfg["rotate"] = False
    return args


def _toggle_linear_rotation(model: torch.nn.Module, matmul_type: str, enable: bool) -> int:
    """Swap plain MX{Int,FP} linear <-> rotate counterpart in place for every
    linear whose qualified name ends in ``.{matmul_type}``. Returns the number
    of swapped modules. ``from_linear`` shares the existing weight Parameter,
    so the swap is O(modules) and does not allocate new weight storage.

    Each module's current class determines which (plain, rotate) pair it
    belongs to — a single search can mix MXInt and MXFP linears.
    """
    suffix = f".{matmul_type}"
    if enable:
        # plain -> rotate
        from_to = _LINEAR_PLAIN_TO_ROTATE
    else:
        # rotate -> plain
        from_to = _LINEAR_ROTATE_TO_PLAIN

    swapped = 0
    # Materialize the list — we mutate the tree as we iterate.
    for name, module in list(model.named_modules()):
        if not name.endswith(suffix):
            continue
        target_cls = from_to.get(type(module))
        if target_cls is None:
            continue  # already in target state, or not the kind we manage
        # Resolve parent + attr.
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        new_module = target_cls.from_linear(module, module.config)
        setattr(parent, attr, new_module)
        swapped += 1
    return swapped


def _toggle_attention_stage(model: torch.nn.Module, stage: str, enable: bool) -> int:
    """Flip the per-stage rotate flag on every rotate-attention instance
    (Qwen3 MXInt / Llama MXInt / Llama MXFP). Returns the number of attention
    modules touched."""
    flag = _ATTN_STAGE_TO_FLAG[stage]
    touched = 0
    for _, module in model.named_modules():
        if isinstance(module, _ROTATE_ATTENTION_CLASSES):
            setattr(module, flag, bool(enable))
            touched += 1
    return touched


def _toggle(model: torch.nn.Module, matmul_type: str, enable: bool) -> int:
    if matmul_type in LINEAR_MATMUL_TYPES:
        return _toggle_linear_rotation(model, matmul_type, enable)
    if matmul_type in ATTENTION_MATMUL_TYPES:
        return _toggle_attention_stage(model, matmul_type, enable)
    raise ValueError(f"Unknown matmul_type: {matmul_type}")


@torch.no_grad()
def _compute_calibration_perplexity(
    model: torch.nn.Module,
    loader,
    device: str,
    label: str,
    score_phase: str = "decode",
) -> float:
    """Mean-NLL-based perplexity over a list of ``(input_ids, target)`` tuples
    (the same shape ``gptq.data_utils.get_loaders`` produces).

    ``score_phase`` forces the runtime phase during scoring. Cache-free ppl
    forwards would otherwise register as "prefill" — under a decode-only
    phase config that bypasses all quantisation, making every candidate look
    identical. Scoring in "decode" runs every token through the decode
    chip's numerics, which is what the rotation decisions are for. (For
    legacy flat configs both phases are identical, so this is a no-op.)
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_batches = len(loader)
    logger.info(
        "    ppl[%s] forward over %d batches (phase=%s)...",
        label, n_batches, score_phase,
    )
    t0 = time.time()
    pbar = tqdm.tqdm(loader, desc=f"ppl[{label}]", total=n_batches, leave=False)
    with force_runtime_phase(score_phase):
        for batch in pbar:
            input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            input_ids = input_ids.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            # HF causal-LM loss is mean NLL over the (seqlen-1) predicted positions.
            n_predicted = input_ids.shape[-1] - 1
            total_nll += outputs.loss.float().item() * n_predicted
            total_tokens += n_predicted
            pbar.set_postfix(running_nll=f"{total_nll/max(total_tokens,1):.4f}")
    pbar.close()
    if total_tokens == 0:
        raise RuntimeError("Empty calibration loader — cannot compute perplexity.")
    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    elapsed = time.time() - t0
    logger.info(
        "    ppl[%s] = %.4f (mean_nll=%.4f, %d tokens, %.1fs)",
        label, ppl, mean_nll, total_tokens, elapsed,
    )
    return ppl


def _search_greedy_forward(
    network, calib_loader, device, matmul_types, baseline_ppl,
    improvement_eps, score_phase,
) -> dict:
    """Greedy forward selection. Each round: try every remaining matmul on
    top of the currently committed set, pick the one with the largest Δ
    against current_ppl (NOT the original baseline), commit it, repeat.

    Stops when best round-Δ ≤ improvement_eps OR when there's nothing left
    to add. No round cap — bounded naturally by ``len(matmul_types)``
    rounds in the worst case. ``current_ppl`` is monotone non-increasing,
    so ``final_ppl`` is guaranteed ≤ baseline.
    """
    logger.info(
        "STEP 3/4 — greedy forward selection (eps=%g, bounded by %d rounds)",
        improvement_eps, len(matmul_types),
    )

    committed: list[str] = []           # winners in commit order
    remaining = list(matmul_types)
    current_ppl = baseline_ppl
    rounds_log: list[dict] = []         # history for the JSON report
    per_type_swap_count: dict[str, int] = {}
    n_trials = 0
    round_idx = 0

    while remaining:
        round_idx += 1
        logger.info(
            "  ROUND %d  committed=%s  current_ppl=%.4f  candidates=%d",
            round_idx, committed, current_ppl, len(remaining),
        )
        round_ppls: dict[str, float] = {}

        for cand_idx, candidate in enumerate(remaining, start=1):
            t_trial = time.time()
            n_touched = _toggle(network, candidate, enable=True)
            per_type_swap_count.setdefault(candidate, n_touched)
            if n_touched == 0:
                logger.warning(
                    "    [%d/%d] %s: no matching modules — skipping.",
                    cand_idx, len(remaining), candidate,
                )
                round_ppls[candidate] = float("nan")
                continue
            ppl = _compute_calibration_perplexity(
                network, calib_loader, device,
                label=f"r{round_idx}_+{candidate}",
                score_phase=score_phase,
            )
            round_ppls[candidate] = ppl
            _toggle(network, candidate, enable=False)
            n_trials += 1
            delta = current_ppl - ppl
            logger.info(
                "    [%d/%d] +%s  ppl=%.4f  Δ=%+.4f  (%.1fs)",
                cand_idx, len(remaining), candidate, ppl, delta,
                time.time() - t_trial,
            )

        # Pick the best candidate this round (largest drop from current_ppl).
        valid = {k: v for k, v in round_ppls.items() if not math.isnan(v)}
        if not valid:
            logger.info("  round %d: all candidates skipped — stopping.", round_idx)
            rounds_log.append({
                "round": round_idx, "committed_before": list(committed),
                "current_ppl_before": current_ppl, "round_ppls": round_ppls,
                "selected": None, "current_ppl_after": current_ppl, "stopped": True,
            })
            break

        best_cand = min(valid, key=lambda k: valid[k])
        best_ppl = valid[best_cand]
        best_delta = current_ppl - best_ppl

        if best_delta <= improvement_eps:
            logger.info(
                "  round %d: best Δ=%+.4f (%s) ≤ eps=%g — stopping.",
                round_idx, best_delta, best_cand, improvement_eps,
            )
            rounds_log.append({
                "round": round_idx, "committed_before": list(committed),
                "current_ppl_before": current_ppl, "round_ppls": round_ppls,
                "selected": None, "current_ppl_after": current_ppl, "stopped": True,
            })
            break

        # Commit the winner: leave it ON in the network and remove from remaining.
        n_touched = _toggle(network, best_cand, enable=True)
        committed.append(best_cand)
        remaining.remove(best_cand)
        prev_ppl = current_ppl
        current_ppl = best_ppl
        logger.info(
            "  round %d COMMIT: +%s  ppl: %.4f -> %.4f  Δ=%+.4f",
            round_idx, best_cand, prev_ppl, current_ppl, best_delta,
        )
        rounds_log.append({
            "round": round_idx, "committed_before": list(committed[:-1]),
            "current_ppl_before": prev_ppl, "round_ppls": round_ppls,
            "selected": best_cand, "current_ppl_after": current_ppl,
            "stopped": False,
        })
    if not remaining:
        logger.info("  exhausted all candidates — stopping.")

    logger.info("STEP 4/4 — final state already committed during greedy loop")
    logger.info("  committed: %s  final_ppl=%.4f", committed, current_ppl)

    return {
        "per_type_swap_count": per_type_swap_count,
        "winners": committed,
        "final_ppl": current_ppl,
        "rounds": rounds_log,
        "n_trials": n_trials,
    }


def rotation_search_transform_pass(network, pass_args):
    """
    Calibration-aware per-matmul rotation search.

    pass_args keys:
        base_quantize_args (dict, required):
            Full pass_args for ``quantize_module_transform_pass``. Should
            represent the desired *non-rotated* quantization profile (each
            attention selector at ``name = "mxint"`` or ``"mxfp"``); the
            search patches them up to the matching ``*_rotate`` selector
            with all stage flags off and then toggles per stage.
        calib_loader (list, required):
            ``[(input_ids, target), ...]`` — the same shape produced by
            ``chop.passes.module.transforms.gptq.data_utils.get_loaders``.
            Used solely for perplexity scoring (no gradients).
        device (str, default "cuda:0"):
            Where the perplexity forwards run. The model is moved here.
        matmul_types (Iterable[str], default ALL_MATMUL_TYPES):
            Subset of the 10 generic matmul types to search over. Fused
            Qwen3-MoE automatically narrows the default to 7 attention types.
        output_json (str | None):
            If given, write a JSON summary (winners, per-round history,
            final ppl) to this path. Also acts as the cache file when
            ``cache_winners=True``.
        improvement_eps (float, default 0.0):
            Only flag a matmul as a winner if ``ppl_baseline - ppl_only_i >
            improvement_eps`` (treats sub-noise improvements as ties).
        cache_winners (bool, default False):
            If True and ``output_json`` is set and the file exists, load the
            saved ``winners`` list from it, apply them via ``_toggle``, and
            skip the entire search (no calib forwards). Mirrors GPTQ's
            checkpoint-resume behaviour. Delete the JSON to invalidate the
            cache and force a re-search.

    Returns:
        ``(network, results)`` — ``results`` keys: ``baseline_ppl``,
        ``final_ppl``, ``winners`` (in commit order), ``rounds`` (per-round
        history of every candidate's ppl + which one was selected),
        ``per_type_swap_count``, ``n_trials``, ``improvement_eps``,
        ``matmul_types_searched``.

    Algorithm: greedy forward selection. Start from no rotation; each round
    try adding every remaining matmul type to the current set, commit the
    one that drops ppl the most, repeat. Stops when the best round-Δ is
    ≤ ``improvement_eps``, or when no candidates are left. Worst-case
    n*(n+1)/2 ppl forwards; typical 25-35 for n=10. ``current_ppl`` is
    monotone non-increasing, so ``final_ppl ≤ baseline_ppl`` by construction.
    """
    base_args = pass_args["base_quantize_args"]
    calib_loader = pass_args["calib_loader"]
    device = pass_args.get("device", "cuda:0")
    explicit_matmul_types = "matmul_types" in pass_args
    matmul_types = tuple(pass_args.get("matmul_types", ALL_MATMUL_TYPES))
    output_json = pass_args.get("output_json", None)
    improvement_eps = float(pass_args.get("improvement_eps", 0.0))
    cache_winners = bool(pass_args.get("cache_winners", False))
    # Scoring phase for the cache-free ppl forwards. Default None = infer
    # from the quantised network after step 1: score in "decode" when any
    # module quantises decode (decode-side deployments; also a no-op for
    # legacy flat configs), else in "prefill" (prefill-quantised deployments
    # with FP decode, where decode scoring would bypass everything).
    score_phase = pass_args.get("score_phase")
    if score_phase not in (None, "prefill", "decode"):
        raise ValueError(f"Unknown score_phase {score_phase!r}")

    for t in matmul_types:
        if t not in ALL_MATMUL_TYPES:
            raise ValueError(
                f"Unknown matmul_type {t!r}; valid types: {ALL_MATMUL_TYPES}"
            )
    matmul_types, rotation_scope = _resolve_rotation_scope(
        network,
        matmul_types,
        explicitly_requested=explicit_matmul_types,
    )

    # Cache check: if a saved decisions file exists, load and apply — skip
    # the whole search. Same spirit as GPTQ's auto_load_quantized_layers.
    cached = None
    if cache_winners and output_json and Path(output_json).exists():
        cached = json.loads(Path(output_json).read_text())
        if cached.get("rotation_scope") != rotation_scope:
            raise ValueError(
                "cached rotation scope does not match this model; invalidate "
                f"{output_json} and rerun calibration"
            )
        logger.info("=" * 64)
        logger.info("CACHE HIT — loading rotation decisions from %s", output_json)
        logger.info(
            "  cached: winners=%s  baseline_ppl=%.4f  final_ppl=%.4f",
            cached.get("winners"),
            cached.get("baseline_ppl", float("nan")),
            cached.get("final_ppl", float("nan")),
        )
        logger.info("  delete the file to invalidate and re-search.")
        logger.info("=" * 64)

    logger.info("=" * 64)
    logger.info("BEGIN — matmul_types=%s", list(matmul_types))
    logger.info("device=%s  improvement_eps=%s  cache_winners=%s",
                device, improvement_eps, cache_winners)
    logger.info("calib batches=%d", len(calib_loader))
    logger.info("=" * 64)

    # Step 1: Build the model in baseline state (rotate class instantiated
    # for attention so we can flip flags later, but every flag is False).
    logger.info("STEP 1/4 — quantize_module_transform_pass (incl. GPTQ)")
    logger.info("  this is the slow step on first run; "
                "resumes fast if checkpoint_dir is populated.")
    t0 = time.time()
    patched_args = _patch_base_args_for_rotate_class(base_args)
    network, _ = quantize_module_transform_pass(network, patched_args)
    network.to(device)
    logger.info("  quantization done in %.1fs", time.time() - t0)

    if score_phase is None:
        from .quantize import _infer_runtime_decode_policy

        policy = _infer_runtime_decode_policy(network)
        score_phase = "decode" if policy in (None, "quantized") else "prefill"
        logger.info("  score_phase inferred: %s (decode_policy=%s)",
                    score_phase, policy)

    # Sanity: count how many of each module class we ended up with.
    from collections import Counter
    cls_count = Counter(
        type(m).__name__ for _, m in network.named_modules()
        if "MX" in type(m).__name__ or "Rotate" in type(m).__name__
    )
    logger.info("  post-quant class counts: %s", dict(cls_count.most_common()))

    # Cache fast-path: apply saved winners directly, skip ppl evals + search.
    if cached is not None:
        logger.info("STEP 2/2 — applying %d cached winners (no calib forwards)",
                    len(cached.get("winners", [])))
        for w in cached.get("winners", []):
            n = _toggle(network, w, enable=True)
            logger.info("  enabling %s: %d modules", w, n)
        results = dict(cached)
        results["from_cache"] = True
        results["matmul_types_searched"] = list(matmul_types)
        results["rotation_scope"] = rotation_scope
        logger.info("-----Rotation Search (cached) Done-----")
        return network, results

    # Step 2: baseline perplexity (no rotation).
    logger.info("STEP 2/4 — baseline ppl (1 forward pass over calib)")
    baseline_ppl = _compute_calibration_perplexity(
        network, calib_loader, device, label="baseline_no_rotate",
        score_phase=score_phase,
    )

    # Steps 3 + 4: greedy forward selection.
    results = _search_greedy_forward(
        network, calib_loader, device, matmul_types, baseline_ppl,
        improvement_eps, score_phase,
    )

    results["baseline_ppl"] = baseline_ppl
    results["improvement_eps"] = improvement_eps
    results["matmul_types_searched"] = list(matmul_types)
    results["rotation_scope"] = rotation_scope

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("saved results to %s", out_path)

    logger.info("-----Rotation Search Done-----")
    logger.info(
        "baseline_ppl=%.4f final_ppl=%.4f winners=%s",
        baseline_ppl, results["final_ppl"], results["winners"],
    )
    return network, results


def dispatch_rotation_search_block(network, base_pass_args: dict, rot_cfg: dict):
    """TOML-block entry point for ``[rotation_search]``.

    Builds the calibration loader, derives ``cache_path`` (defaults to
    ``<gptq.checkpoint_dir>/rotation_decisions.json``), and calls
    ``rotation_search_transform_pass``. The search internally calls
    ``quantize_module_transform_pass`` again with the rotation_search block
    already popped — no recursion.

    Required keys in ``rot_cfg``:
        calib_data:    Calibration spec for ``get_loaders`` (e.g.
                       ``"file:calib/...pt"``). Falls back to
                       ``base_pass_args["gptq"]["dataset"]`` if absent.

    Common optional keys:
        calib_nsamples (default 32),
        calib_seqlen   (default 1024),
        improvement_eps (default 0.0),
        matmul_types   (default = all 10 generic types; 7 for fused Qwen3-MoE),
        cache_path     (default = <gptq.checkpoint_dir>/rotation_decisions.json),
        cache_winners  (default True — set False to force a re-search).

    eval_lm-injected keys (not from TOML):
        device, model_name.
    """
    from ..gptq.data_utils import get_loaders

    device = rot_cfg.get("device", "cuda:0")
    model_name = rot_cfg.get("model_name")
    if model_name is None:
        # GPTQ block usually has it; fall back there for convenience.
        model_name = base_pass_args.get("gptq", {}).get("model_name")
    if model_name is None:
        raise ValueError(
            "[rotation_search] needs model_name (or [gptq].model_name to fall back on)."
        )

    calib_data = rot_cfg.get(
        "calib_data",
        base_pass_args.get("gptq", {}).get("dataset"),
    )
    if calib_data is None:
        raise ValueError(
            "[rotation_search] needs calib_data (or [gptq].dataset to fall back on)."
        )
    calib_loader = get_loaders(
        calib_data,
        nsamples=int(rot_cfg.get("calib_nsamples", 32)),
        seed=0,
        seqlen=int(rot_cfg.get("calib_seqlen", 1024)),
        model=model_name,
    )

    cache_path = rot_cfg.get("cache_path")
    if cache_path is None:
        gptq_dir = base_pass_args.get("gptq", {}).get("checkpoint_dir")
        if gptq_dir:
            cache_path = str(Path(gptq_dir) / "rotation_decisions.json")
        else:
            raise ValueError(
                "[rotation_search] needs cache_path (or [gptq].checkpoint_dir to derive it from)."
            )

    search_args = {
        "base_quantize_args": base_pass_args,
        "calib_loader": calib_loader,
        "device": device,
        "improvement_eps": float(rot_cfg.get("improvement_eps", 0.0)),
        "output_json": cache_path,
        "cache_winners": bool(rot_cfg.get("cache_winners", True)),
        "score_phase": rot_cfg.get("score_phase"),
    }
    if "matmul_types" in rot_cfg:
        search_args["matmul_types"] = list(rot_cfg["matmul_types"])

    return rotation_search_transform_pass(network, search_args)
