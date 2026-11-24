"""Benchmark methods of this sub-project (registered into benchmark/fusion_methods.METHODS on import by bench.py;
they run only when named in --methods).

    nn_oracle64[_<teacher>]   the teacher's mask + displacement pooled to the feature texture and re-applied
                              (ceiling of the representation, uses the teacher; diagnostic only)
    nn_<keep>__<disp>         keep model run + displacement (flow) run, one sample
    nn_<keep>__<disp>_k<K>    K displacement samples, the one with the lowest mixing proxy is used
    nn_<keep>__<disp>_dz1     displacements whose max over the knots is < 1 mm are zeroed (sampling jitter)
    nn_<keep>__nodisp         keep model only (threshold log(w_flip)); nn_<keep>_t<thr>__nodisp: other thresholds
    nn_teacher__<disp>        the teacher FM's keep (afm_a2_guard0.7_slab1) + NN displacement (ablation)
Runs are found by scanning $DATA/runs/*/config.json at import.
"""
import glob
import json
import os
import time

import numpy as np

import common  # noqa: F401
from fusion_methods import METHODS, FusionResult, register_method
from fusion_refine._util import keep_from_mask, initial_keep, root_texels

import features as F

DATA = os.environ.get("DATA", "")
# trained runs: $DATA/runs/<name> (local training) and models/<name> in this folder (versions kept in git);
# a run in $DATA/runs wins over a same-named one in models/
_RUN_DIRS = {}
for _root in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), os.path.join(DATA, "runs")):
    for _c in glob.glob(os.path.join(_root, "*", "config.json")):
        _RUN_DIRS[os.path.basename(os.path.dirname(_c))] = os.path.dirname(_c)


def run_dir(run):
    return _RUN_DIRS.get(run, os.path.join(DATA, "runs", run))
MIN_RET = 0.7
_MODELS = {}


def _retention(keep, init):
    return [float((k & i).sum() / max(i.sum(), 1)) for k, i in zip(keep, init)]


def mixing_proxy(heads, keep, step=8):
    """share of kept (subsampled) points with the other head's kept hair in their 3x3x3 voxel neighbourhood
    (comb.py's mixing test), averaged over both heads. Cheap stand-in for 'same-height crossings'."""
    pts = [h["positions"][k][:, ::step].astype(np.float64) for h, k in zip(heads, keep)]
    if min(len(p) for p in pts) == 0:
        return 0.0
    occ, lo, dims = F._voxel_occupancy(pts)
    fr = []
    for i, P in enumerate(pts):
        v = np.clip(np.floor((P.reshape(-1, 3) - lo) / F.VOX).astype(int), 0, dims - 1)
        fr.append(float((occ[1 - i][tuple(v.T)] >= 2).mean()))
    return float(np.mean(fr))


def _finish(heads, labels, mask, disp, fe, info, disp_scale=F.DISP_SCALE):
    keep = keep_from_mask(heads, mask)
    ret = _retention(keep, initial_keep(heads, labels))
    edited = None
    if disp is not None:
        edited, dinfo = F.apply(heads, keep, disp, fe, disp_scale=disp_scale)
        info["disp"] = dinfo
    info["retention_vs_mask"] = ret
    return FusionResult(keep=keep, refined_labels=mask.astype(np.int16), info=info, heads=edited)


def oracle(heads, labels, ctx, teacher="comb20_slab1", R=F.FEAT_RES, full_mask=False, use_disp=True):
    """teacher output pooled to R (mask: unless full_mask; displacement: always R) and re-applied"""
    t0 = time.time()
    fr = METHODS[teacher](heads, labels, ctx)
    teacher_s = time.time() - t0
    fe = F.features(heads, labels, R)
    mask = fr.refined_labels > 0
    edited = [h["positions"] for h in fr.heads] if fr.heads is not None else [h["positions"] for h in heads]
    y, _ = F.targets(heads, fr.keep, mask, edited, fe)
    up = labels.shape[0] // R
    m = mask if full_mask else np.where(np.repeat(np.repeat(fe["region"], up, 0), up, 1),
                                        F.mask_from_field(y[0], fe["region"], labels.shape[0]), labels == 1)
    return _finish(heads, labels, m, y[1:] if use_disp else None, fe,
                   dict(teacher=teacher, teacher_seconds=round(teacher_s, 1), R=R, full_mask=full_mask))


def _load(run):
    if run not in _MODELS:
        import torch
        rd = run_dir(run)
        cfg = json.load(open(os.path.join(rd, "config.json")))
        if cfg.get("kind") == "keep":
            import keep_model as KM
            net = KM.build(cfg)
        else:
            import model as M
            net = M.build(cfg)
        net.load_state_dict(torch.load(os.path.join(rd, "best.pt"), map_location="cpu"))
        _MODELS[run] = (net.cuda().eval(), cfg)
    return _MODELS[run]


def predict_keep(heads, labels, keep_run, fe=None, thr=None):
    """-> keep per head, info. Strands whose flip logit > thr change sides (kept <-> hidden). Default thr =
    log(w_flip): undoes the flip weighting of the training loss (calibrated probability 0.5)."""
    import torch
    net, cfg = _load(keep_run)
    thr = float(np.log(cfg.get("w_flip", 1.0))) if thr is None else thr
    fe = fe or F.features(heads, labels, cfg["feat_res"])
    sf = F.strand_features(heads, labels, fe)
    init = initial_keep(heads, labels)
    x = torch.from_numpy(fe["x"] / np.array(cfg["feat_scale"], np.float32)[:, None, None])[None].cuda()
    sfa = (np.concatenate(sf) - np.array(cfg["sf_mean"], np.float32)) / np.array(cfg["sf_scale"], np.float32)
    uv = np.concatenate([h["root_uv"] for h in heads]).astype(np.float32)
    with torch.no_grad():
        logit = net(x, torch.from_numpy(sfa)[None].cuda(), torch.from_numpy(uv)[None].cuda())[0].cpu().numpy()
    flip = logit > thr
    n0 = len(heads[0]["root_uv"])
    keep = [init[0] ^ flip[:n0], init[1] ^ flip[n0:]]
    return keep, dict(flips=int(flip.sum()), flip_rate=float(flip.mean()))


def predict_disp(heads, labels, disp_run, fe, k=1, n_steps=16, seed=0):
    import torch
    import model as M
    net, cfg = _load(disp_run)
    x = torch.from_numpy(fe["x"] / np.array(cfg["feat_scale"], np.float32)[:, None, None])[None].cuda()
    g = torch.Generator(device="cuda").manual_seed(seed)
    return M.sample(net, x.expand(k, -1, -1, -1).contiguous(), cfg, n_steps, generator=g).cpu().numpy(), cfg


def run_nn(heads, labels, ctx, keep_run, disp_run, k=1, dead_mm=0.0, thr=None):
    """keep_run: a keep model run or "teacher" (comb20_slab1's keep); disp_run: a displacement run or None"""
    t0 = time.time()
    fe = F.features(heads, labels)
    info = dict(keep_run=keep_run, disp_run=disp_run, k=k, feat_seconds=round(time.time() - t0, 2))
    init = initial_keep(heads, labels)
    t0 = time.time()
    if keep_run == "teacher":
        fr = METHODS["afm_a2_guard0.7_slab1"](heads, labels, ctx)
        keep = fr.keep
    else:
        keep, kinfo = predict_keep(heads, labels, keep_run, fe, thr)
        info.update(kinfo)
    ret = _retention(keep, init)
    if min(ret) < MIN_RET:        # guard: never let a head vanish -> fall back to the drawn split
        info["guard_fallback"] = ret
        keep = init
    info["keep_seconds"] = round(time.time() - t0, 2)
    # refined label map for the report: drawn labels with the texels of flipped strands changed
    mask = (labels == 1).copy()
    for i, (h, kp, ik) in enumerate(zip(heads, keep, init)):
        r, c = root_texels(h["root_uv"], labels.shape[0])
        ch = kp != ik
        mask[r[ch], c[ch]] = kp[ch] if i == 1 else ~kp[ch]   # texel -> the head that keeps it (True = donor)
    edited = None
    if disp_run is not None:
        t0 = time.time()
        fe_d = fe
        if _load(disp_run)[1].get("feat_keep"):     # displacement model conditioned on the kept strands
            fe_d = F.features(heads, labels, keep=keep)
        out, dcfg = predict_disp(heads, labels, disp_run, fe_d, k)
        if dead_mm > 0:
            K = len(F.KNOT_IDX)
            o = out.reshape(k, K, 3, *out.shape[-2:])
            small = np.linalg.norm(o, axis=2, keepdims=True).max(1, keepdims=True) * dcfg["disp_scale"] * 1e3 < dead_mm
            out = np.where(small, 0, o).reshape(out.shape)
        cands = []
        for j in range(k):
            hs, dinfo = F.apply(heads, keep, out[j], fe, disp_scale=dcfg["disp_scale"])
            cands.append((mixing_proxy(hs, keep) if k > 1 else 0.0, j, hs, dinfo))
        score, j, edited, dinfo = min(cands, key=lambda c: c[0])
        info.update(disp=dinfo, picked=j, proxies=[c[0] for c in cands], disp_seconds=round(time.time() - t0, 2))
    info["retention_vs_mask"] = _retention(keep, init)
    return FusionResult(keep=keep, refined_labels=mask.astype(np.int16), info=info, heads=edited)


def _register(name, fn):
    fn.__name__ = fn.__qualname__ = name
    register_method(name, supported_m=(2,))(fn)


_register("nn_oracle64", lambda h, l, c: oracle(h, l, c))
_register("nn_oracle64_guardslab1", lambda h, l, c: oracle(h, l, c, "afm_a2_guard0.7_slab1"))
_register("nn_oracle128", lambda h, l, c: oracle(h, l, c, R=128))
_register("nn_oracle256", lambda h, l, c: oracle(h, l, c, R=256))
_register("nn_oracle_fullmask_disp64", lambda h, l, c: oracle(h, l, c, full_mask=True))
_register("nn_oracle_fullmask_nodisp", lambda h, l, c: oracle(h, l, c, full_mask=True, use_disp=False))

# trained runs: $DATA/runs/<name>/{config.json, best.pt}; config["kind"] = "keep" | "disp"
_runs = dict(keep=[], disp=[])
for _cfgp in sorted(os.path.join(d, "config.json") for d in _RUN_DIRS.values()):
    _run = os.path.basename(os.path.dirname(_cfgp))
    if _run.startswith("_") or not os.path.isfile(os.path.join(os.path.dirname(_cfgp), "best.pt")):
        continue
    _runs.setdefault(json.load(open(_cfgp)).get("kind", "disp"), []).append(_run)
for _kr in _runs["keep"] + ["teacher"]:
    if _kr != "teacher":
        _register(f"nn_{_kr}__nodisp", lambda h, l, c, kr=_kr: run_nn(h, l, c, kr, None))
        for _t in (0.0, 1.0, 3.0):
            _register(f"nn_{_kr}_t{_t:g}__nodisp", lambda h, l, c, kr=_kr, t=_t: run_nn(h, l, c, kr, None, thr=t))
    for _dr in _runs["disp"]:
        _register(f"nn_{_kr}__{_dr}", lambda h, l, c, kr=_kr, dr=_dr: run_nn(h, l, c, kr, dr))
        _register(f"nn_{_kr}__{_dr}_dz1", lambda h, l, c, kr=_kr, dr=_dr: run_nn(h, l, c, kr, dr, dead_mm=1.0))
        for _k in (4, 8):
            _register(f"nn_{_kr}__{_dr}_k{_k}", lambda h, l, c, kr=_kr, dr=_dr, k=_k: run_nn(h, l, c, kr, dr, k=k))


# ---------------------------------------------------------------- direct optimisation of the mixing objective

def run_dopt(heads, labels, ctx, keep_src, **kw):
    """keep from keep_src ("fm" = afm_a2_guard0.7_slab1, or a keep model run), then diffopt.optimize"""
    import diffopt
    t0 = time.time()
    init = initial_keep(heads, labels)
    info = dict(keep_src=keep_src)
    if keep_src == "fm":
        fr = METHODS["afm_a2_guard0.7_slab1"](heads, labels, ctx)
        keep, mask = fr.keep, fr.refined_labels > 0
    else:
        keep, kinfo = predict_keep(heads, labels, keep_src)
        if min(_retention(keep, init)) < MIN_RET:
            keep = init
        mask = labels == 1
        info.update(kinfo)
    info["keep_seconds"] = round(time.time() - t0, 2)
    t0 = time.time()
    knots, oinfo = diffopt.optimize(heads, keep, verbose=ctx.get("verbose", False), **kw)
    edited, dinfo = diffopt.apply_knots(heads, keep, knots)
    info.update(disp=dinfo, opt=oinfo, opt_seconds=round(time.time() - t0, 2), retention_vs_mask=_retention(keep, init))
    return FusionResult(keep=keep, refined_labels=mask.astype(np.int16), info=info, heads=edited)


DOPT_VARIANTS = {
    "": dict(),
    "_mix3": dict(w_mix=3.0),
    "_mix10": dict(w_mix=10.0),
}
for _ks, _kn in (("fm", "fm"), ("k1", "k1")):
    for _suf, _kw in DOPT_VARIANTS.items():
        _register(f"dopt_{_kn}{_suf}", lambda h, l, c, ks=_ks, kw=_kw: run_dopt(h, l, c, ks, **kw))


# ---------------------------------------------------------------- amortised displacement (disp_net, kind "dnet")

def run_dnet(heads, labels, ctx, keep_src, run, refine_iters=0):
    """keep from keep_src ("fm" or a keep model run), displacement from the disp_net run (one forward pass);
    refine_iters > 0: then that many diffopt iterations starting from the prediction"""
    import torch
    import diffopt
    import disp_net as DN
    t0 = time.time()
    init = initial_keep(heads, labels)
    info = dict(keep_src=keep_src, run=run)
    if keep_src == "fm":
        fr = METHODS["afm_a2_guard0.7_slab1"](heads, labels, ctx)
        keep, mask = fr.keep, fr.refined_labels > 0
    else:
        keep, kinfo = predict_keep(heads, labels, keep_src)
        if min(_retention(keep, init)) < MIN_RET:
            keep = init
        mask = labels == 1
        info.update(kinfo)
    info["keep_seconds"] = round(time.time() - t0, 2)
    t0 = time.time()
    if run not in _MODELS:
        rd = run_dir(run)
        cfg = json.load(open(os.path.join(rd, "config.json")))
        net = DN.build(cfg)
        net.load_state_dict(torch.load(os.path.join(rd, "best.pt"), map_location="cpu"))
        _MODELS[run] = (net.cuda().eval(), cfg)
    net, cfg = _MODELS[run]
    st, sf = DN.case_inputs(heads, keep, labels, cfg)
    with torch.no_grad():
        D = net(st, sf)
        loss, parts = diffopt.objective(D, st)
    knots = diffopt.to_world(D, st)
    info.update(net_seconds=round(time.time() - t0, 2), objective=float(loss), mix=float(parts["mix"]))
    if refine_iters:
        knots, oinfo = diffopt.optimize(heads, keep, init=knots, iters=refine_iters, **cfg["obj"])
        info["refine"] = oinfo["hist"]
    edited, dinfo = diffopt.apply_knots(heads, keep, knots)
    info.update(disp=dinfo, disp_seconds=round(time.time() - t0, 2), retention_vs_mask=_retention(keep, init))
    return FusionResult(keep=keep, refined_labels=mask.astype(np.int16), info=info, heads=edited)


for _cfgp in sorted(os.path.join(d, "config.json") for d in _RUN_DIRS.values()):
    _run = os.path.basename(os.path.dirname(_cfgp))
    if _run.startswith("_") or json.load(open(_cfgp)).get("kind") != "dnet":
        continue
    for _ks in ("fm", "k1"):
        _register(f"dnet_{_run}_{_ks}", lambda h, l, c, ks=_ks, r=_run: run_dnet(h, l, c, ks, r))
        _register(f"dnet_{_run}_{_ks}_ref30", lambda h, l, c, ks=_ks, r=_run: run_dnet(h, l, c, ks, r, 30))


# ---------------------------------------------------------------- originals (for renders): one input head alone

def _orig(heads, labels, ctx, h):
    keep = [np.full(len(x["root_uv"]), i == h) for i, x in enumerate(heads)]
    return FusionResult(keep=keep, refined_labels=np.full(labels.shape, h, np.int16), info=dict(original_head=h))


_register("orig_base", lambda h, l, c: _orig(h, l, c, 0))
_register("orig_donor", lambda h, l, c: _orig(h, l, c, 1))
