"""Named combinations of the refinement steps, for callers outside the benchmark (e.g. img2hair_tsai_1.py).

    result = refine(heads, labels, "comb20_slab1", grid, dataset_path, base_kwargs)
    result["keep"]        per head (S_i,) bool: strands that go into the fused hair
    result["mask"]        (res, res) bool refined mask (True = donor), for reference
    result["positions"]   per head (S_i, P, 3): positions after combing (the input arrays when nothing moved)
    result["info"]        numbers of every step

The settings are the ones measured in benchmark/ (LOG.md, checkpoints.md): base_kwargs is the script's FM
settings (img2hair_tsai_1.JOINT_FM_KWARGS); each preset overrides lam with the adaptive one.
"""
from .adaptive import refine_joint_fm
from .comb import comb

GUARD = dict(lam_alpha=2.0, min_retention=0.7, guard_factor=2.0, fast_membership=True)
PRESETS = {
    # adaptive lam + retention guard (benchmark afm_a2_guard0.7_fast)
    "guard": dict(fm=GUARD),
    # + depth-aware crossings, dz = 1 cm (afm_a2_guard0.7_slab1)
    "guard_slab1": dict(fm=dict(GUARD, slab_dz=0.01)),
    # + lateral combing of intruding strands, at most 1 / 2 cm (comb10_slab1 / comb20_slab1)
    "comb10_slab1": dict(fm=dict(GUARD, slab_dz=0.01), comb=0.010),
    "comb20_slab1": dict(fm=dict(GUARD, slab_dz=0.01), comb=0.020),
}


def refine(heads, labels, preset, grid, dataset_path, base_kwargs, cache=None, verbose=False):
    cfg = PRESETS[preset]
    result, info = refine_joint_fm(heads, labels, grid, dataset_path, base_kwargs, cache=cache, verbose=verbose,
                                   **cfg["fm"])
    keep = [result["visible_a"], result["visible_b"]]
    mask = result["binary_mask"].astype(bool)
    positions = [h["positions"] for h in heads]
    out = dict(preset=preset, fm=info)
    if cfg.get("comb"):
        edited, comb_info = comb(heads, keep, mask, d_max=cfg["comb"])
        positions = [h["positions"] for h in edited]
        out["comb"] = comb_info
    return dict(keep=keep, mask=mask, positions=positions, info=out)
