"""Flux / vector-field analysis of a benchmark case, with img2hair_tsai_1.debug_field_analysis itself.

That function compares a fused hair before and after FM. Here "before" = the reference method's result
(mask_only: every strand follows the drawn mask) and "after" = the evaluated method's (tsai_joint_fm), both
rebuilt from the keep masks saved in maps.npz, so the figures are the same as the tsai script's:

    <run_dir>/cases/<case>/analysis/stats.json  (no field_data.npz: ~1.5 GB per case, the report only needs the figures)
    <run_dir>/cases/<case>/analysis/figures/<head>/flux_score.png       |flux| map (flux = divergence of the flow)
    <run_dir>/cases/<case>/analysis/figures/<head>/flux_with_flow.png   |flux| + per control point flow vectors
    <run_dir>/cases/<case>/analysis/figures/<head>/flux_edges.png
    <run_dir>/cases/<case>/analysis/figures/fused_hull_score.png        grid score map before / after
    <run_dir>/cases/<case>/analysis/figures/fused_grid_mask.png, strand_roots.png
    with <head> = base, donor (whole input hairstyles), fused_before (reference), fused_after (method).

Only m = 2. summarize.py runs it for the lowest / highest grid-score case of every hair-length pair.
"""
import os

import numpy as np

import common

FIGURE_HEADS = ("base", "donor", "fused_before", "fused_after")


def analysis_dir(run_dir, cid):
    return os.path.join(run_dir, "cases", cid, "analysis")


def analyze_case(run_dir, cid, reference="mask_only", method="tsai_joint_fm", overwrite=False):
    """Run debug_field_analysis for one case (skipped if done). Returns the analysis dir or None."""
    out = analysis_dir(run_dir, cid)
    if os.path.exists(os.path.join(out, "stats.json")) and not overwrite:
        return out
    cfg = common.read_json(os.path.join(run_dir, "run_config.json"))
    case = common.read_json(os.path.join(cfg["benchmark_dir"], "cases", cid, "case.json"))
    if case["m"] != 2:
        print(f"  {cid}: field analysis needs m = 2, skipped")
        return None
    keeps = {}
    for mt in (reference, method):
        p = os.path.join(run_dir, "cases", cid, mt, "maps.npz")
        if not os.path.exists(p):
            print(f"  {cid}: no {mt} result, field analysis skipped")
            return None
        d = np.load(p)
        keeps[mt] = (d["keep_0"], d["keep_1"])
    res = common.read_json(os.path.join(run_dir, "cases", cid, method, "result.json"))

    # same strands as the run: max_strands subset with seed case_seed + head index
    base, donor = [common.load_hairstyle(h["npz_path"], cfg.get("max_strands"), seed=case["seed"] + k)
                   for k, h in enumerate(case["heads"])]
    (ka0, kb0), (ka, kb) = keeps[reference], keeps[method]
    if ka.size != base["root_uv"].shape[0] or kb.size != donor["root_uv"].shape[0]:
        print(f"  {cid}: keep masks do not match the hairstyles (dataset or max_strands changed?), skipped")
        return None
    # debug_field_analysis' per-strand labels: True = donor; a strand is kept iff its label matches its head
    result = dict(initial_state_a=~ka0, initial_state_b=kb0, state_a=~ka, state_b=kb, visible_a=ka, visible_b=kb,
                  objectives=res.get("method_info", {}).get("objectives", {}))
    import img2hair_tsai_1 as tsai
    print(f"  {cid}: flux / vector field analysis ({reference} -> {method})")
    tsai.debug_field_analysis(base, donor, result, os.path.join(run_dir, "cases", cid), save_field_data=False)
    return out
