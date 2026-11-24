"""Code-path test with random weights (no checkpoints needed): bash ../nn/run.sh ../prior/smoke_test.py. Numbers are meaningless."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from difflocks_prior import DiffLocksPrior
from fusion_refine._util import keep_from_mask, hair_region, boundary
D = os.environ["DATA"]
cdir = f"{D}/val/cases/case_0000"
case = common.read_json(f"{cdir}/case.json")
heads = [common.load_hairstyle(h["npz_path"], 3000, seed=i) for i, h in enumerate(case["heads"])]
labels = common.masks_to_labels([common.load_mask_png(f"{cdir}/{f}") for f in case["mask_files"]])
keep = keep_from_mask(heads, labels == 1)
pos = np.concatenate([h["positions"][k] for h, k in zip(heads, keep)]); uv = np.concatenate([h["root_uv"][k] for h, k in zip(heads, keep)])
t0 = time.time(); pr = DiffLocksPrior(random_init=True); print("load", round(time.time()-t0,1))
t0 = time.time(); tex, info = pr.hair_to_texture(pos, uv); print("texture", tex.shape, info["fill"], info["texels"], round(time.time()-t0,1), "finite", bool(torch.isfinite(tex).all()))
t0 = time.time(); em = pr.elbo_maps(tex, (0.1, 1.0), 2); print("elbo", {k: (v.shape, float(v.mean())) for k, v in em.items()}, round(time.time()-t0,1))
t0 = time.time(); ll = pr.loglik_map(tex, n_steps=3); print("loglik", round(ll["total"],1), round(ll["bits_per_dim"],3), ll["map"].shape, ll["fevals"], round(time.time()-t0,1))
build = pr.texture_builder(pos[:, 0], uv)
P = torch.tensor(pos, device="cuda", requires_grad=True)
t0 = time.time(); t2 = build(P); print("builder max diff vs hair_to_texture (latents):", float((t2[:64]-tex[:64]).abs().max()))
region = hair_region([h["root_uv"] for h in heads]); bnd = boundary(labels == 1, region)
from scipy.ndimage import binary_dilation
band = binary_dilation(bnd, iterations=4) & region
loss = pr.sds_loss(t2, band); loss.backward(); g = P.grad
print("sds", float(loss), "grad finite", bool(torch.isfinite(g).all()), "nonzero strands", int((g.abs().sum((1,2))>0).sum()), round(time.time()-t0,1))
t0 = time.time(); new = pr.sdedit(tex, band, 0.5, 3); print("sdedit changed outside band:", float((new-tex)[:, ~torch.as_tensor(band)].abs().max()), "inside:", float((new-tex)[:, torch.as_tensor(band)].abs().mean()), round(time.time()-t0,1))
rc = np.clip(np.floor(uv*256).astype(int),0,255); lat = new[:64, rc[:,0], rc[:,1]].T
t0 = time.time(); w = pr.decode(lat[:500], uv[:500], pos[:500, 0]); print("decode", w.shape, "root err", float(np.abs(w[:,0]-pos[:500,0]).max()), round(time.time()-t0,1))
has = torch.as_tensor(build.has, device="cuda")
print("builder vs texture on texels WITH strands:", float((t2[:64]-tex[:64])[:, has].abs().max()), " holes:", float((t2[:64]-tex[:64])[:, ~has & torch.as_tensor(region, device='cuda')].abs().max()), "hole texels in region", int((~build.has & region).sum()), "of", int(region.sum()))
