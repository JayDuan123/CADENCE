"""Find one held-out MIMIC-IV-ECG recording + lead where several *described* L6 atoms
(from the paper's 1000-atom LLM run) each lock onto a different part of the beat
(P/PR, QRS, ST, T), and export it for the site's decomposition animation.

Search: random sample of test ECGs, leads II and V2..V5. For each atom and phase bin we
score   sel = (activation mass inside the bin / total mass)  x  (fraction of beats where
the atom fires inside that bin).  An ECG/lead is scored by the weakest of its four best
per-phase atoms; the best ECG/lead is written to static/decomp.json.
"""
import os, sys, json, glob
import numpy as np, torch

FS_ = "/rhf/allocations/wq8/yd68/csfm_embed/fullscale"
REPO = "/rhf/allocations/wq8/yd68/Cardiac-Sensing-FM"
sys.path.insert(0, "/rhf/allocations/wq8/jay/cadence_rebuttal/repro/sae")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/utils")
from sae_fig import BatchTopKSAE
from extract_tokens import build_model, tokens_per_layer, N_PATCH, N_LEAD
import neurokit2 as nk

OUT = sys.argv[1]; NCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 400
LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
TRY_LEADS = [1, 7, 8, 9, 10]
PHASES = [("P wave / PR", -260, -40), ("QRS complex", -40, 90), ("ST segment", 90, 200), ("T wave", 200, 450)]
FS = 250; PATCH = 25
dev = "cpu"; torch.set_num_threads(max(1, os.cpu_count() or 1))

idx = json.load(open("/rhf/allocations/wq8/jay/cadence_site/static/explorer/index.json"))
meta = {a["id"]: a for a in idx}
ids = sorted(meta)

ck = torch.load(FS_ + "/fs_data/sweep/sae_L6_k128_h8192.pt", map_location=dev)
sae = BatchTopKSAE(ck["d"], ck["F"], ck["k"]); sae.load_state_dict(ck["state"]); sae.eval()
m6, s6, d_ = ck["mean"], ck["scale"], ck["d"]
csfm = build_model(REPO + "/pretrained/CSFM_tiny.pth", "Tiny", dev)
sig = np.load(FS_ + "/fs_data/eval/signals.npy", mmap_mode="r")
aidx = torch.tensor(ids)

rng = np.random.RandomState(7)
cands = rng.choice(sig.shape[0], NCAND, replace=False)
best = None
for n, e in enumerate(cands):
    s12 = np.asarray(sig[e], np.float32)
    if not np.isfinite(s12).all() or np.abs(s12).max() == 0: continue
    try:
        _, info = nk.ecg_peaks(nk.ecg_clean(s12[1], sampling_rate=FS), sampling_rate=FS)
        R = np.array(info["ECG_R_Peaks"])
    except Exception:
        continue
    if len(R) < 6 or len(R) > 16: continue
    rr = np.diff(R) / FS
    if rr.std() / rr.mean() > 0.08: continue                       # regular rhythm only
    with torch.no_grad():
        H = tokens_per_layer(csfm, torch.tensor(s12)[None], list(range(12)), {6})[6]
        z = sae.encode_infer((H.reshape(-1, d_) - m6) / s6)[:, aidx].reshape(N_LEAD, N_PATCH, len(ids)).numpy()
    centers = (np.arange(N_PATCH) + 0.5) * PATCH
    near = R[np.abs(centers[:, None] - R[None, :]).argmin(1)]
    off = (centers - near) / FS * 1000                               # ms relative to nearest R
    beat = np.abs(centers[:, None] - R[None, :]).argmin(1)
    inner = (beat > 0) & (beat < len(R) - 1)                          # ignore edge beats
    for l in TRY_LEADS:
        A = z[l]                                                     # (100, natoms)
        tot = A.sum(0) + 1e-9
        picks = []
        for name, lo, hi in PHASES:
            m = (off >= lo) & (off < hi)
            mass = A[m].sum(0) / tot
            hits = [A[(beat == b) & m].max(0, initial=0) > 0 for b in range(1, len(R) - 1)]
            cov = np.mean(hits, 0) if hits else np.zeros(A.shape[1])
            sc = mass * cov * (A.max(0) > 1.0)
            j = int(sc.argmax()); picks.append((float(sc[j]), j, name))
        used = [p[1] for p in picks]
        score = min(p[0] for p in picks) if len(set(used)) == len(used) else 0
        if best is None or score > best[0]:
            best = (score, int(e), l, picks, A, R, s12[l])
            print(f"[{n}] ecg {e} lead {LEADS[l]} score {score:.3f}", [(meta[ids[p[1]]]["name"], round(p[0], 2)) for p in picks], flush=True)

score, e, l, picks, A, R, tr = best
sc = float(np.abs(tr).max()) or 1.0
out = dict(ecg=e, lead=LEADS[l], score=round(score, 3), r=[int(x) for x in R],
           w=np.clip(np.round(tr[::2] / sc * 127), -127, 127).astype(int).tolist(),
           atoms=[dict(id=ids[j], name=meta[ids[j]]["name"], rho=meta[ids[j]]["rho"], phase=name, sel=round(s, 3),
                       act=np.round(A[:, j], 2).tolist()) for s, j, name in picks])
json.dump(out, open(OUT, "w"), separators=(",", ":"))
print("wrote", OUT, out["ecg"], out["lead"], [(a["phase"], a["id"], a["name"]) for a in out["atoms"]])
