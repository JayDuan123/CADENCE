"""Export Atom Explorer data for the CADENCE site.

For the 1000 randomly sampled interpretable L6 atoms of the paper's LLM run (bin1000):
  - LLM name / description / geometry, card metadata (best phenotype + AUROC, lead, phase)
  - per-atom Spearman rho of LLM-predicted vs measured held-out activations
  - top-3 MIMIC-IV-ECG test ECGs by per-ECG max activation; for each, the lead with the
    largest activation mass, its 10 s trace (125 Hz, int8) and per-patch activations (100)
    recomputed with the frozen CSFM + L6 SAE over all 1200 tokens.
Writes static/explorer/index.json and static/explorer/a/<atom>.json.
"""
import os, sys, json, glob
import numpy as np, torch

FS_ = "/rhf/allocations/wq8/yd68/csfm_embed/fullscale"
REPO = "/rhf/allocations/wq8/yd68/Cardiac-Sensing-FM"
sys.path.insert(0, "/rhf/allocations/wq8/jay/cadence_rebuttal/repro/sae")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/utils")
from sae_fig import BatchTopKSAE
from extract_tokens import build_model, tokens_per_layer, N_PATCH, N_LEAD

OUT = sys.argv[1]
LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
NTOP = 3
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))

def rank(a):
    r = np.empty(len(a)); r[np.argsort(a, kind="mergesort")] = np.arange(len(a))
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); return (sums / cnt)[inv]
def spearman(m, p):
    m, p = np.asarray(m, float), np.asarray(p, float)
    if m.std() == 0 or p.std() == 0: return None
    return float(np.corrcoef(rank(m), rank(p))[0, 1])

# ---- descriptions, cards, scores ----
atoms = {}
for sd in sorted(glob.glob(FS_ + "/fs_out/auto_desc/bin1000/s*")):
    cards = {c["atom"]: c for c in json.load(open(sd + "/token_cards.json"))}
    for d in json.load(open(sd + "/token_descriptions.json")):
        j = int(d["atom"]); md = cards[j]["metadata"]
        atoms[j] = dict(id=j, name=d.get("short_name", ""), desc=d.get("morphology_description", ""),
                        geom=d.get("primary_geometry", {}), conf=d.get("confidence", ""),
                        kind=d.get("kind", cards[j].get("kind", "")),
                        lead=md.get("modal_lead"), group=md.get("lead_group"),
                        phase=md.get("phase_hist", {}), pheno=md.get("best_pheno"),
                        auroc=md.get("best_auroc"), enrich=md.get("report_enrichment", []))
    rows = {}
    for ln in open(sd + "/tokscore_pred_primary.jsonl"):
        try: r = json.loads(ln)
        except Exception: continue
        rows.setdefault(int(r["atom"]), []).append(r)
    for j, rs in rows.items():
        if j in atoms:
            atoms[j]["rho"] = spearman([r["measured"] for r in rs], [r["predicted"] for r in rs])
print("atoms", len(atoms), flush=True)

# ---- top ECGs ----
A = np.load(FS_ + "/fs_data/auto/act_ecg_L6.npy", mmap_mode="r")
ids = list(atoms)
Asub = np.asarray(A[:, ids], np.float32)
need = {}
for k, j in enumerate(ids):
    top = np.argsort(-Asub[:, k])[:NTOP]
    atoms[j]["top"] = [int(e) for e in top]
    for e in top: need.setdefault(int(e), []).append(j)
print("unique ECGs", len(need), flush=True)

ck = torch.load(FS_ + "/fs_data/sweep/sae_L6_k128_h8192.pt", map_location=dev)
sae = BatchTopKSAE(ck["d"], ck["F"], ck["k"]).to(dev); sae.load_state_dict(ck["state"]); sae.eval()
m6, s6, d_ = ck["mean"].to(dev), ck["scale"].to(dev), ck["d"]
csfm = build_model(REPO + "/pretrained/CSFM_tiny.pth", "Tiny", dev)
sig = np.load(FS_ + "/fs_data/eval/signals.npy", mmap_mode="r")
chs = list(range(12))

ecgdat = {}
for n, (e, js) in enumerate(sorted(need.items())):
    s12 = np.asarray(sig[e], np.float32)
    with torch.no_grad():
        H = tokens_per_layer(csfm, torch.tensor(s12, device=dev)[None], chs, {6})[6]
        z = sae.encode_infer((H.reshape(-1, d_) - m6) / s6)
        z = z[:, torch.tensor(js, device=dev)].reshape(N_LEAD, N_PATCH, len(js)).cpu().numpy()
    for q, j in enumerate(js):
        am = z[:, :, q]; mass = am.sum(1); l = int(mass.argmax())
        tr = s12[l, ::2]                                   # 250 -> 125 Hz, 1250 samples
        sc = float(np.abs(tr).max()) or 1.0
        ecgdat[(e, j)] = dict(ecg=e, lead=LEADS[l], scale=round(sc, 4),
                              w=np.clip(np.round(tr / sc * 127), -127, 127).astype(int).tolist(),
                              act=np.round(am[l], 2).tolist(),
                              mass=np.round(mass / (mass.sum() or 1), 3).tolist(),
                              peak=round(float(am.max()), 2))
    if n % 200 == 0: print(f"  ecg {n}/{len(need)}", flush=True)

os.makedirs(OUT + "/a", exist_ok=True)
index = []
for j in sorted(atoms):
    a = atoms[j]
    ex = [ecgdat[(e, j)] for e in a.pop("top")]
    json.dump(dict(a, ex=ex), open(f"{OUT}/a/{j}.json", "w"), separators=(",", ":"))
    index.append(dict(id=j, name=a["name"], kind=a["kind"], lead=a["lead"], group=a["group"],
                      pheno=a["pheno"], auroc=a["auroc"], rho=None if a.get("rho") is None else round(a["rho"], 3)))
json.dump(index, open(OUT + "/index.json", "w"), separators=(",", ":"))
print("done", len(index))
