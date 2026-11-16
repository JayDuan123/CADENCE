"""Export data behind paper Fig. 2 (figure2_combined.png) for the site's animated version.
(a) best-atom AUROC per layer (SAE vs dense) from fs_data/steer/layer_fig4cats.npz
(b) linear-probe test AUROC per layer from fs_data/fig5/four_ci.npy
(c,d) the same four ECG examples as fig2_combined_gen.py (MIMIC + PTB-XL, same atoms / leads),
      with per-patch Layer-6 activation recomputed by CSFM + SAE.
"""
import os, sys, csv, json
import numpy as np, torch
from scipy.io import loadmat

FIG = "/rhf/allocations/wq8/jay/cadence_rebuttal/repro/figures"
REPO = "/rhf/allocations/wq8/yd68/Cardiac-Sensing-FM"
PTB = "/rhf/allocations/wq8/yd68/data/ptb-xl"
sys.path.insert(0, "/rhf/allocations/wq8/jay/cadence_rebuttal/repro/sae")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/utils")
from sae_fig import BatchTopKSAE
from extract_tokens import build_model, tokens_per_layer, N_PATCH, N_LEAD
from preprocess import preprocess_ecg

OUT = sys.argv[1]
LEAD = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']; LAY = [1,2,3,4,5,6]
ROWS = [("Phenotype — premature ventricular complex", [("MIMIC",48129460,5530,"I"),("PTB-XL",15035,2375,"V2")]),
        ("Rhythm — atrial fibrillation",            [("MIMIC",41354263,5715,"III"),("PTB-XL",17000,6381,None)])]

FC = np.load(FIG + "/fs_data/steer/layer_fig4cats.npz", allow_pickle=True)
RES = np.load(FIG + "/fs_data/fig5/four_ci.npy", allow_pickle=True).item()
cats = ["phenotype", "morphology", "age"]
a = {k: dict(sae=np.round(np.asarray(FC[f"sae_{k}"], float), 4).tolist(),
             dense=np.round(np.asarray(FC[f"dense_{k}"], float), 4).tolist()) for k in cats}
b = {k: dict(sae=[round(float(RES[(k, "sae", L)][0]), 4) for L in LAY],
             dense=[round(float(RES[(k, "csfm", L)][0]), 4) for L in LAY]) for k in cats}

ck = torch.load(FIG + "/fs_data/sweep/sae_L6_k128_h8192.pt", map_location="cpu")
sae = BatchTopKSAE(ck["d"], ck["F"], ck["k"]); sae.load_state_dict(ck["state"]); sae.eval()
m6, s6, d = ck["mean"], ck["scale"], ck["d"]
csfm = build_model(REPO + "/pretrained/CSFM_tiny.pth", "Tiny", "cpu")
eid = np.array([int(r["study_id"]) for r in csv.DictReader(open(FIG + "/fs_data/eval/ids.csv"))])
sig = np.load(FIG + "/fs_data/eval/signals.npy", mmap_mode="r")

def ptb(ecg_id):
    import glob
    hea = glob.glob(f"{PTB}/g*/HR{ecg_id:05d}.hea")[0]
    lines = open(hea).read().splitlines(); fs = int(lines[0].split()[2])
    names = [ln.split()[-1].upper() for ln in lines[1:13]]
    w = np.asarray(loadmat(hea[:-4] + ".mat")["val"], np.float32)
    if w.shape[0] != 12: w = w.T
    return np.asarray(preprocess_ecg(w[[names.index(L.upper()) for L in LEAD]], fs=fs), np.float32)

ex = []
for title, items in ROWS:
    row = []
    for ds, ident, atom, fixed in items:
        s12 = ptb(ident) if ds == "PTB-XL" else np.asarray(sig[int(np.where(eid == ident)[0][0])], np.float32)
        with torch.no_grad():
            H = tokens_per_layer(csfm, torch.tensor(s12)[None], list(range(12)), {6})[6]
            am = sae.encode_infer((H.reshape(-1, d) - m6) / s6).reshape(N_LEAD, N_PATCH, -1)[:, :, atom].numpy()
        l = LEAD.index(fixed) if fixed else int(am.sum(1).argmax())
        tr = s12[l, ::2]; sc = float(np.abs(tr).max()) or 1.0
        row.append(dict(ds=ds, atom=atom, lead=LEAD[l],
                        w=np.clip(np.round(tr / sc * 127), -127, 127).astype(int).tolist(),
                        act=np.round(am[l], 2).tolist()))
        print(title, ds, atom, LEAD[l], int((am[l] > 0).sum()), flush=True)
    ex.append(dict(title=title, ex=row))

json.dump(dict(layers=LAY, a=a, b=b, ecg=ex), open(OUT, "w"), separators=(",", ":"))
print("wrote", OUT)
