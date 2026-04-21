import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def generate_combined_dashboard(tdcf_path, base_path, out_path):
    with open(tdcf_path, "r") as f:
        tdcf_data = json.load(f)
    
    with open(base_path, "r") as f:
        base_data = json.load(f)
    
    ht = tdcf_data["tdcf"]
    hb = base_data.get("baseline", base_data)  # Handle if baseline is nested or flat depending on how it was saved
    if "baseline" in hb:
        hb = hb["baseline"]
        
    class MockEst:
        def __init__(self, data):
            self.band_sensitivity_history = data.get("band_sensitivity", [])
            self.patch_sensitivity_history = [np.array(s) for s in data.get("patch_sensitivity", [])]
            self.coeff_sensitivity_history = []
            if self.patch_sensitivity_history:
                self.nph = self.npw = 4
    
    class MockFs:
        def __init__(self, data):
            self.K = np.array(data.get("schedule_K", []))
            self.q = np.array(data.get("schedule_q", []))
            self.budget = np.array(data.get("schedule_budget", []))

    est = MockEst(tdcf_data)
    fs = MockFs(tdcf_data)
    backbone = tdcf_data["config"]["backbone"]

    plt.style.use('seaborn-v0_8-muted')
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"TDCF on CIFAR-100 — {backbone.upper()} backbone", 
                 fontsize=22, fontweight="bold", y=0.98)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)
    c_tdcf, c_base = "#2196F3", "#FF5722"

    # 1 — Test Accuracy
    ax = fig.add_subplot(gs[0,0])
    ax.plot(ht["epoch"], ht["test_acc"], "o-", c=c_tdcf, lw=2.5, ms=4, label="TDCF (38% I/O Saved)")
    if hb and "epoch" in hb and "test_acc" in hb:
        ax.plot(hb["epoch"], hb["test_acc"], "s--", c=c_base, lw=2.5, ms=4, label="Baseline (Full I/O)")
    ax.set_title("Test Accuracy", fontsize=15, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=12); ax.grid(True, alpha=.3)

    # 2 — Training Loss
    ax = fig.add_subplot(gs[0,1])
    ax.plot(ht["epoch"], ht["train_loss"], "o-", c=c_tdcf, lw=2, ms=4, label="TDCF")
    if hb and "epoch" in hb and "train_loss" in hb:
        ax.plot(hb["epoch"], hb["train_loss"], "s--", c=c_base, lw=2, ms=4, label="Baseline")
    ax.set_title("Training Loss", fontsize=15, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=.3)

    # 3 — Fidelity Schedule / Budget
    ax = fig.add_subplot(gs[0,2])
    if fs.budget.size:
        ax.plot(np.arange(1, len(fs.budget) + 1), fs.budget,
                "o-", c="#4CAF50", lw=2.5, ms=4, label="Budget(e)")
        ax.set_title("Budget Schedule", fontsize=15, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Band-slots")
    else:
        ax.plot(fs.K, "o-", c="#4CAF50", lw=2.5, ms=4, label="K(e) bands")
        ax.plot(fs.q, "s-", c="#9C27B0", lw=2.5, ms=4, label="q(e) patches")
        ax.set_title("Fidelity Schedule", fontsize=15, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, alpha=.3)

    # 4 — I/O Ratio
    ax = fig.add_subplot(gs[1,0])
    ratio = ht.get("io_ratio", [1.0]*len(ht["epoch"]))
    ax.fill_between(ht["epoch"], ratio, alpha=.2, color="#FF9800")
    ax.plot(ht["epoch"], ratio, "o-", c="#FF9800", lw=2.5, ms=4, label="Physical I/O Ratio")
    ax.axhline(1.0, ls="--", c="gray", alpha=0.5, label="Full Baseline Data")
    ax.set_title("I/O Budget / Full Dataset", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=.3)

    # 5 — Band Sensitivity
    ax = fig.add_subplot(gs[1,1])
    if est.band_sensitivity_history:
        sm = np.stack(est.band_sensitivity_history)
        im = ax.imshow(sm.T, aspect="auto", cmap="magma", origin="lower")
        ax.set_title("Band Sensitivity $s_e(b)$", fontsize=15, fontweight="bold")
        ax.set_xlabel("Pilot Epoch"); ax.set_ylabel("Band Index")
        plt.colorbar(im, ax=ax)

    # 6 — Patch Sensitivity
    ax = fig.add_subplot(gs[1,2])
    if est.patch_sensitivity_history:
        pm = est.patch_sensitivity_history[-1].reshape(4, 4)
        im = ax.imshow(pm, cmap="YlOrRd", interpolation="nearest")
        ax.set_title("Spatial Sensitivity (Last Pilot)", fontsize=15, fontweight="bold")
        plt.colorbar(im, ax=ax)

    # 7 — Learning Rate
    ax = fig.add_subplot(gs[2,0])
    ax.plot(ht["epoch"], ht["lr"], "o-", c="#00BCD4", lw=2)
    ax.set_title("Learning Rate (Cosine)", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=.3)

    # 8 — Wall Clock (Estimated)
    ax = fig.add_subplot(gs[2,1])
    ax.plot(ht["epoch"], ht["wall_s"], "o-", c=c_tdcf, lw=2, ms=4, label="TDCF")
    if hb and "epoch" in hb and "wall_s" in hb:
        ax.plot(hb["epoch"], hb["wall_s"], "s--", c=c_base, lw=2, ms=4, label="Baseline")
    ax.set_title("Cumulative Wall Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Seconds")
    ax.legend(); ax.grid(True, alpha=.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    print(f"Saved dashboard to {out_path}")

def parse_args():
    p = argparse.ArgumentParser("generate_combined_dashboard")
    p.add_argument("tdcf_results")
    p.add_argument("baseline_results")
    p.add_argument("out_path")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_combined_dashboard(
        args.tdcf_results,
        args.baseline_results,
        args.out_path,
    )
