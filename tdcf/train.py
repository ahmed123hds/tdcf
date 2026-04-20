"""
TDCF main training script for MNIST — GPU-optimized, ICLR-quality output.
Full pipeline from Section 4.8: pilot → schedule → adaptive training → eval.
"""
import os, sys, time, json, argparse, logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms as T

from .models import MNISTConvNet, MNISTTinyViT
from .sensitivity import SensitivityEstimator
from .scheduler import FidelityScheduler
from .dataloader import TDCFServer
from .transforms import (
    build_nested_masks, reconstruct_at_fidelity,
    dct2d, idct2d, precompute_dct_dataset,
)

# ── logging ───────────────────────────────────────────────────────────
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fmt = "[%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(
                                      os.path.join(save_dir, "train.log"),
                                      mode="w")])
    return logging.getLogger("TDCF")

# ── args ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("TDCF-MNIST")
    p.add_argument("--backbone", choices=["cnn","vit"], default="cnn")
    p.add_argument("--total_epochs", type=int, default=25)
    p.add_argument("--pilot_epochs", type=int, default=5)
    p.add_argument("--pilot_ratio", type=float, default=0.10)
    p.add_argument("--num_bands", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=7)
    p.add_argument("--eta_f", type=float, default=0.85)
    p.add_argument("--eta_s", type=float, default=0.75)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_workers", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--run_baseline", action="store_true")
    return p.parse_args()

def make_model(name, device):
    m = MNISTConvNet() if name == "cnn" else MNISTTinyViT()
    return m.to(device)

# ── helpers ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); loss_sum = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x); l = criterion(logits, y)
        loss_sum += l.item()*y.size(0)
        correct += (logits.argmax(1)==y).sum().item(); n += y.size(0)
    return loss_sum/n, correct/n

def train_epoch(model, loader, opt, crit, device, server=None):
    model.train(); loss_sum = correct = n = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if server is not None:
            x = server.serve(x)            # fidelity truncation on GPU
        opt.zero_grad(set_to_none=True)
        logits = model(x); loss = crit(logits, y)
        loss.backward(); opt.step()
        loss_sum += loss.item()*y.size(0)
        correct += (logits.argmax(1)==y).sum().item(); n += y.size(0)
    return loss_sum/n, correct/n

# ── pilot ─────────────────────────────────────────────────────────────
def run_pilot(model, pilot_coeffs, pilot_labels, masks_gpu,
              estimator, opt, crit, device, pilot_epochs, bs,
              eta_f, eta_s, log):
    log.info("="*60)
    log.info("PILOT PHASE  (%d epochs, %d samples, device=%s)",
             pilot_epochs, pilot_coeffs.shape[0], device)
    log.info("="*60)
    N = pilot_coeffs.shape[0]
    base_mask = masks_gpu[0].unsqueeze(0).unsqueeze(0)
    for ep in range(pilot_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            c_batch = pilot_coeffs[idx]
            y_batch = pilot_labels[idx]
            x_batch = idct2d(c_batch)      # full-fidelity for pilot

            # ── sensitivity measurement ──
            estimator.measure_coefficient_sensitivity_from_coeffs(
                c_batch, y_batch, model, crit)
            x_base = idct2d(c_batch * base_mask)
            estimator.measure_patch_sensitivity(
                x_batch, x_base, y_batch, model, crit)

            # ── normal training step ──
            opt.zero_grad(set_to_none=True)
            logits = model(x_batch); loss = crit(logits, y_batch)
            loss.backward(); opt.step()

        estimator.finalize_epoch()
        K = estimator.compute_band_cutoff(ep, eta_f)
        q = estimator.compute_patch_quota(ep, eta_s)
        band_s = estimator.band_sensitivity_history[ep]
        log.info("  pilot %d/%d | K=%2d/%d | q=%2d/%d | "
                 "top-band=%.4f  tail-band=%.4f",
                 ep+1, pilot_epochs, K, estimator.num_bands,
                 q, estimator.P, band_s[0], band_s[-1])

# ── main TDCF run ────────────────────────────────────────────────────
def run_tdcf(args, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    log.info("Device: %s  |  CUDA: %s", device,
             torch.cuda.get_device_name(0) if device.type=="cuda" else "N/A")
    t0 = time.time()

    # ── data ──
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,),(0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(args.data_dir, False, download=True, transform=tfm)

    log.info("Pre-computing DCT coefficients on GPU ...")
    train_coeffs, train_labels = precompute_dct_dataset(
        train_ds, device, num_workers=args.data_workers)
    test_coeffs,  test_labels  = precompute_dct_dataset(
        test_ds, device, num_workers=args.data_workers)
    log.info("  train: %s  test: %s  (%.1f MB GPU)",
             train_coeffs.shape, test_coeffs.shape,
             (train_coeffs.nelement()+test_coeffs.nelement())*4/1e6)

    # Test loader (full fidelity, matching the cached representation)
    test_loader = DataLoader(
        TensorDataset(idct2d(test_coeffs), test_labels),
        batch_size=512, shuffle=False)

    # ── model ──
    model = make_model(args.backbone, device)
    nparams = sum(p.numel() for p in model.parameters())
    log.info("Backbone: %s  |  params: %d (%.1fK)", args.backbone,
             nparams, nparams/1e3)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_lr = optim.lr_scheduler.CosineAnnealingLR(opt, args.total_epochs)
    crit = nn.CrossEntropyLoss()

    # ── masks & estimator ──
    masks = build_nested_masks(28, 28, args.num_bands)
    masks_gpu = [m.to(device) for m in masks]

    estimator = SensitivityEstimator(
        28, 28, args.num_bands, args.patch_size, device=device)

    # ── Step 2: pilot ──
    pilot_n = max(int(len(train_ds)*args.pilot_ratio), args.batch_size)
    pilot_idx = np.random.choice(len(train_ds), pilot_n, replace=False)
    pilot_coeffs = train_coeffs[pilot_idx]
    pilot_labels = train_labels[pilot_idx]
    run_pilot(model, pilot_coeffs, pilot_labels, masks_gpu,
              estimator, opt, crit, device, args.pilot_epochs,
              args.batch_size, args.eta_f, args.eta_s, log)

    # ── Step 3: fit schedule ──
    fsched = FidelityScheduler(args.num_bands, estimator.P,
                               args.eta_f, args.eta_s)
    fsched.fit_from_pilot(estimator, args.total_epochs)
    log.info("\n%s\n", fsched.summary())

    # ── Step 4: adaptive training ──
    server = TDCFServer(28, 28, args.num_bands, args.patch_size, device)

    # DataLoader from GPU tensors → just index shuffling
    train_tds = TensorDataset(train_coeffs, train_labels)
    train_loader = DataLoader(train_tds, batch_size=args.batch_size,
                              shuffle=True)

    log.info("="*60)
    log.info("ADAPTIVE TRAINING  (%d epochs)", args.total_epochs)
    log.info("="*60)

    hist = {k: [] for k in ["epoch","K","q","bytes_ratio",
            "train_loss","train_acc","test_loss","test_acc","wall_s","lr"]}

    for ep in range(args.total_epochs):
        K_e, q_e = fsched.get_fidelity(ep)
        if estimator.patch_sensitivity_history:
            ps_idx = min(ep, len(estimator.patch_sensitivity_history) - 1)
            ps = estimator.patch_sensitivity_history[ps_idx]
        else:
            ps = None
        server.set_fidelity(K_e, q_e, ps)
        br = server.get_bytes_ratio()

        tr_l, tr_a = train_epoch(model, train_loader, opt, crit, device, server)
        te_l, te_a = evaluate(model, test_loader, crit, device)
        sched_lr.step()
        lr_now = opt.param_groups[0]["lr"]
        elapsed = time.time()-t0

        for k,v in [("epoch",ep),("K",K_e),("q",q_e),("bytes_ratio",br),
                     ("train_loss",tr_l),("train_acc",tr_a),
                     ("test_loss",te_l),("test_acc",te_a),
                     ("wall_s",elapsed),("lr",lr_now)]:
            hist[k].append(v)

        log.info("  E %2d/%d | K=%2d q=%2d | approx=%.2f | "
                 "tr_loss=%.4f tr_acc=%.4f | te_loss=%.4f te_acc=%.4f | "
                 "lr=%.2e | %.1fs",
                 ep+1, args.total_epochs, K_e, q_e, br,
                 tr_l, tr_a, te_l, te_a, lr_now, elapsed)

    return model, hist, estimator, fsched

# ── baseline ──────────────────────────────────────────────────────────
def run_baseline(args, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,),(0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(args.data_dir, False, download=True, transform=tfm)
    pin_memory = device.type == "cuda"
    tr_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                           num_workers=args.data_workers, pin_memory=pin_memory)
    te_loader = DataLoader(test_ds, 512, num_workers=args.data_workers,
                           pin_memory=pin_memory)
    model = make_model(args.backbone, device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_lr = optim.lr_scheduler.CosineAnnealingLR(opt, args.total_epochs)
    crit = nn.CrossEntropyLoss()

    log.info("="*60); log.info("BASELINE (full fidelity)"); log.info("="*60)
    hist = {k:[] for k in ["epoch","train_loss","train_acc",
                            "test_loss","test_acc","wall_s"]}
    t0 = time.time()
    for ep in range(args.total_epochs):
        tr_l, tr_a = train_epoch(model, tr_loader, opt, crit, device)
        te_l, te_a = evaluate(model, te_loader, crit, device)
        sched_lr.step(); elapsed = time.time()-t0
        for k,v in [("epoch",ep),("train_loss",tr_l),("train_acc",tr_a),
                     ("test_loss",te_l),("test_acc",te_a),("wall_s",elapsed)]:
            hist[k].append(v)
        log.info("  E %2d/%d | tr %.4f | te %.4f | %.1fs",
                 ep+1, args.total_epochs, tr_a, te_a, elapsed)
    return model, hist

# ── visualization ─────────────────────────────────────────────────────
def save_plots(ht, hb, est, fs, save_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("TDCF on MNIST — Training-Dynamics-Coupled Fidelity",
                 fontsize=18, fontweight="bold", y=0.98)
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.32)

    c_tdcf, c_base = "#2196F3", "#FF5722"

    # 1 — test accuracy
    ax = fig.add_subplot(gs[0,0])
    ax.plot(ht["epoch"], ht["test_acc"], "o-", c=c_tdcf, lw=2, label="TDCF")
    if hb: ax.plot(hb["epoch"], hb["test_acc"], "s--", c=c_base, lw=2, label="Baseline")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy"); ax.legend(); ax.grid(True, alpha=.3)

    # 2 — train loss
    ax = fig.add_subplot(gs[0,1])
    ax.plot(ht["epoch"], ht["train_loss"], "o-", c=c_tdcf, lw=2, label="TDCF")
    if hb: ax.plot(hb["epoch"], hb["train_loss"], "s--", c=c_base, lw=2, label="Baseline")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
    ax.legend(); ax.grid(True, alpha=.3)

    # 3 — schedule
    ax = fig.add_subplot(gs[0,2])
    Ks, qs = fs.get_full_schedule()
    ax.plot(Ks, "o-", c="#4CAF50", lw=2, label="K(e) bands")
    ax.plot(qs, "s-", c="#9C27B0", lw=2, label="q(e) patches")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Count")
    ax.set_title("Monotone Fidelity Schedule"); ax.legend(); ax.grid(True, alpha=.3)

    # 4 — bytes ratio
    ax = fig.add_subplot(gs[1,0])
    ax.fill_between(ht["epoch"], ht["bytes_ratio"], alpha=.35, color="#FF9800")
    ax.plot(ht["epoch"], ht["bytes_ratio"], "o-", c="#FF9800", lw=2)
    ax.axhline(1, ls="--", c="gray"); ax.set_ylim(0,1.1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Fraction")
    ax.set_title("Approx. Served / Full"); ax.grid(True, alpha=.3)

    # 5 — band sensitivity heatmap
    ax = fig.add_subplot(gs[1,1])
    if est.band_sensitivity_history:
        sm = np.stack(est.band_sensitivity_history)
        im = ax.imshow(sm.T, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xlabel("Pilot Epoch"); ax.set_ylabel("Band (low→high freq)")
        ax.set_title("Band Sensitivity $s_e(b)$"); plt.colorbar(im, ax=ax)

    # 6 — patch sensitivity map
    ax = fig.add_subplot(gs[1,2])
    if est.patch_sensitivity_history:
        pm = est.patch_sensitivity_history[-1].reshape(est.nph, est.npw)
        im = ax.imshow(pm, cmap="hot", interpolation="nearest")
        ax.set_title("Patch Sensitivity (last pilot)"); plt.colorbar(im, ax=ax)

    # 7 — coefficient sensitivity evolution
    ax = fig.add_subplot(gs[2,0])
    if est.coeff_sensitivity_history:
        for i, ch in enumerate(est.coeff_sensitivity_history):
            ax.plot(np.sort(ch)[::-1], alpha=.7, label=f"pilot {i}")
        ax.set_xlabel("Coeff (sorted)"); ax.set_ylabel("|∂L/∂c_k|")
        ax.set_title("Coefficient Sensitivity Spectrum"); ax.legend(fontsize=7)
        ax.set_yscale("log"); ax.grid(True, alpha=.3)

    # 8 — wall time
    ax = fig.add_subplot(gs[2,1])
    ax.plot(ht["epoch"], ht["wall_s"], "o-", c=c_tdcf, lw=2, label="TDCF")
    if hb: ax.plot(hb["epoch"], hb["wall_s"], "s--", c=c_base, lw=2, label="Baseline")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Seconds")
    ax.set_title("Wall-clock Time"); ax.legend(); ax.grid(True, alpha=.3)

    # 9 — LR schedule
    ax = fig.add_subplot(gs[2,2])
    ax.plot(ht["epoch"], ht["lr"], "o-", c="#00BCD4", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("Learning Rate"); ax.grid(True, alpha=.3)

    fig.savefig(os.path.join(save_dir, "tdcf_results.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    # --- fidelity visualization: example images at different levels ---
    fig2, axes2 = plt.subplots(2, 8, figsize=(20, 5))
    fig2.suptitle("Multi-Fidelity Reconstruction (sample image)", fontsize=14)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,),(0.3081,))])
    test_ds = datasets.MNIST("./data", False, transform=tfm)
    sample_img = test_ds[0][0].unsqueeze(0).to(device)
    masks = build_nested_masks(28, 28, est.num_bands)
    masks_gpu = [m.to(device) for m in masks]
    num_show = min(8, len(masks_gpu))
    step = max(1, len(masks_gpu)//num_show)
    for col in range(num_show):
        lvl = min(col * step, len(masks_gpu)-1)
        recon = reconstruct_at_fidelity(sample_img, lvl, masks_gpu)
        axes2[0, col].imshow(recon[0,0].cpu(), cmap="gray")
        axes2[0, col].set_title(f"Level {lvl}")
        axes2[0, col].axis("off")
        diff = (sample_img - recon).abs()
        axes2[1, col].imshow(diff[0,0].cpu(), cmap="hot")
        axes2[1, col].axis("off")
    axes2[1,0].set_ylabel("Error")
    fig2.savefig(os.path.join(save_dir, "fidelity_levels.png"), dpi=200,
                 bbox_inches="tight")
    plt.close()

# ── main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    log = setup_logger(args.save_dir)

    log.info("="*60)
    log.info("  TDCF — Training-Dynamics-Coupled Fidelity (MNIST)")
    log.info("  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("="*60)
    log.info("Config: %s", json.dumps(vars(args), indent=2))

    model_t, ht, est, fs = run_tdcf(args, log)
    hb = None
    if args.run_baseline:
        _, hb = run_baseline(args, log)

    # ── save ──
    results = {"tdcf": ht, "baseline": hb, "config": vars(args),
               "band_sensitivity": [s.tolist() for s in est.band_sensitivity_history],
               "patch_sensitivity": [s.tolist() for s in est.patch_sensitivity_history],
               "schedule_K": fs.get_full_schedule()[0].tolist(),
               "schedule_q": fs.get_full_schedule()[1].tolist()}
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model_t.state_dict(),
               os.path.join(args.save_dir, "model_tdcf.pt"))

    save_plots(ht, hb, est, fs, args.save_dir)

    log.info("\n" + "="*60)
    log.info("FINAL RESULTS")
    log.info("="*60)
    log.info("  TDCF  test acc:   %.4f", ht["test_acc"][-1])
    if hb:
        log.info("  Base  test acc:   %.4f", hb["test_acc"][-1])
        log.info("  Accuracy drop:    %.4f", hb["test_acc"][-1]-ht["test_acc"][-1])
    log.info("  Avg approx ratio: %.3f", np.mean(ht["bytes_ratio"]))
    log.info("  TDCF wall time:   %.1fs", ht["wall_s"][-1])
    if hb: log.info("  Base wall time:   %.1fs", hb["wall_s"][-1])
    log.info("  Schedule:\n%s", fs.summary())
    log.info("  Results saved to %s/", args.save_dir)

if __name__ == "__main__":
    main()
