#!/usr/bin/env python
"""plot_dist.py — joint-angle distributions of N retargeting arms on paired clips (robot joint space).

    python scripts/skel/plot_dist.py --xml assets/unitree_g1/g1_mocap_29dof.xml \
        --folders $OUT/smpl_robot $OUT/smpl_robot_bone $OUT/skel_robot $OUT/skel_robot_general \
        --names SMPL-Tpose SMPL-bone SKEL-Tpose SKEL-axis --out results/dist/paper.pdf
    ... --all --out results/dist/all_joints.pdf          # every hinge joint, appendix grid
    ... --overlay <clip key> --out results/dist/<clip>.pdf   # the four arms' joint angles over time for one clip

Frames pooled over the clips every folder has. Dashed = joint limits, amber = --sat-deg band inside each limit.
At-limit percentages (within --sat-deg) are printed per joint and arm.
"""
import argparse, glob, os, pickle
import numpy as np
import mujoco as mj

COLORS = ["#c8553d", "#e59a1c", "#3a6ea5", "#2f8f73", "#7b5aa6", "#555555"]
DEFAULT = "left_hip_roll_joint,left_knee_joint,left_ankle_pitch_joint,left_elbow_joint"
TITLES = {"left_hip_roll_joint": "Hip roll (L)", "left_knee_joint": "Knee (L)", "left_ankle_pitch_joint": "Ankle pitch (L)", "left_elbow_joint": "Elbow (L)"}


def clip_key(p):
    k = os.path.splitext(os.path.basename(p))[0]
    for s in ("_skel", "_smplx", "_smpl", "_poses"):
        if k.endswith(s): k = k[: -len(s)]
    return k


def hinge_joints(m):
    js = [(int(m.jnt_qposadr[j]), j) for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
    js.sort()
    return [(mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j), i, float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) for i, (_, j) in enumerate(js)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True); ap.add_argument("--folders", nargs="+", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--joints", default=DEFAULT); ap.add_argument("--all", action="store_true"); ap.add_argument("--overlay", default=None)
    ap.add_argument("--bins", type=int, default=50); ap.add_argument("--sat-deg", type=float, default=2.0)
    ap.add_argument("--width", type=float, default=3.4); ap.add_argument("--height", type=float, default=2.6); ap.add_argument("--fontsize", type=float, default=8)
    ap.add_argument("--out", default="joint_dist.pdf")
    a = ap.parse_args(); assert len(a.folders) == len(a.names)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    m = mj.MjModel.from_xml_path(a.xml); hinges = hinge_joints(m); by_name = {h[0]: h for h in hinges}
    idx = [{clip_key(p): p for p in glob.glob(os.path.join(f, "**", "*.pkl"), recursive=True)} for f in a.folders]
    keys = sorted(set.intersection(*(set(i) for i in idx))); print(f"paired clips: {len(keys)}")
    joints = hinges if a.all else [by_name[j] for j in a.joints.split(",")]
    fs = a.fontsize

    if a.overlay:                                                       # one clip, angle vs time per arm
        ov = clip_key(a.overlay); D = [np.degrees(pickle.load(open(i[ov], "rb"))["dof_pos"]) for i in idx]
        fps = float(pickle.load(open(idx[0][ov], "rb"))["fps"])
        n = len(joints); cols = 2 if n <= 4 else 4; rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 1.8 * rows), dpi=200, squeeze=False)
        for ax, (jn, adr, lo, hi) in zip(axes.ravel(), joints):
            for k, (d, nm) in enumerate(zip(D, a.names)):
                ax.plot(np.arange(len(d)) / fps, d[:, adr], color=COLORS[k], lw=1.0, label=nm)
            for L in (np.degrees(lo), np.degrees(hi)): ax.axhline(L, color="0.25", lw=0.8, ls="--")
            ax.set_title(TITLES.get(jn, jn.replace("_joint", "")), fontsize=fs, fontweight="bold"); ax.tick_params(labelsize=fs - 1)
        for ax in axes.ravel()[n:]: ax.axis("off")
        axes[0, 0].legend(fontsize=fs - 1, frameon=False); axes[-1, 0].set_xlabel("time [s]", fontsize=fs)
        fig.tight_layout(); fig.savefig(a.out); print("wrote", a.out); return

    data = []
    for nm, i in zip(a.names, idx):
        chunks = [pickle.load(open(i[k], "rb"))["dof_pos"] for k in keys]
        chunks = [c for c in chunks if c.shape[1] == len(hinges)]
        data.append(np.degrees(np.concatenate(chunks, 0))); print(f"{nm}: {len(chunks)} clips, {len(data[-1])} frames")

    n = len(joints); cols = 2 if n <= 4 else 5; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(a.width, a.height) if not a.all else (2.0 * cols, 1.6 * rows), dpi=300, squeeze=False)
    for ax, (jn, adr, lo, hi) in zip(axes.ravel(), joints):
        lo_d, hi_d = np.degrees(lo), np.degrees(hi); vals = [d[:, adr] for d in data]
        span = (min([lo_d] + [v.min() for v in vals]), max([hi_d] + [v.max() for v in vals])); pad = 0.03 * (span[1] - span[0])
        edges = np.linspace(span[0], span[1], a.bins + 1)
        for k, (v, nm) in enumerate(zip(vals, a.names)):
            ax.hist(v, bins=edges, density=True, histtype="stepfilled", color=COLORS[k], alpha=0.15, lw=0)
            ax.hist(v, bins=edges, density=True, histtype="step", color=COLORS[k], lw=1.1, label=nm)
        if np.isfinite(lo_d):
            for L in (lo_d, hi_d): ax.axvline(L, color="0.25", lw=0.8, ls="--")
            ax.axvspan(lo_d, lo_d + a.sat_deg, color="#e59a1c", alpha=0.3, lw=0); ax.axvspan(hi_d - a.sat_deg, hi_d, color="#e59a1c", alpha=0.3, lw=0)
        ax.set_xlim(span[0] - pad, span[1] + pad); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=fs - 1, length=2, pad=1)
        ax.set_title(TITLES.get(jn, jn.replace("_joint", "")), fontsize=fs, fontweight="bold", pad=3)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        sat = "  ".join(f"{nm} {100 * ((v < lo_d + a.sat_deg) | (v > hi_d - a.sat_deg)).mean():.1f}%" for v, nm in zip(vals, a.names))
        print(f"{jn:26s} [{lo_d:6.1f}, {hi_d:6.1f}]  at-limit: {sat}")
    for ax in axes.ravel()[n:]: ax.axis("off")
    for ax in axes[-1]: ax.set_xlabel("angle [deg]", fontsize=fs - 1, labelpad=1)
    axes[0, 0].legend(fontsize=fs - 1, frameon=False, handlelength=1.2, borderaxespad=0.2, loc="upper right")
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    stem = os.path.splitext(a.out)[0]; os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=300); print(f"wrote {stem}.pdf/.png")


if __name__ == "__main__":
    main()
