#!/usr/bin/env python
"""Frontal view: the SMPL rest-pose thigh and shank are not collinear in the frontal plane, and the G1 knee (pitch only)
cannot reproduce that bend. Left: SMPL skin + left-leg chain, frontal angles of thigh and shank. Right: G1 with the hip
roll set so its thigh has the SMPL thigh frontal angle; the shank then has the same angle (knee is a sagittal hinge),
while the dashed line shows where the SMPL shank direction would send it. The arc at the G1 knee is the frontal angle
SMPL-bone asks for and the knee cannot deliver; the IK pushes it into hip roll, ankle roll and the residual.

    python scripts/fig_frontal_knee.py --out fig_frontal_knee.png            # next to fig_tpose_models.py / fig_leg_mapping.py
    ... --gender female --smpl_ext npz --xml assets/unitree_g1/g1_mocap_29dof.xml
"""
import argparse, os, sys, tempfile
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.environ.get("FIG_LIB", "")):     # fig_tpose_models.py / fig_leg_mapping.py: same dir, scripts/, or $FIG_LIB
    if _p and _p not in sys.path: sys.path.insert(0, _p)
import fig_tpose_models as T
from fig_leg_mapping import SMPL_LEG, SKEL_LEG, G1_LEG, HUM_LINE, G1_LINE, INK, MUTED, meshes_only, joints_of, height, chain, render_side

RZ90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)       # canonical (facing -y) -> facing +x, so the +x side camera of render_side sees the front


def rot_part(mesh, R):
    mesh.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    if getattr(mesh, "_kind", "") == "spheres": mesh._points = mesh._points @ R.T
    elif getattr(mesh, "_kind", "") == "segments": mesh._pairs = [(R @ p, R @ q) for p, q in mesh._pairs]


def front_deg(v):
    """frontal angle of a leg segment, 0 = straight down, + = outward (person left = +x in the canonical frame)"""
    return np.degrees(np.arctan2(v[0], -v[2]))


def g1_hip_roll_to(m, d, thigh_deg):
    """1-D sweep on left_hip_roll so the left thigh (hip_pitch -> knee anchors) has the given frontal angle (robot frame: left = +y)"""
    import mujoco
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    hr, hp, hy, kn, an = (J(f"left_{n}_joint") for n in ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_pitch"))
    fr = lambda v: np.degrees(np.arctan2(v[1], -v[2]))
    print("G1 zero pose, frontal: thigh from hip_pitch %+.1f, hip_roll %+.1f, hip_yaw %+.1f; shank %+.1f deg" % (
        fr(d.xanchor[kn] - d.xanchor[hp]), fr(d.xanchor[kn] - d.xanchor[hr]), fr(d.xanchor[kn] - d.xanchor[hy]), fr(d.xanchor[an] - d.xanchor[kn])))
    lo, hi = m.jnt_range[hr]; qs = np.linspace(lo, hi, 2001); err = []
    for q in qs:
        d.qpos[m.jnt_qposadr[hr]] = q; mujoco.mj_forward(m, d); v = d.xanchor[kn] - d.xanchor[hr]
        err.append(abs(np.degrees(np.arctan2(v[1], -v[2])) - thigh_deg))
    q = qs[int(np.argmin(err))]; d.qpos[m.jnt_qposadr[hr]] = q; mujoco.mj_forward(m, d)
    print(f"G1 left_hip_roll = {np.degrees(q):+.1f} deg gives thigh frontal angle {thigh_deg:+.1f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl_models", default="assets/body_models"); ap.add_argument("--smpl_ext", default="npz")
    ap.add_argument("--skel_models", default="assets/body_models/skel"); ap.add_argument("--gender", default="male")
    ap.add_argument("--xml", default="assets/unitree_g1/g1_mocap_29dof.xml"); ap.add_argument("--g1_all_geoms", action="store_true")
    ap.add_argument("--g1_qpos0", action="store_true")
    ap.add_argument("--joint_r", type=float, default=0.016); ap.add_argument("--seg_r", type=float, default=0.006)
    ap.add_argument("--smpl_alpha", type=float, default=0.30); ap.add_argument("--g1_alpha", type=float, default=0.55)
    ap.add_argument("--joint_front", type=float, default=0.15); ap.add_argument("--gap", type=float, default=0.12)
    ap.add_argument("--width", type=float, default=5.0); ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=400); ap.add_argument("--fontsize", type=float, default=8)
    ap.add_argument("--human", default="smpl", choices=["smpl", "skel"], help="which human model to put beside the G1")
    ap.add_argument("--skel_alpha", type=float, default=0.16)
    ap.add_argument("--out", default="fig_frontal_knee.png")
    a = ap.parse_args()

    # ---- SMPL rest pose: frontal angles of the left thigh and shank
    HUM = a.human.upper(); LEG = SMPL_LEG if a.human == "smpl" else SKEL_LEG; skin_alpha = a.smpl_alpha if a.human == "smpl" else a.skel_alpha
    smpl = T.smpl_parts(a) if a.human == "smpl" else T.skel_parts(a); P, _ = joints_of(smpl); Jh = {k: P[i] for k, i in LEG.items()}
    thigh = front_deg(Jh["knee"] - Jh["hip"]); shank = front_deg(Jh["ankle"] - Jh["knee"])
    print(f"{HUM} rest, frontal: thigh {thigh:+.1f} deg, shank {shank:+.1f} deg, bend at the knee {shank - thigh:+.1f} deg")
    hum = [(m, rgb, skin_alpha if al < 1 else al) for m, rgb, al in meshes_only(smpl)] + chain(P, list(LEG.values()), a)

    # ---- G1: hip roll so the thigh matches; the shank follows the thigh (pitch-only knee)
    g = T.g1_parts(a, pose=lambda m, d: g1_hip_roll_to(m, d, thigh))
    k = height(smpl) / height(T.g1_parts(a))
    for p in g: T.scale_part(p[0], k)
    names = joints_of(g)[1]; gi = {n: names.index(n) for n in G1_LEG}; Pg, _ = joints_of(g)
    g1 = [(m, rgb, a.g1_alpha) for m, rgb, _ in meshes_only(g)] + chain(Pg, [gi[n] for n in G1_LEG], a)
    g1[-2] = (T.segments(np.asarray([Pg[gi[n]] for n in G1_LEG]), [-1, -1, -1, 1, 3, 4], a.seg_r), T.SEG, 1.0)   # thigh drawn from the hip_roll anchor
    Jg = {n: Pg[gi[n]] for n in G1_LEG}
    print(f"G1 thigh {front_deg(Jg['left_knee_joint'] - Jg['left_hip_roll_joint']):+.1f}, shank {front_deg(Jg['left_ankle_pitch_joint'] - Jg['left_knee_joint']):+.1f} deg (frontal)")

    # ---- to the front camera: rotate, ground, place human then G1 along screen x (= world y after rotation)
    for parts in (hum, g1):
        for p in parts: rot_part(p[0], RZ90)
    def place(parts, y0):
        allv = np.concatenate([p[0].vertices for p in parts]); lo, hi = allv.min(0), allv.max(0)
        for p in parts: T.shift_part(p[0], [-(lo[0] + hi[0]) / 2, y0 - (lo[1] + hi[1]) / 2, -lo[2]])
        return (hi[1] - lo[1]) / 2
    place(hum, 0.0); place(g1, 0.0)
    Ph0 = [p[0] for p in hum if getattr(p[0], "_kind", "") == "spheres"][0]._points; Pg0 = [p[0] for p in g1 if getattr(p[0], "_kind", "") == "spheres"][0]._points
    dy = Ph0[list(LEG).index("pelvis")][1] + a.gap - Pg0[gi["left_hip_roll_joint"]][1] + 0.35      # G1 pelvis ~gap+0.35 m right of the SMPL pelvis (legs only, the T-pose arms are cropped away)
    for p in g1: T.shift_part(p[0], [0, dy, 0])
    for parts in (hum, g1):
        for p in parts:
            if getattr(p[0], "_kind", ""): T.shift_part(p[0], [a.joint_front, 0, 0])
    Ph = [p[0] for p in hum if getattr(p[0], "_kind", "") == "spheres"][0]._points; Jh = {n: Ph[i] for i, n in enumerate(LEG)}
    Pg = [p[0] for p in g1 if getattr(p[0], "_kind", "") == "spheres"][0]._points; Jg = {n: Pg[i] for i, n in enumerate(G1_LEG)}

    # legs only: crop to the leg region in y (screen x), from the ground to a bit above the hips
    ylo, yhi = Jh["pelvis"][1] - 0.42, Jg["left_hip_roll_joint"][1] + 0.36      # right margin: room for the knee-axis label
    zc_top = max(Jh["pelvis"][2], Jg["left_hip_pitch_joint"][2]) + 0.12
    ymag = (zc_top + 0.06) / 2; zc = ymag - 0.06; fovy = np.degrees(2 * np.arctan(ymag / 30.0))
    H = int(round(a.height * a.dpi)) if a.height else 1200; W = int(round(H * (yhi - ylo) / (2 * ymag)))
    cams = [("front", (ylo + yhi) / 2, zc, fovy, W)]
    img = render_side(hum + g1, cams, H, tempfile.mkdtemp(prefix="fig_front_"))[0]

    # ---- annotate
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    fig_w = a.width; fig_h = a.height or fig_w * H / W; fig = plt.figure(figsize=(fig_w, fig_h), dpi=a.dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(img); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    cy = (ylo + yhi) / 2; m_px = H / (2 * ymag)
    px = lambda p: (W / 2 + (p[1] - cy) * m_px, H / 2 - (p[2] - zc) * m_px)
    lw, fs = 0.7, a.fontsize
    for h, gname in zip(("hip", "knee", "ankle"), ("left_hip_roll_joint", "left_knee_joint", "left_ankle_pitch_joint")):
        (u1, v1), (u2, v2) = px(Jh[h]), px(Jg[gname]); ax.plot([u1, u2], [v1, v2], ls=(0, (2, 2)), lw=lw, color=MUTED)
    # SMPL: extend the thigh line through the knee, arc to the shank = frontal bend at the knee
    def extend(p_from, p_to, L):
        d = (p_to - p_from); d = d / np.linalg.norm(d); return p_to + d * L
    u0, v0 = px(Jh["knee"]); ue, ve = px(extend(Jh["hip"], Jh["knee"], 0.25))
    ax.plot([u0, ue], [v0, ve], ls=(0, (3, 2)), lw=lw, color=HUM_LINE)
    bend = shank - thigh
    ax.text(u0 + 0.05 * m_px * np.sign(bend if bend else 1), v0 + 0.14 * m_px, f"{abs(bend):.0f}°", fontsize=fs, color=HUM_LINE, ha="center")
    ax.text(*px(Jh["hip"] + np.array([0, 0.14, -0.10])), f"thigh {thigh:+.0f}°", fontsize=fs * 0.85, color=HUM_LINE, ha="left", va="center")
    ax.text(*px(Jh["knee"] + np.array([0, 0.14, -0.14])), f"shank {shank:+.0f}°", fontsize=fs * 0.85, color=HUM_LINE, ha="left", va="center")
    # G1: requested shank direction (SMPL shank frontal angle) vs the one the pitch-only knee gives
    kg = Jg["left_knee_joint"]; sh = np.array([0, np.sin(np.radians(shank)), -np.cos(np.radians(shank))])   # screen-x = +y after RZ90
    sign = np.sign(Jg["left_ankle_pitch_joint"][1] - Jg["left_hip_roll_joint"][1]) or 1.0                   # outward side of the left leg on screen
    sh[1] *= sign
    u0, v0 = px(kg); ue, ve = px(kg + sh * 0.40); ax.plot([u0, ue], [v0, ve], ls=(0, (3, 2)), lw=lw * 1.3, color=HUM_LINE)
    um, vm = px(kg + sh * 0.26); ax.text(um - sign * 0.05 * m_px, vm, f"{HUM} shank\ndirection", fontsize=fs * 0.85, color=HUM_LINE, ha="right" if sign > 0 else "left", va="center")
    ax.text(u0 + sign * 0.06 * m_px, v0 + 0.16 * m_px, f"{abs(bend):.0f}°", fontsize=fs, color=HUM_LINE, ha="center")
    # knee hinge axis: horizontal in the frontal plane
    ax.plot([u0 - 0.07 * m_px, u0 + 0.07 * m_px], [v0, v0], lw=lw * 1.6, color=G1_LINE)
    ax.text(u0 + sign * 0.10 * m_px, v0 - 0.02 * m_px, "knee axis\n(pitch only)", fontsize=fs * 0.85, color=G1_LINE, ha="left" if sign > 0 else "right", va="center")
    ax.text(px(Jh["pelvis"])[0], 0.015 * H, f"{HUM} rest pose, front", ha="center", va="top", fontsize=fs)      # centred on each pelvis
    ax.text((px(Jg["left_hip_roll_joint"])[0] + px(Jg["right_hip_roll_joint"])[0]) / 2 if "right_hip_roll_joint" in Jg else px(Jg["left_hip_roll_joint"])[0] - 0.12 * m_px,
            0.015 * H, f"G1, hip roll = {HUM} thigh", ha="center", va="top", fontsize=fs)
    stem = os.path.splitext(a.out)[0]; os.makedirs(os.path.dirname(stem) or ".", exist_ok=True); fig.savefig(stem + ".png", dpi=a.dpi); fig.savefig(stem + ".pdf")
    print(f"wrote {stem}.png/.pdf")


if __name__ == "__main__":
    main()