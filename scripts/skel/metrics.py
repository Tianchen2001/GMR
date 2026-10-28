#!/usr/bin/env python
"""metrics.py — evaluate N retargeting arms on the same clips (reference-level, no policy).

    python scripts/skel/metrics.py --xml assets/unitree_g1/g1_mocap_29dof.xml \
        --folders  $OUT/smpl_robot $OUT/smpl_robot_bone $OUT/skel_robot $OUT/skel_robot_general \
        --names    SMPL-Tpose SMPL-bone SKEL-Tpose SKEL-axis \
        --src_folders $SMPL_SRC $SMPL_SRC $SKEL_SRC $SKEL_SRC \
        --out results/metrics

Per clip and arm (all on the retargeted reference):
    limit_nmr      fraction of joint-frames within 0.05 rad of a position limit (NMR convention)
    limit_phuma    fraction of joint-frames outside the central 98% of the range (PHUMA convention)
    selfcol        fraction of frames with a self-collision between non-adjacent bodies (MuJoCo contacts)
    jerk           RMS of the third finite difference of joint angles [rad/s^3]
    jumps          number of single-step joint changes above 0.5 rad
    ori_res        mean orientation residual of the IK targets [deg]: angle between the target link orientation
                   (R_human * rot_offset, as GMR builds it) and the link orientation reached by the solver
    pos_res        same for positions [cm], after removing the per-clip constant shift of HEIGHT_ADJUST / ROOT_ORIGIN
    + per-joint limit fractions (both conventions) and per-body residuals
Outputs: per_clip.csv, per_joint_limits.csv, per_body_residuals.csv, summary.md (means, and paired differences
vs the first arm with bootstrap 95% CIs). Fidelity needs the human source (--src_folders); omit it to skip.
"""
import argparse, csv, glob, os, pickle, sys
from collections import defaultdict
import numpy as np
import mujoco as mj

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NMR_EPS, PHUMA_FRAC, JUMP_RAD = 0.05, 0.98, 0.5


# ---------------------------------------------------------------- helpers
def clip_key(path):
    k = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_skel", "_smplx", "_smpl", "_poses"):
        if k.endswith(suf): k = k[: -len(suf)]
    return k


def index_pkls(folder):
    return {clip_key(p): p for p in glob.glob(os.path.join(folder, "**", "*.pkl"), recursive=True)}


def find_src(folder, key):
    hits = [p for p in glob.glob(os.path.join(folder, "**", f"{key}*.npz"), recursive=True) if not p.endswith("_stagei.npz")]
    return hits[0] if hits else None


class Robot:
    def __init__(self, xml):
        self.m = mj.MjModel.from_xml_path(xml); self.d = mj.MjData(self.m)
        hinge = [(int(self.m.jnt_qposadr[j]), j) for j in range(self.m.njnt) if self.m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
        hinge.sort(); self.jadr = [a for a, _ in hinge]; self.jid = [j for _, j in hinge]
        self.jnames = [mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_JOINT, j) for j in self.jid]
        self.lo = np.array([self.m.jnt_range[j][0] for j in self.jid]); self.hi = np.array([self.m.jnt_range[j][1] for j in self.jid])
        self.bname = {b: mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, b) for b in range(self.m.nbody)}
        self.bid = {n: b for b, n in self.bname.items()}
        self.parent = self.m.body_parentid

    def adjacent(self, b1, b2):
        return b1 == b2 or self.parent[b1] == b2 or self.parent[b2] == b1 or self.parent[b1] == self.parent[b2]

    def set_frame(self, root_pos, root_rot_xyzw, dof):
        q = self.d.qpos; q[:3] = root_pos; q[3:7] = root_rot_xyzw[[3, 0, 1, 2]]; q[7:] = dof
        mj.mj_forward(self.m, self.d)

    def self_collision(self):
        for i in range(self.d.ncon):
            c = self.d.contact[i]; b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0: continue                      # world / floor
            if not self.adjacent(b1, b2): return True
        return False

    def link_pose(self, name):
        b = self.bid[name]; return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()


def ang_deg(R1, R2):
    return np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


# ---------------------------------------------------------------- kinematic metrics
def kinematic_metrics(robot, md):
    dof, fps = np.asarray(md["dof_pos"]), float(md["fps"])
    T, J = dof.shape
    near_lo = dof < robot.lo + NMR_EPS; near_hi = dof > robot.hi - NMR_EPS
    rng = robot.hi - robot.lo; mid = (robot.hi + robot.lo) / 2
    out_phuma = np.abs(dof - mid) > PHUMA_FRAC * rng / 2
    per_joint = {"nmr": (near_lo | near_hi).mean(0), "phuma": out_phuma.mean(0)}
    d3 = np.diff(dof, n=3, axis=0) * fps ** 3 if T > 3 else np.zeros((1, J))
    jumps = int((np.abs(np.diff(dof, axis=0)) > JUMP_RAD).sum())
    col = 0
    for t in range(T):
        robot.set_frame(md["root_pos"][t], np.asarray(md["root_rot"][t]), dof[t])
        col += robot.self_collision()
    return {"limit_nmr": per_joint["nmr"].mean(), "limit_phuma": per_joint["phuma"].mean(), "selfcol": col / T,
            "jerk": float(np.sqrt((d3 ** 2).mean())), "jumps": jumps, "frames": T}, per_joint


# ---------------------------------------------------------------- fidelity (IK residuals vs the human targets)
def fidelity(robot, md, src_path, robot_name):
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from retarget_dataset import load_frames
    key = md.get("src_human") or None
    if key is None: raise SystemExit("pkl has no src_human field; re-run retarget_dataset.py or pass --no_fidelity")
    base = key.replace("_male", "").replace("_female", "")
    frames, fps, height_auto, key2, _ = load_frames(src_path, base, robot_name, int(round(float(md["fps"]))))
    T = min(len(frames), len(md["dof_pos"]))
    g = GMR(src_human=key2, tgt_robot=robot_name, actual_human_height=md.get("actual_human_height", height_auto))
    tasks = list(g.tasks1) + [t for t in g.tasks2 if t.frame_name not in {u.frame_name for u in g.tasks1}]
    ori, pos = defaultdict(list), defaultdict(list)
    shift = None
    for t in range(T):
        g.update_targets({b: [np.asarray(v[0]), np.asarray(v[1])] for b, v in frames[t].items()})
        robot.set_frame(md["root_pos"][t], np.asarray(md["root_rot"][t]), md["dof_pos"][t])
        tg = {}
        for task in tasks:
            T_wt = task.transform_target_to_world
            tg[task.frame_name] = (np.asarray(T_wt.translation()), np.asarray(T_wt.rotation().as_matrix()))
        if shift is None:                                              # constant per-clip shift from HEIGHT_ADJUST / ROOT_ORIGIN
            p_fk, _ = robot.link_pose(g.robot_root_name if hasattr(g, "robot_root_name") else "pelvis")
            shift = p_fk - tg[g.robot_root_name if hasattr(g, "robot_root_name") else "pelvis"][0]
        for name, (p_t, R_t) in tg.items():
            p_fk, R_fk = robot.link_pose(name)
            ori[name].append(ang_deg(R_fk, R_t)); pos[name].append(100 * np.linalg.norm(p_fk - shift - p_t))
    per_body = {n: (float(np.mean(ori[n])), float(np.mean(pos[n]))) for n in ori}
    return {"ori_res": float(np.mean([v[0] for v in per_body.values()])),
            "pos_res": float(np.mean([v[1] for v in per_body.values()]))}, per_body


# ---------------------------------------------------------------- summary
def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    if len(x) < 2: return (np.nan, np.nan)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]; return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True); ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--folders", nargs="+", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--src_folders", nargs="*", default=None, help="human npz folders, one per arm (enables fidelity)")
    ap.add_argument("--no_fidelity", action="store_true"); ap.add_argument("--limit", type=int, default=0, help="first N paired clips (debug)")
    ap.add_argument("--out", default="results/metrics")
    a = ap.parse_args()
    assert len(a.folders) == len(a.names)
    do_fid = bool(a.src_folders) and not a.no_fidelity
    if do_fid: assert len(a.src_folders) == len(a.folders)
    os.makedirs(a.out, exist_ok=True)
    robot = Robot(a.xml)

    idx = [index_pkls(f) for f in a.folders]
    keys = sorted(set.intersection(*(set(i) for i in idx)))
    print(f"paired clips: {len(keys)}  (per arm: {[len(i) for i in idx]})")
    if a.limit: keys = keys[: a.limit]

    rows, pj_rows, pb_rows = [], [], []
    for n, key in enumerate(keys):
        for arm, name in enumerate(a.names):
            md = pickle.load(open(idx[arm][key], "rb"))
            km, per_joint = kinematic_metrics(robot, md)
            row = {"clip": key, "arm": name, **km}
            if do_fid:
                src = find_src(a.src_folders[arm], key)
                if src is None: print(f"  [warn] no source for {key} in {a.src_folders[arm]}")
                else:
                    try:
                        fm, per_body = fidelity(robot, md, src, a.robot); row.update(fm)
                        pb_rows += [{"clip": key, "arm": name, "body": b, "ori_res_deg": o, "pos_res_cm": p} for b, (o, p) in per_body.items()]
                    except Exception as e:
                        print(f"  [warn] fidelity failed for {key}/{name}: {e}")
            rows.append(row)
            pj_rows += [{"clip": key, "arm": name, "joint": j, "nmr": float(per_joint["nmr"][i]), "phuma": float(per_joint["phuma"][i])}
                        for i, j in enumerate(robot.jnames)]
        if (n + 1) % 20 == 0: print(f"  {n + 1}/{len(keys)}")

    def dump(path, rs):
        if not rs: return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
    dump(f"{a.out}/per_clip.csv", rows); dump(f"{a.out}/per_joint_limits.csv", pj_rows); dump(f"{a.out}/per_body_residuals.csv", pb_rows)

    metrics = [m for m in ("limit_nmr", "limit_phuma", "selfcol", "jerk", "jumps", "ori_res", "pos_res") if any(m in r for r in rows)]
    by = {(r["clip"], r["arm"]): r for r in rows}
    lines = [f"# Retargeting metrics ({len(keys)} paired clips)\n", "## Means\n", "| arm | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for name in a.names:
        vals = [np.nanmean([by[(k, name)].get(m, np.nan) for k in keys]) for m in metrics]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4g}" for v in vals) + " |")
    lines += ["", f"## Paired difference vs {a.names[0]} (mean [95% bootstrap CI]; negative = lower than {a.names[0]})\n",
              "| arm | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for name in a.names[1:]:
        cells = []
        for m in metrics:
            dlt = [by[(k, name)].get(m, np.nan) - by[(k, a.names[0])].get(m, np.nan) for k in keys]
            dlt = [x for x in dlt if np.isfinite(x)]; lo, hi = boot_ci(dlt)
            cells.append(f"{np.mean(dlt):+.3g} [{lo:+.3g}, {hi:+.3g}]" if dlt else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    # per-joint limit table (NMR), joints as rows
    lines += ["", "## At-limit fraction per joint (NMR, within 0.05 rad)\n", "| joint | " + " | ".join(a.names) + " |", "|---|" + "---|" * len(a.names)]
    pj = defaultdict(list)
    for r in pj_rows: pj[(r["joint"], r["arm"])].append(r["nmr"])
    for j in robot.jnames:
        lines.append(f"| {j} | " + " | ".join(f"{100 * np.mean(pj[(j, n)]):.1f}%" for n in a.names) + " |")
    open(f"{a.out}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(a.names)])); print(f"wrote {a.out}/")


if __name__ == "__main__":
    main()
