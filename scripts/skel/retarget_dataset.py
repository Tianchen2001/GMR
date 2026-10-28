#!/usr/bin/env python
"""retarget_dataset.py — one driver for every human source: SMPL (T-pose or bone-direction offsets) and SKEL
(T-pose or axis-anchored offsets). Replaces smpl_to_robot_dataset.py, skel_to_robot_dataset.py and the src_human
edit of the upstream smplx_to_robot_dataset.py. Everything after the loader (warm-up, IK loop, FK, HEIGHT_ADJUST,
ROOT_ORIGIN_OFFSET, pkl fields, filters) is shared, so a difference between arms is the source, not the driver.

    python scripts/skel/retarget_dataset.py --src_human smpl         --src_folder <amass smpl npz dir> --tgt_folder <out>
    python scripts/skel/retarget_dataset.py --src_human smpl_bone    --src_folder <same>              --tgt_folder <out>
    python scripts/skel/retarget_dataset.py --src_human skel         --src_folder <skel npz dir>      --tgt_folder <out> --index index.csv
    python scripts/skel/retarget_dataset.py --src_human skel_general --src_folder <skel npz dir>      --tgt_folder <out> --index index.csv

src_human families (params.IK_CONFIG_DICT keys):
    smpl*   native SMPL loader (utils/smpl_native.py), frames via utils/smpl.get_smplx_data_offline_fast
    skel*   utils/skel.py; the clip's gender is inserted into the key (skel_general -> skel_general_male), and the
            orientation convention (rest-normalized vs absolute) is read from the config's "rot_offset_mode".
--height: actual_human_height handed to GMR, which scales the whole scale table by height / human_height_assumption.
    auto  = the loader's value (1.66 + 0.1 * beta0; SKEL keeps SMPL's beta0, so all arms get the same number)
    none  = disable the height ratio;  <float> = fixed value for every clip.
--no_init: skip warm_start and retarget from qpos0 like upstream GMR.
--index: csv with columns flat,key mapping flattened SKEL basenames back to the AMASS key (filter only).
"""
import argparse, csv, gc, os, pathlib, pickle, time, multiprocessing as mp
import numpy as np
import torch
from natsort import natsorted
from rich import print
import psutil

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.warmup import warm_start

REPO = pathlib.Path(__file__).resolve().parents[2]
BODY_MODELS = REPO / "assets" / "body_models"
HARD_MOTIONS = [REPO / "assets" / "hard_motions" / "0.txt", REPO / "assets" / "hard_motions" / "1.txt"]
EXCLUDE = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
CARRY = ("grf", "cop", "contact_torque", "tau_gt_det", "q_gt", "qd_gt", "dof_names", "mass_kg", "height_m", "fps",
         "grf_valid_mask", "n_contacts", "activity_class", "study", "subject_pure")          # ForceBody extras


# ---------------------------------------------------------------- sources
def family(src_human):
    if src_human.startswith("smpl"): return "smpl"
    if src_human.startswith("skel"): return "skel"
    raise SystemExit(f"--src_human {src_human!r}: expected smpl* or skel*")


def resolve_key(src_human, gender):
    """skel_general -> skel_general_<gender> when such a key exists; everything else unchanged."""
    for cand in (f"{src_human}_{gender}", src_human.replace("skel_general", f"skel_general_{gender}", 1)):
        if cand in IK_CONFIG_DICT: return cand
    if src_human in IK_CONFIG_DICT: return src_human
    raise SystemExit(f"--src_human {src_human!r} not in params.IK_CONFIG_DICT ({sorted(IK_CONFIG_DICT)})")


def load_frames(path, src_human, robot, tgt_fps):
    """-> (frames, aligned_fps, height, resolved_key, extra). frames: list of {body: (pos, quat_wxyz)}."""
    if family(src_human) == "smpl":
        from general_motion_retargeting.utils.smpl_native import load_smpl_file
        from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast
        data, model, out, height = load_smpl_file(path, BODY_MODELS, model_type="smpl", ext="npz")
        frames, fps = get_smplx_data_offline_fast(data, model, out, tgt_fps=tgt_fps)
        return frames, fps, height, resolve_key(src_human, None), {}
    from general_motion_retargeting.utils.skel import load_skel_file, get_skel_data_offline_fast, rest_normalize_for_config
    src = np.load(path, allow_pickle=True)
    gender = str(src["gender"]).strip().lower()
    key = resolve_key(src_human, gender)
    rest_norm = rest_normalize_for_config(IK_CONFIG_DICT[key][robot])
    data, model, out, height = load_skel_file(path, BODY_MODELS / "skel", rest_normalize=rest_norm)
    frames, fps = get_skel_data_offline_fast(data, model, out, tgt_fps=tgt_fps)
    extra = {k: src[k] for k in CARRY if k in src.files}
    if extra: extra["up_axis"] = str(src["up_axis"]) if "up_axis" in src.files else "z"
    return frames, fps, height, key, extra


# ---------------------------------------------------------------- one clip
def wait_for_memory(path, threshold_gb=30):
    for n in range(11):
        if psutil.virtual_memory().available / 2**30 >= threshold_gb: return True
        print(f"[PAUSE] {path}: low memory, pause {n}"); time.sleep(120)
    print(f"[ERROR] {path}: memory still low after 10 pauses"); return False


def process_file(src_path, tgt_path, robot, src_human, tgt_fps, height_mode, no_init, tgt_folder, total):
    if not wait_for_memory(src_path): return
    try:
        frames, fps, height_auto, key, extra = load_frames(src_path, src_human, robot, tgt_fps)
    except Exception as e:
        print(f"Error loading {src_path}: {e}"); return
    height = {"auto": height_auto, "none": None}.get(height_mode, None)
    if height_mode not in ("auto", "none"): height = float(height_mode)

    retargeter = GMR(src_human=key, tgt_robot=robot, actual_human_height=height)
    if not no_init:
        warm_start(retargeter, frames[0])
    qpos = np.array([retargeter.retarget(f).copy() for f in frames])

    device = "cuda:0"
    km = KinematicsModel(retargeter.xml_file, device=device)
    root_pos, root_rot, dof_pos = qpos[:, :3], qpos[:, 3:7], qpos[:, 7:]
    root_rot = root_rot[:, [1, 2, 3, 0]]                                      # wxyz -> xyzw (pkl convention)
    T = len(qpos)
    zero_p = torch.zeros((T, 3), device=device); zero_q = torch.zeros((T, 4), device=device); zero_q[:, -1] = 1.0
    dof_t = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    local_body_pos, _ = km.forward_kinematics(zero_p, zero_q, dof_t)
    body_pos, _ = km.forward_kinematics(torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                                        torch.from_numpy(root_rot).to(device=device, dtype=torch.float), dof_t)
    root_pos[:, 2] -= torch.min(body_pos[..., 2]).item()                     # HEIGHT_ADJUST
    root_pos[:, :2] -= root_pos[0, :2]                                        # ROOT_ORIGIN_OFFSET

    motion = {"fps": fps, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof_pos,
              "local_body_pos": local_body_pos.detach().cpu().numpy(), "link_body_list": km.body_names,
              "src_human": key, "actual_human_height": height, "init": not no_init}
    if extra: motion["source_extra"] = extra
    os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
    with open(tgt_path, "wb") as f: pickle.dump(motion, f)
    done = sum(len([x for x in fs if x.endswith(".pkl")]) for _, _, fs in os.walk(tgt_folder))
    print(f"Processed {done}/{total}: {tgt_path}")
    torch.cuda.empty_cache(); gc.collect()


# ---------------------------------------------------------------- main
def hard_motion_names():
    names = []
    for p in HARD_MOTIONS:
        if not p.exists(): continue
        for line in open(p):
            if "Motion:" in line:
                names.append(line.split(":")[1].strip().split(",")[0].strip().split(".")[0])
    return set(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--src_human", required=True, help="smpl | smpl_bone | skel | skel_general (or any IK_CONFIG_DICT key)")
    ap.add_argument("--src_folder", required=True); ap.add_argument("--tgt_folder", required=True)
    ap.add_argument("--fps", type=int, default=30, help="target fps; >= source fps keeps the source rate")
    ap.add_argument("--height", default="auto", help="auto | none | <meters>")
    ap.add_argument("--no_init", action="store_true"); ap.add_argument("--index", default=None)
    ap.add_argument("--override", action="store_true"); ap.add_argument("--num_cpus", type=int, default=4)
    a = ap.parse_args()

    fam = family(a.src_human)
    for g in (("male", "female") if fam == "skel" else (None,)):
        k = resolve_key(a.src_human, g); print(f"src_human {a.src_human} ({g}) -> {k}: {IK_CONFIG_DICT[k][a.robot]}")
    key_of = {r["flat"]: r["key"] for r in csv.DictReader(open(a.index))} if a.index else {}
    hard = hard_motion_names()

    jobs, dropped = [], []
    for d, _, fs in os.walk(a.src_folder):
        for fn in natsorted(fs):
            if fn.endswith("_stagei.npz") or not fn.endswith((".npz", ".pkl")): continue
            src = os.path.join(d, fn); tgt = src.replace(a.src_folder, a.tgt_folder).replace(".npz", ".pkl")
            flat = fn.split(".")[0]
            flat = flat[:-5] if flat.endswith("_skel") else flat
            name = os.path.basename(key_of.get(flat, flat)).split(".")[0]
            hit = "hard_motions" if name in hard else next((c for c in EXCLUDE if c in name), None)
            if hit: dropped.append((name, hit)); continue
            if os.path.exists(tgt) and not a.override: continue
            jobs.append((src, tgt, a.robot, a.src_human, a.fps, a.height, a.no_init, a.tgt_folder))
    for n, why in dropped: print(f"    dropped [{why}] {n}")
    if fam == "skel" and not key_of: print("    [warning] no --index: filter ran on flattened names")
    print(f"{len(jobs)} clips to process ({len(dropped)} dropped); {a.num_cpus} CPUs")
    with mp.Pool(a.num_cpus) as pool:
        pool.starmap(process_file, [j + (len(jobs),) for j in jobs])
    print("Done. Saved to", a.tgt_folder)


if __name__ == "__main__":
    main()
