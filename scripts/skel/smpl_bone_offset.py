#!/usr/bin/env python
"""smpl_bone_offset.py — fourth arm: SMPL + bone-direction correspondence.

Keeps the GMR T-pose rot_offset (so the twist about each bone is inherited from the T-pose alignment) and
replaces only the SWING: the G1 link direction at T-pose is rotated by the minimal rotation onto the SMPL rest
bone direction (joint-center to joint-center, same rule as the SKEL axis-anchored arm). Per link:

    rot_offset_new = rot_offset_Tpose * S_local,   S_local = R_link_T^T * R_swing(d_r -> d_h) * R_link_T

with R_link_T the G1 T-pose link orientation and d_r its long axis (B_link[:,0] from the *_frames.json written by
anat_rot_offset.py, i.e. the same robot-side definition as the SKEL axis-anchored arm; world axis as fallback) and d_h the SMPL rest bone direction converted to the mj world (x fwd, y left, z up).
Because the swing is minimal, the component about the bone (twist) is untouched.

Run in SKEL-fit-work/skelfit:
    python smpl_bone_offset.py out/skel_to_g1_general_male_frames.json \
        ../../GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json \
        /path/to/SMPL_MALE.npz --out ../../GMR/general_motion_retargeting/ik_configs/smplx_bone_to_g1.json
Then point smpl_to_robot_dataset.py at the new config (same loader, same everything else).
Prints the swing angle per link = the tilt this arm removes relative to the T-pose arm.
"""
import argparse, json, copy
import numpy as np
from scipy.spatial.transform import Rotation as Rot

C_MJ2SMPL = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])     # smpl = C @ mj
C_SMPL2MJ = C_MJ2SMPL.T

# SMPL(-X) joint indices (first 22 shared by SMPL and SMPL-X)
J = dict(pelvis=0, left_hip=1, right_hip=2, spine1=3, left_knee=4, right_knee=5, spine2=6, left_ankle=7, right_ankle=8,
         spine3=9, left_foot=10, right_foot=11, neck=12, left_shoulder=16, right_shoulder=17, left_elbow=18,
         right_elbow=19, left_wrist=20, right_wrist=21)

# G1 link -> (SMPL bone start, SMPL bone end), (G1 link, G1 child body giving the link direction at T-pose)
BONES = {
    "torso_link":              (("spine1", "neck"),              ("torso_link", None)),          # None: use +z (vertical)
    "left_hip_yaw_link":       (("left_hip", "left_knee"),       ("left_hip_yaw_link", "left_knee_link")),
    "right_hip_yaw_link":      (("right_hip", "right_knee"),     ("right_hip_yaw_link", "right_knee_link")),
    "left_hip_roll_link":      (("left_hip", "left_knee"),       ("left_hip_roll_link", "left_knee_link")),   # old naming
    "right_hip_roll_link":     (("right_hip", "right_knee"),     ("right_hip_roll_link", "right_knee_link")),
    "left_knee_link":          (("left_knee", "left_ankle"),     ("left_knee_link", "left_ankle_pitch_link")),
    "right_knee_link":         (("right_knee", "right_ankle"),   ("right_knee_link", "right_ankle_pitch_link")),
    "left_shoulder_yaw_link":  (("left_shoulder", "left_elbow"), ("left_shoulder_yaw_link", "left_elbow_link")),
    "right_shoulder_yaw_link": (("right_shoulder", "right_elbow"), ("right_shoulder_yaw_link", "right_elbow_link")),
    "left_elbow_link":         (("left_elbow", "left_wrist"),    ("left_elbow_link", "left_wrist_yaw_link")),
    "right_elbow_link":        (("right_elbow", "right_wrist"),  ("right_elbow_link", "right_wrist_yaw_link")),
    # kept at the T-pose offset (reported as 'kept T-pose'): hands (no finger joint in the 22-joint set),
    # pelvis (no long bone; SMPL pelvis->spine1 is an artifact of joint placement), feet (no heel landmark; ankle->foot
    # joint line points 30 deg down). Bone direction is only applied where SMPL has a long bone: thigh, shank, upper arm, forearm, torso.
}


WORLD_AXIS = {"pelvis": [0, 0, 1], "torso_link": [0, 0, 1],
              "left_hip_yaw_link": [0, 0, -1], "right_hip_yaw_link": [0, 0, -1], "left_hip_roll_link": [0, 0, -1], "right_hip_roll_link": [0, 0, -1],
              "left_knee_link": [0, 0, -1], "right_knee_link": [0, 0, -1],
              "left_toe_link": [1, 0, 0], "right_toe_link": [1, 0, 0], "left_ankle_roll_link": [1, 0, 0], "right_ankle_roll_link": [1, 0, 0],
              "left_shoulder_yaw_link": [0, 1, 0], "right_shoulder_yaw_link": [0, -1, 0],
              "left_elbow_link": [0, 1, 0], "right_elbow_link": [0, -1, 0]}


def unit(v):
    return v / np.linalg.norm(v)


def swing(a, b):
    """minimal rotation taking unit vector a onto unit vector b (3x3)."""
    a, b = unit(a), unit(b); v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else Rot.from_rotvec(np.pi * unit(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]))).as_matrix()
    k = v / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def smpl_rest_joints(npz, betas):
    m = np.load(npz, allow_pickle=True)
    v = np.array(m["v_template"], dtype=np.float64)
    if betas is not None and "shapedirs" in m:
        sd = np.array(m["shapedirs"], dtype=np.float64)[:, :, :len(betas)]
        v = v + sd @ np.asarray(betas, dtype=np.float64)
    Jr = m["J_regressor"]
    Jr = Jr.toarray() if hasattr(Jr, "toarray") else np.array(Jr, dtype=np.float64)
    return Jr @ v                                                   # (24 or 55, 3), SMPL world (y up, z fwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", help="*_frames.json from anat_rot_offset.py (for G1 T-pose bodies_mj)")
    ap.add_argument("old_config", help="GMR smpl(x)_to_g1.json with T-pose rot_offsets")
    ap.add_argument("smpl_npz", help="SMPL or SMPL-X model npz (v_template, J_regressor[, shapedirs])")
    ap.add_argument("--betas", type=float, nargs="*", default=None, help="subject betas (default: mean shape)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.load(open(a.frames)); bodies = d["_g1"]["bodies_mj"]
    cfg = json.load(open(a.old_config)); tab = cfg["ik_match_table1"]
    Jh = C_SMPL2MJ @ smpl_rest_joints(a.smpl_npz, a.betas).T        # (3, N) in mj world
    Jh = Jh.T
    new = copy.deepcopy(cfg)
    print(f"{'link':26s} {'swing [deg]':>12s}  robot axis from")
    for link, entry in tab.items():
        if link not in BONES or link not in bodies:
            print(f"{link:26s} {'kept T-pose':>12s}"); continue
        (h0, h1), (g0, g1) = BONES[link]
        d_h = Jh[J[h1]] - Jh[J[h0]]
        R_T = np.array(bodies[link]["R"])
        if link in d and "B_link" in d[link]:                                   # robot long axis: same definition as the SKEL arm
            d_r = R_T @ np.array(d[link]["B_link"])[:, 0]; src = "B_link"
        else:                                                                   # fallback: world axis of the T-pose link
            d_r = np.array(WORLD_AXIS[link], float); src = "world"
        S_local = R_T.T @ swing(d_r, d_h) @ R_T
        q_old = np.array(entry[4], dtype=np.float64)                 # wxyz
        R_old = Rot.from_quat(q_old[[1, 2, 3, 0]]).as_matrix()
        R_new = R_old @ S_local
        q_new = Rot.from_matrix(R_new).as_quat()[[3, 0, 1, 2]]       # back to wxyz
        if q_new[0] < 0: q_new = -q_new
        new["ik_match_table1"][link][4] = [float(x) for x in q_new]
        ang = np.degrees(np.arccos(np.clip((np.trace(S_local) - 1) / 2, -1, 1)))
        print(f"{link:26s} {ang:12.1f}  {src}")
    json.dump(new, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()