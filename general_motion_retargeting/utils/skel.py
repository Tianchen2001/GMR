"""SKEL source loader.  ->  general_motion_retargeting/utils/skel.py

Deliberately shaped like `utils/smpl.py`: same three entry points, same
signatures, same return tuples, so the SKEL driver reads line-for-line like
`smplx_to_robot_dataset.py` and the two can be diffed.

    load_skel_file(skel_file, skel_body_model_path)
        -> skel_data, body_model, skel_output, actual_human_height
    get_skel_data(skel_data, body_model, skel_output, curr_frame)
        -> {body_name: (position, quaternion WXYZ)}
    get_skel_data_offline_fast(skel_data, body_model, skel_output, tgt_fps=30)
        -> [frame dicts], aligned_fps

SKEL is its own source, not a corrected SMPL: the bodies below are SKEL's own
24 and the ik config maps robot links straight onto them.

Two things this does NOT do
---------------------------
* **No frame conversion.** The fitted params live in the AMASS world already
  (`aligner._init_params` sets `trans_skel = trans_smpl` and seeds the global
  orient from SMPL), so the forward pass comes out Z-up and correctly headed.
  The Y-up flip and 180 deg yaw a ForceBody loader needs belong to that
  dataset, not to SKEL.

* **No joint-tree composition.** `utils/smpl.py` has to walk the kinematic
  tree to build global joint orientations; SKEL returns `joints_ori` directly.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from skel.skel_model import SKEL

#: SKEL's own joint order (kin_skel.py::skel_joints_name). Right side first.
SKEL_BODY_NAMES = [
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
    "lumbar_body", "thorax", "head",
    "scapula_r", "humerus_r", "ulna_r", "radius_r", "hand_r",
    "scapula_l", "humerus_l", "ulna_l", "radius_l", "hand_l",
]

#: forward in chunks; SKEL always returns the 6890-vertex skin mesh, so a long
#: clip in one call is a needless couple of hundred MB.
FORWARD_BATCH = 256

#: Emit each joint's orientation RELATIVE TO THIS SUBJECT'S OWN REST POSE,
#: i.e. `R_i(t) @ R_i(rest)^T`, instead of the raw anatomical world frame.
#:
#: Why. SMPL-X's rest pose has every joint at identity, so its joint frames are
#: shape-independent and one `rot_offset` serves every subject. SKEL's frames
#: are anatomical, so their constant part moves with betas: measured across
#: five ACCAD subjects the humerus/ulna rest frames disagree by 8.8-9.9 deg,
#: MORE than those joints vary within a clip. A single global rot_offset would
#: therefore inject a per-subject orientation bias larger than the effect the
#: study is trying to measure.
#:
#: Dividing out the subject's own rest pose removes exactly that constant and
#: nothing else - every time-varying difference between SKEL and SMPL survives
#: untouched. It is the convention that is subject-specific, not the motion,
#: and a comparison must never rest on a convention.
#:
#: With this on, D = R_skel^T R_smplx collapses to ONE constant shared by all
#: 14 bodies and all subjects (SMPL-X's C). That is the acceptance test.
#:
#: Two consistent pairings with the ik config (never mix them):
#:   rest_normalize=True   + config with rot_offset = F^h(0)·Bᵀ            ("subject" mode, one config per subject)
#:   rest_normalize=False  + config with rot_offset = rest_oriᵀ·F^h(0)·Bᵀ  ("general" mode, one config for everyone;
#:                           the subject's own rest orientation rides along inside joints_ori)
#: `rest_normalize_for_config()` reads the config's "rot_offset_mode" field and picks the right one.
REST_NORMALIZE = True          # default when the caller passes nothing (backward compatible)


def rest_normalize_for_config(ik_config_path):
    """True for a subject-mode config, False for a general-mode config, REST_NORMALIZE if the field is absent
    (old hand-written configs are subject-convention)."""
    import json
    mode = json.load(open(ik_config_path)).get("rot_offset_mode")
    return {"subject": True, "general": False}.get(mode, REST_NORMALIZE)


class SkelOutput:
    """Stands in for smplx's output object: whole-sequence joints and frames."""

    def __init__(self, joints, joints_ori):
        self.joints = joints              # (T, 24, 3)
        self.joints_ori = joints_ori      # (T, 24, 3, 3)


def load_skel_file(skel_file, skel_body_model_path=None, device="cpu", rest_normalize=None):
    """Load one gathered `*_skel.npz` and run SKEL forward over the sequence.

    `actual_human_height` uses GMR's own heuristic, `1.66 + 0.1 * betas[0]`, on
    the SAME betas[0] the SMPL-X arm saw: SKEL keeps the first 10 SMPL shape
    coefficients verbatim, so both arms get an identical height and therefore
    an identical `ratio` into the scale table. Nothing has to be measured and
    the two arms cannot drift apart on this.
    """
    skel_data = np.load(skel_file, allow_pickle=True)

    gender = str(skel_data["gender"]).strip().lower()
    if gender not in ("male", "female"):
        raise ValueError(f"SKEL has male/female only, got {gender!r}")
    body_model = (SKEL(gender, model_path=skel_body_model_path)
                  if skel_body_model_path else SKEL(gender)).to(device)

    poses = np.asarray(skel_data["poses"], np.float32)          # (T, 46)
    trans = np.asarray(skel_data["trans"], np.float32)          # (T, 3)
    betas = np.asarray(skel_data["betas"], np.float32).reshape(-1, 10)

    num_frames = poses.shape[0]
    joints = np.zeros((num_frames, 24, 3))
    joints_ori = np.zeros((num_frames, 24, 3, 3))
    for a in range(0, num_frames, FORWARD_BATCH):
        b = min(a + FORWARD_BATCH, num_frames)
        bet = betas[a:b] if len(betas) == num_frames else np.repeat(betas[:1], b - a, 0)
        with torch.no_grad():
            out = body_model(poses=torch.from_numpy(poses[a:b]).to(device),
                             betas=torch.from_numpy(bet).to(device),
                             trans=torch.from_numpy(trans[a:b]).to(device),
                             poses_type="skel",
                             skelmesh=False)
        joints[a:b] = out.joints.detach().cpu().numpy()[:, :24]
        joints_ori[a:b] = out.joints_ori.detach().cpu().numpy()[:, :24]

    if rest_normalize is None:
        rest_normalize = REST_NORMALIZE
    if rest_normalize:
        with torch.no_grad():
            rest = body_model(poses=torch.zeros(1, 46, dtype=torch.float32).to(device),
                              betas=torch.from_numpy(betas[:1]).to(device),
                              trans=torch.zeros(1, 3, dtype=torch.float32).to(device),
                              poses_type="skel", skelmesh=False)
        rest_ori = rest.joints_ori.detach().cpu().numpy()[0, :24]     # (24,3,3)
        joints_ori = np.einsum("tnij,nkj->tnik", joints_ori, rest_ori)

    skel_output = SkelOutput(joints, joints_ori)
    actual_human_height = 1.66 + 0.1 * float(betas[0, 0])
    return skel_data, body_model, skel_output, actual_human_height


def get_skel_data(skel_data, body_model, skel_output, curr_frame):
    """
    Must return a dictionary with the following structure:
    {
        "pelvis": (position, orientation),
        "femur_r": (position, orientation),
        ...
    }
    """
    joints = skel_output.joints[curr_frame]
    joints_ori = skel_output.joints_ori[curr_frame]

    result = {}
    for i, joint_name in enumerate(SKEL_BODY_NAMES):
        rot = R.from_matrix(joints_ori[i])
        result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))
    return result


def get_skel_data_offline_fast(skel_data, body_model, skel_output, tgt_fps=30):
    """
    Must return a dictionary with the following structure:
    {
        "pelvis": (position, orientation),
        "femur_r": (position, orientation),
        ...
    }
    """
    src_fps = float(skel_data["mocap_framerate"])
    num_frames = skel_output.joints.shape[0]
    joints = skel_output.joints
    joints_ori = skel_output.joints_ori

    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        frame_skip = int(src_fps / tgt_fps)
        new_num_frames = num_frames // frame_skip
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames - 1, new_num_frames)

        # Interpolate joint orientations using SLERP
        ori_interp = []
        for i in range(joints_ori.shape[1]):
            rots = R.from_matrix(joints_ori[:, i])
            slerp = Slerp(original_time, rots)
            ori_interp.append(slerp(target_time).as_matrix())
        joints_ori = np.stack(ori_interp, axis=1)

        # Interpolate joint positions using linear interpolation
        joints = np.stack(
            [[np.interp(target_time, original_time, joints[:, i, k])
              for k in range(3)] for i in range(joints.shape[1])], axis=0
        ).transpose(2, 0, 1)

        aligned_fps = len(target_time) / num_frames * src_fps
    else:
        aligned_fps = src_fps

    skel_data_frames = []
    for curr_frame in range(len(joints)):
        result = {}
        for i, joint_name in enumerate(SKEL_BODY_NAMES):
            rot = R.from_matrix(joints_ori[curr_frame, i])
            result[joint_name] = (joints[curr_frame, i],
                                  rot.as_quat(scalar_first=True))
        skel_data_frames.append(result)

    return skel_data_frames, aligned_fps