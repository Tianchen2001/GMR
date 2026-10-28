"""Root initialization + warm-up for GeneralMotionRetargeting. Upstream starts every clip from qpos0 (T-pose facing +x)
and runs only a few damped IK steps per frame; when the first human frame faces elsewhere the orientation error is
near 180 deg, its log map is ill-conditioned, and the solver spreads the rotation over root / waist / hips into a
twisted local minimum that per-frame warm starts never leave. Shared by every loader (SMPL, SKEL, ...)."""
import numpy as np


def _copy_frame(frame):
    return {b: [np.asarray(v[0]), np.asarray(v[1])] for b, v in frame.items()}   # update_targets mutates in place


def warm_start(retargeter, frame0, iters=50):
    """Place the floating base at the first-frame root target (after scale/offset), zero the joints, then solve the
    first frame `iters` times so the chain settles before the real sequence starts."""
    retargeter.update_targets(_copy_frame(frame0))
    pos, quat = retargeter.scaled_human_data[retargeter.human_root_name]          # quat wxyz = MuJoCo order
    q = retargeter.configuration.data.qpos.copy()
    q[:] = 0.0
    q[:3] = pos
    q[3:7] = quat / np.linalg.norm(quat)
    retargeter.configuration.update(q)
    for _ in range(iters):
        retargeter.retarget(_copy_frame(frame0))
