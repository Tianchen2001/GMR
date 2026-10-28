"""Native SMPL loader (same interface as load_smplx_file in smpl.py: returns (data, body_model, output, height)).
Kept in its own module so upstream utils/smpl.py stays untouched.
"""
import numpy as np
import smplx
import torch

def load_smpl_file(smpl_file, smpl_body_model_path, model_type="smpl", ext="pkl", chunk=512):
    import pickle
    if str(smpl_file).endswith(".pkl"):
        with open(smpl_file, "rb") as f:
            d = pickle.load(f)
        d = {k: (np.asarray(v) if not isinstance(v, str) else v) for k, v in d.items()}
    else:
        z = np.load(smpl_file, allow_pickle=True); d = {k: z[k] for k in z.files}
    if "poses" not in d:
        raise ValueError(f"{smpl_file}: 需要 AMASS 的 poses 格式 (SMPL/SMPL-H), 有 {list(d)[:8]}")
    P = np.asarray(d["poses"], np.float32); T = len(P)
    gender = str(d.get("gender", "neutral")).strip().lower()
    betas = np.ravel(np.asarray(d["betas"], np.float32))
    nb = 10 if model_type == "smpl" else 16
    betas = np.pad(betas, (0, max(0, nb - len(betas))))[:nb]
    if model_type == "smpl" and ext == "npz":
        # smplx 的 SMPL 类把文件名写死成 SMPL_{GENDER}.pkl (忽略 ext), 而原版 pkl 要 chumpy. 绕过去: 自己读 npz, 用 data_struct 传进去.
        import os
        from smplx.utils import Struct
        npz = os.path.join(str(smpl_body_model_path), "smpl", f"SMPL_{gender.upper()}.npz")
        z = np.load(npz, allow_pickle=True)
        data_struct = Struct(**{k: z[k] for k in z.files})
        body_model = smplx.create(str(smpl_body_model_path), model_type, gender=gender, num_betas=nb, batch_size=chunk, data_struct=data_struct)
    else:
        body_model = smplx.create(str(smpl_body_model_path), model_type, gender=gender, num_betas=nb, use_pca=False, batch_size=chunk, ext=ext)
    joints, full_pose = [], []
    for a in range(0, T, chunk):
        sl = slice(a, min(a + chunk, T)); n = sl.stop - sl.start; pad = chunk - n
        def t(x, w):
            x = torch.tensor(np.asarray(x)[sl], dtype=torch.float32)
            return torch.cat([x, torch.zeros(pad, w)], 0) if pad else x
        kw = dict(betas=torch.tensor(betas).expand(chunk, -1), global_orient=t(P[:, :3], 3), transl=t(d["trans"], 3), return_full_pose=True)
        if model_type == "smpl":
            body = np.zeros((T, 69), np.float32); w = min(69, P.shape[1] - 3); body[:, :w] = P[:, 3:3 + w]
            kw["body_pose"] = t(body, 69)
        else:
            kw.update(body_pose=t(P[:, 3:66], 63), left_hand_pose=torch.zeros(chunk, 45), right_hand_pose=torch.zeros(chunk, 45))
        with torch.no_grad():
            o = body_model(**kw)
        joints.append(o.joints[:n]); full_pose.append(o.full_pose[:n])

    class _Out:
        pass
    smpl_output = _Out()
    smpl_output.joints = torch.cat(joints); smpl_output.full_pose = torch.cat(full_pose)
    smpl_output.global_orient = torch.tensor(P[:, :3]); smpl_output.betas = torch.tensor(betas)[None]
    smpl_data = dict(d); smpl_data["pose_body"] = P[:, 3:66]
    fps = d.get("mocap_frame_rate", d.get("mocap_framerate", None))
    if fps is None:
        raise ValueError(f"{smpl_file}: 没有 mocap_framerate")
    smpl_data["mocap_frame_rate"] = np.array(float(fps))
    human_height = 1.66 + 0.1 * float(betas[0])       # 和 load_smplx_file / SKEL loader 同一个启发式, 两臂一致
    return smpl_data, body_model, smpl_output, human_height
