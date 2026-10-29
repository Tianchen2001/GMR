#!/usr/bin/env python
"""render.py — N retargeting arms side by side on the same clip (MuJoCo offscreen), video or single frame.

    python scripts/skel/render.py --clip <clip key> \
        --folders $OUT/smpl_robot $OUT/smpl_robot_bone $OUT/skel_robot $OUT/skel_robot_general \
        --names SMPL-Tpose SMPL-bone SKEL-Tpose SKEL-axis --out results/video/<clip>.mp4
    ... --frame 120 --out results/video/<clip>_f120.png      # one frame as PNG
    ... --azimuth 90 --elevation -15 --distance 2.2           # camera (follows the pelvis of each arm)

Input pkls: {fps, root_pos, root_rot (xyzw), dof_pos}. The robot xml is loaded as is (g1_mocap_29dof.xml already has
floor, lights and sky). Panels are time-aligned by frame index (clips truncated to the shortest).
"""
import argparse, glob, os, pickle
import numpy as np
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont

def clip_key(p):
    k = os.path.splitext(os.path.basename(p))[0]
    for s in ("_skel", "_smplx", "_smpl", "_poses"):
        if k.endswith(s): k = k[: -len(s)]
    return k


def load_model(xml):
    return mj.MjModel.from_xml_path(os.path.abspath(xml))   # the GMR xml carries its own floor, lights and sky; adding a second scene z-fights the floor


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(f): return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="assets/unitree_g1/g1_mocap_29dof.xml")
    ap.add_argument("--clip", required=True); ap.add_argument("--folders", nargs="+", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--frame", type=int, default=None, help="render this frame only (PNG)")
    ap.add_argument("--size", type=int, nargs=2, default=[480, 480]); ap.add_argument("--fps_out", type=float, default=None)
    ap.add_argument("--azimuth", type=float, default=135); ap.add_argument("--elevation", type=float, default=-20); ap.add_argument("--distance", type=float, default=2.6)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args(); assert len(a.folders) == len(a.names)
    a.clip = clip_key(a.clip)

    mds = []
    for f in a.folders:
        hit = [p for p in glob.glob(os.path.join(f, "**", "*.pkl"), recursive=True) if clip_key(p) == clip_key(a.clip)]
        if not hit: raise SystemExit(f"{a.clip}: not found in {f}")
        mds.append(pickle.load(open(hit[0], "rb")))
    T = min(len(m["dof_pos"]) for m in mds); fps = float(mds[0]["fps"])
    frames = [a.frame] if a.frame is not None else list(range(a.start, T if a.end is None else min(a.end, T)))

    m = load_model(a.xml); d = mj.MjData(m); W, H = a.size
    r = mj.Renderer(m, height=H, width=W); cam = mj.MjvCamera(); cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = a.azimuth, a.elevation, a.distance
    pelvis = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "pelvis"); fnt = font(max(14, W // 20))

    def panel(md, t):
        q = d.qpos; q[:3] = md["root_pos"][t]; q[3:7] = np.asarray(md["root_rot"][t])[[3, 0, 1, 2]]; q[7:] = md["dof_pos"][t]
        mj.mj_forward(m, d); cam.lookat[:] = d.xpos[pelvis] + np.array([0, 0, -0.1])
        r.update_scene(d, cam); return r.render().copy()

    writer = None
    if a.frame is None:
        import imageio                                                     # needs imageio-ffmpeg (pip install imageio[ffmpeg])
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        writer = imageio.get_writer(a.out, fps=a.fps_out or fps, codec="libx264", quality=8, macro_block_size=1)
    for t in frames:
        canvas = Image.new("RGB", (W * len(mds), H), "white"); draw = ImageDraw.Draw(canvas)
        for k, (md, nm) in enumerate(zip(mds, a.names)):
            canvas.paste(Image.fromarray(panel(md, t)), (k * W, 0))
            draw.rectangle([k * W, 0, k * W + W, int(H * 0.09)], fill="white"); draw.text((k * W + 10, 6), nm, fill="black", font=fnt)
        draw.text((10, H - 24), f"t = {t / fps:5.2f} s", fill="black", font=font(14))
        if writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True); canvas.save(a.out); print("wrote", a.out); return
        writer.append_data(np.asarray(canvas))
    writer.close(); print(f"wrote {a.out} ({len(frames)} frames @ {a.fps_out or fps:g} fps)")


if __name__ == "__main__":
    main()