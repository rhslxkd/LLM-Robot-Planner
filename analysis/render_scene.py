"""
render_scene.py ─ 실험 씬(점유 지도를 상자로 세운 MuJoCo 씬)을 로봇이 출발점에 선 상태로 렌더 (그림 2의 ③ 칸).

  MUJOCO_GL=egl python analysis/render_scene.py replica_apartment_0_p0 results/figs/scene_apartment_0_p0.png

씬 XML은 indoor/build_scene.py가 만든 dial_mpc/dial_mpc/models/unitree_go2/<scene>.xml을 쓴다.
카메라는 analysis/make_fig1.py의 오른쪽 그림과 같은 설정이고, 초록 공 = 목표점.
"""
import os, sys, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import mujoco
from PIL import Image
from analysis.collision_replay import build_replay_model
scene, out = sys.argv[1], sys.argv[2]
sm = json.load(open(os.path.join(REPO, "data", scene, "scene_meta.json")))
m = build_replay_model(os.path.join(REPO, "dial_mpc", "dial_mpc", "models", "unitree_go2", f"{scene}.xml"))
S = 900; m.vis.global_.offwidth = m.vis.global_.offheight = S
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
o = d.qpos[:2].copy(); g = o + np.asarray(sm["goal"])
lo, hi = np.minimum(o, g), np.maximum(o, g); span = max(hi - lo) + 1.0
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2 - 0.08*span, 0]; cam.azimuth, cam.elevation = 90, -58; cam.distance = 1.45*span + 1.5
r = mujoco.Renderer(m, S, S); r.update_scene(d, cam)
sc = r.scene
def sphere(p, rad, rgba):
    gg = sc.geoms[sc.ngeom]; mujoco.mjv_initGeom(gg, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad,0,0]), np.array(p,float), np.eye(3).ravel(), np.array(rgba,np.float32)); sc.ngeom += 1
sphere([g[0], g[1], 0.1], 0.14, (0.1, 0.85, 0.25, 1))
Image.fromarray(r.render()).save(out); print(out)
