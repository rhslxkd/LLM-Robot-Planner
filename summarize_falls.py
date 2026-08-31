import glob
import csv
import numpy as np

SCENES = ["R002", "R003", "R004", "R006", "R008", "R009", "R010", "R011", "R015"]


def analyze(states_path):
    data = np.load(states_path, allow_pickle=True)
    t = data[:, 0]
    qpos = data[:, 1:20]
    z = qpos[:, 2]
    quat = qpos[:, 3:7]  # w,x,y,z (mujoco convention)
    x, y = quat[:, 1], quat[:, 2]
    R22 = 1 - 2 * (x ** 2 + y ** 2)
    tilt_done = R22 < 0
    height_done = z < 0.18
    done = tilt_done | height_done
    if done.any():
        idx = int(np.argmax(done))
        reasons = []
        if tilt_done[idx]:
            reasons.append("tilt")
        if height_done[idx]:
            reasons.append("height")
        return True, int(t[idx]), float(z[idx]), float(R22[idx]), "+".join(reasons)
    else:
        return False, None, float(z[-1]), float(R22[-1]), ""


rows = []
for s in SCENES:
    scene = f"oracle_scene_{s}"
    outdir = f"data/{scene}/batch"
    matches = sorted(glob.glob(f"{outdir}/*_states.npy"))
    if not matches:
        rows.append([scene, "NO_STATES", "", "", "", ""])
        continue
    states_path = matches[-1]
    fell, step, z_val, r22_val, reason = analyze(states_path)
    rows.append([
        scene,
        "FALL" if fell else "OK",
        step if step is not None else "",
        round(z_val, 3),
        round(r22_val, 3),
        reason,
    ])

with open("dial_batch_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "result", "fall_step", "final_or_fall_z", "final_or_fall_R22", "reason"])
    w.writerows(rows)

print("scene,result,fall_step,z,R22,reason")
for r in rows:
    print(",".join(str(x) for x in r))

n_ok = sum(1 for r in rows if r[1] == "OK")
n_fall = sum(1 for r in rows if r[1] == "FALL")
n_missing = sum(1 for r in rows if r[1] == "NO_STATES")
print(f"\nOK={n_ok}, FALL={n_fall}, NO_STATES={n_missing} (총 {len(rows)}개)")
print("saved: dial_batch_results.csv")
