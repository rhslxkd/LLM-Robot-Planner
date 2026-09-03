import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
VARIANT = "waypoint_gen_v1"
MANIFEST = "data/random_batch_manifest_v2.csv"

with open(MANIFEST) as f:
    scenes = [row["scene"] for row in csv.DictReader(f)]

ok, skipped = 0, 0
for scene in scenes:
    out_dir = f"data/{scene}/{VARIANT}"
    path_file = f"{out_dir}/last_judged_path.json"
    log_file = f"{out_dir}/log.txt"
    image_path = f"data/{scene}/oracle.png"
    try:
        with open(path_file) as f:
            coords = json.load(f)
        with open(log_file) as f:
            passed_line = f.readline().strip()
    except FileNotFoundError:
        print(f"skip (no data): {scene}")
        skipped += 1
        continue

    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    rx, ry = ROBOT_PX
    xs = [rx + c["x"] * PPM for c in coords]
    ys = [ry - c["y"] * PPM for c in coords]
    ax.plot(xs, ys, "r-", linewidth=2)
    ax.scatter(xs, ys, c="yellow", s=50, zorder=5)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(i), (x, y), color="white", fontsize=11, fontweight="bold")
    ax.plot(rx, ry, "bo", markersize=10)
    ax.set_title(f"{scene} / {VARIANT} -- {passed_line}")
    out_path = f"{out_dir}/verdict.png"
    plt.savefig(out_path)
    plt.close()
    ok += 1

print(f"\n완료: {ok}개 verdict.png 생성, {skipped}개 스킵")
