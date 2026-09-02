import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROBOT_PX = (421.0, 540.0)
PPM = 150.0

VARIANTS = ["no_vlm_tight", "no_vlm_06"]
SCENES = ["R000", "R001", "R002", "R003", "R004", "R006", "R008",
          "R009", "R010", "R011", "R012", "R013", "R015"]

for variant in VARIANTS:
    for s in SCENES:
        scene = f"oracle_scene_{s}"
        path_file = f"data/{scene}/{variant}/last_judged_path.json"
        log_file = f"data/{scene}/{variant}/log.txt"
        image_path = f"data/{scene}/oracle.png"
        try:
            with open(path_file) as f:
                coords = json.load(f)
            with open(log_file) as f:
                passed_line = f.readline().strip()
        except FileNotFoundError:
            print(f"skip (no data): {scene}/{variant}")
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
        ax.set_title(f"{scene} / {variant} -- {passed_line}")
        out_path = f"data/{scene}/{variant}/verdict.png"
        plt.savefig(out_path)
        plt.close()
        print(f"OK: {out_path}")
