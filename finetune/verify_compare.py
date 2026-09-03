"""
val 8개 held-out 씬에서 GT(안전보정 경로) vs 사전학습 모델 vs fine-tuned 모델의
raw Neural A* 출력 경로를 oracle.png 위에 겹쳐 그려서 시각 비교.
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_neural_astar_step as nas

from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CACHE_PATH = os.path.join(HERE, "cache/dataset.npz")
PRETRAINED_CKPT = os.path.join(REPO_ROOT, nas.CKPT_PATH)
FINETUNED_CKPT = os.path.join(HERE, "checkpoints")
OUT_DIR = os.path.join(HERE, "verify")

GRID = nas.GRID
grid_to_full = nas.grid_to_full
extract_ordered_path = nas.extract_ordered_path


class SceneDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.map_designs = d["map_designs"]
        self.start_maps = d["start_maps"]
        self.goal_maps = d["goal_maps"]
        self.opt_trajs = d["opt_trajs"]
        self.scenes = d["scenes"]

    def __len__(self):
        return len(self.map_designs)

    def __getitem__(self, idx):
        return idx


def grid_path_to_px(mask_2d, start_grid, goal_grid, w, h):
    wp_grid = extract_ordered_path(mask_2d, start_grid, goal_grid)
    if not wp_grid:
        return [], []
    xs, ys = [], []
    for gx, gy in wp_grid:
        px, py = grid_to_full(gx, gy, w, h)
        xs.append(px); ys.append(py)
    return xs, ys


def main():
    full_ds = SceneDataset(CACHE_PATH)
    n_val = max(1, int(len(full_ds) * 0.1))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    val_indices = list(val_ds)
    print(f"val 씬: {[full_ds.scenes[i] for i in val_indices]}")

    model_pre = NeuralAstar(g_ratio=0.5, encoder_arch="CNN")
    model_pre.load_state_dict(load_from_ptl_checkpoint(PRETRAINED_CKPT))
    model_pre.eval()

    model_ft = NeuralAstar(g_ratio=0.5, encoder_arch="CNN")
    model_ft.load_state_dict(load_from_ptl_checkpoint(FINETUNED_CKPT))
    model_ft.eval()

    os.makedirs(OUT_DIR, exist_ok=True)

    for idx in val_indices:
        scene = str(full_ds.scenes[idx])
        oracle_path = os.path.join(REPO_ROOT, f"data/{scene}/oracle.png")
        img = mpimg.imread(oracle_path)
        h_img, w_img = img.shape[0], img.shape[1]

        map_t = torch.from_numpy(full_ds.map_designs[idx]).float()[None]
        start_t = torch.from_numpy(full_ds.start_maps[idx]).float()[None]
        goal_t = torch.from_numpy(full_ds.goal_maps[idx]).float()[None]
        opt_traj = full_ds.opt_trajs[idx][0]

        start_grid = tuple(int(v) for v in np.argwhere(full_ds.start_maps[idx][0])[0][::-1])
        goal_grid = tuple(int(v) for v in np.argwhere(full_ds.goal_maps[idx][0])[0][::-1])

        with torch.no_grad():
            out_pre = model_pre(map_t, start_t, goal_t)
            out_ft = model_ft(map_t, start_t, goal_t)

        mask_pre = (out_pre.paths[0, 0].numpy() > 0.5)
        mask_ft = (out_ft.paths[0, 0].numpy() > 0.5)
        mask_gt = (opt_traj > 0.5)

        gt_xs, gt_ys = grid_path_to_px(mask_gt, start_grid, goal_grid, w_img, h_img)
        pre_xs, pre_ys = grid_path_to_px(mask_pre, start_grid, goal_grid, w_img, h_img)
        ft_xs, ft_ys = grid_path_to_px(mask_ft, start_grid, goal_grid, w_img, h_img)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        if gt_xs:
            ax.plot(gt_xs, gt_ys, color="lime", linewidth=3, linestyle="--", label="GT (safety-corrected)", zorder=4)
        if pre_xs:
            ax.plot(pre_xs, pre_ys, color="blue", linewidth=2, label="pretrained (before)", zorder=5)
        if ft_xs:
            ax.plot(ft_xs, ft_ys, color="red", linewidth=2, label="fine-tuned (after)", zorder=6)
        ax.legend(loc="upper right")
        ax.set_title(scene)
        out_path = os.path.join(OUT_DIR, f"{scene}_compare.png")
        plt.savefig(out_path)
        plt.close()
        print(f"저장: {out_path}")

    print(f"\n완료: {len(val_indices)}개 비교 이미지 -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
