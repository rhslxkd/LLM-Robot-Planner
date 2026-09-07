"""
val held-out 씬에서 Neural A* 인코더가 예측하는 cost map과 histories를
pretrained vs fine-tuned 비교. diff(fine-tuned - pretrained)까지 같이 그려서
코너(꺾이는 지점)에서 뭐가 바뀌었는지 진단.
--tag로 어떤 학습 결과를 볼지 선택. 결과는 finetune/gradient_maps/{tag}/ 밑에 저장.

사용: python finetune/visualize_gradient.py --tag combined
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import run_neural_astar_step as nas
from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PRETRAINED_CKPT = os.path.join(REPO_ROOT, nas.CKPT_PATH)


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


def upsample(arr32, h_img, w_img):
    zy, zx = h_img / arr32.shape[0], w_img / arr32.shape[1]
    return zoom(arr32, (zy, zx), order=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    args = parser.parse_args()
    tag = args.tag
    cache_path = os.path.join(HERE, "cache", f"dataset_{tag}.npz")
    finetuned_ckpt = os.path.join(HERE, "checkpoints", tag)
    out_dir = os.path.join(HERE, "gradient_maps", tag)
    print(f"[visualize_gradient] tag={tag}  dataset={cache_path}  ckpt={finetuned_ckpt}  -> {out_dir}")

    full_ds = SceneDataset(cache_path)
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
    model_ft.load_state_dict(load_from_ptl_checkpoint(finetuned_ckpt))
    model_ft.eval()

    os.makedirs(out_dir, exist_ok=True)

    for idx in val_indices:
        scene = str(full_ds.scenes[idx])
        oracle_path = os.path.join(REPO_ROOT, f"data/{scene}/oracle.png")
        img = mpimg.imread(oracle_path)
        h_img, w_img = img.shape[0], img.shape[1]

        map_t = torch.from_numpy(full_ds.map_designs[idx]).float()[None]
        start_t = torch.from_numpy(full_ds.start_maps[idx]).float()[None]
        goal_t = torch.from_numpy(full_ds.goal_maps[idx]).float()[None]

        with torch.no_grad():
            cost_pre = model_pre.encode(map_t, start_t, goal_t)[0, 0].numpy()
            cost_ft = model_ft.encode(map_t, start_t, goal_t)[0, 0].numpy()
            out_pre = model_pre(map_t, start_t, goal_t)
            out_ft = model_ft(map_t, start_t, goal_t)

        hist_pre = out_pre.histories[0, 0].numpy()
        hist_ft = out_ft.histories[0, 0].numpy()

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        rows = [
            ("cost map", cost_pre, cost_ft),
            ("histories", hist_pre, hist_ft),
        ]
        for r, (label, pre, ft) in enumerate(rows):
            diff = ft - pre
            for c, (title, arr, cmap, vlim) in enumerate([
                (f"pretrained {label}", pre, "jet", None),
                (f"fine-tuned [{tag}] {label}", ft, "jet", None),
                (f"diff (ft - pre) {label}", diff, "bwr", np.abs(diff).max() + 1e-6),
            ]):
                ax = axes[r, c]
                ax.imshow(img)
                heat = upsample(arr, h_img, w_img)
                kwargs = dict(cmap=cmap, alpha=0.55, extent=[0, w_img, h_img, 0])
                if vlim is not None:
                    kwargs["vmin"], kwargs["vmax"] = -vlim, vlim
                im = ax.imshow(heat, **kwargs)
                ax.set_title(title, fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"{scene}  (tag={tag})", fontsize=14)
        out_path = os.path.join(out_dir, f"{scene}_gradient.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"저장: {out_path}")

    print(f"\n완료: {len(val_indices)}개 -> {out_dir}/")


if __name__ == "__main__":
    main()
