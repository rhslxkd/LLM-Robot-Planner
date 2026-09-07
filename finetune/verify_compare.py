"""
사전학습 체크포인트 vs fine-tuned 체크포인트를 동일한 held-out val 씬에 돌려서
loss / p_opt(최적경로 길이 일치율) / p_exp(탐색 효율) 비교.
train.py와 완전히 동일한 random_split(seed=0) 사용 -> 정확히 같은 val 샘플.
--tag로 어떤 학습 결과(dataset_{tag}.npz / checkpoints/{tag}/)를 검증할지 선택.

사용: python finetune/verify_compare.py --tag combined
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import run_neural_astar_step as nas

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), nas.CKPT_PATH
)


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
        return (
            torch.from_numpy(self.map_designs[idx]).float(),
            torch.from_numpy(self.start_maps[idx]).float(),
            torch.from_numpy(self.goal_maps[idx]).float(),
            torch.from_numpy(self.opt_trajs[idx]).float(),
        )


def evaluate(model, loader):
    vanilla = VanillaAstar()
    model.eval()
    losses, p_opts, p_exps = [], [], []
    with torch.no_grad():
        for map_designs, start_maps, goal_maps, opt_trajs in loader:
            outputs = model(map_designs, start_maps, goal_maps)
            loss = torch.nn.L1Loss()(outputs.histories, opt_trajs).item()

            va_outputs = vanilla(map_designs, start_maps, goal_maps)
            pathlen_astar = va_outputs.paths.sum((1, 2, 3)).numpy()
            pathlen_model = outputs.paths.sum((1, 2, 3)).numpy()
            p_opt = (pathlen_astar == pathlen_model).mean()

            exp_astar = va_outputs.histories.sum((1, 2, 3)).numpy()
            exp_na = outputs.histories.sum((1, 2, 3)).numpy()
            p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

            losses.append(loss)
            p_opts.append(p_opt)
            p_exps.append(p_exp)
    return np.mean(losses), np.mean(p_opts), np.mean(p_exps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2",
                         help="검증할 학습 결과의 이름표 (build_dataset.py/train.py --tag와 일치)")
    args = parser.parse_args()
    tag = args.tag
    cache_path = os.path.join(HERE, "cache", f"dataset_{tag}.npz")
    finetuned_ckpt = os.path.join(HERE, "checkpoints", tag)
    print(f"[verify_compare] tag={tag}  dataset={cache_path}  ckpt={finetuned_ckpt}")

    full_ds = SceneDataset(cache_path)
    n_val = max(1, int(len(full_ds) * 0.1))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    print(f"val set: {n_val}개 씬")

    for label, ckpt_path in [("사전학습 (fine-tune 전)", PRETRAINED_CKPT), (f"fine-tuned [{tag}]", finetuned_ckpt)]:
        model = NeuralAstar(g_ratio=0.5, encoder_arch="CNN")
        model.load_state_dict(load_from_ptl_checkpoint(ckpt_path))
        loss, p_opt, p_exp = evaluate(model, val_loader)
        print(f"[{label}] loss={loss:.5f}  p_opt={p_opt:.3f}  p_exp={p_exp:.3f}")


if __name__ == "__main__":
    main()
