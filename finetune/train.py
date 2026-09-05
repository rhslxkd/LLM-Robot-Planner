"""
Neural A* fine-tuning: 사전학습 체크포인트에서 시작해서 우리 100씬 배치 데이터로
낮은 LR로 소규모 fine-tune. build_dataset.py로 만든 cache/dataset.npz를 사용.
"""
import os, sys, argparse, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import run_neural_astar_step as nas

from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint, PlannerModule

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "cache/dataset.npz")
CKPT_DIR = os.path.join(HERE, "checkpoints")


class SceneDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.map_designs = d["map_designs"]
        self.start_maps = d["start_maps"]
        self.goal_maps = d["goal_maps"]
        self.opt_trajs = d["opt_trajs"]

    def __len__(self):
        return len(self.map_designs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.map_designs[idx]).float(),
            torch.from_numpy(self.start_maps[idx]).float(),
            torch.from_numpy(self.goal_maps[idx]).float(),
            torch.from_numpy(self.opt_trajs[idx]).float(),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: {CACHE_PATH} 없음 - 먼저 build_dataset.py 실행 필요")
        sys.exit(1)

    full_ds = SceneDataset(CACHE_PATH)
    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    print(f"train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = NeuralAstar(g_ratio=0.5, encoder_arch="CNN")
    model.load_state_dict(load_from_ptl_checkpoint(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), nas.CKPT_PATH
    )))

    config = OmegaConf.create({"params": {"lr": args.lr}})
    module = PlannerModule(model, config)

    # neural-astar conda env의 PyTorch 빌드가 sm_120(RTX 5060 Ti)을 지원하지 않아
    # torch.cuda.is_available()==True여도 커널 실행이 불가능함 (확인됨: RuntimeError).
    # 모델이 391K params로 작아서 CPU로도 충분히 빠르게 돌아감 -> CPU 강제.
    use_gpu = False
    print(f"device: {'cuda:' + torch.cuda.get_device_name(0) if use_gpu else 'cpu'}")
    if use_gpu:
        free_mb = torch.cuda.mem_get_info()[0] / 1024**2
        print(f"GPU free memory before training: {free_mb:.0f} MiB")

    os.makedirs(CKPT_DIR, exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=CKPT_DIR,
        log_every_n_steps=1,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
    )
    t0 = time.time()
    trainer.fit(module, train_loader, val_loader)
    print(f"학습 소요시간: {time.time()-t0:.1f}s")

    final_ckpt = os.path.join(CKPT_DIR, "finetuned_final.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"저장 완료: {final_ckpt}")


if __name__ == "__main__":
    main()
