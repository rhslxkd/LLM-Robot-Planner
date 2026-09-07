"""
Neural A* fine-tuning: 사전학습 체크포인트에서 시작해서 우리 배치 데이터로
낮은 LR로 소규모 fine-tune. --tag로 build_dataset.py가 만든
finetune/cache/dataset_{tag}.npz를 선택하고, 결과 체크포인트는
finetune/checkpoints/{tag}/finetuned_final.ckpt로 tag별로 분리 저장한다.

사용: python finetune/train.py --tag combined --epochs 30 --lr 1e-4
"""
import os, sys, json, argparse, time
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
    parser.add_argument("--tag", type=str, default="v2",
                         help="build_dataset.py --tag와 일치해야 함 (finetune/cache/dataset_{tag}.npz를 읽음)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    tag = args.tag
    cache_path = os.path.join(HERE, "cache", f"dataset_{tag}.npz")
    ckpt_dir = os.path.join(HERE, "checkpoints", tag)
    print(f"[train] tag={tag}  dataset={cache_path}  -> {ckpt_dir}/finetuned_final.ckpt")

    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} 없음 - 먼저 build_dataset.py --tag {tag} 실행 필요")
        sys.exit(1)

    full_ds = SceneDataset(cache_path)
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

    os.makedirs(ckpt_dir, exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=ckpt_dir,
        log_every_n_steps=1,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
    )
    t0 = time.time()
    trainer.fit(module, train_loader, val_loader)
    elapsed = time.time() - t0
    print(f"학습 소요시간: {elapsed:.1f}s")

    final_ckpt = os.path.join(ckpt_dir, "finetuned_final.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"저장 완료: {final_ckpt}")

    meta_path = os.path.join(ckpt_dir, "train_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "tag": tag,
            "dataset": cache_path,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "val_ratio": args.val_ratio,
            "n_train": n_train,
            "n_val": n_val,
            "elapsed_s": round(elapsed, 1),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"메타 저장: {meta_path}")


if __name__ == "__main__":
    main()
