#!/bin/bash
set -e

echo "=== 0. 폴더 준비 + 기존 v2 결과물 마이그레이션 (있으면만, 없으면 skip) ==="
mkdir -p finetune/cache finetune/checkpoints finetune/verify finetune/gradient_maps

if [ -f finetune/cache/dataset.npz ] && [ ! -f finetune/cache/dataset_v2.npz ]; then
    mv -n finetune/cache/dataset.npz finetune/cache/dataset_v2.npz
    echo "  migrated: cache/dataset.npz -> cache/dataset_v2.npz"
fi
if [ -f finetune/checkpoints/finetuned_final.ckpt ] && [ ! -d finetune/checkpoints/v2 ]; then
    mkdir -p finetune/checkpoints/v2
    mv -n finetune/checkpoints/finetuned_final.ckpt finetune/checkpoints/v2/finetuned_final.ckpt
    echo "  migrated: checkpoints/finetuned_final.ckpt -> checkpoints/v2/finetuned_final.ckpt"
fi

echo ""
echo "=== 1. build_dataset.py 재작성 (--manifest, --tag) ==="
cat > finetune/build_dataset.py << 'BD_EOF'
"""
manifest CSV에 성공(generator_passed=True, dial_mpc_ok=True)으로 기록된 씬들을
모아 Neural A* fine-tuning용 (map, start, goal, opt_traj) npz 캐시로 빌드한다.
run_neural_astar_step.py의 좌표변환(full_to_grid)/그리드 로직을 그대로 재사용해서
추론 파이프라인과 좌표계가 절대 어긋나지 않도록 함.

--manifest/--tag로 어떤 manifest를 어떤 이름표(tag)로 캐싱할지 선택한다.
같은 tag로 다시 돌리면 항상 같은 파일을 덮어쓰고(멱등), 다른 tag의 결과와는
절대 안 섞인다 -- finetune/cache/dataset_{tag}.npz + dataset_{tag}.meta.json.

사용: python finetune/build_dataset.py --manifest data/random_batch_manifest_combined.csv --tag combined
"""
import sys, os, csv, json, argparse, time
from collections import Counter
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import run_neural_astar_step as nas  # GRID, ROBOT_PX, PPM, full_to_grid, CKPT_PATH 재사용

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "data/random_batch_manifest_v2.csv")
VARIANT = "waypoint_gen_v1"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

GRID = nas.GRID
ROBOT_PX = nas.ROBOT_PX
PPM = nas.PPM
full_to_grid = nas.full_to_grid


def build_map_design(oracle_path):
    img = Image.open(oracle_path).convert("RGB")
    w, h = img.size
    arr = np.array(img)
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    red_mask = (r > 180) & (g < 120) & (b < 120)
    cell_h, cell_w = h / GRID, w / GRID
    obs = np.zeros((GRID, GRID), dtype=np.float32)
    for gy in range(GRID):
        y0, y1 = int(gy * cell_h), int((gy + 1) * cell_h)
        for gx in range(GRID):
            x0, x1 = int(gx * cell_w), int((gx + 1) * cell_w)
            obs[gy, gx] = 1.0 if red_mask[y0:y1, x0:x1].any() else 0.0
    return (1.0 - obs).astype(np.float32), w, h  # 1=통행가능 (Neural A* 컨벤션)


def world_to_grid(x, y, w, h):
    px = ROBOT_PX[0] + x * PPM
    py = ROBOT_PX[1] - y * PPM
    gx, gy = full_to_grid(px, py, w, h)
    gx = min(max(int(round(gx)), 0), GRID - 1)
    gy = min(max(int(round(gy)), 0), GRID - 1)
    return gx, gy


def onehot(gx, gy):
    m = np.zeros((GRID, GRID), dtype=np.float32)
    m[gy, gx] = 1.0
    return m


def bresenham(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


def rasterize_path(grid_pts):
    mask = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(len(grid_pts) - 1):
        for gx, gy in bresenham(*grid_pts[i], *grid_pts[i + 1]):
            mask[gy, gx] = 1.0
    if grid_pts:
        gx, gy = grid_pts[-1]
        mask[gy, gx] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST,
                         help="사용할 manifest CSV 경로 (기본값: v2)")
    parser.add_argument("--tag", type=str, default="v2",
                         help="이 데이터셋의 이름표. finetune/cache/dataset_{tag}.npz로 저장됨")
    args = parser.parse_args()
    manifest_path = args.manifest
    tag = args.tag
    out_path = os.path.join(CACHE_DIR, f"dataset_{tag}.npz")
    meta_path = os.path.join(CACHE_DIR, f"dataset_{tag}.meta.json")
    print(f"[build_dataset] tag={tag}  manifest={manifest_path}  -> {out_path}")

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    included, skipped = [], []
    map_list, start_list, goal_list, traj_list, scene_list = [], [], [], [], []

    for row in rows:
        scene = row["scene"]
        if row.get("generator_passed") != "True" or row.get("dial_mpc_ok") != "True":
            skipped.append((scene, "gate_failed"))
            continue
        oracle_path = os.path.join(REPO_ROOT, f"data/{scene}/oracle.png")
        path_file = os.path.join(REPO_ROOT, f"data/{scene}/{VARIANT}/last_judged_path.json")
        if not (os.path.exists(oracle_path) and os.path.exists(path_file)):
            skipped.append((scene, "missing_file"))
            continue

        map_design, w, h = build_map_design(oracle_path)
        with open(path_file) as f:
            coords = json.load(f)
        if len(coords) < 2:
            skipped.append((scene, "path_too_short"))
            continue

        goal_x, goal_y = float(row["goal_x"]), float(row["goal_y"])
        start_gx, start_gy = world_to_grid(0.0, 0.0, w, h)
        goal_gx, goal_gy = world_to_grid(goal_x, goal_y, w, h)
        grid_pts = [world_to_grid(c["x"], c["y"], w, h) for c in coords]
        opt_traj = rasterize_path(grid_pts)

        map_list.append(map_design)
        start_list.append(onehot(start_gx, start_gy))
        goal_list.append(onehot(goal_gx, goal_gy))
        traj_list.append(opt_traj)
        scene_list.append(scene)
        included.append(scene)

    print(f"포함: {len(included)}개, 스킵: {len(skipped)}개")
    skip_counts = dict(Counter(r for _, r in skipped))
    if skipped:
        print(f"  스킵 사유: {skip_counts}")
    if not included:
        print("ERROR: 포함된 샘플이 0개 - manifest/경로 확인 필요")
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(
        out_path,
        map_designs=np.stack(map_list)[:, None, :, :],
        start_maps=np.stack(start_list)[:, None, :, :],
        goal_maps=np.stack(goal_list)[:, None, :, :],
        opt_trajs=np.stack(traj_list)[:, None, :, :],
        scenes=np.array(scene_list),
    )
    with open(meta_path, "w") as f:
        json.dump({
            "tag": tag,
            "manifest": manifest_path,
            "n_included": len(included),
            "n_skipped": len(skipped),
            "skip_reasons": skip_counts,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"저장 완료: {out_path} ({len(included)} samples, shape={map_list[0].shape})")
    print(f"메타 저장: {meta_path}")


if __name__ == "__main__":
    main()
BD_EOF
echo "  작성 완료: finetune/build_dataset.py"

echo ""
echo "=== 2. train.py 재작성 (--tag) ==="
cat > finetune/train.py << 'TR_EOF'
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
TR_EOF
echo "  작성 완료: finetune/train.py"

echo ""
echo "=== 3. verify_compare.py 재작성 (--tag) ==="
cat > finetune/verify_compare.py << 'VC_EOF'
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
VC_EOF
echo "  작성 완료: finetune/verify_compare.py"

echo ""
echo "=== 4. eval_compare.py 재작성 (--tag) ==="
cat > finetune/eval_compare.py << 'EC_EOF'
"""
val 8개 held-out 씬에서 GT(정답 경로 마스크) vs 사전학습 모델 vs fine-tuned 모델의
raw Neural A* 출력 경로를 oracle.png 위에 겹쳐 그려서 시각 비교.
--tag로 어떤 학습 결과를 볼지 선택. 결과는 finetune/verify/{tag}/ 밑에 저장돼서
다른 tag의 결과와 절대 안 섞인다.

사용: python finetune/eval_compare.py --tag combined
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
import run_neural_astar_step as nas

from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PRETRAINED_CKPT = os.path.join(REPO_ROOT, nas.CKPT_PATH)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    args = parser.parse_args()
    tag = args.tag
    cache_path = os.path.join(HERE, "cache", f"dataset_{tag}.npz")
    finetuned_ckpt = os.path.join(HERE, "checkpoints", tag)
    out_dir = os.path.join(HERE, "verify", tag)
    print(f"[eval_compare] tag={tag}  dataset={cache_path}  ckpt={finetuned_ckpt}  -> {out_dir}")

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
            ax.plot(ft_xs, ft_ys, color="red", linewidth=2, label=f"fine-tuned [{tag}]", zorder=6)
        ax.legend(loc="upper right")
        ax.set_title(f"{scene}  (tag={tag})")
        out_path = os.path.join(out_dir, f"{scene}_compare.png")
        plt.savefig(out_path)
        plt.close()
        print(f"저장: {out_path}")

    print(f"\n완료: {len(val_indices)}개 비교 이미지 -> {out_dir}/")


if __name__ == "__main__":
    main()
EC_EOF
echo "  작성 완료: finetune/eval_compare.py"

echo ""
echo "=== 5. visualize_gradient.py 재작성 (--tag) ==="
cat > finetune/visualize_gradient.py << 'VG_EOF'
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
VG_EOF
echo "  작성 완료: finetune/visualize_gradient.py"

echo ""
echo "=== 6. pilot10_compare.py 재작성 (--ckpt-tag, + core/ sys.path 누락 버그 수정) ==="
cat > finetune/pilot10_compare.py << 'P10_EOF'
"""
학습에 전혀 안 쓰인 새 씬 10개를 생성해서, 우리 production 파이프라인 그대로
(Neural A* -> waypoint_generator 보정) pretrained/fine-tuned 체크포인트 각각으로
raw proposal을 뽑고, GT(=pretrained proposal을 보정한 안전경로)와 3-way 비교.
vlm_court env에서 실행 (run_random_batch_v2.py와 동일 환경).

--ckpt-tag로 어떤 fine-tuned 체크포인트(finetune/checkpoints/{tag}/)를 볼지 선택.
결과는 finetune/verify/pilot_{tag}/ 밑에 저장돼서 eval_compare.py의
finetune/verify/{tag}/ 결과와 안 섞인다.

사용: python finetune/pilot10_compare.py --ckpt-tag combined
"""
import os, sys, argparse, subprocess, json, random, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (파일 위치 기준 자동 계산, 컴퓨터마다 안전)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "core"))  # core/ 이동 후 누락돼있던 sys.path (이번에 같이 수정)
os.chdir(REPO)
from waypoint_generator import generate_waypoints
from generate_random_baffle_maze import generate as generate_slalom, generate_boxes

ROBOT_PX = (421.0, 540.0)
PPM = 150.0
NEURAL_ASTAR_ENV = "neural-astar"
N_SCENES = 10
SEED_BASE = 90000  # 기존 배치(0~299)와 절대 안 겹치는 범위


def run_neural_astar_step(scene, goal_x, goal_y, ckpt_override=None, timeout=120):
    env = {**os.environ}
    if ckpt_override:
        env["NEURAL_ASTAR_CKPT_OVERRIDE"] = ckpt_override
    cmd = ["conda", "run", "-n", NEURAL_ASTAR_ENV, "--no-capture-output",
           "python", "core/run_neural_astar_step.py",
           "--scene", scene, "--goal-x", str(goal_x), "--goal-y", str(goal_y)]
    try:
        proc = subprocess.run(cmd, cwd=REPO, timeout=timeout, capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        print("  ⚠️ timeout")
        return None
    proposal_path = f"data/{scene}/neural_astar/coordinate_proposal.json"
    if proc.returncode != 0 or not os.path.exists(proposal_path):
        print(f"  ⚠️ 실패 (returncode={proc.returncode}): {proc.stdout[-300:]}")
        return None
    with open(proposal_path) as f:
        return json.load(f)


def to_px(coords):
    rx, ry = ROBOT_PX
    xs = [rx + c["x"] * PPM for c in coords]
    ys = [ry - c["y"] * PPM for c in coords]
    return xs, ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-tag", type=str, default="v2",
                         help="비교할 fine-tuned 체크포인트 이름표 (finetune/checkpoints/{tag}/)")
    args = parser.parse_args()
    tag = args.ckpt_tag
    finetuned_ckpt_dir = os.path.join(REPO, "finetune/checkpoints", tag)
    out_dir = os.path.join(REPO, "finetune/verify", f"pilot_{tag}")
    print(f"[pilot10_compare] ckpt_tag={tag}  ckpt={finetuned_ckpt_dir}  -> {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i in range(N_SCENES):
        seed = SEED_BASE + i
        scene = f"oracle_scene_PILOT{i:02d}"
        print(f"\n=== [{scene}] (seed={seed}) ===")
        layout_rng = random.Random(seed * 7919 + 13)
        use_boxes = layout_rng.random() < 0.5
        xml, meta = (generate_boxes(seed, scene) if use_boxes else generate_slalom(seed, scene))
        goal_x, goal_y = meta["goal_x"], meta["goal_y"]
        with open(f"dial_mpc/dial_mpc/models/unitree_go2/{scene}.xml", "w") as f:
            f.write(xml)

        env = {**os.environ, "MUJOCO_GL": "egl"}
        render_proc = subprocess.run(["python", "core/oracle_gen.py", f"{scene}.xml"],
                                      cwd=REPO, capture_output=True, text=True, timeout=60, env=env)
        image_path = f"data/{scene}/oracle.png"
        if not os.path.exists(image_path):
            print(f"  ⚠️ 렌더링 실패: {render_proc.stderr[-300:]}")
            continue

        pretrained_proposal = run_neural_astar_step(scene, goal_x, goal_y)
        if pretrained_proposal is None:
            print("  ⚠️ pretrained 추론 실패, 스킵"); continue

        finetuned_proposal = run_neural_astar_step(scene, goal_x, goal_y, ckpt_override=finetuned_ckpt_dir)
        if finetuned_proposal is None:
            print("  ⚠️ fine-tuned 추론 실패, 스킵"); continue

        final_coords, passed, _ = generate_waypoints(image_path, pretrained_proposal, ROBOT_PX, PPM)
        print(f"  GT(waypoint_generator) passed={passed}")

        gt_xs, gt_ys = to_px(final_coords)
        pre_xs, pre_ys = to_px(pretrained_proposal)
        ft_xs, ft_ys = to_px(finetuned_proposal)

        img = mpimg.imread(image_path)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.plot(gt_xs, gt_ys, color="lime", linewidth=3, linestyle="--", label="GT (safety-corrected)", zorder=4)
        ax.plot(pre_xs, pre_ys, color="blue", linewidth=2, label="pretrained (before)", zorder=5)
        ax.plot(ft_xs, ft_ys, color="red", linewidth=2, label=f"fine-tuned [{tag}]", zorder=6)
        ax.legend(loc="upper right")
        ax.set_title(f"{scene}  passed={passed}  (ckpt_tag={tag})")
        out_path = os.path.join(out_dir, f"{scene}_pilot.png")
        plt.savefig(out_path)
        plt.close()
        print(f"  저장: {out_path}")
        results.append(scene)

    print(f"\n완료: {len(results)}/{N_SCENES}개 -> {out_dir}/")


if __name__ == "__main__":
    main()
P10_EOF
echo "  작성 완료: finetune/pilot10_compare.py"

echo ""
echo "=== 7. 문법 검증 ==="
for f in finetune/build_dataset.py finetune/train.py finetune/verify_compare.py finetune/eval_compare.py finetune/visualize_gradient.py finetune/pilot10_compare.py; do
    python3 -c "import ast; ast.parse(open('$f').read())" && echo "  OK: $f" || echo "  !! SYNTAX ERROR: $f"
done

echo ""
echo "=== 8. --tag/--manifest 옵션 존재 확인 ==="
grep -n '"--tag"\|"--manifest"\|"--ckpt-tag"' finetune/build_dataset.py finetune/train.py finetune/verify_compare.py finetune/eval_compare.py finetune/visualize_gradient.py finetune/pilot10_compare.py

echo ""
echo "=== 완료 ==="
echo "예시: v2로 기존 재현 -> python finetune/build_dataset.py --tag v2 && python finetune/train.py --tag v2"
echo "예시: combined로 새 실험 -> python finetune/build_dataset.py --manifest data/random_batch_manifest_combined.csv --tag combined && python finetune/train.py --tag combined"
