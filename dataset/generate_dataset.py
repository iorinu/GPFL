# MNIST / Fashion-MNIST / CIFAR-100 を Dirichlet non-IID 分割で
# クライアントごとに npz として書き出すジェネレータ。
#
# 出力先: dataset/<name>-<alpha>-npz/{train,test}/{idx}.npz
#   name   : "mnist" | "fmnist" | "Cifar100"
#            (main.py の分岐: "mnist" -> 1ch, "Cifar" -> 3ch を尊重)
#   npz    : {"data": {"x": (N,C,H,W) float32 [-1,1], "y": (N,) int64}}
#            data_utils.read_data がこの形式を tolist() で読む前提。
#
# 使い方:
#   uv run python generate_dataset.py --name fmnist   --num_clients 20 --alpha 0.1
#   uv run python generate_dataset.py --name Cifar100 --num_clients 20 --alpha 0.1 --num_classes 100
#
# 依存: torch, torchvision, numpy
# 既に生成済みなら再生成をスキップ(--force で強制)。

import argparse
import json
import os
from typing import Callable

import numpy as np
import torch
from torchvision import datasets, transforms


# 3種類のデータセットに対する共通仕様
# num_classes は「デフォルト値」で、CLIから上書きも可
DATASETS = {
    "mnist":    {"cls": datasets.MNIST,        "channels": 1, "num_classes": 10},
    "fmnist":   {"cls": datasets.FashionMNIST, "channels": 1, "num_classes": 10},
    "Cifar100": {"cls": datasets.CIFAR100,     "channels": 3, "num_classes": 100},
}


def build_transform(channels: int) -> Callable:
    # 既存 mnist-0.1-npz と揃えて mean=0.5, std=0.5 → 値域 [-1, 1]
    # チャンネル数だけタプル長を変える
    mean = (0.5,) * channels
    std  = (0.5,) * channels
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_split(name: str, root: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """torchvision から画像を読み込み、(N,C,H,W) float32 と (N,) int64 を返す。"""
    spec = DATASETS[name]
    tfm  = build_transform(spec["channels"])
    ds   = spec["cls"](root=root, train=train, download=True, transform=tfm)

    # DataLoader を使うとバッチ処理でメモリに乗せやすい
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def dirichlet_partition(
    y: np.ndarray, num_clients: int, num_classes: int, alpha: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """
    label-skew Dirichlet 分割 (PFLlib と同じ考え方)。
    クラスごとに Dirichlet(alpha) を引き、そのクラスのサンプルをクライアント間で
    比率通りに分配する。alpha が小さいほど各クライアントは少数クラスに偏る。
    """
    client_idx: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        # クライアント数ぶんの比率をサンプリング
        proportions = rng.dirichlet([alpha] * num_clients)
        # 累積で分割点を作る
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for i, part in enumerate(np.split(idx_c, splits)):
            client_idx[i].extend(part.tolist())

    # 各クライアント内はシャッフルしておく
    out = []
    for lst in client_idx:
        arr = np.array(lst, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def save_shards(out_dir: str, split: str, x: np.ndarray, y: np.ndarray, parts: list[np.ndarray]) -> list[list[list[int]]]:
    """クライアント別に {split}/{idx}.npz を保存し、config 用のサンプル分布を返す。"""
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    dist_all = []
    for i, idxs in enumerate(parts):
        xi = x[idxs]
        yi = y[idxs]
        data = {"x": xi, "y": yi}
        # PFLlib と同じ [{"data": dict}] 形式で保存
        np.savez_compressed(os.path.join(split_dir, f"{i}.npz"), data=data)

        # config.json 用: [[class, count], ...] の並び
        classes, counts = np.unique(yi, return_counts=True)
        dist_all.append([[int(c), int(n)] for c, n in zip(classes, counts)])
    return dist_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=list(DATASETS.keys()),
                        help="生成対象のデータセット名")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=None,
                        help="省略時はデータセット既定 (mnist/fmnist:10, Cifar100:100)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Dirichlet の集中度パラメータ。小さいほど non-IID")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw_root", type=str, default="./rawdata",
                        help="torchvision の生データ置き場")
    parser.add_argument("--force", action="store_true",
                        help="既に生成済みでも上書きする")
    args = parser.parse_args()

    spec = DATASETS[args.name]
    num_classes = args.num_classes if args.num_classes is not None else spec["num_classes"]

    # 出力先: mnist-0.1-npz と同じ命名規則
    out_dir = os.path.join(os.path.dirname(__file__), f"{args.name}-{args.alpha}-npz")
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path) and not args.force:
        print(f"[skip] 既に生成済み: {out_dir} (--force で再生成)")
        return

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[load] {args.name} を torchvision から取得中...")
    x_tr, y_tr = load_split(args.name, args.raw_root, train=True)
    x_te, y_te = load_split(args.name, args.raw_root, train=False)
    print(f"  train: x={x_tr.shape} y={y_tr.shape} / test: x={x_te.shape} y={y_te.shape}")

    print(f"[partition] Dirichlet(alpha={args.alpha}) で {args.num_clients} クライアントへ分割...")
    parts_tr = dirichlet_partition(y_tr, args.num_clients, num_classes, args.alpha, rng)
    parts_te = dirichlet_partition(y_te, args.num_clients, num_classes, args.alpha, rng)

    print(f"[save] {out_dir} に書き出し中...")
    dist_tr = save_shards(out_dir, "train", x_tr, y_tr, parts_tr)
    _       = save_shards(out_dir, "test",  x_te, y_te, parts_te)

    # config.json: 既存 mnist-0.1-npz と同じキー構成
    config = {
        "num_clients": args.num_clients,
        "num_classes": num_classes,
        "non_iid": True,
        "balance": False,
        "partition": "dir",
        "Size of samples for labels in clients": dist_tr,
        "alpha": args.alpha,
        "batch_size": 10,
    }
    with open(cfg_path, "w") as f:
        json.dump(config, f)

    total_tr = sum(len(p) for p in parts_tr)
    total_te = sum(len(p) for p in parts_te)
    print(f"[done] train samples={total_tr}, test samples={total_te}")
    print(f"       main.py からは -data {os.path.basename(out_dir)} -nb {num_classes} で参照")


if __name__ == "__main__":
    main()
