from __future__ import annotations

import argparse
import os
import subprocess


REPO_URL = "https://gitee.com/mindspore/models.git"
DEFAULT_REF = "r1.5"
DEFAULT_TARGET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".external", "mindspore-models")
)
SPARSE_PATHS = (
    "/README.md",
    "/official/cv/lenet/train.py",
    "/official/cv/lenet/eval.py",
    "/official/cv/lenet/src/lenet.py",
)


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch(repo_url: str, ref: str, target_dir: str) -> None:
    if os.path.exists(target_dir):
        print(f"官方 models 参考目录已存在，刷新稀疏检出: {target_dir}")
    else:
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                ref,
                repo_url,
                target_dir,
            ]
        )

    run(["git", "sparse-checkout", "set", "--no-cone", *SPARSE_PATHS], cwd=target_dir)
    run(["git", "rev-parse", "HEAD"], cwd=target_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="浅克隆 MindSpore 官方 models 仓库参考脚本")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--target-dir", default=DEFAULT_TARGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch(args.repo_url, args.ref, os.path.abspath(args.target_dir))


if __name__ == "__main__":
    main()
