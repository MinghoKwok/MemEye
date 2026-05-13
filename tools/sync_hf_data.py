import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://huggingface.co/datasets/MemEyeBench/MemEye"
DEFAULT_REPO_DIR = Path.home() / ".cache" / "memeye_hf" / "MemEye"
DEFAULT_LOCAL_DATA_ROOT = Path(__file__).resolve().parent / "data"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def capture(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Missing HF token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in your environment.")
    return token


def authed_repo_url(repo_url: str, token: str) -> str:
    if "://" not in repo_url:
        raise ValueError(f"Unsupported repo url: {repo_url}")
    prefix, rest = repo_url.split("://", 1)
    return f"{prefix}://user:{token}@{rest}"


def ensure_clone(repo_url: str, repo_dir: Path, token: str) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if (repo_dir / ".git").exists():
        run(["git", "remote", "set-url", "origin", authed_repo_url(repo_url, token)], cwd=repo_dir)
        run(["git", "pull", "--ff-only", "origin", "main"], cwd=repo_dir)
        return
    if repo_dir.exists():
        raise RuntimeError(f"Target repo_dir exists but is not a git repo: {repo_dir}")
    run(["git", "clone", "--depth", "1", authed_repo_url(repo_url, token), str(repo_dir)])


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def print_status(repo_dir: Path) -> None:
    try:
        out = capture(["git", "status", "--short", "--untracked-files=all"], cwd=repo_dir)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to read git status from {repo_dir}") from exc
    print(out if out else "clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync MemEye benchmark data with the HF dataset repo.")
    parser.add_argument("action", choices=["pull", "push", "status"])
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--local-data-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT)
    parser.add_argument("--commit-message", default="Sync MemEye data")
    parser.add_argument("--push", action="store_true", help="Push after staging and commit when action=push.")
    parser.add_argument("--git-user-name", default="")
    parser.add_argument("--git-user-email", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = require_hf_token()
    repo_dir = args.repo_dir.resolve()
    local_data_root = args.local_data_root.resolve()

    ensure_clone(args.repo_url, repo_dir, token)

    repo_data_root = repo_dir / "data"
    if args.action == "status":
        print_status(repo_dir)
        return

    if args.action == "pull":
        copy_tree(repo_data_root, local_data_root)
        print(f"Pulled HF data into {local_data_root}")
        return

    if args.action == "push":
        copy_tree(local_data_root, repo_data_root)
        status = capture(["git", "status", "--short", "--untracked-files=all"], cwd=repo_dir)
        print(status if status else "clean")
        if not status:
            return
        run(["git", "add", "data"], cwd=repo_dir)
        if args.git_user_name:
            run(["git", "config", "user.name", args.git_user_name], cwd=repo_dir)
        if args.git_user_email:
            run(["git", "config", "user.email", args.git_user_email], cwd=repo_dir)
        run(["git", "commit", "-m", args.commit_message], cwd=repo_dir)
        if args.push:
            run(["git", "push", "origin", "main"], cwd=repo_dir)
        return


if __name__ == "__main__":
    main()
