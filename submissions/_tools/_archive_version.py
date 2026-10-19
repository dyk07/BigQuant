"""归档指定 commit 的提交包到 submissions/<name>/。

用法:
  python _archive_version.py <commit> <name> [pkg_subdir]

从 git 仓库导出 <commit> 时刻的 <pkg_subdir>(默认 cnn_3f) 全部文件,
写到 submissions/<name>/。用于把历史提交逐一恢复成独立文件夹, 方便回滚。
"""
import subprocess
import sys
import os

ROOT = r"C:\Users\Kelvin\Documents\GitHub\BigQuant"


def run_bin(cmd):
    r = subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{r.stderr.decode('utf-8', 'replace')}")
    return r.stdout


def run(cmd):
    return run_bin(cmd).decode("utf-8", "replace")


def main():
    commit = sys.argv[1]
    name = sys.argv[2]
    pkg = sys.argv[3] if len(sys.argv) > 3 else "cnn_3f"

    # 列出该 commit 下 pkg 的全部文件
    out = run(f'git ls-tree -r --name-only {commit} submissions/{pkg}')
    files = [l.strip() for l in out.splitlines() if l.strip()]
    if not files:
        raise SystemExit(f"[{commit}] submissions/{pkg} 无文件")

    dest = os.path.join(ROOT, "submissions", name)
    os.makedirs(dest, exist_ok=True)

    for f in files:
        rel = os.path.relpath(f, f"submissions/{pkg}")
        if rel.startswith(".."):
            continue
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        content = run_bin(f'git show {commit}:{f}')
        with open(target, "wb") as fh:
            fh.write(content)
        print(f"  {rel}  ({len(content):,} bytes)")

    print(f"[archive] {commit} -> submissions/{name}/  ({len(files)} files)")


if __name__ == "__main__":
    main()
