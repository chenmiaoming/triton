"""One image-build transaction: actual cache counters around native install."""
from pathlib import Path
import subprocess
import sys


def main():
    output = Path("/opt/phase6-build-evidence")
    output.mkdir(parents=True, exist_ok=True)
    for name, args in (("ccache_version.txt", ["ccache", "--version"]),
                       ("ccache_before.txt", ["ccache", "--print-stats"])):
        (output / name).write_text(subprocess.check_output(args, text=True))
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "triton", "pytorch-triton"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "python/requirements.txt"],
                   cwd="/opt/triton-src", check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "-v"],
                   cwd="/opt/triton-src", check=True)
    for name, args in (("ccache_after.txt", ["ccache", "--print-stats"]),
                       ("ccache_summary.txt", ["ccache", "-s"])):
        (output / name).write_text(subprocess.check_output(args, text=True))


if __name__ == "__main__":
    main()
