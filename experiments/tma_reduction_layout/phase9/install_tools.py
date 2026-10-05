"""Read-only CUDA13 disassembly tools layered over the unchanged native image."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path("/opt/phase9-cuda-tools")
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
PACKAGES = {
    "cuobjdump": ("cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-13.0.85-archive.tar.xz",
                  "bd624dcb2089842add8f293efab1d21aa076c98b36d8dcb64347fe08fb03315d"),
    "nvdisasm": ("cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-13.0.85-archive.tar.xz",
                 "0541e0230f724a43d67288ef882c63351d0c302bf567591b7620d40f93c2c93c"),
}


def main():
    (ROOT / "bin").mkdir(parents=True, exist_ok=True)
    manifest = urllib.request.urlopen(BASE + "redistrib_13.0.2.json", timeout=60).read()
    (ROOT / "redistrib_13.0.2.json").write_bytes(manifest)
    records = {}
    for name, (relative, expected) in PACKAGES.items():
        data = urllib.request.urlopen(BASE + relative, timeout=60).read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError("Official tool archive SHA mismatch: " + name)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tar:
            member = next(x for x in tar.getmembers() if x.name.endswith("/bin/" + name))
            binary = tar.extractfile(member).read()
            path = ROOT / "bin" / name
            path.write_bytes(binary)
            path.chmod(0o755)
        records[name] = {"url": BASE + relative, "archive_SHA256": actual,
                         "binary_SHA256": hashlib.sha256(binary).hexdigest(), "binary_path": str(path)}
    (ROOT / "installation.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print("CUDA13.0.85 cuobjdump/nvdisasm installed with verified official archive hashes; no native build")


if __name__ == "__main__":
    main()
