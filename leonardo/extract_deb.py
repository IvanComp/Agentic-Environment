import gzip
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path


def version_key(version):
    return [int(part) if part.isdigit() else part for part in re.split(r"[^0-9A-Za-z]+", version)]


def latest(packages_path, package):
    with gzip.open(packages_path, "rt", encoding="utf-8", errors="replace") as handle:
        blocks = handle.read().split("\n\n")
    candidates = []
    for block in blocks:
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(" "))
        if fields.get("Package") == package and "Filename" in fields:
            candidates.append((version_key(fields.get("Version", "")), fields["Filename"]))
    if not candidates:
        raise SystemExit(f"{package} not found in {packages_path}")
    print(max(candidates)[1])


def members(deb_path):
    data = Path(deb_path).read_bytes()
    if not data.startswith(b"!<arch>\n"):
        raise SystemExit(f"{deb_path} is not a deb archive")
    offset = 8
    while offset + 60 <= len(data):
        header = data[offset:offset + 60]
        name = header[:16].decode().strip().rstrip("/")
        size = int(header[48:58].decode().strip())
        start = offset + 60
        yield name, data[start:start + size]
        offset = start + size + (size % 2)


def extract(deb_path, destination):
    Path(destination).mkdir(parents=True, exist_ok=True)
    for name, payload in members(deb_path):
        if not name.startswith("data.tar"):
            continue
        if name.endswith(".zst"):
            archive = Path(destination) / name
            archive.write_bytes(payload)
            subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", destination], check=True)
            archive.unlink()
        else:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as tar:
                tar.extractall(destination)
        return
    raise SystemExit(f"no data archive in {deb_path}")


if __name__ == "__main__":
    if sys.argv[1] == "latest":
        latest(sys.argv[2], sys.argv[3])
    else:
        extract(sys.argv[2], sys.argv[3])
