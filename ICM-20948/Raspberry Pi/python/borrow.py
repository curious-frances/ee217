#!/usr/bin/env python3
import csv
import re
from pathlib import Path

G = 9.81  # convert accel from "g" to m/s^2

TIME_RE = re.compile(r"^time:\s*([-\d.]+)\s*$")
A_RE    = re.compile(r"^a:\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*$")
GY_RE   = re.compile(r"^g:\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*$")


def parse_triplets(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    while i + 2 < len(lines):
        m_t = TIME_RE.match(lines[i])
        m_a = A_RE.match(lines[i + 1])
        m_g = GY_RE.match(lines[i + 2])
        if not (m_t and m_a and m_g):
            i += 1
            continue

        t = float(m_t.group(1))
        ax, ay, az = map(float, m_a.groups())
        gx, gy, gz = map(float, m_g.groups())
        yield (t, ax, ay, az, gx, gy, gz)
        i += 3


def convert_file(src_path: Path, dst_path: Path):
    text = src_path.read_text(errors="ignore")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time",
            "ax_raw", "ay_raw", "az_raw",
            "ax", "ay", "az",
            "gx", "gy", "gz"
        ])

        for t, ax_g, ay_g, az_g, gx, gy, gz in parse_triplets(text):
            ax = ax_g * G
            ay = ay_g * G
            az = az_g * G

            w.writerow([
                f"{t:.6f}",
                f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}",
                f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}",
                f"{gx:.6f}", f"{gy:.6f}", f"{gz:.6f}",
            ])


def main():
    in_dir = Path("/Users/francesraphael/school/ee217/project/ee217/ICM-20948/Raspberry Pi/python/borrowed-data")
    out_dir = Path("/Users/francesraphael/school/ee217/project/ee217/ICM-20948/Raspberry Pi/python")

    if not in_dir.exists():
        raise SystemExit(f"Input dir not found: {in_dir}")

    for src in sorted(in_dir.iterdir()):
        if not src.is_file():
            continue
        if src.suffix.lower() == ".csv":
            continue

        dst = out_dir / f"{src.stem}.csv"
        convert_file(src, dst)
        print(f"Wrote: {dst}")

    print("Done!")


if __name__ == "__main__":
    main()
