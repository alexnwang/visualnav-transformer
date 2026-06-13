"""Scale the per-frame playback duration of an animated WebP in place (byte-level),
without decoding/re-encoding pixels. Each ANMF chunk carries a 24-bit LE frame-duration
field at payload offset 12; we multiply each by `factor` (clamped to 24-bit).

Used to make slow-mo variants of the action webps without re-rendering.
"""
import argparse, os, shutil, struct, glob


def scale_file(src, dst, factor):
    b = bytearray(open(src, "rb").read())
    assert b[0:4] == b"RIFF" and b[8:12] == b"WEBP", f"not a webp: {src}"
    o, n = 12, 0
    while o + 8 <= len(b):
        fourcc = b[o:o + 4]
        size = struct.unpack("<I", b[o + 4:o + 8])[0]
        if fourcc == b"ANMF":
            do = o + 8 + 12  # duration field: 3 bytes LE at payload offset 12
            dur = b[do] | b[do + 1] << 8 | b[do + 2] << 16
            new = min(0xFFFFFF, int(round(dur * factor)))
            b[do] = new & 0xFF
            b[do + 1] = (new >> 8) & 0xFF
            b[do + 2] = (new >> 16) & 0xFF
            n += 1
        o += 8 + size + (size & 1)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    open(dst, "wb").write(b)
    return n


def main(args):
    if os.path.isdir(args.src):
        # mirror tree: scale *.webp, copy other files as-is
        for root, _, files in os.walk(args.src):
            rel = os.path.relpath(root, args.src)
            for f in files:
                s = os.path.join(root, f)
                d = os.path.join(args.dst, rel, f)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if f.endswith(".webp"):
                    nf = scale_file(s, d, args.factor)
                    print(f"scaled {nf} frames x{args.factor}: {os.path.relpath(d, args.dst)}", flush=True)
                else:
                    shutil.copy2(s, d)
    else:
        nf = scale_file(args.src, args.dst, args.factor)
        print(f"scaled {nf} frames x{args.factor}: {args.dst}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", help="webp file or directory tree")
    p.add_argument("dst", help="output file or directory")
    p.add_argument("--factor", type=float, default=1.5)
    main(p.parse_args())
