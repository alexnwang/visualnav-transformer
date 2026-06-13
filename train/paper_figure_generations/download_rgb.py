"""Download ONLY the video_main_rgb mp4 (1408x1408 @30fps head-cam RGB) for given
Nymeria tracks, straight from the download_url in all_data.json. This is the minimal
download for high-res frames (~650-1000MB/track); the VRS route to scene RGB needs the
12-17GB data.vrs and is avoided.
"""
import argparse, json, os, hashlib, urllib.request, shutil


def sha1(path, buf=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    seqs = json.load(open(args.json))["sequences"]
    os.makedirs(args.out, exist_ok=True)
    for track in args.tracks:
        e = seqs[track]["video_main_rgb"]
        dst_dir = os.path.join(args.out, track)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "video_main_rgb.mp4")
        if os.path.isfile(dst) and os.path.getsize(dst) == e["file_size_bytes"]:
            print(f"[skip] {track} (exists, size ok)", flush=True)
            continue
        print(f"[get] {track}  {e['file_size_bytes']/1e6:.0f} MB", flush=True)
        with urllib.request.urlopen(e["download_url"], timeout=120) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        got = sha1(dst)
        ok = got == e["sha1sum"]
        print(f"[done] {track}  sha1 {'OK' if ok else 'MISMATCH! ' + got}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="/home/anw2067/visualnav-transformer/data_jsons/all_data.json")
    p.add_argument("--out", default="/home/anw2067/scratch/temp_nymeria_large_videos")
    p.add_argument("--tracks", nargs="+", required=True)
    main(p.parse_args())
