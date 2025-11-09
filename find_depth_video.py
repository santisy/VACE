#!/usr/bin/env python3

import argparse
import glob
import math
import os
import re
import shutil
import sys

import cv2

IDENTIFIER_PATTERN = re.compile(
    r'obj(?P<object_id>[^_/]+)_joint(?P<joint_id>\d+)_prompt(?P<prompt_idx>\d+)_video(?P<video_idx>\d+)'
)


def parse_generated_identifier(path_fragment: str) -> dict:
    """
    Extract the object, joint, prompt, and video indices from a generated video path.
    """
    match = IDENTIFIER_PATTERN.search(path_fragment)
    if not match:
        raise ValueError(
            f"Could not locate an identifier like "
            f"'obj<id>_joint<id>_prompt<id>_video<id>' in '{path_fragment}'"
        )
    return {
        "object_id": match.group("object_id"),
        "joint_id": int(match.group("joint_id")),
        "prompt_idx": int(match.group("prompt_idx")),
        "video_idx": int(match.group("video_idx")),
        "tag": match.group(0),
    }


def locate_depth_video(input_root: str, object_id: str, video_idx: int, use_color_direct: bool) -> str:
    """
    Reproduce the ordering used in batch_run.py to locate the source depth (or color) video.
    """
    pattern = "color_*.mp4" if use_color_direct else "depth_*.mp4"
    object_path = os.path.join(input_root, object_id)
    if not os.path.isdir(object_path):
        raise FileNotFoundError(f"Object directory '{object_path}' does not exist")

    video_paths = glob.glob(os.path.join(object_path, pattern))
    if not video_paths:
        raise FileNotFoundError(
            f"No videos matching '{pattern}' found under '{object_path}'"
        )

    try:
        return video_paths[video_idx]
    except IndexError as exc:
        raise IndexError(
            f"video_idx {video_idx} exceeds available videos ({len(video_paths)}) in '{object_path}'"
        ) from exc


def copy_video(src: str, dst_dir: str, tag: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f"{tag}_source_depth.mp4")
    shutil.copy2(src, dst_path)
    return dst_path


def copy_generated_video(src: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst_path)
    return dst_path


def copy_mask(source_video_path: str, dst_dir: str, tag: str) -> str | None:
    mask_path = (
        source_video_path.replace("depth", "mask")
        .replace("color", "mask")
        .replace(".mp4", ".png")
    )
    if not os.path.exists(mask_path):
        return None
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f"{tag}_mask.png")
    shutil.copy2(mask_path, dst_path)
    return dst_path


def _frame_requests(frame_count: int) -> dict[int, list[str]]:
    if frame_count <= 0:
        frame_count = 1
    mid = max(frame_count // 2, 0)
    last = max(frame_count - 1, 0)
    requests: dict[int, list[str]] = {}
    for label, idx in (("first", 0), ("middle", mid), ("last", last)):
        requests.setdefault(idx, []).append(label)
    return requests


def save_key_frames(video_path: str, dst_dir: str, tag: str, variant: str) -> list[str]:
    os.makedirs(dst_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video '{video_path}'")

    frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0 if math.isnan(frame_count_raw) else int(frame_count_raw)

    outputs: list[str] = []
    if frame_count <= 0:
        frames: list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames found in '{video_path}'")
        requests = _frame_requests(len(frames))
        for idx, labels in requests.items():
            frame = frames[idx]
            for label in labels:
                dst_path = os.path.join(dst_dir, f"{tag}_{variant}_{label}.png")
                if not cv2.imwrite(dst_path, frame):
                    raise RuntimeError(f"Failed to write frame '{dst_path}'")
                outputs.append(dst_path)
        return outputs

    requests = _frame_requests(frame_count)
    for idx, labels in requests.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Failed to read frame {idx} from '{video_path}'")
        for label in labels:
            dst_path = os.path.join(dst_dir, f"{tag}_{variant}_{label}.png")
            if not cv2.imwrite(dst_path, frame):
                cap.release()
                raise RuntimeError(f"Failed to write frame '{dst_path}'")
            outputs.append(dst_path)
    cap.release()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Locate the depth video that produced a generated output, copy all related RGB clips and mask, "
            "and save first/middle/last frames for both RGB and depth videos."
        )
    )
    parser.add_argument("--input_root", required=True, help="Root directory passed to batch_run.py")
    parser.add_argument(
        "--generated",
        required=True,
        help="Path or name of the generated video (e.g. results/obj42_joint00_prompt000_video000/out_video.mp4)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the depth video should be copied",
    )
    parser.add_argument(
        "--use_color_direct",
        action="store_true",
        help="Set if batch_run.py was executed with --use_color_direct",
    )

    args = parser.parse_args()

    try:
        identifier = parse_generated_identifier(args.generated)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    try:
        depth_path = locate_depth_video(
            args.input_root,
            identifier["object_id"],
            identifier["video_idx"],
            args.use_color_direct,
        )
    except (FileNotFoundError, IndexError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    shared_tag = f"obj{identifier['object_id']}_joint{identifier['joint_id']:02d}_video{identifier['video_idx']:03d}"

    try:
        copied_depth_path = copy_video(depth_path, args.output_dir, shared_tag)
    except OSError as exc:
        print(f"Failed to copy '{depth_path}' to '{args.output_dir}': {exc}", file=sys.stderr)
        sys.exit(1)

    copied_mask = copy_mask(depth_path, args.output_dir, shared_tag)

    generated_video_path = args.generated
    if not os.path.isabs(generated_video_path):
        generated_video_path = os.path.abspath(generated_video_path)
    if not os.path.exists(generated_video_path):
        print(f"Generated video '{args.generated}' not found on disk", file=sys.stderr)
        sys.exit(1)

    generated_dir = os.path.dirname(generated_video_path)
    pattern = os.path.join(
        generated_dir,
        f"obj{identifier['object_id']}_joint{identifier['joint_id']:02d}_prompt*_video{identifier['video_idx']:03d}.mp4",
    )
    related_videos = sorted(glob.glob(pattern))
    if not related_videos:
        related_videos = [generated_video_path]

    saved_generated_frames: list[str] = []
    copied_generated_videos: list[str] = []
    for rgb_path in related_videos:
        try:
            rgb_identifier = parse_generated_identifier(rgb_path)
        except ValueError:
            continue
        try:
            copied_rgb = copy_generated_video(rgb_path, args.output_dir)
        except OSError as exc:
            print(f"Failed to copy '{rgb_path}' to '{args.output_dir}': {exc}", file=sys.stderr)
            continue

        copied_generated_videos.append(copied_rgb)
        prompt_tag = f"{shared_tag}_prompt{rgb_identifier['prompt_idx']:03d}"
        try:
            saved_generated_frames.extend(
                save_key_frames(rgb_path, args.output_dir, prompt_tag, "rgb")
            )
        except RuntimeError as exc:
            print(exc, file=sys.stderr)

    try:
        depth_frames = save_key_frames(depth_path, args.output_dir, shared_tag, "depth")
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"Copied depth '{depth_path}' -> '{copied_depth_path}'")
    if copied_mask:
        print(f"Copied mask '{copied_mask}'")
    else:
        print("Mask file not found; skipped copying.")
    print("Copied RGB videos:")
    for video_path in copied_generated_videos:
        print(f"  {video_path}")
    print("Saved RGB frames:")
    for frame_path in saved_generated_frames:
        print(f"  {frame_path}")
    print("Saved depth frames:")
    for frame_path in depth_frames:
        print(f"  {frame_path}")


if __name__ == "__main__":
    main()
