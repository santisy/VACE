#!/usr/bin/env python3

import argparse
import glob
import os
import re
import shutil
import sys

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locate the depth video that produced a generated output and copy it to a folder."
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

    try:
        copied_path = copy_video(depth_path, args.output_dir, identifier["tag"])
    except OSError as exc:
        print(f"Failed to copy '{depth_path}' to '{args.output_dir}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Copied '{depth_path}' -> '{copied_path}'")


if __name__ == "__main__":
    main()
