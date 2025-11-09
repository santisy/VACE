#!/usr/bin/env python3

import argparse
import glob
import math
import os
import random
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


def load_prompt_dict(input_root: str, object_id: str) -> dict[int, list[tuple[str, str]]]:
    object_path = os.path.join(input_root, object_id)
    prompts_file = os.path.join(object_path, "prompts.txt")
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file '{prompts_file}' not found")

    prompt_dict: dict[int, list[tuple[str, str]]] = {}
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            non_empty = [x for x in parts if x != ""]
            if not non_empty:
                continue
            try:
                joint_id = int(non_empty[0])
            except ValueError:
                continue
            if len(parts) <= 2:
                continue
            original = parts[2].strip()
            simplified = parts[3].strip() if len(parts) > 3 else ""
            if not simplified:
                simplified = original
            prompt_list = prompt_dict.get(joint_id, [])
            prompt_list.append((original, simplified))
            prompt_dict[joint_id] = prompt_list
    return prompt_dict


def select_prompts_for_video(
    object_id: str,
    joint_id: int,
    video_idx: int,
    prompt_dict: dict[int, list[tuple[str, str]]],
    max_prompt_index: int,
    max_prompts_override: int,
) -> list[tuple[str, str]]:
    available_prompts = prompt_dict.get(joint_id, [])
    if not available_prompts:
        raise ValueError(f"No prompts available for joint {joint_id}")

    target_count = max(max_prompt_index + 1, max_prompts_override)
    if target_count <= 0:
        target_count = max_prompt_index + 1
    if target_count <= 0:
        target_count = 1

    sample_count = min(len(available_prompts), target_count)
    if sample_count <= 0:
        raise ValueError("Unable to determine prompt sample size")

    seed = hash((object_id, joint_id, video_idx)) % (2**32)
    rng = random.Random(seed)
    indices = rng.sample(range(len(available_prompts)), sample_count)
    return [available_prompts[i] for i in indices]


def _frame_requests(frame_count: int) -> dict[int, list[str]]:
    if frame_count <= 0:
        frame_count = 1
    last = max(frame_count - 1, 0)
    first = min(4, last)  # Prefer the 5th frame; fall back if video is shorter
    mid = max(frame_count // 2, 0)
    if mid < first:
        mid = first
    requests: dict[int, list[str]] = {}
    for label, idx in (("first", first), ("middle", mid), ("last", last)):
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
            "save first/middle/last frames for both RGB and depth videos, and write both original and simplified prompts."
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
    parser.add_argument(
        "--max_prompts_per_video",
        type=int,
        default=0,
        help=(
            "Override for the number of prompts sampled per video. Defaults to the highest "
            "prompt index encountered + 1."
        ),
    )
    parser.add_argument(
        "--simplified_run",
        action="store_true",
        help="Set if the original batch run used --simplified (so simplified prompts were fed to the model).",
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
    related_paths = sorted(glob.glob(pattern))
    if not related_paths:
        related_paths = [generated_video_path]

    related_infos: list[tuple[str, dict]] = []
    max_prompt_idx = -1
    for rgb_path in related_paths:
        try:
            rgb_identifier = parse_generated_identifier(rgb_path)
        except ValueError:
            continue
        related_infos.append((rgb_path, rgb_identifier))
        max_prompt_idx = max(max_prompt_idx, rgb_identifier["prompt_idx"])

    if not related_infos:
        related_infos.append((generated_video_path, identifier))
        max_prompt_idx = max(max_prompt_idx, identifier["prompt_idx"])

    prompts_used_text: dict[int, str] = {}
    prompts_original_text: dict[int, str] = {}
    prompts_simplified_text: dict[int, str] = {}
    prompts_error: str | None = None
    prompt_file_path = os.path.join(args.output_dir, f"{shared_tag}_prompts.txt")
    if max_prompt_idx < 0:
        max_prompt_idx = 0
    try:
        prompt_dict = load_prompt_dict(args.input_root, identifier["object_id"])
        selected_prompts = select_prompts_for_video(
            identifier["object_id"],
            identifier["joint_id"],
            identifier["video_idx"],
            prompt_dict,
            max_prompt_idx,
            args.max_prompts_per_video,
        )
        for idx, (original_text, simplified_text) in enumerate(selected_prompts):
            prompts_original_text[idx] = original_text
            prompts_simplified_text[idx] = simplified_text
            prompts_used_text[idx] = simplified_text if args.simplified_run else original_text
    except (FileNotFoundError, ValueError) as exc:
        prompts_error = str(exc)

    prompt_records: list[tuple[int, str | None, str | None, str]] = []
    saved_generated_frames: list[str] = []
    copied_generated_videos: list[str] = []
    for rgb_path, rgb_identifier in related_infos:
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

        prompt_idx = rgb_identifier["prompt_idx"]
        prompt_original = prompts_original_text.get(prompt_idx)
        prompt_simplified = prompts_simplified_text.get(prompt_idx)
        prompt_records.append(
            (
                prompt_idx,
                prompt_original,
                prompt_simplified,
                os.path.basename(rgb_path),
            )
        )

    prompt_file_written: str | None = None
    missing_prompt_indices: list[int] = []
    if prompt_records and prompts_used_text:
        prompt_records_sorted = sorted(prompt_records, key=lambda item: item[0])
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(
                "# prompt_index\tfilename\toriginal_prompt\tsimplified_prompt\tused_prompt\n"
            )
            for idx, original_text, simplified_text, filename in prompt_records_sorted:
                used_text = prompts_used_text.get(idx)
                if used_text is None:
                    missing_prompt_indices.append(idx)
                orig = original_text or ""
                simp = simplified_text or ""
                used = used_text or ""
                f.write(f"prompt{idx:03d}\t{filename}\t{orig}\t{simp}\t{used}\n")
        if not missing_prompt_indices:
            prompt_file_written = prompt_file_path
    elif prompt_records and prompts_error:
        missing_prompt_indices = [idx for idx, _, _, _ in prompt_records]

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
    if prompt_file_written:
        print(f"Saved prompts to '{prompt_file_written}'")
    elif prompt_records:
        if prompts_error:
            print(f"Prompts not saved: {prompts_error}")
        elif missing_prompt_indices:
            print(
                f"Prompts saved with missing entries for indices {missing_prompt_indices} at '{prompt_file_path}'"
            )


if __name__ == "__main__":
    main()
