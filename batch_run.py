import argparse
import gc
import glob
import math
import os
import subprocess
import uuid
import shutil
import sys
import time
import pandas as pd
import random

import torch
from hoss_utils import CephS3Storage
from utils import save_first_frame
from vace.vace_wan_modified import get_wan_model
from vace.vace_wan_modified import save_video

parser = argparse.ArgumentParser()

parser.add_argument("--input_root", type=str, required=True)
parser.add_argument("--output_root", type=str, default="results")
parser.add_argument("--batch_id", type=int, default=0)
parser.add_argument("--batch_n", type=int, default=1)
parser.add_argument("--dry_run_meta", action="store_true")
parser.add_argument("--reorg", action="store_true")
parser.add_argument("--max_prompts_per_video", type=int, default=3)  # New parameter

args = parser.parse_args()

if args.reorg:
    oss = CephS3Storage(
        key=os.environ.get("HOSS_KEY"),
        secret=os.environ.get("HOSS_SECRET"),
        endpoint_url=os.environ.get("ENDPOINT_URL"),
        default_bucket="yangdingdong"
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    TEMP_DIR = f"./temp_results/mytmp_{timestamp}_{uuid.uuid4().hex[:6]}"
    df = pd.DataFrame(columns=['video', 'prompt', 'input_image'])

base_command = "python vace/vace_wan_inference.py --model_name 'vace-14B' --ckpt_dir models/Wan2.1-VACE-14B --src_video {0} --prompt \"{1}\" --save_dir {2}"

input_root = args.input_root
object_dirs = os.listdir(input_root)
wan_vace = get_wan_model()

# Step 1: Collect all tasks (object_id, joint_id, video_idx, prompt_idx, video_path, prompt)
all_tasks = []
for object_dir in object_dirs:
    object_id = object_dir
    object_path = os.path.join(input_root, object_dir)
    video_paths = glob.glob(os.path.join(object_path, "*.mp4"))
    
    # Load prompts
    prompt_dict = {}
    with open(os.path.join(object_path, "prompts.txt"), "r") as f:
        prompts = f.readlines()
    for p_ in prompts:
        p_split = p_.split("\t")
        p_split = [x for x in p_split if x != '']
        joint_id = int(p_split[0])
        prompt = p_.split("\t")[2].strip("\n")
        prompt_list = prompt_dict.get(joint_id, [])
        prompt_list.append(prompt)
        prompt_dict[joint_id] = prompt_list
    
    # Create tasks for each video-prompt combination
    for j, video_path in enumerate(video_paths):
        joint_id = int(os.path.basename(video_path).split("_")[2].lstrip("joint"))
        available_prompts = prompt_dict[joint_id]
        
        # Sample prompts instead of using all
        num_prompts = min(len(available_prompts), args.max_prompts_per_video)
        selected_prompts = random.sample(available_prompts, num_prompts)
        
        for i, prompt in enumerate(selected_prompts):
            all_tasks.append((object_id, joint_id, j, i, video_path, prompt))

# Step 2: Split tasks evenly across batches
batch_len = math.ceil(len(all_tasks) / float(args.batch_n))
batch_tasks = all_tasks[args.batch_id * batch_len: (args.batch_id + 1) * batch_len]

# Step 3: Process batched tasks
for object_id, joint_id, video_idx, prompt_idx, video_path, prompt in batch_tasks:
    out_str = f"obj{object_id}_joint{joint_id:02d}_prompt{prompt_idx:03d}_video{video_idx:03d}"
    
    if args.reorg:
        output_dir = TEMP_DIR
        output_root = args.output_root
        os.makedirs(output_root, exist_ok=True)
    else:
        output_base = os.path.join(args.output_root, f"obj{object_id}")
        os.makedirs(output_base, exist_ok=True)
        output_dir = os.path.join(output_base, out_str)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not args.dry_run_meta:
        if not args.reorg:
            with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
                f.write(prompt)
        src_video, src_mask, src_ref_images = wan_vace.prepare_source([video_path,],
                                                                      [None,],
                                                                      [None,],
                                                                      81,
                                                                      (480, 832),
                                                                      0
                                                                      )
        video = wan_vace.generate(
            prompt,
            src_video,
            src_mask,
            src_ref_images,
            size=(480, 832),
            frame_num=81,
            shift=16,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            seed=random.randint(0, sys.maxsize),
            offload_model=False)

        save_video(video, src_video, src_mask, output_dir)
        #subprocess.run(base_command.format(video_path, prompt, output_dir), shell=True)

    if args.reorg:
        out_video_path = os.path.join(output_dir, "out_video.mp4")
        dst_video_path = os.path.join(output_root, f"{out_str}.mp4")
        if not args.dry_run_meta:
            shutil.copy(out_video_path, dst_video_path)
            first_frame_path = save_first_frame(out_video_path, output_root, output_name=f"{out_str}_frame0.png")
        else:
            first_frame_path = f"{out_str}_frame0.png"

        if os.path.exists(dst_video_path):
            new_row = {'video': os.path.basename(dst_video_path), 
                       'prompt': prompt,
                       'input_image': os.path.basename(first_frame_path)}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    gc.collect()
    torch.cuda.empty_cache()

if args.reorg:
    df.to_csv(os.path.join(output_root, 'metadata.csv'), index=False)

    #if not args.dry_run_meta:
    #    shutil.rmtree(TEMP_DIR)
    #    oss.upload_directory(local_dir=output_root,
    #                        remote_prefix=f"ti2v/{os.path.basename(output_root)}",
    #                        recursive=True)
    #    shutil.rmtree(output_root)