import argparse
import gc
import glob
import math
import os
import uuid
import shutil
import sys
import time
import pandas as pd
import random
from multiprocessing import Pool, cpu_count
from functools import partial

import torch
from hoss_utils import CephS3Storage
from utils import save_first_frame

parser = argparse.ArgumentParser()

parser.add_argument("--input_root", type=str, required=True)
parser.add_argument("--output_root", type=str, default="results")
parser.add_argument("--batch_id", type=int, default=0)
parser.add_argument("--batch_n", type=int, default=1)
parser.add_argument("--dry_run_meta", action="store_true")
parser.add_argument("--reorg", action="store_true")
parser.add_argument("--max_prompts_per_video", type=int, default=3)
parser.add_argument("--num_processes", type=int, default=None, help="Number of CPU processes for dry_run_meta (default: CPU count)")
parser.add_argument("--out_meta", type=str, default="metadata.csv")
parser.add_argument("--simplified", action="store_true")
parser.add_argument("--use_color_direct", action="store_true")

args = parser.parse_args()

if not args.dry_run_meta and not args.use_color_direct:
    from vace.vace_wan_modified import get_wan_model
    from vace.vace_wan_modified import save_video

if args.reorg:
    oss = CephS3Storage(
        key=os.environ.get("HOSS_KEY"),
        secret=os.environ.get("HOSS_SECRET"),
        endpoint_url=os.environ.get("ENDPOINT_URL"),
        default_bucket="yangdingdong"
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    TEMP_DIR = f"./temp_results/mytmp_{timestamp}_{uuid.uuid4().hex[:6]}"
    df = pd.DataFrame(columns=['video', 'prompt', 'input_image', 'ref_mask'])

base_command = "python vace/vace_wan_inference.py --model_name 'vace-14B' --ckpt_dir models/Wan2.1-VACE-14B --src_video {0} --prompt \"{1}\" --save_dir {2}"

def process_task_dry_run(task_data, args):
    """Process a single task in dry run mode (CPU only)"""
    object_id, joint_id, video_idx, prompt_idx, video_path, prompt = task_data
    out_str = f"obj{object_id}_joint{joint_id:02d}_prompt{prompt_idx:03d}_video{video_idx:03d}"
    
    if args.reorg:
        output_dir = args.temp_dir  # Pass temp_dir through args
        output_root = args.output_root
        os.makedirs(output_root, exist_ok=True)
    else:
        output_base = os.path.join(args.output_root, f"obj{object_id}")
        os.makedirs(output_base, exist_ok=True)
        output_dir = os.path.join(output_base, out_str)
    
    os.makedirs(output_dir, exist_ok=True)
    
    result_row = None
    if args.reorg:
        dst_video_path = os.path.join(output_root, f"{out_str}.mp4")
        first_frame_path = f"{out_str}_frame0.png"
        
        # Derive mask path from video_path and copy to output
        mask_path = video_path.replace("depth", "mask").replace("color", "mask").replace(".mp4", ".png")
        dst_mask_path = os.path.join(output_root, f"{out_str}_mask.png")
        shutil.copy(mask_path, dst_mask_path)
        
        # In dry run, we just create the metadata without actual files
        result_row = {
            'video': os.path.basename(dst_video_path), 
            'prompt': prompt,
            'input_image': os.path.basename(first_frame_path),
            'ref_mask': os.path.basename(dst_mask_path)
        }
    
    return result_row

def process_task_color_direct(task_data, args):
    """Process a single task by directly copying color videos without V2V model"""
    object_id, joint_id, video_idx, prompt_idx, video_path, prompt = task_data
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
    
    if not args.reorg:
        with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
            f.write(prompt)
    
    result_row = None
    if args.reorg:
        dst_video_path = os.path.join(output_root, f"{out_str}.mp4")
        shutil.copy(video_path, dst_video_path)
        first_frame_path = save_first_frame(video_path, output_root, output_name=f"{out_str}_frame0.png")

        # Derive mask path from video_path and copy to output
        mask_path = video_path.replace("color", "mask").replace(".mp4", ".png")
        dst_mask_path = os.path.join(output_root, f"{out_str}_mask.png")
        shutil.copy(mask_path, dst_mask_path)

        if os.path.exists(dst_video_path):
            result_row = {
                'video': os.path.basename(dst_video_path), 
                'prompt': prompt,
                'input_image': os.path.basename(first_frame_path),
                'ref_mask': os.path.basename(dst_mask_path)
            }
    else:
        shutil.copy(video_path, os.path.join(output_dir, "out_video.mp4"))
        save_first_frame(video_path, output_dir, output_name="frame0.png")
        mask_path = video_path.replace("color", "mask").replace(".mp4", ".png")
        shutil.copy(mask_path, os.path.join(output_dir, "mask.png"))
    
    return result_row

def process_task_full(task_data, args, wan_vace):
    """Process a single task with full GPU processing"""
    object_id, joint_id, video_idx, prompt_idx, video_path, prompt = task_data
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
    
    result_row = None
    if args.reorg:
        out_video_path = os.path.join(output_dir, "out_video.mp4")
        dst_video_path = os.path.join(output_root, f"{out_str}.mp4")
        shutil.copy(out_video_path, dst_video_path)
        first_frame_path = save_first_frame(out_video_path, output_root, output_name=f"{out_str}_frame0.png")

        # Derive mask path from video_path and copy to output
        mask_path = video_path.replace("depth", "mask").replace("color", "mask").replace(".mp4", ".png")
        dst_mask_path = os.path.join(output_root, f"{out_str}_mask.png")
        shutil.copy(mask_path, dst_mask_path)

        if os.path.exists(dst_video_path):
            result_row = {
                'video': os.path.basename(dst_video_path), 
                'prompt': prompt,
                'input_image': os.path.basename(first_frame_path),
                'ref_mask': os.path.basename(dst_mask_path)
            }

    gc.collect()
    torch.cuda.empty_cache()
    
    return result_row

# Main processing logic
input_root = args.input_root
object_dirs = os.listdir(input_root)

# Step 1: Collect all tasks (object_id, joint_id, video_idx, prompt_idx, video_path, prompt)
all_tasks = []
for object_dir in object_dirs:
    object_id = object_dir
    object_path = os.path.join(input_root, object_dir)
    
    if args.use_color_direct:
        video_paths = glob.glob(os.path.join(object_path, "color_*.mp4"))
    else:
        #TODO: Fix the bugs
        video_paths = glob.glob(os.path.join(object_path, "depth_*.mp4"))
        #video_paths = glob.glob(os.path.join(object_path, "*.mp4"))
    
    # Load prompts
    prompt_dict = {}
    with open(os.path.join(object_path, "prompts.txt"), "r") as f:
        prompts = f.readlines()
    for p_ in prompts:
        p_split = p_.split("\t")
        p_split = [x for x in p_split if x != '']
        joint_id = int(p_split[0])
        if not args.simplified:
            prompt = p_.split("\t")[2].strip("\n")
        else:
            prompt = p_.split("\t")[3].strip("\n")
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
if args.dry_run_meta:
    # Use multiprocessing for dry run
    num_processes = args.num_processes or cpu_count()
    print(f"Using {num_processes} CPU processes for dry_run_meta")
    
    # Add temp_dir to args for multiprocessing
    if args.reorg:
        args.temp_dir = TEMP_DIR
    
    # Create partial function with fixed args
    process_func = partial(process_task_dry_run, args=args)
    
    # Process tasks in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_func, batch_tasks)
    
    # Collect results
    if args.reorg:
        for result_row in results:
            if result_row:
                df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)

elif args.use_color_direct:
    # Single-threaded processing for direct color video copying
    for task_data in batch_tasks:
        result_row = process_task_color_direct(task_data, args)
        
        if args.reorg and result_row:
            df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)

else:
    # Single-threaded processing for GPU-based tasks
    wan_vace = get_wan_model()
    
    for task_data in batch_tasks:
        result_row = process_task_full(task_data, args, wan_vace)
        
        if args.reorg and result_row:
            df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)

if args.reorg:
    df.to_csv(os.path.join(args.output_root, args.out_meta), index=False)