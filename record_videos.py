#!/usr/bin/env python3
"""
Record gameplay videos of trained and random agents playing Atari Enduro.

This script records videos of a random agent and a trained DQN agent playing Enduro.
Bulk mode generates videos without overlays for data collection.

Usage:
    # Record bulk videos (default: 10 videos, 50% random)
    python record_videos.py --bulk

    # Record 30 bulk videos, 30% random
    python record_videos.py --bulk --total_videos 30 --percent_random 30

    # Use specific checkpoint
    python record_videos.py --bulk --skill_level 1

    # Save to different directory
    python record_videos.py --bulk --output_dir /scratch/users/axb2032/videos

    # Enable debug mode
    python record_videos.py --bulk --debug

    # Adjust action randomness
    python record_videos.py --bulk --temperature 0.5

Arguments:
    --bulk: Enable bulk video generation mode (no overlays)
    --total_videos: Total number of videos to generate in bulk mode (default: 10)
    --percent_random: Percentage of random agent videos in bulk mode (0-100, default: 50)
    --output_dir: Directory to save videos (default: 'videos')
    --skill_level: Skill level checkpoint to use (0-3). If none, uses latest checkpoint.
    --no_sync: Use a temporary directory to avoid sync issues (default: False)
    --temperature: Softmax temperature for action selection (default: None, uses greedy/random)
    --debug: Enable debug logging for troubleshooting (default: False)

Output:
    - Bulk random agent videos: bulk_random_agent_X.mp4
    - Bulk trained agent videos: bulk_trained_agent_[checkpoint]_X.mp4
    - PNG frame backups: bulk_[agent]_[checkpoint]_X_frames/frame_XXXXX.png
    where [checkpoint] is 'latest' or 'skill_N' depending on --skill_level
"""

import os
import cv2
import numpy as np
import torch
import argparse
import tempfile
import shutil
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Disable Dynamo to avoid JIT errors
from atari_env import AtariEnduroEnv
from dqn_agent import DQNAgent
from random_agent import RandomAgent

def record_episode_no_overlay(env, agent, video_writer, temperature=None, frame_size=None, debug=False, png_dir=None):
    """Record a single episode without text overlays for bulk mode."""
    obs, info = env.reset()
    # Initialize state stack with preprocessed observations for DQN
    if agent.__class__.__name__ == 'DQNAgent':
        preprocessed_obs = env.preprocess_observation(obs)
        state_stack = np.stack([preprocessed_obs] * 8, axis=0)  # (8, 84, 84)
    else:
        state_stack = np.stack([obs] * 8, axis=0)  # (8, 210, 160, 3) for RandomAgent
    frame_num = 0
    max_steps = 10000  # Match train_dqn.py max_steps
    frames_written = 0

    while True:
        if frame_num > max_steps:
            if debug:
                print(f"Episode terminated at max_steps={max_steps}")
            break

        # Get raw RGB frame for video
        try:
            raw_frame = env.env.render()
            if debug and frame_num % 100 == 0:
                print(f"Frame {frame_num} shape: {raw_frame.shape if raw_frame is not None else 'None'}")
        except Exception as e:
            if debug:
                print(f"Error rendering frame {frame_num}: {e}")
            raw_frame = None

        if raw_frame is None or raw_frame.size == 0:
            if debug:
                print(f"Warning: Empty frame at step {frame_num}. Using dummy frame.")
            raw_frame = np.zeros((210, 160, 3), dtype=np.uint8)

        # Resize frame if needed
        if frame_size and hasattr(video_writer, 'write'):
            expected_width, expected_height = frame_size
            actual_height, actual_width = raw_frame.shape[:2]
            if actual_width != expected_width or actual_height != expected_height:
                if expected_width > 0 and expected_height > 0:
                    raw_frame = cv2.resize(raw_frame, (expected_width, expected_height))
                    if debug:
                        print(f"Resized frame to {frame_size}")

        # Write frame to video
        try:
            video_writer.write(raw_frame)
            frames_written += 1
        except Exception as e:
            if debug:
                print(f"Error writing frame {frame_num}: {e}")

        # Save frame as PNG
        if png_dir:
            png_path = os.path.join(png_dir, f"frame_{frame_num:05d}.png")
            cv2.imwrite(png_path, raw_frame)

        # Select action
        if hasattr(agent, 'select_action'):
            if agent.__class__.__name__ == 'DQNAgent':
                action = agent.select_action(state_stack, mode='greedy' if temperature is None else 'softmax', temperature=temperature)
            else:
                action = agent.select_action(temperature=temperature)
        else:
            action = agent.select_action()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        frame_num += 1
        if agent.__class__.__name__ == 'DQNAgent':
            # Update state stack with preprocessed observation
            next_preprocessed_obs = env.preprocess_observation(next_obs)
            state_stack = np.roll(state_stack, shift=-1, axis=0)
            state_stack[-1] = next_preprocessed_obs
        if terminated or truncated:
            try:
                raw_frame = env.env.render()
                if raw_frame is not None and raw_frame.size > 0:
                    if frame_size:
                        actual_height, actual_width = raw_frame.shape[:2]
                        if actual_width != expected_width or actual_height != expected_height:
                            raw_frame = cv2.resize(raw_frame, (expected_width, expected_height))
                    video_writer.write(raw_frame)
                    frames_written += 1
                    if png_dir:
                        png_path = os.path.join(png_dir, f"frame_{frame_num:05d}.png")
                        cv2.imwrite(png_path, raw_frame)
                if debug:
                    print(f"Final frame shape: {raw_frame.shape if raw_frame is not None else 'None'}")
            except Exception as e:
                if debug:
                    print(f"Error rendering final frame: {e}")
            break

    if debug:
        print(f"Episode completed - wrote {frames_written} frames")
    return frames_written

def record_bulk_videos(
    total_videos=10,
    output_dir='videos',
    percent_random=50,
    skill_level=None,
    no_sync=False,
    temperature=None,
    debug=False
):
    """Generate bulk videos without overlays for Enduro."""
    # Setup temporary directory if no_sync
    temp_dir = None
    working_dir = output_dir
    if no_sync:
        temp_dir = tempfile.mkdtemp()
        working_dir = temp_dir
        print(f"Using temporary directory: {temp_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize environment (return_rgb=True for raw frames)
    env = AtariEnduroEnv(return_rgb=True)
    env.reset()
    try:
        test_frame = env.env.render()
        frame_size = (test_frame.shape[1], test_frame.shape[0]) if test_frame is not None else (160, 210)
        if debug:
            print(f"Test frame shape: {test_frame.shape if test_frame is not None else 'None'}")
    except Exception as e:
        if debug:
            print(f"Error rendering test frame: {e}")
        frame_size = (160, 210)
    print(f"[Bulk] Using frame size: {frame_size}")

    # Initialize agents
    random_agent = RandomAgent(n_actions=9)  # Enduro has 9 actions
    dqn_agent = DQNAgent(n_actions=9, state_shape=(8, 84, 84), prioritized=True, per_alpha=0.6, per_beta=0.4)

    # Load checkpoint
    checkpoint_name = 'latest'
    if skill_level is not None:
        model_path = os.path.join('/scratch/users/axb2032/world_model/checkpoints', f'dqn_skill_{skill_level}.pth')
        checkpoint_name = f'skill_{skill_level}'
    else:
        model_path = os.path.join('/scratch/users/axb2032/world_model/checkpoints', 'dqn_latest.pth')
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            # Strip '_orig_mod.' prefix from state_dict keys
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
            dqn_agent.policy_net.load_state_dict(state_dict)
            dqn_agent.target_net.load_state_dict(state_dict)
            dqn_agent.policy_net.eval()
            print(f"Model loaded from {model_path}. Device: {next(dqn_agent.policy_net.parameters()).device}")
        else:
            print(f"Warning: No checkpoint found at {model_path}. Using untrained model for trained agent videos.")
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}. Using untrained model for trained agent videos.")

    # Select codec
    fps = 30
    codecs = [('MJPG', '.avi'), ('XVID', '.avi'), ('mp4v', '.mp4')]
    working_codec = None
    for codec, ext in codecs:
        try:
            test_path = os.path.join(working_dir, f'test{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            test_writer = cv2.VideoWriter(test_path, fourcc, fps, frame_size)
            if not test_writer.isOpened():
                if debug:
                    print(f"Codec {codec} failed to open.")
                continue
            dummy_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            test_writer.write(dummy_frame)
            test_writer.release()
            if os.path.exists(test_path) and os.path.getsize(test_path) > 1000:
                working_codec = (codec, ext)
                os.remove(test_path)
                if debug:
                    print(f"Codec {codec} succeeded.")
                break
            if os.path.exists(test_path):
                os.remove(test_path)
        except Exception as e:
            if debug:
                print(f"Codec {codec} failed: {e}")
    if working_codec is None:
        print("Warning: No working codec found. Using MJPG as fallback.")
        working_codec = ('MJPG', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*working_codec[0])
    file_ext = working_codec[1]
    print(f"Using codec: {working_codec[0]} with extension {file_ext}")

    # Calculate video split
    num_random = int(total_videos * percent_random / 100)
    num_trained = total_videos - num_random
    print(f"[Bulk] Generating {num_random} random agent videos and {num_trained} trained agent videos.")

    # Record random agent videos
    for i in range(num_random):
        video_path = os.path.join(working_dir, f'bulk_random_agent_{i+1}{file_ext}')
        png_dir = os.path.join(working_dir, f'bulk_random_agent_{i+1}_frames')
        os.makedirs(png_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {video_path}")
            video_writer = None
        frames_written = record_episode_no_overlay(
            env, random_agent, video_writer, temperature=temperature,
            frame_size=frame_size, debug=debug, png_dir=png_dir
        )
        if video_writer:
            video_writer.release()
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        print(f"[Bulk] Random agent video {i+1} - Frames: {frames_written}, Size: {file_size} bytes, Frames dir: {png_dir}")

    # Record trained agent videos
    for i in range(num_trained):
        video_path = os.path.join(working_dir, f'bulk_trained_agent_{checkpoint_name}_{i+1}{file_ext}')
        png_dir = os.path.join(working_dir, f'bulk_trained_agent_{checkpoint_name}_{i+1}_frames')
        os.makedirs(png_dir, exist_ok=True)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {video_path}")
            video_writer = None
        frames_written = record_episode_no_overlay(
            env, dqn_agent, video_writer, temperature=temperature,
            frame_size=frame_size, debug=debug, png_dir=png_dir
        )
        if video_writer:
            video_writer.release()
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        print(f"[Bulk] Trained agent video {i+1} - Frames: {frames_written}, Size: {file_size} bytes, Frames dir: {png_dir}")

    env.close()

    # Copy files if no_sync
    if no_sync:
        print(f"Copying files to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(temp_dir):
            src_path = os.path.join(temp_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Copied directory: {filename}")
            elif os.path.getsize(src_path) > 1000:
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {filename} ({os.path.getsize(dst_path)} bytes)")
            else:
                print(f"Skipped: {filename}")
        shutil.rmtree(temp_dir)
    print("[Bulk] All videos generated successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record gameplay videos for Atari Enduro')
    parser.add_argument('--bulk', action='store_true', help='Enable bulk video generation without overlays')
    parser.add_argument('--total_videos', type=int, default=10, help='Total number of videos in bulk mode')
    parser.add_argument('--percent_random', type=float, default=50, help='Percentage of random agent videos (0-100)')
    parser.add_argument('--output_dir', type=str, default='videos', help='Directory to save videos')
    parser.add_argument('--skill_level', type=int, choices=[0, 1, 2, 3], help='Skill level checkpoint (0-3)')
    parser.add_argument('--no_sync', action='store_true', help='Use temporary directory')
    parser.add_argument('--temperature', type=float, default=None, help='Softmax temperature for action selection')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.bulk:
        record_bulk_videos(
            total_videos=args.total_videos,
            output_dir=args.output_dir,
            percent_random=args.percent_random,
            skill_level=args.skill_level,
            no_sync=args.no_sync,
            temperature=args.temperature,
            debug=args.debug
        )
    else:
        print("Error: Only bulk mode is supported. Use --bulk flag.")
