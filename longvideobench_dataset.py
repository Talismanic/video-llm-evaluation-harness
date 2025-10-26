"""
    This Dataset is created for project LongVideoBench. 
    ===
    Paper: LongVideoBench: A Comprehensive Benchmark for Long Video Understanding
    url: https://arxiv.org/abs/2409.11422
    data: https://huggingface.co/datasets/longvideo-bench/LongVideoBench
    code: https://github.com/DAMO-NLP-SG/VideoLLaMA2/pull/32
    "LongVideoBench" includes two version: V1 and V2
    V1 version Annotation is in JSON format, which is used for training. 
    V2 version Annotation is in JSONL format, which is used for evaluation.
    V1 version contains 650 videos, V2 version contains 3745 videos.
    This dataset will automatically use the V2 version.
    ===
    New Dataset Stats: 
    24 Tasks and 6 Categories of understanding capabilities in 9 video domains with 3745 videos.
    6 Categories:
        1. Perception: find_detail, repeat_count, scene_transition, summary, subtle_action, object_interact
        2. Comprehension: attribute, bridging, conflict_resolve, recognition, misra, state_change, video_completion
        3. Reasoning: interview, why, future_event, rationale, scene_movement
        4. Creative: story_generate, ads, description_generate, next_qa
        5. Knowledge: social_norm, world_knowledge
        6. Hallucination: hallucination
===
    Data Splits:
    - test: 3745 videos, ~71GB total size

    ===
    Usage:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("longvideo-bench/LongVideoBench", "LongVideoBench")
    ```
"""

import json
import os
from typing import Dict, List, Any, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import decord
import numpy as np
import math


class LongVideoBenchDataCollator:
    """Collator for LongVideoBench dataset"""
    def __call__(self, features):
        return features


class LongVideoBenchDataset(Dataset):
    """
    PyTorch dataset for LongVideoBench
    """
    
    def __init__(
        self,
        data_path: str,
        video_dir: str,
        num_frames: int = 8,
        max_txt_len: int = 512,
        video_first: bool = True,
        video_only: bool = False,
        use_subtitles: bool = False,
        en_subtitles: bool = False,
        split: str = "test",
        video_processor=None,
    ):
        """
        Args:
            data_path: Path to the annotation JSON file
            video_dir: Root directory containing video files
            num_frames: Number of frames to sample from video
            max_txt_len: Maximum length of text input
            video_first: Whether video should be processed first
            video_only: Whether to process only video (no text)
            use_subtitles: Whether to use subtitle information
            en_subtitles: Whether to use English subtitles only
            split: Data split to use (train/test)
            video_processor: Function to process video frames
        """
        self.data_path = data_path
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.max_txt_len = max_txt_len
        self.video_first = video_first
        self.video_only = video_only
        self.use_subtitles = use_subtitles
        self.en_subtitles = en_subtitles
        self.split = split
        self.video_processor = video_processor
        
        # Load annotations
        with open(data_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter by split if provided
        if split and isinstance(self.annotations, list):
            if "dataset" in self.annotations[0]:
                self.annotations = [item for item in self.annotations if item.get("dataset", "").lower() == split.lower()]
        
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data after loading"""
        # Handle the case where annotations might be nested
        if isinstance(self.annotations, dict):
            self.annotations = self.annotations.get("data", self.annotations)
        
        # Ensure annotations is a list
        if not isinstance(self.annotations, list):
            self.annotations = [self.annotations]
        
        print(f"Loaded {len(self.annotations)} annotations for {self.split} split")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Extract basic information
        video_id = annotation.get("video", annotation.get("video_id", ""))
        question = annotation.get("question", annotation.get("caption", ""))
        answer = annotation.get("answer", annotation.get("response", ""))
        
        # Task information
        task = annotation.get("task", "unknown")
        category = annotation.get("category", "unknown")
        
        # Load video
        video_path = os.path.join(self.video_dir, video_id)
        if not os.path.exists(video_path):
            video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(self.video_dir, f"{video_id}.mkv")
        if not os.path.exists(video_path):
            video_path = os.path.join(self.video_dir, f"{video_id}.avi")
            
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            return self._create_empty_sample(annotation)
        
        # Load video frames
        video_data, video_time = self._load_video(video_path)
        
        # Process subtitles if requested
        subtitle_text = ""
        if self.use_subtitles:
            subtitle_text = self._get_subtitles(annotation)
        
        # Build conversation
        conversation = self._build_conversation(question, subtitle_text)
        
        sample = {
            "video_id": video_id,
            "video": video_data,
            "video_time": video_time,
            "question": question,
            "answer": answer,
            "conversation": conversation,
            "task": task,
            "category": category,
        }
        
        # Add raw annotation for reference
        sample["raw_annotation"] = annotation
        
        return sample
    
    def _load_video(self, video_path, num_frames=None):
        """Load video and sample frames"""
        if num_frames is None:
            num_frames = self.num_frames
            
        try:
            # Use decord to load video
            vr = decord.VideoReader(video_path)
            num_total_frames = len(vr)
            
            if num_total_frames <= num_frames:
                sampled_indices = list(range(num_total_frames))
            else:
                # Uniform sampling
                sampling_interval = num_total_frames / num_frames
                sampled_indices = [
                    int(i * sampling_interval) for i in range(num_frames)
                ]
            
            video_features = vr.get_batch(sampled_indices).asnumpy()
            video_features = torch.from_numpy(video_features)
            
            # Apply video processor if provided
            if self.video_processor:
                video_features = self.video_processor(video_features)
            
            # Calculate video duration
            video_fps = vr.get_avg_fps()
            video_time = num_total_frames / video_fps
            
            return video_features, video_time
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            return self._create_empty_video(), 0
    
    def _create_empty_video(self):
        """Create empty video tensor when loading fails"""
        return torch.zeros(self.num_frames, 3, 224, 224)  # Assuming standard ViT input size
    
    def _create_empty_sample(self, annotation):
        """Create empty sample when data is missing"""
        return {
            "video_id": annotation.get("video", "empty"),
            "video": self._create_empty_video(),
            "video_time": 0,
            "question": annotation.get("question", ""),
            "answer": annotation.get("answer", ""),
            "conversation": "",
            "task": annotation.get("task", "unknown"),
            "category": annotation.get("category", "unknown"),
            "raw_annotation": annotation,
        }
    
    def _get_subtitles(self, annotation):
        """Extract subtitle text from annotation"""
        subtitles = annotation.get("subtitles", "")
        if isinstance(subtitles, dict):
            if self.en_subtitles and "en" in subtitles:
                return subtitles["en"]
            elif "text" in subtitles:
                return subtitles["text"]
            else:
                return str(list(subtitles.values())[0]) if subtitles else ""
        return str(subtitles) if subtitles else ""
    
    def _build_conversation(self, question, subtitle_text=""):
        """Build conversation format text"""
        if self.video_only:
            return ""
        
        parts = []
        
        # Add video indication
        if not self.video_first:
            parts.append("<video>")
        
        # Add subtitles if available
        if subtitle_text and len(subtitle_text.strip()) > 0:
            parts.append(f"Subtitles: {subtitle_text}")
        
        # Add question
        parts.append(question)
        
        # Join conversation
        conv_text = " ".join(parts)
        
        # Truncate if too long
        if len(conv_text) > self.max_txt_len:
            conv_text = conv_text[:self.max_txt_len-3] + "..."
            
        return conv_text


class LongVideoBenchDataModule:
    """
    Data module for LongVideoBench dataset for PyTorch Lightning
    """
    
    def __init__(
        self,
        data_path: str,
        video_dir: str,
        num_frames: int = 8,
        batch_size: int = 1,
        max_txt_len: int = 512,
        num_workers: int = 4,
        video_processor=None,
    ):
        self.save_hyperparameters()
        self.data_path = data_path
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.max_txt_len = max_txt_len
        self.num_workers = num_workers
        self.video_processor = video_processor
    
    def setup(self, stage=None):
        """Setup dataset for training/testing"""
        if stage == "fit" or stage is None:
            self.test_dataset = LongVideoBenchDataset(
                data_path=self.data_path,
                video_dir=self.video_dir,
                num_frames=self.num_frames,
                max_txt_len=self.max_txt_len,
                video_processor=self.video_processor,
                split="test",
            )
    
    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=LongVideoBenchDataCollator(),
        )


def load_longvideobench_dataset(
    data_path: str = None,
    video_dir: str = None,
    split: str = "test",
    num_frames: int = 8,
    **kwargs
):
    """
    Simple wrapper to load LongVideoBench dataset
    """
    if data_path is None:
        # Use Hugging Face dataset path by default
        from datasets import load_dataset
        dataset = load_dataset("longvideo-bench/LongVideoBench", "LongVideoBench", split=split)
        return dataset
    else:
        dataset = LongVideoBenchDataset(
            data_path=data_path,
            video_dir=video_dir,
            num_frames=num_frames,
            split=split,
            **kwargs
        )
        return dataset