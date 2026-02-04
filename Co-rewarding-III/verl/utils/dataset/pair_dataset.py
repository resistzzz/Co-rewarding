from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import torch
import copy
import os
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig, ListConfig
import datasets
import logging
import re
import random

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)



def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def pair_collate_fn(batch: list[dict]) -> dict:
    """
    Collate a batch of pairs: each item is {"ori": {...}, "aug": {...}}
    Returns:
        {
            "ori": batched_ori_dict,
            "aug": batched_aug_dict,
        }
    """
    ori_batch = [item["ori"] for item in batch]
    aug_batch = [item["aug"] for item in batch]
    return {
        "ori": collate_fn(ori_batch),  # 调用你原来的collate_fn
        "aug": collate_fn(aug_batch),
    }


class PairAugmentationDatasetV1(Dataset):
    def __init__(
        self,
        ori_data_files: Union[str, List[str]],
        aug_data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(ori_data_files, (List, ListConfig)):
            ori_data_files = [ori_data_files]
        if not isinstance(aug_data_files, (List, ListConfig)):
            aug_data_files = [aug_data_files]

        self.ori_data_files = copy.deepcopy(ori_data_files)
        self.aug_data_files = copy.deepcopy(aug_data_files)
        self.original_ori_data_files = copy.deepcopy(ori_data_files)
        self.original_aug_data_files = copy.deepcopy(aug_data_files)
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # 新增
        self.p = config.get("p", 0.5)
        seed = config.get("seed", 54)
        self._rng = random.Random(seed) if seed is not None else random.Random()
        print(f"PairDataset: p={self.p}, seed={seed}")
        
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
    
    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        ori_data_files = self.ori_data_files if not use_origin_parquet else self.original_ori_data_files
        for i, parquet_file in enumerate(ori_data_files):
            self.ori_data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)
        
        aug_data_files = self.aug_data_files if not use_origin_parquet else self.original_aug_data_files
        for i, parquet_file in enumerate(aug_data_files):
            self.aug_data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)
    
    def _read_files_and_tokenize(self):
        # 加载原始和增强数据集
        ori_dataframes = []
        for parquet_file in self.ori_data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            ori_dataframes.append(dataframe)
        ori_dataframe = datasets.concatenate_datasets(ori_dataframes)

        aug_dataframes = []
        for parquet_file in self.aug_data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            aug_dataframes.append(dataframe)
        aug_dataframe = datasets.concatenate_datasets(aug_dataframes)

        assert len(ori_dataframe) == len(aug_dataframe), f"The length is not consistency for raw data: {len(ori_dataframe)} vs {len(aug_dataframe)}"
        print(f"PairAugmentationDataset: total pairs = {len(ori_dataframe)}")

        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        def is_short(prompt):
            return len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True)) <= self.max_prompt_length

        # filter out too long prompts
        if self.filter_overlong_prompts:
            keep_indices = []
            for i in range(len(ori_dataframe)):
                ori_prompt = ori_dataframe[i][prompt_key]
                aug_prompt = aug_dataframe[i][prompt_key]
                if is_short(ori_prompt) and is_short(aug_prompt):
                    keep_indices.append(i)
            # breakpoint()
            print(f"PairAugmentationDataset: filtered valid pairs = {len(keep_indices)} / {len(ori_dataframe)}")
            ori_dataframe = ori_dataframe.select(keep_indices)
            aug_dataframe = aug_dataframe.select(keep_indices)
            assert len(ori_dataframe) == len(aug_dataframe), f"The length is not consistency after filtering overlong: {len(ori_dataframe)} vs {len(aug_dataframe)}"

        self.ori_dataframe = ori_dataframe
        self.aug_dataframe = aug_dataframe
        self.keep_indices = keep_indices
    
    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_ori_data_files") and not hasattr(self, "original_aug_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")


    def __len__(self):
        return len(self.ori_dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def _single_item_process(self, row_dict: dict):
        row_dict = copy.deepcopy(row_dict)
        messages = self._build_messages(row_dict)
        model_inputs = {}
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images
            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]
            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )           
        
        # position_ids
        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
     
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict.get("data_source", "N/A"))
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict    

    def __getitem__(self, idx):
        return {
            "ori": self._single_item_process(self.ori_dataframe[idx]),
            "aug": self._single_item_process(self.aug_dataframe[idx])
        }

    def __getstate__(self):
        """
        For multi-process DataLoader. Remove large/unsaveable objects.
        """
        state = self.__dict__.copy()
        # HuggingFace datasets对象不可序列化，自动去除，恢复时再加载
        for key in ["ori_dataframe", "aug_dataframe"]:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        """
        Called on unpickling; reload datasets.
        """
        self.__dict__.update(state)
        self._read_files_and_tokenize()