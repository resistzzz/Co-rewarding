# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, Counter

import torch
from typing import Optional
import random

from verl import DataProto
from verl.utils.reward_score import default_compute_score

from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def majority_vote(ans_list, empty_value=''):
    """
    从答案列表中投票选出最多的那个答案。
    如果全是空/None，则返回 empty_value（可以指定为 0、''、'[NO_ANSWER]'等）。
    """
    ans_list = [a for a in ans_list if a is not None and str(a).strip() != '']
    if not ans_list:
        return empty_value  # 或 return 0, 或 return '[NO_ANSWER]'
    return Counter(ans_list).most_common(1)[0][0]


def _answers_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    sa, sb = str(a).strip(), str(b).strip()
    if sa == '' or sb == '':
        return False
    return sa == sb

@register("majority_voting")
class MajorityVotingManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        print(f"++++++Loaded Majority Voting reward manager")

    def _extract_valid_response_str(self, item):
        # 从 item 提取有效 response 的 tokenizer.decode 字符串
        prompt_ids = item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = item.batch["attention_mask"][:prompt_length].sum()
        response_ids = item.batch["responses"]
        valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return response_str

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # breakpoint()
        uid2answers = defaultdict(list)
        all_answers = []
        for i, item in enumerate(data):
            uid = item.non_tensor_batch["uid"]
            ans = self.extract_answer(self._extract_valid_response_str(item))
            uid2answers[uid].append(ans)
            all_answers.append(ans)
        
        # 2. 做 majority vote，得到每个 uid 的 pseudo label
        uid2pseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in uid2answers.items()}
 
        # 3. 写 reward tensor
        reward_extra_info = {}
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # ori 用 aug 的 pseudo label 计算 reward
        # 统计指标
        pseudo_nonempty = 0

        # pseudo label vs gt
        gt_total_pseudo = 0
        gt_pseudo_correct = 0
        gt_empty = 0

        # reward通过率
        pass_response = 0
        valid_response = 0
        
        N = len(data) 
        for i in range(N):
            item = data[i]
            ans = all_answers[i]
            pseudo_label = uid2pseudo[item.non_tensor_batch["uid"]]
            if pseudo_label != '':
                pseudo_nonempty += 1
            rm = item.non_tensor_batch.get("reward_model", {})
            gt = rm.get("ground_truth", None)
            gt_total_pseudo += 1
            if gt is not None and str(gt).strip() != '':
                if _answers_equal(gt, pseudo_label):
                    gt_pseudo_correct += 1
            else:
                gt_empty += 1
                
            valid_response_length = item.batch["attention_mask"][item.batch["prompts"].shape[-1]:].sum().item()
            reward = 0.0
            if pseudo_label != '':
                valid_response += 1
                if ans == pseudo_label:
                    pass_response += 1
                    reward = 1.0
                    reward_tensor[i, valid_response_length - 1] = reward

        pseudo_nonempty_rate = pseudo_nonempty / N
        pseudo_acc = gt_pseudo_correct / gt_total_pseudo
        pass_acc = pass_response / valid_response

        metrics = {
            "coreward/pseudo_nonempty_rate": pseudo_nonempty_rate,
            "coreward/gt_empty": gt_empty,
            "coreward/pseudo_equal_gt": gt_pseudo_correct,
            "coreward/pseudo_acc": pseudo_acc,
            "coreward/pass_response": pass_response,
            "coreward/valid_response": valid_response,
            "coreward/pass_acc": pass_acc,
        }
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {'metrics': metrics},
            }
        else:
            return reward_tensor

    def extract_answer(self, solution_str: str):
        from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
        answer = ''
        try:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
        except Exception as e:
            print(e)
        return answer