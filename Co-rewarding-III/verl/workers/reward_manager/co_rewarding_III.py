

from collections import defaultdict, Counter

import torch
from typing import Optional, Any

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


@register("co_rewarding_III")
class CoRewardingIII(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

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
    
    def _extract_valid_ref_response_str(self, item):
        # 从 item 提取有效 response 的 tokenizer.decode 字符串
        prompt_ids = item.batch["ref_prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = item.batch["ref_attention_mask"][:prompt_length].sum()
        response_ids = item.batch["ref_responses"]
        valid_response_length = item.batch["ref_attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return response_str

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

    def __call__(self, data_ori: DataProto, data_aug: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        if "ref_responses" not in data_ori.batch.keys():
            raise ValueError("ref_responses is not in data_ori")
        if "ref_responses" not in data_aug.batch.keys():
            raise ValueError("ref_responses is not in data_aug")

        #  1. Extract answers
        ori_uid2answers = defaultdict(list)
        ori_uid2policy_answers = defaultdict(list)
        ori_all_policy_answers = []
        for i, item in enumerate(data_ori):
            uid = item.non_tensor_batch["uid"]
            ans = self.extract_answer(self._extract_valid_ref_response_str(item))
            ori_uid2answers[uid].append(ans)
            
            ans_policy = self.extract_answer(self._extract_valid_response_str(item))
            ori_uid2policy_answers[uid].append(ans_policy)
            ori_all_policy_answers.append(ans_policy)
        
        aug_uid2answers = defaultdict(list)
        aug_uid2policy_answers = defaultdict(list)
        aug_all_policy_answers = []
        for i, item in enumerate(data_aug):
            uid = item.non_tensor_batch["uid"]
            ans = self.extract_answer(self._extract_valid_ref_response_str(item))
            aug_uid2answers[uid].append(ans)
            
            ans_policy = self.extract_answer(self._extract_valid_response_str(item))
            aug_uid2policy_answers[uid].append(ans_policy)
            aug_all_policy_answers.append(ans_policy)
        
        # check uid
        # Check if uid sets are consistent
        ori_uids = set(ori_uid2answers.keys())
        aug_uids = set(aug_uid2answers.keys())
        assert ori_uids == aug_uids, (
            f"UID mismatch between ori_uid2pseudo and aug_uid2pseudo.\n"
            f"Missing in ori: {aug_uids - ori_uids}\n"
            f"Missing in aug: {ori_uids - aug_uids}"
        )
        
        # 2. majority voting for extracting pseudo labels, cross agreement
        ori_uid2pseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in aug_uid2answers.items()}
        aug_uid2pseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in ori_uid2answers.items()}
        
        ori_uid2policypseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in aug_uid2policy_answers.items()}
        aug_uid2policypseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in ori_uid2policy_answers.items()}

        # 3. obtain reward tensor
        reward_tensor_ori = torch.zeros_like(data_ori.batch["responses"], dtype=torch.float32)
        reward_tensor_aug = torch.zeros_like(data_aug.batch["responses"], dtype=torch.float32)
        reward_extra_info_ori, reward_extra_info_aug = {}, {}
        
        ori_pseudo_nonempty, aug_pseudo_nonempty = 0, 0
        ori_gt_empty, aug_gt_empty = 0, 0
        ori_pseudo_correct, aug_pseudo_correct = 0, 0
        ori_policypseudo_correct, aug_policypseudo_correct = 0, 0
        ori_pseudo_pass, aug_pseudo_pass = 0, 0
        
        N = len(data_ori)
        for i in range(N):
            # for data_ori
            item = data_ori[i]
            ans = ori_all_policy_answers[i]
            pseudo_label = ori_uid2pseudo[item.non_tensor_batch["uid"]]
            pseudo_policy_label = ori_uid2policypseudo[item.non_tensor_batch["uid"]]
            if pseudo_label != '':
                ori_pseudo_nonempty += 1
            rm = item.non_tensor_batch.get("reward_model", {})
            gt = rm.get("ground_truth", None)
            if gt is not None and str(gt).strip() != '':
                if _answers_equal(gt, pseudo_label):
                    ori_pseudo_correct += 1
                if _answers_equal(gt, pseudo_policy_label):
                    ori_policypseudo_correct += 1
            else:
                ori_gt_empty += 1
                
            valid_response_length = item.batch["attention_mask"][item.batch["prompts"].shape[-1]:].sum().item()
            
            reward = 0.0
            if pseudo_label != '':
                if ans == pseudo_label:
                    ori_pseudo_pass += 1
                    reward = 1.0
            
            if valid_response_length > 0:
                reward_tensor_ori[i, valid_response_length - 1] = reward
            
            # for data_aug
            item = data_aug[i]
            ans = aug_all_policy_answers[i]
            pseudo_label = aug_uid2pseudo[item.non_tensor_batch["uid"]]
            pseudo_policy_label = aug_uid2policypseudo[item.non_tensor_batch["uid"]]
            if pseudo_label != '':
                aug_pseudo_nonempty += 1
            rm = item.non_tensor_batch.get("reward_model", {})
            gt = rm.get("ground_truth", None)
            if gt is not None and str(gt).strip() != '':
                if _answers_equal(gt, pseudo_label):
                    aug_pseudo_correct += 1
                if _answers_equal(gt, pseudo_policy_label):
                    aug_policypseudo_correct += 1
            else:
                aug_gt_empty += 1
                
            valid_response_length = item.batch["attention_mask"][item.batch["prompts"].shape[-1]:].sum().item()
            
            reward = 0.0
            if pseudo_label != '':
                if ans == pseudo_label:
                    aug_pseudo_pass += 1
                    reward = 1.0
            
            if valid_response_length > 0:
                reward_tensor_aug[i, valid_response_length - 1] = reward 
                
            
        ori_pseudo_nonempty_rate = ori_pseudo_nonempty / N
        aug_pseudo_nonempty_rate = aug_pseudo_nonempty / N
        
        ori_pseudo_acc = ori_pseudo_correct / N
        aug_pseudo_acc = aug_pseudo_correct / N
        ori_policypseudo_acc = ori_policypseudo_correct / N
        aug_policypseudo_acc = aug_policypseudo_correct / N

        ori_pseudo_pass_rate = ori_pseudo_pass / N
        aug_pseudo_pass_rate = aug_pseudo_pass / N

        metrics = {
            "coreward/ori_pseudo_nonempty_rate": ori_pseudo_nonempty_rate,
            "coreward/aug_pseudo_nonempty_rate": aug_pseudo_nonempty_rate,
            "coreward/ori_pseudo_count": ori_pseudo_correct,
            "coreward/ori_pseudo_count": aug_pseudo_correct,
            "coreward/ori_pseudo_acc": ori_pseudo_acc,
            "coreward/aug_pseudo_acc": aug_pseudo_acc,
            "coreward/ori_policypseudo_acc": ori_policypseudo_acc,
            "coreward/aug_policypseudo_acc": aug_policypseudo_acc,
            "coreward/ori_pseudo_pass_rate": ori_pseudo_pass_rate,
            "coreward/aug_pseudo_pass_rate": aug_pseudo_pass_rate,
        }

        if return_dict:
            return {
                "reward_tensor_ori": reward_tensor_ori,
                "reward_tensor_aug": reward_tensor_aug,
                "reward_extra_info": {"metrics": metrics},
            }
        else:
            return reward_tensor_ori, reward_tensor_aug

