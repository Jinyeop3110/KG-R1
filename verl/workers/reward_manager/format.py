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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import qa_em_format


def _select_rm_score_fn(data_source):
    """Select the appropriate reward scoring function based on data source."""
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa', 'webqsp', 'cwq', 'kgR1_webqsp', 'kgR1_cwq']:
        return qa_em_format.compute_score_em
    else:
        raise NotImplementedError(f"Data source '{data_source}' not supported for format reward manager")


class FormatRewardManager:
    """The format-aware reward manager with support for structured scoring."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0.,
                 otc_scaling=None, max_turns=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        
        # Format-specific scoring parameters
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        
        # Ignore PPO-specific parameters that are not used by FormatRewardManager
        # otc_scaling and max_turns are passed from default config but not needed here
        self.retrieval_score = retrieval_score
        self.format_score = format_score

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards with format-aware scoring."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # Convert to long integers if they're floats (can happen with search generation)
            if sequences.dtype != torch.long:
                sequences = sequences.long()
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # select rm_score function based on data source
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            # Use custom compute_score function if provided, otherwise use format-specific scoring
            if self.compute_score is not None:
                extra_info = data_item.non_tensor_batch.get("extra_info", None)
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            else:
                # Use the format-specific scoring function
                compute_score_fn = _select_rm_score_fn(data_source)
                score = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth,
                    structure_format_score=self.structure_format_score,
                    final_format_score=self.final_format_score,
                    retrieval_score=self.retrieval_score,
                    format_score=self.format_score
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[sequences]", sequences_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
