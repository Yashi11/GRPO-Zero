import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset
from deepseek_vl2.utils.io import load_pil_images

from data_types import MiniBatch

USER_TEMPLATE = (
    "You are a fact-checking assistant. Your task is to verify the authenticity of news using a reasoning-first approach. "
    "First, think through the evidence and rationale, then provide a clear final verdict."
    "A news story is presented as a query consisting of an image <image>\n and a caption <|ref|>({query_caption})<|/ref|>. "
    "Your task is to determine whether this news is factual ('TRUE') or fake ('FALSE'). "
    "You are provided with supporting evidence, including a set of images ({evidence_images}) and related text descriptions ({evidence_text}).\n\n"

    "Use all available evidence to assess whether the query image and caption align with the facts. "
    "Check for visual inconsistencies, mismatches in dates or events, and contradictions between the query and the evidence. "
    "Your reasoning process must be written inside <think> </think> tags. Show how the visual and textual elements support or refute the claim.\n\n"

    "Finally, provide your verdict in <answer> </answer> tags.\n"
    "Example: <answer> FALSE </answer>"
)

RESPONSE_PROMPT = "Let me analyze this step by step.\n<think>"

class CustomDataset(Dataset):
    """Prepare Countdown Tasks for training"""

    def __init__(
        self,
        vl_processor,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_csv(data_path)
        data = data[["Evidence_Summary", "Query_Image", "Query_Caption", "Evidence_Image", "Target"]]
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.vl_processor = vl_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        # Process inputs for the model
        processed = self.encode_prefix(
            query_image=item["Query_Image"],
            query_caption=item["Query_Caption"],
            evidence_images=item["Evidence_Image"],
            evidence_text=item["Evidence_Summary"]
        )
        # Add target separately for reward calculation only
        processed["target"] = item["Target"]
        return processed

    def encode_prefix(self, query_image: str, query_caption: str, evidence_images: str, evidence_text: str):
        """Prefix is the *actual* input to the model."""
        # Convert string representation of evidence images list to actual list
        evidence_image_list = ast.literal_eval(evidence_images)
        
        # Prepare images list with query image first
        images = [query_image] + evidence_image_list
        
        # Format evidence images placeholder text
        evidence_images_text = ""
        for _ in range(len(evidence_image_list)):
            evidence_images_text += "<image>"
        
        # Prepare conversation format for DeepSeek VL2
        conversation = [
            {
                "role": "<|User|>",
                "content": USER_TEMPLATE.format(
                    query_caption=query_caption,
                    evidence_images=evidence_images_text,
                    evidence_text=evidence_text
                ),
                "images": images,
            },
            {"role": "<|Assistant|>", "content": RESPONSE_PROMPT},
        ]

        # Load and prepare images - keep them as PIL images
        pil_images = load_pil_images(conversation)
        
        # Let the processor handle image transformations
        inputs = self.vl_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""  # Empty system prompt as it's not supported
        )
        
        return {
            "inputs": inputs,
            "conversation": conversation,
            "images": images
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        return MiniBatch(
            inputs=[item["inputs"] for item in batch],
            conversations=[item["conversation"] for item in batch],
            images=[item["images"] for item in batch],
            targets=[item["target"] for item in batch]  # Add targets to batch
        )


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward


def answer_reward_function(
    response: str, target: str = None
) -> float:
    """
    Evaluates if the model's fact-checking verdict matches the target label.
    
    Args:
        response: Model's response containing <answer>TRUE</answer> or <answer>FALSE</answer>
        target: Ground truth label ('TRUE', 'miscaptioned', or 'out-of-context')
    
    Returns:
        1.0 if verdict matches target (TRUE for TRUE, FALSE for miscaptioned/out-of-context)
        0.0 if no match or invalid format
    """
    answer_regex = r"<answer>\s*(TRUE|FALSE)\s*<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1).strip()
    if not answer_content:
        return 0.0

    # Convert target to expected answer
    expected_answer = "TRUE" if target == "TRUE" else "FALSE"
    
    # Check if model's answer matches expected answer
    if answer_content == expected_answer:
        return 1.0
    
    return 0.0


def reward_function(
    response: str,
    target: str = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for fact-checking.

    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function(response, end_token)
    answer_reward = answer_reward_function(response, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
