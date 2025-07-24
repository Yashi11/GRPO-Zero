# GRPO-Zero: Fact-Checking with DeepSeek-VL2

GRPO training with minimal dependencies for fact-checking tasks using vision-language models. This implementation supports both binary (TRUE/FALSE) and multiclass (TRUE/out-of-context/miscaptioned) classification for news verification using DeepSeek-VL2.

## Quick Start

For the easiest setup and training experience, use the provided setup script:

```bash
# Setup environment and run training in one command
./setup_and_run.sh && ./run_train.sh --config config_24GB.yaml
```

This script will:
- Initialize conda environment
- Install all dependencies including PyTorch and DeepSeek-VL2
- Create necessary directories
- Prepare the environment for training

## Data Requirements

For custom fact-checking tasks, you need to prepare your data directories:

```
GRPO-Zero/
├── filtered/          # Evidence data repositories
│   ├── [data_repo_1]/ # Your evidence data repositories go here
│   └── [data_repo_2]/
├── images/            # Query and evidence images
│   ├── query_*.jpg    # Query images for fact-checking
│   └── evidence_*.jpg # Evidence images for verification
└── ...
```

**Important**: Make sure to place your data repositories under the `filtered/` directory and all images (both query and evidence) under the `images/` directory before starting training.

## Switching to Multiclass Classification

The current implementation uses binary classification (TRUE/FALSE). To switch to multiclass classification with three categories ("TRUE", "out-of-context", "miscaptioned"), you need to modify `customdata_task.py`:

### 1. Update USER_TEMPLATE

Change the classification instructions in the `USER_TEMPLATE`:

```python
USER_TEMPLATE = (
    "You are a fact-checking assistant. Your task is to verify the authenticity of news using a reasoning-first approach. "
    "First, think through the evidence and rationale, then provide a clear final verdict."
    "A news story is presented as a query consisting of an image <image>\n and a caption <|ref|>({query_caption})<|/ref|>. "
    "Your task is to determine whether this news is 'TRUE', 'out-of-context', or 'miscaptioned'. "
    "You are provided with supporting evidence, including a set of images ({evidence_images}) and related text descriptions ({evidence_text}).\n\n"

    "Use all available evidence to assess whether the query image and caption align with the facts. "
    "Check for visual inconsistencies, mismatches in dates or events, and contradictions between the query and the evidence. "
    "- 'TRUE': The image and caption are factually accurate and properly matched\n"
    "- 'out-of-context': The image is real but used in wrong context or timeframe\n" 
    "- 'miscaptioned': The image is real but the caption is incorrect or misleading\n\n"
    "Your reasoning process must be written inside <think> </think> tags. Show how the visual and textual elements support or refute the claim.\n\n"

    "Finally, provide your verdict in <answer> </answer> tags.\n"
    "Example: <answer> out-of-context </answer>"
)
```

### 2. Update Answer Reward Function

Modify the `answer_reward_function` to handle three classes:

```python
def answer_reward_function(response: str, target: str = None) -> float:
    """
    Evaluates if the model's fact-checking verdict matches the target label.
    
    Args:
        response: Model's response containing <answer>CLASS</answer>
        target: Ground truth label ('TRUE', 'miscaptioned', or 'out-of-context')
    
    Returns:
        1.0 if verdict matches target exactly
        0.0 if no match or invalid format
    """
    answer_regex = r"<answer>\s*(TRUE|out-of-context|miscaptioned)\s*<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1).strip()
    if not answer_content:
        return 0.0

    # Direct exact match for multiclass
    if answer_content == target:
        return 1.0
    
    return 0.0
```

### 3. Update Format Reward Function (Optional)

You may also want to update the regex in `format_reward_function` to reflect the new answer format:

```python
def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    # Update the full format regex to include multiclass options
    full_format_regex = r"^<think>.*?<\/think>\n<answer>\s*(TRUE|out-of-context|miscaptioned)\s*<\/answer>$"
    # ... rest of the function remains the same
```

These changes will enable proper multiclass classification with appropriate reward calculation for each of the three categories.