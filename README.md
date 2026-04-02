# HeRoN: A Multi Agent RL-LLM Framework for Adaptive NPC Decision Making
Non-Player Characters (NPCs) play a central role in modern video games, in fluencing both immersion and narrative depth. However, traditional design approaches, from rule-based systems to utility-driven AI, often fail to produce adaptive and contextually coherent behaviors. Recent progress in Reinforcement Learning (RL) and Large Language Models (LLMs) has opened new opportunities for improving NPC decision-making, but both face key limitations: RL struggles with training efficiency and generalization, while LLMs are prone to hallucinations and context drift. In this work, we introduce HeRoN, a multi-agent architecture that integrates RL and LLMs to produce NPCs with more strategic and contextually relevant behaviors. HeRoN combines three components: (i) the NPC, an RL-driven agent whose policy is iteratively refined via LLM-generated critiques; (ii) the Helper, an LLM operating in zero-shot reasoning mode to generate diverse, context-aware action strategies; and (iii) the Reviewer, a lightweight, fine-tuned LLM that evaluates and refines the Helper’s suggestions, ensuring strategic consistency and alignment with game-specific constraints. We evaluate HeRoN in a custom turn-based battle environment, demonstrating superior performance over standard RL baselines in strategy refinement, learning efficiency, adaptability, and contextual decision-making.

## Motivation

Recent progress in RL and LLMs makes hybrid game AI systems increasingly feasible, but each approach still has practical limitations when used in isolation:

- RL can achieve strong policies, yet training may be sample-inefficient and sensitive to scenario changes.
- LLMs can reason over context, but they may hallucinate, drift from constraints, or generate inconsistent tactical suggestions.

HeRoN addresses this through mediated interaction: RL remains the policy backbone, while language models contribute strategic guidance and quality control.

## HeRoN Architecture

HeRoN separates responsibilities into three components:

- NPC: the learning agent (RL-based) that selects actions in the environment.
- Helper: a zero-shot LLM that proposes candidate strategies and contextual action ideas.
- Reviewer: a lightweight fine-tuned model that critiques and refines Helper outputs before they influence the NPC update loop.

In practice, this split improves coherence and stability compared to naive single-model integrations, because generation (Helper) and validation (Reviewer) are explicitly decoupled.

## Evaluation Environments

The project is designed for experiments in two domains:

- Custom turn-based battle environment (implemented in this repository).

## Reproducibility Guide

### 1. Requirements

Recommended:

- Python 3.10 or newer
- A clean virtual environment
- GPU optional (training can still run on CPU, with longer times)

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Project Structure (What to Edit)

The key folders for replication are:

- `classes/`: core game and agent logic.
	- `agent.py`: NPC implementation.
	- `environment.py`, `game.py`, `inventory.py`, `magic.py`: environment mechanics and battle rules.
- `HeRoN/`: main experiment configurations and training variants.
- `reviewer/`: Reviewer training and inference pipeline.
- `dataset Reviewer/`: scripts and resources for Reviewer dataset generation.
- `baseline helper/` and `baseline RL/`: reference baselines for comparison.

If you want to change game constraints described in the paper (actions, rewards, state design, combat rules), start from `classes/` first.

### 3. Configure the Helper LLM (LM Studio)

To run Helper-based experiments, configure a local or remote LLM endpoint via LM Studio.

1. Install and run [LM Studio](https://lmstudio.ai/).
2. Load the model you want to test for Helper.
3. In HeRoN training scripts, set:
	 - `SERVER_API_HOST` to your API endpoint.
	 - `model = client.llm.model("YOUR_LLM_NAME")` to the loaded model identifier.

Apply the same configuration consistently across the scripts you use for train/test so results remain comparable.

### 4. Train the Reviewer

Reviewer-related files are located in `reviewer/`. If you need a custom training set, generate it from `dataset Reviewer/`.

After training, reference your Reviewer checkpoint in the HeRoN scripts using:

```python
AutoTokenizer.from_pretrained("YOUR_MODEL_PATH")
T5ForConditionalGeneration.from_pretrained("YOUR_MODEL_PATH")
```

Use the same checkpoint for all runs in a comparison batch; mixing reviewer checkpoints across runs can make metrics hard to interpret.

### 5. Train the NPC

Training configurations are available in `HeRoN/` (for example final, initial, and random variants).

Before launching training:

1. Confirm Helper endpoint/model are correctly set.
2. Confirm Reviewer tokenizer/model paths are correctly set.
3. Update output names in plotting/export utilities (for example in `plot_training` and `export_success_rate`) to avoid overwriting previous experiments.

Implementation notes:

- `DQNAgent` is the NPC learner.
- `IntructorAgent` is the Reviewer-side interface used in the training loop.
- Trained NPC checkpoints are saved in Keras format.

### 6. Test a Trained NPC

For evaluation, run the testing script used by your experiment branch (for example, `old_results/testing_model.py` in this repository snapshot).

Before testing:

1. Set the checkpoint name/path of the trained NPC model.
2. Update plot/export names to keep outputs separated by run.
3. Reuse the same Helper/Reviewer settings used during training whenever you are measuring reproducibility.

## Practical Notes for Reliable Replication

- Keep a fixed seed policy across repeated runs when possible.
- Store every run with unique output names (JSON, CSV, figures, model files).
- Avoid changing environment rules and reward shaping mid-comparison.
- Track which Helper model and Reviewer checkpoint were used for each result file.
