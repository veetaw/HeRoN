import re
import json
import torch
import pandas as pd
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer
from tqdm import tqdm


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# --------------------------------------------------------
# REWARD FUNCTION (tuo codice, corretto solo dove serviva)
# --------------------------------------------------------
def calculate_reward(ideal_instructions, suggested_action):
    print("Calculating reward...")
    print("Ideal Instructions:", ideal_instructions, "Suggested Action:", suggested_action, flush=False)
    try:
        ideal_dict = json.loads(ideal_instructions)
        ideal_attacker = ideal_dict.get("attacker", "")
        ideal_supporter = ideal_dict.get("supporter", "")
    except (json.JSONDecodeError, TypeError):
        return -10.0
    
    match_ideal_att = re.search(r'\[(.*?)\]', ideal_attacker)
    ideal_action_att = match_ideal_att.group(1).strip().lower() if match_ideal_att else ""
    
    match_ideal_sup = re.search(r'\[(.*?)\]', ideal_supporter)
    ideal_action_sup = match_ideal_sup.group(1).strip().lower() if match_ideal_sup else ""
    
    print("Ideal Attacker Action:", ideal_action_att, "Ideal Supporter Action:", ideal_action_sup, "Suggested action:", suggested_action, flush=True)
    try:
        suggested_dict = json.loads(suggested_action)
        suggested_att = suggested_dict.get("attacker", "")
        suggested_sup = suggested_dict.get("supporter", "")
    except (json.JSONDecodeError, TypeError):
        match_att = re.search(r'\[(.*?)\]', suggested_action)
        suggested_att = match_att.group(1).strip().lower() if match_att else ""
        return -5.0
    
    match_att = re.search(r'\[(.*?)\]', suggested_att)
    suggested_action_att = match_att.group(1).strip().lower() if match_att else ""
    
    match_sup = re.search(r'\[(.*?)\]', suggested_sup)
    suggested_action_sup = match_sup.group(1).strip().lower() if match_sup else ""
    
    def normalize_action(action):
        action = action.lower().strip()
        mapping = {
            "meteor": "meteor spell",
            "cura": "cura spell",
            "blizzard": "blizzard spell",
            "thunder": "thunder spell",
            "fire": "fire spell"
        }
        return mapping.get(action, action)
    
    suggested_action_att = normalize_action(suggested_action_att)
    suggested_action_sup = normalize_action(suggested_action_sup)
    ideal_action_att = normalize_action(ideal_action_att)
    ideal_action_sup = normalize_action(ideal_action_sup)
    
    reward_att = 5.0 if suggested_action_att == ideal_action_att else -5.0
    reward_sup = 5.0 if suggested_action_sup == ideal_action_sup else -5.0
    
    return float(reward_att + reward_sup)


# --------------------------------------------------------
# DEVICE
# --------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------
# LOAD MODEL + TOKENIZER
# --------------------------------------------------------
from transformers import AutoTokenizer

MODEL_PATH = "/Users/giuseppepiosorrentino/HeronBase/content/flan_t5_small_reviewer"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False   # <- questo forza il tokenizer "slow" compatibile con spiece.model
)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(MODEL_PATH).to(device)


# --------------------------------------------------------
# LOAD DATASET
# --------------------------------------------------------
CSV_PATH = "/Users/giuseppepiosorrentino/HeronBase/game_scenarios_dataset_4.csv"
df = pd.read_csv(CSV_PATH)

df["input"] = df["prompt"] + " " + df["response"]

dataset = Dataset.from_pandas(df[["input", "instructions"]])


# --------------------------------------------------------
# PPO CONFIG
# --------------------------------------------------------
ppo_config = PPOConfig(
    learning_rate=5e-7,
    ppo_epochs=1,
    mini_batch_size=1,
    batch_size=1,
)


ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer
)


generation_kwargs = {
    "temperature": 0.4,
    "top_k": 50,
    "top_p": 0.8,
    "max_new_tokens": 128
}


# --------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------
def train_ppo(epochs):
    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch+1}/{epochs} =====\n")

        for batch in tqdm(dataset, desc="Training"):
            query = batch["input"]
            target = batch["instructions"]

            # 1) Encode input
            query_tensor = tokenizer(query, return_tensors="pt").input_ids.to(device)

            # 2) Model generates response
            response_tensor = ppo_trainer.generate(
                query_tensor,
                **generation_kwargs
            )

            response_text = tokenizer.decode(response_tensor[0], skip_special_tokens=True)

            # 3) Compute reward
            reward_value = calculate_reward(target, response_text)
            reward_tensor = torch.tensor([reward_value], dtype=torch.float32)  # NON .to(device)

            # 4) PPO step
            stats = ppo_trainer.step(
                queries=[query_tensor[0]],
                responses=[response_tensor[0]],
                rewards=[reward_tensor]
            )

            print(f"Reward: {reward_value}")
            print(f"KL: {stats['objective/kl']:.4f} | Returns: {stats['ppo/returns/mean']:.4f}")

        print("Epoch complete.")

    print("\nSaving PPO model...")
    ppo_trainer.save_pretrained("/Users/giuseppepiosorrentino/HeronBase/content/flan_t5_small_reviewer_ppo")
    print("Done!")


# --------------------------------------------------------
# RUN
# --------------------------------------------------------
train_ppo(1)