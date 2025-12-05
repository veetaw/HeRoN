import re
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from tqdm import tqdm
import copy

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def calculate_reward(ideal_instructions, suggested_action):
    """
    Calcola il reward comparando le azioni ideali con quelle suggerite.
    """
    try:
        ideal_dict = json.loads(ideal_instructions)
        ideal_attacker_text = ideal_dict.get("attacker", "")
        ideal_supporter_text = ideal_dict.get("supporter", "")
    except (json.JSONDecodeError, TypeError):
        return 0.0
    
    # Estrai azioni ideali tra parentesi quadre
    match_ideal_att = re.search(r'\[(.*?)\]', str(ideal_attacker_text))
    ideal_action_att = match_ideal_att.group(1).strip().lower() if match_ideal_att else ""
    
    match_ideal_sup = re.search(r'\[(.*?)\]', str(ideal_supporter_text))
    ideal_action_sup = match_ideal_sup.group(1).strip().lower() if match_ideal_sup else ""

    if not ideal_action_att and not ideal_action_sup:
        return 0.0
    
    # Parse suggested actions
    suggested_att_raw = ""
    suggested_sup_raw = ""
    
    try:
        suggested_dict = json.loads(str(suggested_action))
        suggested_att_raw = suggested_dict.get("attacker", "")
        suggested_sup_raw = suggested_dict.get("supporter", "")
    except (json.JSONDecodeError, TypeError):
        try:
            fixed_json = "{" + str(suggested_action) + "}"
            suggested_dict = json.loads(fixed_json)
            suggested_att_raw = suggested_dict.get("attacker", "")
            suggested_sup_raw = suggested_dict.get("supporter", "")
        except (json.JSONDecodeError, TypeError):
            suggested_text = str(suggested_action)
            att_match = re.search(r'"attacker":\s*"[^"]*\[([^\]]+)\]', suggested_text)
            suggested_att_raw = att_match.group(1).strip() if att_match else ""
            
            sup_match = re.search(r'"supporter":\s*"[^"]*\[([^\]]+)\]', suggested_text)
            suggested_sup_raw = sup_match.group(1).strip() if sup_match else ""
    
    def extract_action_from_text(text):
        if not text:
            return ""
        match = re.search(r'\[(.*?)\]', str(text))
        return match.group(1).strip().lower() if match else str(text).lower().strip()
    
    def normalize_action(action):
        action = str(action).lower().strip() if action else ""
        mapping = {
            "meteor": "meteor spell",
            "cura": "cura spell",
            "blizzard": "blizzard spell", 
            "thunder": "thunder spell",
            "fire": "fire spell",
        }
        return mapping.get(action, action)
    
    suggested_att_action = extract_action_from_text(suggested_att_raw)
    suggested_sup_action = extract_action_from_text(suggested_sup_raw)
    
    suggested_att = normalize_action(suggested_att_action)
    suggested_sup = normalize_action(suggested_sup_action)
    ideal_action_att = normalize_action(ideal_action_att)
    ideal_action_sup = normalize_action(ideal_action_sup)
    
    # Calcola reward
    reward_att = 0.0
    reward_sup = 0.0
    
    if ideal_action_att:
        reward_att = 5.0 if (suggested_att and suggested_att == ideal_action_att) else -2.0
        
    if ideal_action_sup:
        reward_sup = 5.0 if (suggested_sup and suggested_sup == ideal_action_sup) else -2.0
    
    total_reward = float(reward_att + reward_sup)
    
    if np.isnan(total_reward) or total_reward is None:
        return 0.0
    
    return total_reward

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
MODEL_PATH = "/Users/giuseppepiosorrentino/HeronBase/content/flan_t5_small_reviewer"
OUTPUT_PATH = "/Users/giuseppepiosorrentino/HeronBase/content/flan_t5_small_reviewer_ppo2"

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
ref_model = copy.deepcopy(model).to(device)
print(f"Model loaded successfully!")

# Load dataset
CSV_PATH = "/Users/giuseppepiosorrentino/HeronBase/game_scenarios_dataset_4.csv"
df = pd.read_csv(CSV_PATH)
df["input_text"] = df["prompt"] + " " + df["response"]
df["target_text"] = df["instructions"]
dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

# PPO Configuration
class PPOConfig:
    learning_rate = 1e-5
    ppo_epochs = 4
    mini_batch_size = 2
    batch_size = 8
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 1.0
    kl_penalty = 0.02

config = PPOConfig()

# Value Head
class ValueHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, 256)
        self.linear2 = torch.nn.Linear(256, 1)
    
    def forward(self, hidden_states):
        x = F.relu(self.linear1(hidden_states))
        return self.linear2(x).squeeze(-1)

# PPO Trainer
class SimplePPOTrainer:
    def __init__(self, model, ref_model, tokenizer, config, device):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        hidden_size = model.config.d_model
        self.value_head = ValueHead(hidden_size).to(device)
        
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate
        )
    
    def generate_response(self, query_ids, max_length=128):
        with torch.no_grad():
            outputs = self.model.generate(
                query_ids,
                max_length=max_length,
                num_beams=1,
                temperature=1.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return outputs
    
    def compute_kl_penalty(self, query_ids, response_ids):
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=query_ids,
                decoder_input_ids=response_ids,
                return_dict=True
            ).logits
        
        model_logits = self.model(
            input_ids=query_ids,
            decoder_input_ids=response_ids,
            return_dict=True
        ).logits
        
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        model_log_probs = F.log_softmax(model_logits, dim=-1)
        
        kl = torch.sum(
            torch.exp(ref_log_probs) * (ref_log_probs - model_log_probs),
            dim=-1
        )
        return kl.mean()
    
    def ppo_step(self, query_ids, response_ids, rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        response_only_ids = response_ids
        
        try:
            outputs = self.model(
                input_ids=query_ids,
                decoder_input_ids=response_only_ids[:, :-1] if response_only_ids.shape[1] > 1 else response_only_ids,
                return_dict=True,
                output_hidden_states=True
            )
        except Exception as e:
            return self._get_zero_stats()
        
        logits = outputs.logits
        decoder_hidden_states = outputs.decoder_hidden_states[-1]
        values = self.value_head(decoder_hidden_states[:, -1, :])
        
        rewards_flat = rewards.squeeze()
        values_flat = values.squeeze()
        advantages = rewards_flat - values_flat.detach()
        
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        value_loss = F.mse_loss(values_flat, rewards_flat)
        policy_loss = -advantages.mean() if not torch.isnan(advantages).any() else torch.tensor(0.0, device=self.device)
        
        total_loss = policy_loss + self.config.value_coef * value_loss
        total_loss = torch.clamp(total_loss, min=-1e3, max=1e3)
        
        self.model.train()
        self.value_head.train()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        return {
            "loss": float(total_loss.item()) if not torch.isnan(total_loss) else 0.0,
            "policy_loss": float(policy_loss.item()) if not torch.isnan(policy_loss) else 0.0,
            "value_loss": float(value_loss.item()) if not torch.isnan(value_loss) else 0.0,
            "kl": 0.0,
            "reward": float(rewards_flat.mean().item()) if not torch.isnan(rewards_flat.mean()) else 0.0,
            "advantage": float(advantages.mean().item()) if not torch.isnan(advantages.mean()) else 0.0
        }
    
    def _get_zero_stats(self):
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0, "reward": 0.0, "advantage": 0.0}

# Training
TEST_MODE = False
MAX_SAMPLES = 5 if TEST_MODE else len(dataset)
NUM_EPOCHS = 1 if TEST_MODE else 3

trainer = SimplePPOTrainer(model, ref_model, tokenizer, config, device)

for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1}/{NUM_EPOCHS}")
    
    epoch_stats = {
        "loss": [], "policy_loss": [], "value_loss": [],
        "kl": [], "reward": [], "advantage": []
    }
    
    dataset_subset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    
    for batch_idx, sample in enumerate(tqdm(dataset_subset)):
        query = sample["input_text"]
        target = sample["target_text"]
        
        query_ids = tokenizer(
            query,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).input_ids.to(device)
        
        response_output = trainer.generate_response(query_ids)
        response_ids = response_output.sequences if hasattr(response_output, 'sequences') else response_output
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        reward = calculate_reward(target, response_text)
        
        if np.isnan(reward) or reward is None:
            reward = 0.0
        
        response_tensor = response_ids if isinstance(response_ids, torch.Tensor) else torch.tensor(response_ids).to(device)
        stats = trainer.ppo_step(query_ids, response_tensor, [reward])
        
        for key in epoch_stats:
            epoch_stats[key].append(stats[key])
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}: Reward {reward:.2f}, Loss {stats['loss']:.4f}")
        
        if TEST_MODE and batch_idx >= (MAX_SAMPLES - 1):
            break
    
    # Epoch summary
    print(f"\nEPOCH {epoch+1} SUMMARY:")
    for key in epoch_stats:
        if epoch_stats[key]:
            mean_val = np.mean(epoch_stats[key])
            print(f"   {key}: {mean_val:.4f}")

model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
torch.save(trainer.value_head.state_dict(), f"{OUTPUT_PATH}/value_head.pt")
print("PPO training completed!")