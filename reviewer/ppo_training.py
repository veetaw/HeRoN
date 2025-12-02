import re
import json
import torch
import pandas as pd
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer
from tqdm import tqdm


def processes_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['instructions']]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def calculate_reward(ideal_instructions, suggested_action):
    """
    Calcola reward comparando le azioni ideali (da instructions JSON) 
    con le azioni suggerite dal modello.
    
    Args:
        ideal_instructions: JSON string con {"attacker": "...", "supporter": "..."}
        suggested_action: testo suggerito dal modello
    
    Returns:
        reward: float (somma reward attacker + supporter)
    """
    try:
        # Parse ideal instructions JSON
        ideal_dict = json.loads(ideal_instructions)
        ideal_attacker = ideal_dict.get("attacker", "")
        ideal_supporter = ideal_dict.get("supporter", "")
    except (json.JSONDecodeError, TypeError):
        return -10.0
    
    # Estrai azioni tra parentesi quadre
    match_ideal_att = re.search(r'\[(.*?)\]', ideal_attacker)
    ideal_action_att = match_ideal_att.group(1).strip().lower() if match_ideal_att else ""
    
    match_ideal_sup = re.search(r'\[(.*?)\]', ideal_supporter)
    ideal_action_sup = match_ideal_sup.group(1).strip().lower() if match_ideal_sup else ""
    
    # Parse suggested action (assumi sia JSON string)
    try:
        suggested_dict = json.loads(suggested_action)
        suggested_att = suggested_dict.get("attacker", "")
        suggested_sup = suggested_dict.get("supporter", "")
    except (json.JSONDecodeError, TypeError):
        # Fallback: cerca azioni tra parentesi nel testo
        match_att = re.search(r'\[(.*?)\]', suggested_action)
        suggested_att = match_att.group(1).strip().lower() if match_att else ""
        # Non c'è info per supporter, penalizza
        return -5.0
    
    # Estrai azioni suggerite tra parentesi quadre
    match_att = re.search(r'\[(.*?)\]', suggested_att)
    suggested_action_att = match_att.group(1).strip().lower() if match_att else ""
    
    match_sup = re.search(r'\[(.*?)\]', suggested_sup)
    suggested_action_sup = match_sup.group(1).strip().lower() if match_sup else ""
    
    # Normalizza varianti comuni (es. "meteor" -> "meteor spell")
    def normalize_action(action):
        action = action.lower().strip()
        if action == "meteor":
            action = "meteor spell"
        elif action == "cura":
            action = "cura spell"
        elif action == "blizzard":
            action = "blizzard spell"
        elif action == "thunder":
            action = "thunder spell"
        elif action == "fire":
            action = "fire spell"
        return action
    
    suggested_action_att = normalize_action(suggested_action_att)
    suggested_action_sup = normalize_action(suggested_action_sup)
    ideal_action_att = normalize_action(ideal_action_att)
    ideal_action_sup = normalize_action(ideal_action_sup)
    
    # Calcola reward separati
    reward_att = 5.0 if suggested_action_att == ideal_action_att else -5.0
    reward_sup = 5.0 if suggested_action_sup == ideal_action_sup else -5.0
    
    # Reward totale
    total_reward = reward_att + reward_sup
    
    return float(total_reward)


def collators(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("") # Insert Reviewer fine tuning path
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("").to(device) # Insert Reviewer fine tuning path

df = pd.read_csv('') # Insert dataset

df['input'] = df['prompt'] + " " + df['response']

dataset = Dataset.from_pandas(df[['input', 'instructions']])
tokenizer_dataset = dataset.map(processes_function, batched=True)

ppo_config = PPOConfig(
    learning_rate=5e-7,
    ppo_epochs=1,
    mini_batch_size=1,
    batch_size=1
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

generation_kwards = {
    "temperature": 0.4,
    "top_k": 50,
    "top_p": 0.8,
}


def train_ppo(epochs):
    for i in tqdm(range(epochs)):
        for batch in dataset:
            input_tensor = []
            response_tensor = []
            reward_tensor = []
            game_input = batch['input']
            targets = batch['instructions']

            inputs = tokenizer(game_input, return_tensors="pt").to(device).input_ids
            input_tensor.append(inputs[0])

            response = ppo_trainer.generate(inputs[0], **generation_kwards)
            response_tensor.append(response[0])

            response_text = tokenizer.decode(response[0], skip_special_tokens=True)

            reward = calculate_reward(targets, response_text)
            reward_tensor.append((torch.tensor(reward, dtype=torch.float)))

            stats = ppo_trainer.step(input_tensor, response_tensor, reward_tensor)
            print(f"objective/kl: {stats['objective/kl']}")
            print(f"ppo/returns/mean: {stats['ppo/returns/mean']}")
            print(f"ppo/policy/advantages_mean: {stats['ppo/policy/advantages_mean']}")
            print(f"Reward: {reward}")
        print("epoch complete")

    ppo_trainer.save_pretrained("/Users/macstudio/Desktop/Tesi_magistrale-main/flan-t5-large-ppo")

train_ppo(3)

