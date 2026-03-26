import re
import json
import torch
import pandas as pd
from datasets import Dataset
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoTokenizer
from tqdm import tqdm

debug_answers = []  # detection error


def json_parser(text, label="", debug=False):
    if not isinstance(text, str) or not text.strip():
        if debug:
            print(f"[DEBUG {label}] Empty or invalid text:", text)
        return None

    original = text

    try:
        return json.loads(text)
    except Exception:
        pass

    # fix errori nel csv
    try:
        fixed = text.replace('""', '"')
        return json.loads(fixed)
    except Exception:
        pass

    try:
        fixed = text.replace('""', '"').strip()
        if not fixed.startswith("{"):
            fixed = "{" + fixed
        if not fixed.endswith("}"):
            fixed = fixed + "}"
        return json.loads(fixed)
    except Exception:
        pass

    try:
        attacker = re.search(
            r'["\']attacker["\']\s*:\s*["\']([^"\']+)["\']', original
        )
        supporter = re.search(
            r'["\']supporter["\']\s*:\s*["\']([^"\']+)["\']', original
        )
        result = {}
        if attacker:
            result["attacker"] = attacker.group(1)
        if supporter:
            result["supporter"] = supporter.group(1)
        if result:
            if debug:
                print(f"[DEBUG {label}] Parsed via regex:", result)
            return result
    except Exception:
        pass

    if debug:
        print("\nJSON PARSE FAILED]", label)
        print(original)
        print("-" * 60)

    return None


def extract_action(text):
    if not isinstance(text, str):
        return ""
    match = re.search(r"\[(.*?)\]", text)
    return match.group(1).strip().lower() if match else ""


def normalize_action(action):
    mapping = {
        "meteor": "meteor spell",
        "cura": "cura spell",
        "blizzard": "blizzard spell",
        "thunder": "thunder spell",
        "fire": "fire spell",
    }
    return mapping.get(action, action)

# calcolo reward


def calculate_reward(ideal_json, model_output, debug=False):
    ideal = json_parser(ideal_json, "IDEAL", debug)
    pred = json_parser(model_output, "MODEL", debug)

    # allucinazioni
    if not isinstance(ideal, dict) or not isinstance(pred, dict):
        if debug:
            print("[DEBUG REWARD] Parsed output is not dict → penalty")
            print("IDEAL:", ideal)
            print("PRED :", pred)
        return -15.0

    total_reward = 0.0

    for role in ["attacker", "supporter"]:
        ideal_raw = ideal.get(role, "")
        pred_raw = pred.get(role, "")

        ideal_action = normalize_action(extract_action(ideal_raw))
        pred_action = normalize_action(extract_action(pred_raw))

        if debug:
            print(f"[DEBUG {role.upper()}]")
            print("  Ideal raw:", ideal_raw)
            print("  Pred raw :", pred_raw)
            print("  Ideal action:", ideal_action)
            print("  Pred action :", pred_action)

        if not ideal_action:
            continue

        if pred_action == ideal_action:
            total_reward += 5.0
        else:
            total_reward -= 2.0

    if debug:
        print("[DEBUG TOTAL REWARD]:", total_reward)
        print("=" * 70)

    global debug_answers
    if (total_reward <= 3 and len(debug_answers) <= 50) or total_reward == -15.0:
        debug_answers.append(
            {'ideal': ideal, 'pred': pred, 'total_reward': total_reward})

        with open('debug_answers.json', 'w') as f:
            f.write(json.dumps(debug_answers))

    return total_reward


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "C:\\Users\\daisl\\PycharmProjects\\HeRoN\\model_directory"
DATASET_PATH = "C:\\Users\\daisl\\PycharmProjects\\HeRoN\\dataset Reviewer\\game_scenarios_dataset_4.csv"
OUTPUT_PATH = "C:\\Users\\daisl\\PycharmProjects\\HeRoN\\result_ppo"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    MODEL_PATH).to(device)

df = pd.read_csv(DATASET_PATH)
df["input"] = df["prompt"] + " " + df["response"]

dataset = Dataset.from_pandas(df[["input", "instructions"]])

ppo_config = PPOConfig(
    learning_rate=5e-7,
    ppo_epochs=1,
    mini_batch_size=1,
    batch_size=1,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)


generation_kwargs = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "do_sample": True,
}


def train_ppo(epochs):
    step = 0

    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch + 1} =====")

        for sample in tqdm(dataset):
            step += 1
            debug = (step % 25 == 0)  # debug vale true ogni 25 step

            query = sample["input"]
            target = sample["instructions"]

            query_tensors = tokenizer(
                query, return_tensors="pt"
            ).input_ids.to(device)

            response_tensors = ppo_trainer.generate(
                query_tensors[0], **generation_kwargs
            )

            response_text = tokenizer.decode(
                response_tensors[0], skip_special_tokens=True
            )

            reward = calculate_reward(
                target, response_text, debug=debug
            )

            reward_tensor = torch.tensor(reward, dtype=torch.float).to(device)

            stats = ppo_trainer.step(
                [query_tensors[0]],
                [response_tensors[0]],
                [reward_tensor],
            )

            print(
                f"Reward: {reward:>5.1f} | "
                f"KL: {stats['objective/kl']:.4f} | "
                f"Adv: {stats['ppo/policy/advantages_mean']:.4f}"
            )

        print("Epoch complete")

    ppo_trainer.save_pretrained(OUTPUT_PATH)


train_ppo(3)
