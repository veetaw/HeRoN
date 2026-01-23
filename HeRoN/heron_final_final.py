import json

import numpy as np
import matplotlib.pyplot as plt
from HeRoN.utils import build_prompt
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.agent import DQNAgent
from classes.environment import BattleEnv
import pandas as pd
import re
#from openai import OpenAI
import action_score as score
from classes.instructor_agent import InstructorAgent
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import os
from classes.games import *

from classes.support_agent import DQNSupportAgent

# Reviewer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REVIEWER POST PPO
MODEL_PATH = "C:\\Users\\daisl\\PycharmProjects\\HeRoN\\result_ppo"   # o il path assoluto
OUTPUT_DIRECTORY = "Heron_Final_test"
tokenizer_instruction = AutoTokenizer.from_pretrained(MODEL_PATH)
model_instruction = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH
).to(device)

# tokenizer_instruction = None
# model_instruction = None

SERVER_API_HOST = "http://localhost:1234/v1"

from openai import OpenAI

client = OpenAI(base_url=SERVER_API_HOST, api_key="lm-studio")


def get_llm_response(input_text):
    response_obj = client.chat.completions.create(
        model="gemma-3-27b-it-qat",
        messages=[{"role": "user", "content": input_text}],
        temperature=0.7,
        max_tokens=100
    )
    return response_obj.choices[0].message.content


def parse_llm_json(llm_text):
    patterns = {
            "attacker": r'"attacker"\s*:\s*"([^"]+)"',
            "supporter": r'"supporter"\s*:\s*"([^"]+)"',
        }

    result = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, llm_text, re.DOTALL)
        result[key] = match.group(1).strip() if match else None

    return result

def train_dqn(episodes, batch_size=32, attacker_path=None, support_path=None):
    # environment settings

    players = [player1, player2]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    # NPC

    attacker_agent = DQNAgent(
        env.get_state_size_of_player(PLAYER_1_NAME),
        env.get_action_size(0),
        attacker_path
    )
    supporter_agent = DQNSupportAgent(
        env.get_state_size_of_player(PLAYER_2_NAME),
        env.get_action_size(1),
        support_path
    )

    # Reviewer
    # TODO: INSTRUCTOR AGENT VA RIFATTO
    instructor_agent = InstructorAgent(model_instruction, tokenizer_instruction, device)

    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []
    allucination = 0
    total_agent_wins = 0
    total_enemy_wins = 0

    for ep in range(episodes):
        state_global = env.reset()
        state_attacker = state_global[PLAYER_1_NAME]
        state_attacker = np.reshape(state_attacker, [1, env.get_state_size_of_player(PLAYER_1_NAME)])
        state_support = state_global[PLAYER_2_NAME]
        state_support = np.reshape(state_support, [1, env.get_state_size_of_player(PLAYER_2_NAME)])

        done = False
        total_reward_support = 0
        total_reward_attacker = 0

        moves = 0
        match_score_attacker = []
        match_score_support = []

        last_enemy_move = "No action"

        score.reset_quantities()

        threshold = 1.0
        decay = 0.1  # 0.2
        suggestion = 0

        state_size_attacker = env.get_state_size_of_player(PLAYER_1_NAME)
        state_size_support = env.get_state_size_of_player(PLAYER_2_NAME)

        while not done:
            player_attacker = players[0]
            player_support = players[1]

            p = np.random.rand()

            # Inizializza le variabili PRIMA del blocco if
            attacker_action = None
            support_action = None
            match_attacker = None
            match_support = None
            attacker_scores = {}
            support_scores = {}
            print("PORCODIOOOOO CAMBIA DOPO")
            # quando deve esplorare e quando no
            if p > threshold or ep < 1:
                # Description of environment and Helper action #
                suggestion += 1
                game_description_attacker = env.describe_game_state_attacker(last_enemy_move)
                game_description_supporter = env.describe_game_state_supporter(last_enemy_move)

                input_text = f"""You are a game assistant coordinating 2 players (attacker and supporter) in battle against an enemy.
                    
                    GAME SETUP:
                    - Supporter: max {PLAYER_2_HEALTH} HP, max {PLAYER_2_MP} MP
                    - Attacker: max {PLAYER_1_HEALTH} HP, max {PLAYER_1_MP} MP
                    - Enemy: max {ENEMY_HEALTH} HP, max {ENEMY_MP} MP

                    CURRENT STATE:
                    - Attacker: {game_description_attacker}
                    - Supporter: {game_description_supporter}
                    - Enemy last move: {last_enemy_move}

                    ROLES:
                    - Attacker: Focuses on dealing damage to enemy
                    - Supporter: Can attack OR heal (self/mate/both) based on necessity

                    REQUIRED OUTPUT (valid JSON only):
                    {{
                        "attacker": "ACTION",
                        "supporter": "ACTION",
                    }}

                    Respond ONLY with the JSON object, no additional text."""
              
                llm_response = get_llm_response(input_text)

                # risposta del reviewer
                game_description = (
                    env.get_compact_state() +
                    f"Last enemy move: {last_enemy_move}."
                )
                print("GAME DESCRIPTION PER REVIEWER: ", game_description)
                response = instructor_agent.generate_suggestion(game_description, str(parse_llm_json(llm_response)))
                revise = f"""Given the attacker's game state: '{game_description_attacker}'
                            and the supporter's game state: '{game_description_supporter}'.
                            Your initial response was '{llm_response}'.
                            Considering the suggestion given by a reviewer: {{{response}}},
                            please provide your revised actions.

                            JSON schema (strict):
                            {{
                              "attacker": "string, one of allowed actions",
                              "supporter": "string, one of allowed actions",
                              "reason_action_attacker": "string, max 40 words",
                              "reason_action_supporter": "string, max 40 words"
                            }}

                            Respond ONLY with the JSON object, no additional text. /no_think"""

                new_llm_response = get_llm_response(revise)

                try:
                    response_dict = parse_llm_json(new_llm_response)

                    llm_requested_attacker = response_dict.get("attacker", "").strip()
                    attacker_action = map_llm_action_to_attacker_action(llm_requested_attacker)
                    if attacker_action is not None:
                        if attacker_action != -1:
                            if attacker_action == "elixer":
                                attacker_action = "elixir"
                            match_attacker = map_action_attack(attacker_action)
                            attacker_scores = score.calculate_scores_attacker(
                                player_attacker.get_hp(),
                                player_attacker.get_mp(),
                                enemies[0].get_hp()
                            )
                            #print(f"npc 1 esegue {match_attacker}")
                        else:
                            pass
                            #print("MARIA è MORTA, QUINDI NESSUNA AZIONE")
                    else:
                        attacker_action = attacker_agent.act(state_attacker, env, 0)
                        match_attacker = map_action_attack(attacker_action)
                        attacker_scores = score.calculate_scores_attacker(
                            player_attacker.get_hp(),
                            player_attacker.get_mp(),
                            enemies[0].get_hp()
                        )
                        allucination += 1
                        #print(
                           # f"AZIONE CHIESTA DALL'LLM PER Maria: {llm_requested_attacker} (NON VALIDA), AZIONE ESEGUITA DA Maria: {match_attacker} (da DQN)")

                    # Extract and map supporter action
                    llm_requested_support = response_dict.get("supporter", "").strip()
                    support_action = map_llm_action_to_supporter_action(llm_requested_support)
                    if support_action is not None:
                        if support_action != -1:
                            if support_action == "elixer":
                                support_action = "elixir"
                            match_support = map_action_support(support_action)
                            support_scores = score.calculate_scores_support(
                                player_support.get_hp(),
                                player_attacker.get_hp(),
                                player_support.get_mp(),
                                enemies[0].get_hp()
                            )
                           # print(f"npc 2 esegue: {match_support}")
                        else:
                            pass
                            #print("JUANA è MORTA, QUINDI NESSUNA AZIONE")
                    else:
                        support_action = supporter_agent.act(state_support, env, 1)
                        match_support = map_action_support(support_action)
                        support_scores = score.calculate_scores_support(
                            player_support.get_hp(),
                            player_attacker.get_hp(),
                            player_support.get_mp(),
                            enemies[0].get_hp()
                        )
                        allucination += 1
                        #print(
                            #f"AZIONE CHIESTA DALL'LLM PER Juana: {llm_requested_support} (NON VALIDA), AZIONE ESEGUITA DA Juana: {match_support} (da DQN)")
                except Exception as e:
                    print("Eccezione: ", e)
                    # In caso di eccezione, usa DQN
                    attacker_action = attacker_agent.act(state_attacker, env, 0)
                    match_attacker = map_action_attack(attacker_action)
                    attacker_scores = score.calculate_scores_attacker(
                        player_attacker.get_hp(),
                        player_attacker.get_mp(),
                        enemies[0].get_hp()
                    )

                    support_action = supporter_agent.act(state_support, env, 1)
                    match_support = map_action_support(support_action)
                    support_scores = score.calculate_scores_support(
                        player_support.get_hp(),
                        player_attacker.get_hp(),
                        player_support.get_mp(),
                        enemies[0].get_hp()
                    )
            else:
                # Quando NON si usa l'LLM (p <= threshold o e >= 180), usa DQN

                attacker_action = attacker_agent.act(state_attacker, env, 0)
                match_attacker = map_action_attack(attacker_action)
                attacker_scores = score.calculate_scores_attacker(
                    player_attacker.get_hp(),
                    player_attacker.get_mp(),
                    enemies[0].get_hp()
                )

                support_action = supporter_agent.act(state_support, env, 1)
                match_support = map_action_support(support_action)
                support_scores = score.calculate_scores_support(
                    player_support.get_hp(),
                    player_attacker.get_hp(),
                    player_support.get_mp(),
                    enemies[0].get_hp()
                )
            if match_attacker is not None and match_attacker != "no_action":
                match_score_attacker.append(round(attacker_scores.get(match_attacker, 0), 2))
            if match_support is not None and match_support != "no_action":
                match_score_support.append(round(support_scores.get(match_support, 0), 2))

            next_state, reward_attacker, reward_support, done, a_win, last_enemy_move, __ = env.step(attacker_action,
                                                                                                     support_action)

            if match_attacker is not None and match_attacker != "no_action":
                score.update_quantity(match_attacker, player_attacker.get_mp(), 0)
            if match_support is not None and match_support != "no_action":
                score.update_quantity(match_support, player_support.get_mp(), 1)

            total_reward_attacker += reward_attacker
            total_reward_support += reward_support

            next_state_attacker = np.reshape(next_state[PLAYER_1_NAME], [1, state_size_attacker])
            next_state_support = np.reshape(next_state[PLAYER_2_NAME], [1, state_size_support])

            if match_attacker is not None and match_attacker != "no_action":
                attacker_agent.remember(state_attacker, attacker_action, reward_attacker, next_state_attacker, done)
            if match_support is not None and match_support != "no_action":
                supporter_agent.remember(state_support, support_action, reward_support, next_state_support, done)

            state_attacker = next_state_attacker
            state_support = next_state_support

            moves += 1

            threshold = max(0, threshold - decay)

            if len(attacker_agent.memory) > batch_size:
                attacker_agent.replay(batch_size, env, 0)

            if len(supporter_agent.memory) > batch_size:
                supporter_agent.replay(batch_size, env, 1)

            if done:
                result = "VICTORY" if a_win else "DEFEAT"
                
                survivors = []
                if player_attacker.get_hp() > 0:
                    survivors.append(f"{PLAYER_1_NAME} (HP: {player_attacker.get_hp()})")
                if player_support.get_hp() > 0:
                    survivors.append(f"{PLAYER_2_NAME} (HP: {player_support.get_hp()})")
                
                survivor_text = ", ".join(survivors) if survivors else "Nessuno"
                
                print(f"\n{'='*70}")
                print(f"  {result}  |  Episode {ep+1}/{episodes}")
                print(f"{'='*70}")
                print(f"  Attacker Reward: {total_reward_attacker:>6.0f}  |  Moves: {moves}")
                print(f"  Support Reward:  {total_reward_support:>6.0f}  |  Epsilon: ATK={attacker_agent.epsilon:.3f}, SUP={supporter_agent.epsilon:.3f}")
                print(f"  Sopravvissuti: {survivor_text}")
                
                if a_win:
                    total_agent_wins += 1
                else:
                    total_enemy_wins += 1
                
                win_rate = total_agent_wins / (ep + 1)
                print(f"  Win Rate: {total_agent_wins}/{ep+1} ({100*win_rate:.1f}%)")
                print(f"{'='*70}\n")
                
                break
        print(f"Vittorie agente: {total_agent_wins}, vittorie nemico: {total_enemy_wins}")

        # Salva reward e mosse per l'episodio
        rewards_per_episode.append({
            'attacker': total_reward_attacker,
            'support': total_reward_support,
            'combined': total_reward_attacker + total_reward_support
        })
        agent_moves_per_episode.append(moves)
        agent_wins.append(1 if a_win else 0)
        enemy_wins.append(0 if a_win else 1)
        success_rate.append(total_agent_wins / (ep + 1))

        action_scores.append({
            'attacker': np.mean(match_score_attacker),
            'support': np.mean(match_score_support),
            'combined': (np.mean(match_score_attacker) + np.mean(match_score_support)) / 2
        })
    avg_reward_attacker = np.mean([r['attacker'] for r in rewards_per_episode])
    avg_reward_support = np.mean([r['support'] for r in rewards_per_episode])
    avg_reward_combined = np.mean([r['combined'] for r in rewards_per_episode])
    avg_moves = np.mean(agent_moves_per_episode)
    avg_score_attacker = np.mean([s['attacker'] for s in action_scores])
    avg_score_support = np.mean([s['support'] for s in action_scores])
    avg_score_combined = np.mean([s['combined'] for s in action_scores])
    
    print("\n=== TRAINING SUMMARY ===")
    print(f"Average reward (Attacker): {avg_reward_attacker:.2f}")
    print(f"Average reward (Support): {avg_reward_support:.2f}")
    print(f"Average reward (Combined): {avg_reward_combined:.2f}")
    print(f"Average moves: {avg_moves:.2f}")
    print(f"Average score (Attacker): {avg_score_attacker:.4f}")
    print(f"Average score (Support): {avg_score_support:.4f}")
    print(f"Average score (Combined): {avg_score_combined:.4f}")

    
    attacker_agent.save(OUTPUT_DIRECTORY + "/MODELLO_HELPER_B_ATTACKER")
    supporter_agent.save(OUTPUT_DIRECTORY + "/MODELLO_HELPER_B_SUPPORTER")

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores, allucination


def plot_training(
    rewards,
    agent_wins,
    enemy_wins,
    moves,
    success_rate,
    action_scores,
    hallucinations,
    output_dir=OUTPUT_DIRECTORY
):
    os.makedirs(output_dir, exist_ok=True)
    reward_attacker = [r['attacker'] for r in rewards]
    reward_support = [r['support'] for r in rewards]
    reward_combined = [r['combined'] for r in rewards]
    plt.figure(figsize=(8, 6))
    plt.plot(reward_attacker, label='Attacker Reward', color='red', alpha=0.7)
    plt.plot(reward_support, label='Support Reward', color='blue', alpha=0.7)
    plt.plot(reward_combined, label='Combined Reward', color='green', linewidth=2)

    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Train_reward_DQN.png"))
    plt.close()
    cumulative_agent_wins = np.cumsum(agent_wins).tolist()
    cumulative_enemy_wins = np.cumsum(enemy_wins).tolist()
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')
    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Train_cumulative_Win_DQN.png"))
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Train_moves_DQN.png"))
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Train_success_rate_DQN.png"))
    plt.close()
    attacker_scores = [s['attacker'] for s in action_scores]
    support_scores = [s['support'] for s in action_scores]
    combined_scores = [s['combined'] for s in action_scores]

    plt.figure(figsize=(8, 6))
    plt.plot(attacker_scores, label='Attacker Score', color='red', alpha=0.7)
    plt.plot(support_scores, label='Support Score', color='blue', alpha=0.7)
    plt.plot(combined_scores, label='Combined Score', color='green', linewidth=2)

    plt.title('Action Scores per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Score_DQN_separated.png"))
    plt.close()
    raw_data = {
        "rewards": rewards,
        "agent_wins": agent_wins,
        "enemy_wins": enemy_wins,
        "moves": moves,
        "success_rate": success_rate,
        "action_scores": action_scores,
        "cumulative_agent_wins": cumulative_agent_wins,
        "cumulative_enemy_wins": cumulative_enemy_wins,
        "hallucinations": hallucinations
    }

    json_path = os.path.join(output_dir, "training_data.json")
    with open(json_path, "w") as f:
        json.dump(raw_data, f, indent=4)

    print(f"Plots and raw data saved in: {output_dir}")



def export_success_rate(success_rate):
    df = pd.DataFrame({
        "Episode": list(range(1, len(success_rate) + 1)),
        "Success Rate": success_rate
    })

    df.to_csv('success_rate_model_llama_1000_final.csv', index=False)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    print(OUTPUT_DIRECTORY)


    # Train the agent
    rewards, agent_wins, enemy_wins, moves, success_rate, action_scores, hallucinations = train_dqn(episodes=2)

    # Plot dei risultati
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores, hallucinations)
    export_success_rate(success_rate)