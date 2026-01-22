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

from classes.support_agent import DQNSupportAgent

# TODO:
# Reviewer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REVIEWER POST PPO
MODEL_PATH = "C:\\Users\\daisl\\PycharmProjects\\HeRoN\\result_ppo"   # o il path assoluto

tokenizer_instruction = AutoTokenizer.from_pretrained(MODEL_PATH)
model_instruction = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH
).to(device)

# tokenizer_instruction = None
# model_instruction = None

GROQ_API_KEY = None

NPC_1_NAME = "Maria"
NPC_2_NAME = "Juana"
ENEMY_NAME = "Antonio"


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

    # Trova il primo '{' e l'ultimo '}' nel testo
    start = llm_text.find('{')
    end = llm_text.rfind('}')

    if start == -1 or end == -1 or start > end:
        return None  # Nessun JSON trovato

    json_str = llm_text[start:end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def map_llm_action_to_attacker_action(llm_response):
    action = llm_response.strip().lower()
    if action == "attack":
        return 0
    elif action == "fire spell":
        return 1
    elif action == "thunder spell":
        return 2
    elif action == "blizzard spell":
        return 3
    elif action == "meteor spell":
        return 4
    elif action == "cura spell":
        return 5
    elif action == "potion":
        return 6
    elif action == "grenade":
        return 7
    elif action == "elixir" or action == "elixer":
        return 8
    elif action == "no_action":
        return "no_action"
    return None


def map_llm_action_to_supporter_action(llm_response):
    action = llm_response.strip().lower()
    if action == "attack":
        return 0
    elif action == "fire spell":
        return 1
    elif action == "cura spell":
        return 2
    elif action == "cura_tot":
        return 3
    elif action == "splash":
        return 4
    elif action == "cura_m":
        return 5
    elif action == "cura_totm":
        return 6
    elif action == "potion":
        return 7
    elif action == "grenade":
        return 8
    elif action == "elixir" or action == "elixer":
        return 9
    elif action == "no_action":
        return "no_action"
    return None


# implementa map_llm_action_to_agent_action per il supporter
def map_action_attack(action):
    """Mappa l'indice azione al nome per ATTACKER"""
    actions_map = {
        0: 'attack',
        1: 'fire spell',
        2: 'thunder spell',
        3: 'blizzard spell',
        4: 'meteor spell',
        5: 'cura spell',
        6: 'potion',
        7: 'grenade',
        8: 'elixir'
    }
    return actions_map.get(action, 'attack')


# aggiungere la nuova map action del supporter

def map_action_support(action):
    """Mappa l'indice azione al nome per SUPPORT"""
    actions_map = {
        0: 'attack',
        1: 'fire spell',
        2: 'cura spell',  # Auto-cure (white)
        3: 'cura_tot',  # Cura entrambi (white_tot)
        4: 'splash',  # Cura entrambi meno (white_tot)
        5: 'cura_m',  # Cura mate (white_m)
        6: 'cura_totm',  # Cura mate tanto (white_m)
        7: 'potion',
        8: 'grenade',
        9: 'elixir'
    }
    return actions_map.get(action, 'attack')


def train_dqn(episodes, batch_size=32):
    # environment settings
    attacker_spells = [fire, thunder, blizzard, meteor, cura]

    # SUPPORT spells (offensivi base + cure variegate)
    support_spells = [fire, cura_support, cura_tot, splash, cura_m, cura_totm]

    player_items = [
        {"item": potion, "quantity": 3},
        {"item": grenade, "quantity": 2},
        {"item": hielixer, "quantity": 1}
    ]
    player1 = Person(NPC_1_NAME, 2600, 120, 300, 34, attacker_spells, player_items)
    player2 = Person(NPC_2_NAME, 2300, 180, 120, 50, support_spells, player_items)
    enemy1 = Person(ENEMY_NAME, 6000, 400, 470, 25, [fire, cura], [])

    players = [player1, player2]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    # NPC

    attacker_agent = DQNAgent(
        env.get_state_size_of_player(NPC_1_NAME),
        env.get_action_size(0),
        None
    )
    supporter_agent = DQNSupportAgent(
        env.get_state_size_of_player(NPC_2_NAME),
        env.get_action_size(1),
        None
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

    for episode in range(episodes):
        state_global = env.reset()
        state_attacker = state_global[NPC_1_NAME]
        state_attacker = np.reshape(state_attacker, [1, env.get_state_size_of_player(NPC_1_NAME)])
        state_support = state_global[NPC_2_NAME]
        state_support = np.reshape(state_support, [1, env.get_state_size_of_player(NPC_2_NAME)])

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

        state_size_attacker = env.get_state_size_of_player(NPC_1_NAME)
        state_size_support = env.get_state_size_of_player(NPC_2_NAME)

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

            # quando deve esplorare e quando no
            if p > threshold and episode < 600:
                # Description of environment and Helper action #
                suggestion += 1
                game_description_attacker = env.describe_game_state_attacker(last_enemy_move)
                game_description_supporter = env.describe_game_state_supporter(last_enemy_move)

                input_text = build_prompt(game_description_attacker, game_description_supporter, last_enemy_move)

                llm_response = get_llm_response(input_text)

                # risposta del reviewer
                game_description = (
            env.get_compact_state() +
            f"Last enemy move: {last_enemy_move}."
        )
                print(game_description)
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
                    response_dict = parse_llm_json(llm_response)

                    llm_requested_attacker = response_dict.get("attacker", "").strip()
                    attacker_action = map_llm_action_to_attacker_action(llm_requested_attacker)
                    if attacker_action is not None:
                        if attacker_action != "no_action":
                            if attacker_action == "elixer":
                                attacker_action = "elixir"
                            match_attacker = map_action_attack(attacker_action)
                            attacker_scores = score.calculate_scores_attacker(
                                player_attacker.get_hp(),
                                player_attacker.get_mp(),
                                enemies[0].get_hp()
                            )
                            print(f"npc 1 esegue {match_attacker}")
                        else:
                            print("MARIA è MORTA, QUINDI NESSUNA AZIONE")
                    else:
                        attacker_action = attacker_agent.act(state_attacker, env, 0)
                        match_attacker = map_action_attack(attacker_action)
                        attacker_scores = score.calculate_scores_attacker(
                            player_attacker.get_hp(),
                            player_attacker.get_mp(),
                            enemies[0].get_hp()
                        )
                        allucination += 1
                        print(
                            f"AZIONE CHIESTA DALL'LLM PER Maria: {llm_requested_attacker} (NON VALIDA), AZIONE ESEGUITA DA Maria: {match_attacker} (da DQN)")

                    # Extract and map supporter action
                    llm_requested_support = response_dict.get("supporter", "").strip()
                    support_action = map_llm_action_to_supporter_action(llm_requested_support)
                    if support_action is not None:
                        if support_action != "no_action":
                            if support_action == "elixer":
                                support_action = "elixir"
                            match_support = map_action_support(support_action)
                            support_scores = score.calculate_scores_support(
                                player_support.get_hp(),
                                player_attacker.get_hp(),
                                player_support.get_mp(),
                                enemies[0].get_hp()
                            )
                            print(f"npc 2 esegue: {match_support}")
                        else:
                            print("JUANA è MORTA, QUINDI NESSUNA AZIONE")
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
                        print(
                            f"AZIONE CHIESTA DALL'LLM PER Juana: {llm_requested_support} (NON VALIDA), AZIONE ESEGUITA DA Juana: {match_support} (da DQN)")
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
                # Quando NON si usa l'LLM (p <= threshold o e >= 600), usa DQN
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

            match_score_attacker.append(round(attacker_scores.get(match_attacker, 0), 2))
            match_score_support.append(round(support_scores.get(match_support, 0), 2))

            # Salva gli HP prima dell'azione
            hp_before_enemy = enemies[0].get_hp()
            hp_before_maria = player_attacker.get_hp()
            hp_before_juana = player_support.get_hp()

            next_state, reward_attacker, reward_support, done, a_win, last_enemy_move, __ = env.step(attacker_action,
                                                                                                     support_action)

            # Calcola i danni/cure effettuati
            hp_after_enemy = enemies[0].get_hp()
            hp_after_maria = player_attacker.get_hp()
            hp_after_juana = player_support.get_hp()

            score.update_quantity(match_attacker, player_attacker.get_mp(), 0)
            score.update_quantity(match_support, player_support.get_mp(), 1)

            total_reward_attacker += reward_attacker
            total_reward_support += reward_support

            next_state_attacker = np.reshape(next_state['Maria'], [1, state_size_attacker])
            next_state_support = np.reshape(next_state['Juana'], [1, state_size_support])

            attacker_agent.remember(state_attacker, attacker_action, reward_attacker, next_state_attacker, done)
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
                    survivors.append(f"Maria (HP: {player_attacker.get_hp()})")
                if player_support.get_hp() > 0:
                    survivors.append(f"Juana (HP: {player_support.get_hp()})")
                survivor_text = ", ".join(survivors) if survivors else "Nessuno"

                print(f"\n{'=' * 70}")
                print(f"  {result}  |  Episode {episode + 1}/{episodes}")
                print(f"{'=' * 70}")
                print(f"  Attacker Reward: {total_reward_attacker:>6.0f}  |  Moves: {moves}")
                print(
                    f"  Support Reward:  {total_reward_support:>6.0f}  |  Epsilon: ATK={attacker_agent.epsilon:.3f}, SUP={supporter_agent.epsilon:.3f}")
                print(f"  Sopravvissuti: {survivor_text}")

                if a_win:
                    total_agent_wins += 1
                else:
                    total_enemy_wins += 1

                win_rate = total_agent_wins / (episode + 1)
                print(f"  Win Rate: {total_agent_wins}/{episode + 1} ({100 * win_rate:.1f}%)")
                print(f"{'=' * 70}\n")
                break
        print(f"Vittorie agente: {total_agent_wins}, vittorie nemico: {total_enemy_wins}")

        # TODO: vedere che altri dati ci servono per i grafici
        rewards_per_episode.append({
            'attacker': total_reward_attacker,
            'support': total_reward_support,
            'combined': total_reward_attacker + total_reward_support
        })
        agent_moves_per_episode.append(moves)

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

    attacker_agent.save("MODELLO_LLM_E_REVIEWER_ATTACKER")  # save the agent model
    supporter_agent.save("MODELLO_LLM_E_REVIEWER_SUPPORTER")  # save the agent model

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores


def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores):
    plt.figure(figsize=(8, 6))
    reward_attacker = [r['attacker'] for r in rewards]
    reward_support = [r['support'] for r in rewards]
    reward_combined = [r['combined'] for r in rewards]

    plt.plot(reward_attacker, label='Attacker Reward', color='red', alpha=0.7)
    plt.plot(reward_support, label='Support Reward', color='blue', alpha=0.7)
    plt.plot(reward_combined, label='Combined Reward', color='green', linewidth=2)

    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.legend()
    plt.savefig("Train_reward_DQN_final.png")

    plt.figure(figsize=(8, 6))
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)

    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')

    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Train_cumulative_Win_DQN_final.png")

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Train_moves_DQN_final.png")

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Train_success_rate_DQN_final.png")

    plt.figure(figsize=(8, 6))
    attacker_scores = [s['attacker'] for s in action_scores]
    support_scores = [s['support'] for s in action_scores]
    combined_scores = [s['combined'] for s in action_scores]

    plt.plot(attacker_scores, label='Attacker Score', color='red', alpha=0.7)
    plt.plot(support_scores, label='Support Score', color='blue', alpha=0.7)
    plt.plot(combined_scores, label='Combined Score', color='green', linewidth=2)

    plt.title('Action Scores per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.legend()
    plt.savefig("Score_DQN_separated_final.png")


def export_success_rate(success_rate):
    df = pd.DataFrame({
        "Episode": list(range(1, len(success_rate) + 1)),
        "Success Rate": success_rate
    })

    df.to_csv('success_rate_model_llama_1000_final.csv', index=False)


if __name__ == "__main__":
    # Spell offensivi (già esistenti)
    fire = Spell("Fire", 25, 600, "black")
    thunder = Spell("Thunder", 30, 700, "black")
    blizzard = Spell("Blizzard", 35, 800, "black")
    meteor = Spell("Meteor", 40, 1000, "black")

    # Spell curativi per ATTACKER
    cura = Spell("Cura", 32, 1500, "white")

    # Spell curativi per SUPPORT
    cura_support = Spell("Cura", 32, 1200, "white")  # Auto-cure
    cura_tot = Spell("Cura Tot", 30, 700, "white_tot")  # Cura entrambi
    splash = Spell("Splash", 18, 450, "white_tot")  # Cura entrambi (meno potente)
    cura_m = Spell("Cura M", 28, 1300, "white_m")  # Cura il mate
    cura_totm = Spell("Cura TotM", 36, 1700, "white_m")  # Cura di più il mate

    potion = Item("Potion", "potion", "Heals 50 HP", 50)
    hielixer = Item("MegaElixer", "elixir", "Fully restores party's HP/MP", 9999)
    grenade = Item("Grenade", "attack", "Deals 500 damage", 500)

    # Train the agent
    rewards, agent_wins, enemy_wins, moves, success_rate, action_scores = train_dqn(episodes=120)

    # Plot dei risultati
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores)
    export_success_rate(success_rate)