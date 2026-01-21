import numpy as np
import matplotlib.pyplot as plt
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.agent import DQNAgent
from classes.support_agent import DQNSupportAgent
from classes.environment import BattleEnv
import pandas as pd
import re
import action_score as score
import json
from classes.games import *

import os


GROQ_API_KEY = None

try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get("GQ_KEY")
except ImportError:
    GROQ_API_KEY = os.getenv("GQ_KEY")


if GROQ_API_KEY:
    from groq import Groq
    def get_llm_response(input_text):
        client = Groq(api_key=GROQ_API_KEY)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": input_text}],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content
else: 
    SERVER_API_HOST = "http://127.0.0.1:1234/v1"

    try:
        import lmstudio as lms
        lms.get_default_client(SERVER_API_HOST)

        def get_llm_response(input_text):
            with lms.Client() as client:
                model = client.llm.model("") # Helper model
                llm_response = model.respond(input_text)

            llm_response = str(llm_response)
            return llm_response

    except ImportError:
        from openai import OpenAI
        client = OpenAI(base_url=SERVER_API_HOST, api_key="lm-studio")

        def get_llm_response(input_text):
            response_obj = client.chat.completions.create(
                model="qwen/qwen3-vl-4b",
                messages=[{"role": "user", "content": input_text}],
                temperature=0.7,
                max_tokens=100
            )
            return response_obj.choices[0].message.content

OUTPUT_DIRECTORY = "test2_resultsHelperB_test"

def train_dqn(episodes, batch_size=32, attacker_path=None, support_path=None):

    players = [player1, player2]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)

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
    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []
    allucination = 0
    total_agent_wins = 0
    total_enemy_wins = 0

    for e in range(episodes):
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
        state_size_attacker = env.get_state_size_of_player(PLAYER_1_NAME)
        state_size_support = env.get_state_size_of_player(PLAYER_2_NAME)
        while not done:
            player_attacker = players[0]
            player_support = players[1]

            attacker_action = None
            support_action = None
            game_description_attacker = env.describe_game_state_attacker(None)
            game_description_supporter = env.describe_game_state_supporter(None)
            prompt =f"""You are a game assistant coordinating 2 players (attacker and supporter) in battle against an enemy.

                        GAME SETUP:
                        Supporter: max {PLAYER_2_HEALTH} HP, max {PLAYER_2_MP} MP
                        Attacker: max {PLAYER_1_HEALTH} HP, max {PLAYER_1_MP} MP
                        Enemy: max {ENEMY_HEALTH} HP, max {ENEMY_MP}  MP

                        CURRENT STATE:
                        Attacker: {game_description_attacker}
                        Supporter: {game_description_supporter}
                        Enemy last move: {last_enemy_move}

                        ROLES:
                        Attacker: Focuses on dealing damage to enemy
                        Supporter: Can attack OR heal (self/mate/both) based on necessity

                        REQUIRED OUTPUT (valid JSON only):
                        {{
                        "attacker": "ACTION",
                        "supporter": "ACTION",
                        "reason_action_attacker": "Max 40 words explaining why this action",
                        "reason_action_supporter": "Max 40 words explaining why this action"
                        }}

                        Respond ONLY with the JSON object, no additional text."""
            if e==0 and moves==0:
                print(f"\n[Prompt]\n{prompt}\n")
            llm_response = get_llm_response(prompt)
            print(f"[BOTH] LLM response: {llm_response}")
            try:
                response_json = re.search(r'\{.*\}', llm_response, re.DOTALL).group(0)
                response_dict = json.loads(response_json)

                # Extract and map attacker action
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
                        print(f"AZIONE CHIESTA DALL'LLM PER {PLAYER_1_NAME}: {llm_requested_attacker}, AZIONE ESEGUITA DA {PLAYER_1_NAME}: {match_attacker}")
                    else:
                        print(f"{PLAYER_1_NAME} è MORTA, QUINDI NESSUNA AZIONE")
                else:
                    attacker_action = attacker_agent.act(state_attacker, env, 0)
                    match_attacker = map_action_attack(attacker_action)
                    attacker_scores = score.calculate_scores_attacker(
                        player_attacker.get_hp(), 
                        player_attacker.get_mp(), 
                        enemies[0].get_hp()
                    )
                    allucination += 1
                    print(f"AZIONE CHIESTA DALL'LLM PER {PLAYER_1_NAME}: {llm_requested_attacker} (NON VALIDA), AZIONE ESEGUITA DA {PLAYER_1_NAME}: {match_attacker} (da DQN)")

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
                        print(f"AZIONE CHIESTA DALL'LLM PER {PLAYER_2_NAME}: {llm_requested_support}, AZIONE ESEGUITA DA {PLAYER_2_NAME}: {match_support}")
                    else:
                        print(f"{PLAYER_2_NAME} è MORTA, QUINDI NESSUNA AZIONE")
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
                    print(f"AZIONE CHIESTA DALL'LLM PER {PLAYER_2_NAME}: {llm_requested_support} (NON VALIDA), AZIONE ESEGUITA DA {PLAYER_2_NAME}: {match_support} (da DQN)")
            except Exception as e:
                print("Errore, ", e)
            match_score_attacker.append(round(attacker_scores.get(match_attacker, 0), 2))
            match_score_support.append(round(support_scores.get(match_support, 0), 2))

            # Salva gli HP prima dell'azione
            hp_before_enemy = enemies[0].get_hp()
            hp_before_p1 = player_attacker.get_hp()
            hp_before_p2 = player_support.get_hp()

            next_state, reward_attacker, reward_support, done, a_win, last_enemy_move, __ = env.step(attacker_action, support_action)
            
            # Calcola i danni/cure effettuati
            hp_after_enemy = enemies[0].get_hp()
            hp_after_p1 = player_attacker.get_hp()
            hp_after_p2 = player_support.get_hp()
            
            damage_to_enemy_p1 = hp_before_enemy - hp_after_enemy if hp_before_enemy > hp_after_enemy else 0
            damage_to_enemy_p2 = 0  # Verrà calcolato sotto se {PLAYER_2_NAME} attacca
            
            print(f"\n[Move {moves:02d}] Ep {e+1}/{episodes}")
            
            # Log azione {PLAYER_1_NAME}
            if player_attacker.get_hp() > 0 and attacker_action != "no_action":
                if "attack" in match_attacker or "spell" in match_attacker or "grenade" in match_attacker:
                    print(f"  {PLAYER_1_NAME} usa {match_attacker} su Enemy per {damage_to_enemy_p1} dmg, nuovi HP di Enemy: {hp_after_enemy}")
                elif "cura" in match_attacker or "potion" in match_attacker:
                    heal_p1 = hp_after_p1 - hp_before_p1
                    print(f"  {PLAYER_1_NAME} usa {match_attacker} su se stessa per {heal_p1} HP, nuovi HP di {PLAYER_1_NAME}: {hp_after_p1}")
                elif "elixir" in match_attacker:
                    print(f"  {PLAYER_1_NAME} usa {match_attacker}, HP/MP completamente ripristinati")
            
            # Log azione {PLAYER_2_NAME}
            if player_support.get_hp() > 0 and support_action != "no_action":
                if "attack" in match_support or match_support == "fire spell" or "grenade" in match_support:
                    # Se {PLAYER_2_NAME} ha attaccato, calcola il danno (HP enemy è già cambiato da {PLAYER_1_NAME})
                    # Dobbiamo stimare il danno di {PLAYER_2_NAME}
                    print(f"  {PLAYER_2_NAME} usa {match_support} su Enemy, nuovi HP di Enemy: {hp_after_enemy}")
                elif "cura" in match_support or "splash" in match_support or "potion" in match_support:
                    heal_p2 = hp_after_p2 - hp_before_p2
                    heal_p1_from_p2 = hp_after_p1 - hp_before_p1 if hp_after_p1 > hp_before_p1 else 0
                    
                    if "tot" in match_support or "splash" in match_support:
                        print(f"  {PLAYER_2_NAME} usa {match_support} su entrambi: {PLAYER_1_NAME} +{heal_p1_from_p2} HP, {PLAYER_2_NAME} +{heal_p2} HP")
                    elif "_m" in match_support:
                        print(f"  {PLAYER_2_NAME} usa {match_support} su {PLAYER_1_NAME} per {heal_p1_from_p2} HP, nuovi HP di {PLAYER_1_NAME}: {hp_after_p1}")
                    else:
                        print(f"  {PLAYER_2_NAME} usa {match_support} su se stessa per {heal_p2} HP, nuovi HP di {PLAYER_2_NAME}: {hp_after_p2}")
                elif "elixir" in match_support:
                    print(f"  {PLAYER_2_NAME} usa {match_support}, HP/MP di tutti completamente ripristinati")
            
            status_atk = "ATK" if player_attacker.get_hp() > 0 else "DEAD"
            status_sup = "SUP" if player_support.get_hp() > 0 else "DEAD"
            
            #print(f"  [{status_atk}] {match_attacker:<18} (HP: {player_attacker.get_hp():>4}, MP: {player_attacker.get_mp():>3}) → score: {attacker_scores.get(match_attacker, 0):.2f}")
            #print(f"  [{status_sup}] {match_support:<18} (HP: {player_support.get_hp():>4}, MP: {player_support.get_mp():>3}) → score: {support_scores.get(match_support, 0):.2f}")
            #print(f"  Enemy HP: {enemies[0].get_hp():>4}/{enemies[0].maxhp}")

            #print(f"  Reward ATK: {reward_attacker:+4d} | SUP: {reward_support:+4d}")
 
            score.update_quantity(match_attacker, player_attacker.get_mp(), 0)
            score.update_quantity(match_support, player_support.get_mp(), 1)

            total_reward_attacker += reward_attacker
            total_reward_support += reward_support

            next_state_attacker = np.reshape(next_state[PLAYER_1_NAME], [1, state_size_attacker])
            next_state_support = np.reshape(next_state[PLAYER_2_NAME], [1, state_size_support])

            attacker_agent.remember(state_attacker, attacker_action, reward_attacker, next_state_attacker, done)
            supporter_agent.remember(state_support, support_action, reward_support, next_state_support, done)

            state_attacker = next_state_attacker
            state_support = next_state_support

            moves += 1

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
                print(f"  {result}  |  Episode {e+1}/{episodes}")
                print(f"{'='*70}")
                print(f"  Attacker Reward: {total_reward_attacker:>6.0f}  |  Moves: {moves}")
                print(f"  Support Reward:  {total_reward_support:>6.0f}  |  Epsilon: ATK={attacker_agent.epsilon:.3f}, SUP={supporter_agent.epsilon:.3f}")
                print(f"  Sopravvissuti: {survivor_text}")
                
                if a_win:
                    total_agent_wins += 1
                else:
                    total_enemy_wins += 1
                
                win_rate = total_agent_wins / (e + 1)
                print(f"  Win Rate: {total_agent_wins}/{e+1} ({100*win_rate:.1f}%)")
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


    #if (e + 1) % 200 == 0:
    #    save_path = f"model_dqn_episode_{e + 1}"
    #    print(f"Saving model to {save_path}...")
    #    agent.save(save_path)
    
    attacker_agent.save("MODELLO_NO_LLM_ATTACKER") # save the agent model
    supporter_agent.save("MODELLO_NO_LLM_SUPPORT") # save the agent model

    #append_csv("reward_per_episode.csv", rewards_per_episode, "Reward")
    #append_csv("agent_wins.csv", agent_wins, "Wins")
    #append_csv("enemy_wins.csv", enemy_wins, "Wins")
    #append_csv("agent_moves.csv", agent_moves_per_episode, "Moves")
    #append_csv("success_rate.csv", success_rate, "Rate")

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores



def plot_training(
    rewards,
    agent_wins,
    enemy_wins,
    moves,
    success_rate,
    action_scores,
    output_dir=OUTPUT_DIRECTORY
):
    """
    Plots training statistics and saves raw data to JSON.

    Args:
        rewards (list[dict])
        agent_wins (list[int])
        enemy_wins (list[int])
        moves (list[int])
        success_rate (list[float])
        action_scores (list[dict])
        output_dir (str): directory where plots and json will be saved
    """
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
        "cumulative_enemy_wins": cumulative_enemy_wins
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

    df.to_csv('success_rate_model_llama_1000.csv', index=False)


def append_csv(path, data, column_name):
    df = pd.DataFrame({
        "Episode": list(range(1, 1 + len(data))),
        column_name: data
    })
    df.to_csv(path, index=False)


if __name__ == "__main__":


    # Train the agent
    rewards, agent_wins, enemy_wins, moves, success_rate, action_scores = train_dqn(episodes=1)
    
    # Plot dei risultati
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores)
    export_success_rate(success_rate)
