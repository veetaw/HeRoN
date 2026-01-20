import numpy as np
import matplotlib.pyplot as plt
from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item
from classes.environment import BattleEnv
from classes.agent import DQNAgent
import pandas as pd
import os
import action_score as score
import re
from classes.games import *
from classes.support_agent import DQNSupportAgent
import json

OUTPUT_DIRECTORY = "test1_results_test"


# Main loop with training
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
        score.reset_quantities()
        state_size_attacker = env.get_state_size_of_player(PLAYER_1_NAME)
        state_size_support = env.get_state_size_of_player(PLAYER_2_NAME)

        while not done:
            attacker_action = attacker_agent.act(state_attacker, env, 0)
            support_action = supporter_agent.act(state_support, env, 1)
            
            # è null quando è morto
            if attacker_action is None:
                attacker_action = 0
            
            if support_action is None:
                support_action = 0
            
            match_attacker = map_action_attack(attacker_action)
            match_support = map_action_support(support_action)
            
            player_attacker = players[0]
            player_support = players[1]
            
            if player_attacker.get_hp() > 0:
                attacker_scores = score.calculate_scores_attacker(
                    player_attacker.get_hp(), 
                    player_attacker.get_mp(), 
                    enemies[0].get_hp()
                )
            else:
                attacker_scores = {match_attacker: 0}
            
            if player_support.get_hp() > 0:
                support_scores = score.calculate_scores_support(
                    player_support.get_hp(), 
                    player_attacker.get_hp(),
                    player_support.get_mp(), 
                    enemies[0].get_hp()
                )
            else:
                support_scores = {match_support: 0}
            
            match_score_attacker.append(round(attacker_scores.get(match_attacker, 0), 2))
            match_score_support.append(round(support_scores.get(match_support, 0), 2))
            
            if moves % 3 == 0 or moves == 0:
                status_atk = "DEAD" if player_attacker.get_hp() <= 0 else "ATK"
                status_sup = "DEAD" if player_support.get_hp() <= 0 else "SUP"
                
                print(f"\n[Move {moves:02d}] Ep {e+1}/{episodes}")
                #print(f"  [{status_atk}] {match_attacker:<18} (HP: {player_attacker.get_hp():>4}, MP: {player_attacker.get_mp():>3}) → score: {attacker_scores.get(match_attacker, 0):.2f}")
                #print(f"  [{status_sup}] {match_support:<18} (HP: {player_support.get_hp():>4}, MP: {player_support.get_mp():>3}) → score: {support_scores.get(match_support, 0):.2f}")
                #print(f"  Enemy HP: {enemies[0].get_hp():>4}/{enemies[0].maxhp}")

            next_state, reward_attacker, reward_support, done, a_win, _, __ = env.step(attacker_action, support_action)
            
            #print(f"  Reward ATK: {reward_attacker:+4d} | SUP: {reward_support:+4d}")
            
            score.update_quantity(match_attacker, player_attacker.get_mp(), 0)
            score.update_quantity(match_support, player_support.get_mp(), 1)

            total_reward_attacker += reward_attacker
            total_reward_support += reward_support
            
            next_state_attacker = np.reshape(next_state[PLAYER_1_NAME], [1, state_size_attacker])
            next_state_support = np.reshape(next_state[PLAYER_2_NAME], [1, state_size_support])
            
            next_valid_attacker = env.get_valid_actions(0)
            next_valid_support = env.get_valid_actions(1)
            
            attacker_agent.remember(state_attacker, attacker_action, reward_attacker, next_state_attacker, done, next_valid_attacker)
            supporter_agent.remember(state_support, support_action, reward_support, next_state_support, done, next_valid_support)
            
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
        
        if attacker_agent.epsilon > attacker_agent.epsilon_min:
            attacker_agent.epsilon *= attacker_agent.epsilon_decay
        
        if supporter_agent.epsilon > supporter_agent.epsilon_min:
            supporter_agent.epsilon *= supporter_agent.epsilon_decay
        
        print(f"Vittorie agente: {total_agent_wins}, vittorie nemico: {total_enemy_wins}")

        rewards_per_episode.append({
            'attacker': total_reward_attacker,
            'support': total_reward_support,
            'combined': total_reward_attacker + total_reward_support
        })
        agent_moves_per_episode.append(moves)
        agent_wins.append(1 if a_win else 0)
        enemy_wins.append(0 if a_win else 1)
        success_rate.append(total_agent_wins / (e + 1))
        
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

    attacker_agent.save(OUTPUT_DIRECTORY + "/MODELLO_NO_LLM_ATTACKER")
    supporter_agent.save(OUTPUT_DIRECTORY + "/MODELLO_NO_LLM_SUPPORT")

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

    df.to_csv('train_success_rate_model_dqn_1000.csv', index=False)


def append_csv(path, data, column_name):
    df = pd.DataFrame({
        "Episode": list(range(1, 1 + len(data))),
        column_name: data
    })
    df.to_csv(path, index=False)


def load_csv_series(filename, column):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return df[column].tolist()
    return []


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    print(OUTPUT_DIRECTORY)
    attacker_path =  "/Users/giuseppepiosorrentino/HeronBase/test1_results/MODELLO_NO_LLM_ATTACKER" # Percorso del modello da caricare, se esistente
    support_path =   "/Users/giuseppepiosorrentino/HeronBase/test1_results/MODELLO_NO_LLM_SUPPORT"  # Percorso del modello da caricare, se esistente
    # Train the agent

    rewards, agent_wins, enemy_wins, moves, success_rate, action_scores = train_dqn(episodes=2, attacker_path=attacker_path, support_path=support_path)
    
    # Plot dei risultati
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores)
    
    # Esporta success rate
    export_success_rate(success_rate)
    
    print("\nTraining completato!")
    print("Grafici salvati:")
    print("   - " + OUTPUT_DIRECTORY + "/Train_reward_DQN.png")
    print("   - " + OUTPUT_DIRECTORY + "/Train_cumulative_Win_DQN.png")
    print("   - " + OUTPUT_DIRECTORY + "/Train_moves_DQN.png")
    print("   - " + OUTPUT_DIRECTORY + "/Train_success_rate_DQN.png")
    print("   - " + OUTPUT_DIRECTORY + "/Score_DQN_separated.png")
    print("Modelli salvati:")
    print("   - " + OUTPUT_DIRECTORY + "/MODELLO_NO_LLM_ATTACKER.h5")
    print("   - " + OUTPUT_DIRECTORY + "/MODELLO_NO_LLM_SUPPORT.h5")