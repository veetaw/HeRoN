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

def map_action(action):
    if action == 0:
        return "attack"
    elif action == 1:
        return "fire spell"
    elif action == 2:
        return "thunder spell"
    elif action == 3:
        return "blizzard spell"
    elif action == 4:
        return "meteor spell"
    elif action == 5:
        return "cura spell"
    elif action == 6:
        return "potion"
    elif action == 7:
        return "grenade"
    elif action == 8:
        return "elixir"
    return None


# Main loop with training
def train_dqn(episodes, batch_size=32, load_model_path=None):
    # Environment settings
    player_spells = [fire, thunder, blizzard, meteor, cura]
    player_items = [{"item": potion, "quantity": 3}, {"item": grenade, "quantity": 2},
                    {"item": hielixer, "quantity": 1}]
    
    # Usa .copy() per dare a ogni player una copia indipendente degli item
    player1 = Person("Valos", 3260, 132, 300, 34, player_spells, player_items.copy())
    player2 = Person("Cristo", 3260, 132, 300, 34, player_spells, player_items.copy())
    enemy1 = Person("Magus", 4000, 701, 525, 25, [fire, cura], [])

    players = [player1, player2]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    
    # Due agenti DQN - uno per ogni player
    agent1 = DQNAgent(env.state_size, env.action_size, load_model_path)
    agent2 = DQNAgent(env.state_size, env.action_size, load_model_path)

    rewards_per_episode = []
    agent_wins = []
    enemy_wins = []
    agent_moves_per_episode = []
    success_rate = []
    action_scores = []

    total_agent_wins = 0

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0
        moves = 0
        match_score = []
        score.reset_quantity()
        
        while not done:
            # Player 1 (Valos) sceglie l'azione
            action1 = agent1.act(state, env)
            match1 = map_action(action1)
            total_score1 = score.calculate_scores(players[0].get_hp(), players[0].get_mp(), enemies[0].get_hp())
            match_score.append(round(total_score1.get(match1), 2))

            # Player 2 (Cristo) sceglie l'azione
            action2 = agent2.act(state, env)
            match2 = map_action(action2)
            total_score2 = score.calculate_scores(players[1].get_hp(), players[1].get_mp(), enemies[0].get_hp())
            match_score.append(round(total_score2.get(match2), 2))
            
            # Esegue entrambe le azioni nell'ambiente
            next_state, reward, done, a_win, e_win, enemy_choice = env.step([action1, action2])
            
            # Aggiorna le quantità per entrambi i player
            score.updage_quantity(match1, players[0].get_mp())
            score.updage_quantity(match2, players[1].get_mp())

            total_reward += reward
            next_state = np.reshape(next_state, [1, env.state_size])
            
            # Entrambi gli agenti salvano l'esperienza
            agent1.remember(state, action1, reward, next_state, done)
            agent2.remember(state, action2, reward, next_state, done)
            
            state = next_state
            moves += 1

            # Entrambi gli agenti fanno replay
            if len(agent1.memory) > batch_size:
                agent1.replay(batch_size, env)
            if len(agent2.memory) > batch_size:
                agent2.replay(batch_size, env)

            if done:
                print(f"Episode: {e}/{episodes}, Score: {total_reward}, Moves: {moves}")
                print(f"Epsilon - Agent1: {agent1.epsilon:.4f}, Agent2: {agent2.epsilon:.4f}")
                
                if a_win:
                    agent_wins.append(1)
                    enemy_wins.append(0)
                    total_agent_wins += 1
                else:
                    agent_wins.append(0)
                    enemy_wins.append(1)

                success_rate.append(total_agent_wins / (e + 1))
                print("Vittorie agente: ", agent_wins.count(1), " Vittorie nemico: ", enemy_wins.count(1))
                
        rewards_per_episode.append(total_reward)
        agent_moves_per_episode.append(moves)
        action_scores.append(np.mean(match_score))
        
    print("Average rewards: ", np.mean(rewards_per_episode))
    print("Average moves: ", np.mean(agent_moves_per_episode))
    print("Average move score: ", np.mean(action_scores))

    # Salva entrambi i modelli separatamente
    agent1.save("ModelloNoLLM_Player1")
    agent2.save("ModelloNoLLM_Player2")

    return rewards_per_episode, agent_wins, enemy_wins, agent_moves_per_episode, success_rate, action_scores


# Plotting function
def plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, match_score):
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.savefig("Train_reward_DQN.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    cumulative_agent_wins = np.cumsum(agent_wins)
    cumulative_enemy_wins = np.cumsum(enemy_wins)

    plt.plot(cumulative_agent_wins, label="Agent Wins (Cumulative)", color='green')
    plt.plot(cumulative_enemy_wins, label="Enemy Wins (Cumulative)", color='red')

    plt.legend()
    plt.title('Cumulative Wins of Agent vs Enemy per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Wins')
    plt.savefig("Train_cumulative_Win_DQN.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Train_moves_DQN.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Train_success_rate_DQN.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(match_score)
    plt.title('Score mosse per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Score')
    plt.savefig("Score_DQN.png")
    plt.close()


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
    # Spells and items setup
    fire = Spell("Fire", 25, 600, "black")
    thunder = Spell("Thunder", 30, 700, "black")
    blizzard = Spell("Blizzard", 35, 800, "black")
    meteor = Spell("Meteor", 40, 1000, "black")
    cura = Spell("Cura", 32, 1500, "white")

    potion = Item("Potion", "potion", "Heals 50 HP", 50)
    hielixer = Item("MegaElixer", "elixer", "Fully restores party's HP/MP", 9999)
    grenade = Item("Grenade", "attack", "Deals 500 damage", 500)

    # Train the agents
    rewards, agent_wins, enemy_wins, moves, success_rate, match_score = train_dqn(episodes=10)
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, match_score)

    export_success_rate(success_rate)