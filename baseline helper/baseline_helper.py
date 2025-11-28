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


def map_llm_action_to_attacker_action(llm_response):
    match = re.search(r'\[(.*?)\]', llm_response)
    if match:
        action = match.group(1).strip().lower()
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
    return None

def map_llm_action_to_supporter_action(llm_response):
    match = re.search(r'\[(.*?)\]', llm_response)
    if match:
        action = match.group(1).strip().lower()
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

#aggiungere la nuova map action del supporter

def map_action_support(action):
    """Mappa l'indice azione al nome per SUPPORT"""
    actions_map = {
        0: 'attack',
        1: 'fire spell',
        2: 'cura spell',     # Auto-cure (white)
        3: 'cura_tot',       # Cura entrambi (white_tot)
        4: 'splash',         # Cura entrambi meno (white_tot)
        5: 'cura_m',         # Cura mate (white_m)
        6: 'cura_totm',      # Cura mate tanto (white_m)
        7: 'potion',
        8: 'grenade',
        9: 'elixir'
    }
    return actions_map.get(action, 'attack')

def train_dqn(episodes, batch_size=32, load_model_path=None):
    #environment settings
    attacker_spells = [fire, thunder, blizzard, meteor, cura]
    
    # SUPPORT spells (offensivi base + cure variegate)
    support_spells = [fire, cura_support, cura_tot, splash, cura_m, cura_totm]

    player_items = [
        {"item": potion, "quantity": 3}, 
        {"item": grenade, "quantity": 2},
        {"item": hielixer, "quantity": 1}
    ]
    player1 = Person("Maria", 2600, 120, 300, 34, attacker_spells, player_items)
    player2 = Person("Juana", 2300, 180, 100, 50, support_spells, player_items)
    enemy1 = Person("Antonio", 8000, 701, 525, 25, [fire, cura], [])

    players = [player1, player2]
    enemies = [enemy1]

    env = BattleEnv(players, enemies)
    #NPC
    # al posto di env.state_size va messo len(env.get state (index)), idem per actiopn size
    attacker_agent = DQNAgent(
        env.get_state_size_of_player('Maria'), 
        env.get_action_size(0), 
        load_model_path
    )
    supporter_agent = DQNSupportAgent(
        env.get_state_size_of_player('Juana'), 
        env.get_action_size(1), 
        load_model_path
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
        state_attacker = state_global['Maria']
        state_attacker = np.reshape(state_attacker, [1, env.get_state_size_of_player('Maria')])
        state_support = state_global['Juana']
        state_support = np.reshape(state_support, [1, env.get_state_size_of_player('Juana')])

        done = False
        total_reward_support = 0
        total_reward_attacker = 0
        moves = 0
        match_score_attacker = []
        match_score_support = []
        last_enemy_move = "No action"
        score.reset_quantities()
        state_size_attacker = env.get_state_size_of_player('Maria')
        state_size_support = env.get_state_size_of_player('Juana')
        while not done:
            player_attacker = players[0]
            player_support = players[1]

            attacker_action = None
            support_action = None

            """ Prompt separati per i due agenti
            if player_attacker.get_hp() > 0:
                game_description_attacker = env.describe_game_state_attacker(last_enemy_move)
                input_text_attacker = "You are a game asstitant for player." \
                                    "The battle involves with 2 players (attacker and supporter) and an enemy. Both player and enemy are characterised by:" \
                                    "HP (health point), MP (magic point), Attack points, Defence points, Available magic points " \
                                    "and Available items. " \
                                    "For spells, the use of MP is necessary, while items have limited availability. " \
                                    "Now you are the attacker, the player can have a maximum of 2600 HP and 120 MP, while the enemy 8000 HP and 701 MP. " \
                                    f"Given the game state '{game_description_attacker}', what is the next action to take? " \
                                    "Write only the chosen action in square brackets and " \
                                    "explain your reasoning briefly, max 50 words. /no_think"
                
                llm_response_attacker = get_llm_response(input_text_attacker)
                llm_response = re.sub(r"<think>.*?</think>", "", llm_response_attacker, flags=re.DOTALL).strip()
                print(f"[ATK] LLM response: {llm_response_attacker}")

                attacker_action = map_llm_action_to_attacker_action(llm_response)
    
                if attacker_action != None:
                    match_attacker = re.search(r'\[(.*?)\]', llm_response)
                    match_attacker = match_attacker.group(1).strip().lower()
                    if match_attacker == "elixer":
                        match_attacker = "elixir"
                    attacker_scores = score.calculate_scores_attacker(
                        player_attacker.get_hp(), 
                        player_attacker.get_mp(), 
                        enemies[0].get_hp()
                    )
                else:
                    attacker_action = attacker_agent.act(state_attacker, env, 0)
                    match_attacker = map_action_attack(attacker_action)
                    attacker_scores = score.calculate_scores_attacker(
                        player_attacker.get_hp(), 
                        player_attacker.get_mp(), 
                        enemies[0].get_hp()
                    )
                    allucination += 1
            else:
                attacker_action = None
                match_attacker = "attack"
                attacker_scores = {match_attacker: 0}

            if player_support.get_hp() > 0:
                game_description_supporter = env.describe_game_state_supporter(last_enemy_move)
                input_text_supporter = "You are a game asstitant for player." \
                                    "The battle involves with 2 players (attacker and supporter) and an enemy. Both player and enemy are characterised by:" \
                                    "HP (health point), MP (magic point), Attack points, Defence points, Available magic points " \
                                    "and Available items. " \
                                    "For spells, the use of MP is necessary, while items have limited availability. " \
                                    "Now you are the SUPPORTER, the player can have a maximum of 2300 HP and 180 MP, your mate can have a maximum of 2600 HP and 120 MP, while the enemy 8000 HP and 701 MP. " \
                                    "Since you are the supporter, you can both attack the enemy and cure either you or the mate or both by choosing wisely (only when necessary, you can also attack if the mate does not need curing) " \
                                    f"Given the mate's game state: '{game_description_attacker}', and " \
                                    f"your game state: '{game_description_supporter}', what is the next action to take? " \
                                    "Write only the chosen action in square brackets and " \
                                    "explain your reasoning briefly, max 50 words. /no_think"

                llm_response_support = get_llm_response(input_text_supporter)
                llm_response = re.sub(r"<think>.*?</think>", "", llm_response_support, flags=re.DOTALL).strip()
                print(f"[SUP] LLM response: {llm_response_support}")

                support_action = map_llm_action_to_supporter_action(llm_response)

                if support_action != None:
                    match_support = re.search(r'\[(.*?)\]', llm_response)
                    match_support = match_support.group(1).strip().lower()
                    if match_support == "elixer":
                        match_support = "elixir"
                    support_scores = score.calculate_scores_support(
                        player_support.get_hp(), 
                        player_attacker.get_hp(),
                        player_support.get_mp(), 
                        enemies[0].get_hp()
                    )
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
            else:
                support_action = None
                match_support = "attack"
                support_scores = {match_support: 0}
             Prompt separati per i due agenti - END """
            game_description_attacker = env.describe_game_state_attacker(None)
            game_description_supporter = env.describe_game_state_supporter(None)
            prompt =f"""You are a game assistant coordinating 2 players (attacker and supporter) in battle against an enemy.

                        GAME SETUP:
                        Supporter: max 2300 HP, 180 MP
                        Attacker: max 2600 HP, 120 MP
                        Enemy: 8000 HP, 701 MP

                        CURRENT STATE:
                        Attacker: {game_description_attacker}
                        Supporter: {game_description_supporter}
                        Enemy last move: {last_enemy_move}

                        ROLES:
                        Attacker: Focuses on dealing damage to enemy
                        Supporter: Can attack OR heal (self/mate/both) based on necessity

                        REQUIRED OUTPUT (valid JSON only):
                        {{
                        "attacker": "[ACTION]",
                        "supporter": "[ACTION]",
                        "reason_action_attacker": "Max 40 words explaining why this action",
                        "reason_action_supporter": "Max 40 words explaining why this action"
                        }}
                        note the parenthesis in the action, they're needed
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
                        print(f"AZIONE CHIESTA DALL'LLM PER Maria: {llm_requested_attacker}, AZIONE ESEGUITA DA Maria: {match_attacker}")
                else:
                    attacker_action = attacker_agent.act(state_attacker, env, 0)
                    match_attacker = map_action_attack(attacker_action)
                    attacker_scores = score.calculate_scores_attacker(
                        player_attacker.get_hp(), 
                        player_attacker.get_mp(), 
                        enemies[0].get_hp()
                    )
                    allucination += 1
                    print(f"AZIONE CHIESTA DALL'LLM PER Maria: {llm_requested_attacker} (NON VALIDA), AZIONE ESEGUITA DA Maria: {match_attacker} (da DQN)")

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
                        print(f"AZIONE CHIESTA DALL'LLM PER Juana: {llm_requested_support}, AZIONE ESEGUITA DA Juana: {match_support}")
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
                    print(f"AZIONE CHIESTA DALL'LLM PER Juana: {llm_requested_support} (NON VALIDA), AZIONE ESEGUITA DA Juana: {match_support} (da DQN)")
            except Exception as e:
                print("Errore, ", e)
            match_score_attacker.append(round(attacker_scores.get(match_attacker, 0), 2))
            match_score_support.append(round(support_scores.get(match_support, 0), 2))

            next_state, reward_attacker, reward_support, done, a_win, last_enemy_move, __ = env.step(attacker_action, support_action)
            print(f"\n[Move {moves:02d}] Ep {e+1}/{episodes}")
            
            status_atk = "ATK" if player_attacker.get_hp() > 0 else "DEAD"
            status_sup = "SUP" if player_support.get_hp() > 0 else "DEAD"
            
            print(f"  [{status_atk}] {match_attacker:<18} (HP: {player_attacker.get_hp():>4}, MP: {player_attacker.get_mp():>3}) → score: {attacker_scores.get(match_attacker, 0):.2f}")
            print(f"  [{status_sup}] {match_support:<18} (HP: {player_support.get_hp():>4}, MP: {player_support.get_mp():>3}) → score: {support_scores.get(match_support, 0):.2f}")
            print(f"  Enemy HP: {enemies[0].get_hp():>4}/{enemies[0].maxhp}")

            print(f"  Reward ATK: {reward_attacker:+4d} | SUP: {reward_support:+4d}")
 
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
    plt.savefig("Train_reward_DQN.png")

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

    plt.figure(figsize=(8, 6))
    plt.plot(moves)
    plt.title('Number of Moves per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Moves')
    plt.savefig("Train_moves_DQN.png")

    plt.figure(figsize=(8, 6))
    plt.plot(success_rate, label="Success Rate", color='blue')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig("Train_success_rate_DQN.png")

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
    plt.savefig("Score_DQN_separated.png")



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
    rewards, agent_wins, enemy_wins, moves, success_rate, action_scores = train_dqn(episodes=1)
    
    # Plot dei risultati
    plot_training(rewards, agent_wins, enemy_wins, moves, success_rate, action_scores)
    export_success_rate(success_rate)
