import csv
import random
import json


actions_attacker = {
    'attack': {'damage': 300, 'mp_cost': 0, 'heal': 0},
    'fire spell': {'damage': 600, 'mp_cost': 25, 'heal': 0},
    'thunder spell': {'damage': 700, 'mp_cost': 30, 'heal': 0},
    'blizzard spell': {'damage': 800, 'mp_cost': 35, 'heal': 0},
    'meteor spell': {'damage': 1000, 'mp_cost': 40, 'heal': 0},
    'cura spell': {'damage': 0, 'mp_cost': 32, 'heal': 1500},
    'potion': {'damage': 0, 'mp_cost': 0, 'heal': 50},
    'grenade': {'damage': 500, 'mp_cost': 0, 'heal': 0},
    'elixer': {'damage': 0, 'mp_cost': 0, 'heal': 'full'}
}
actions_supporter = {
    'attack': {'damage': 200, 'mp_cost': 0, 'heal': 0},
    'fire spell': {'damage': 400, 'mp_cost': 20, 'heal': 0},
    'attack': {'damage': 250, 'mp_cost': 0, 'heal': 0, 'quantity': 1},
    'cura spell': {'damage': 0, 'mp_cost': 32, 'heal': 1200, 'quantity': 1},  # Auto-cure
    'cura_tot': {'damage': 0, 'mp_cost': 30, 'heal': 700, 'quantity': 1},     # Cura entrambi
    'splash': {'damage': 0, 'mp_cost': 18, 'heal': 450, 'quantity': 1},       # Cura entrambi ma meno
    'cura_m': {'damage': 0, 'mp_cost': 28, 'heal': 1300, 'quantity': 1},      # Cura il mate
    'cura_totm': {'damage': 0, 'mp_cost': 36, 'heal': 1700, 'quantity': 1},   # Cura di più il mate
}


def generate_game_scenario():
    attacker_hp = random.randint(10, 2600)
    attacker_mp = random.randint(0, 120)
    supporter_hp = random.randint(10, 2300)
    supporter_mp = random.randint(0, 180)
    enemy_hp = random.randint(10, 8000)
    enemy_mp = random.randint(0, 701)
    available_items_attacker = {
        'potion': random.randint(0, 3),
        'grenade': random.randint(0, 2),
        'elixer': random.randint(0, 1)
    }
    available_items_supporter = {
        'potion': random.randint(0, 3),
        'grenade': random.randint(0, 2),
        'elixer': random.randint(0, 1)
    }
    last_enemy_move = random.choice(['attack', 'fire spell', 'cura spell'])
    available_actions_attacker = {action: info for action, info in actions_attacker.items() if attacker_mp >= info['mp_cost']}
    available_actions_supporter = {action: info for action, info in actions_supporter.items() if supporter_mp >= info['mp_cost']}

    return {
        'attacker_hp': attacker_hp,
        'attacker_mp': attacker_mp,
        'supporter_hp': supporter_hp,
        'supporter_mp': supporter_mp,
        'enemy_hp': enemy_hp,
        'enemy_mp': enemy_mp,
        'available_items_attacker': available_items_attacker,
        'available_items_supporter': available_items_supporter,
        'last_enemy_move': last_enemy_move,
        'available_actions_attacker': available_actions_attacker,
        'available_actions_supporter': available_actions_supporter
    }


def generate_response(scenario):
    action_attacker = random.choice(list(scenario['available_actions_attacker'].keys()))
    action_supporter = random.choice(list(scenario['available_actions_supporter'].keys()))
    response = {
        "attacker": action_attacker,
        "supporter": action_supporter
    }
    return json.dumps(response)


def generate_instructions(scenario, response):
    attacker_hp = scenario['attacker_hp']
    attacker_mp = scenario['attacker_mp']
    supporter_hp = scenario['supporter_hp']
    supporter_mp = scenario['supporter_mp']
    enemy_hp = scenario['enemy_hp']
    
    # Logica per ATTACKER
    instructions_attacker = ""
    if attacker_hp < 800 and attacker_mp >= 32:
        instructions_attacker = "Attacker: [cura spell] because HP is low"
    elif attacker_hp < 500:
        if scenario['available_items_attacker']['elixer'] > 0:
            instructions_attacker = "Attacker: [elixer] to fully restore MP and HP"
        elif scenario['available_items_attacker']['potion'] > 0:
            instructions_attacker = "Attacker: [potion] to restore some HP"
        else:
            instructions_attacker = "Attacker: [attack] as no healing items are available"
    else:
        # Scegli il miglior attacco disponibile
        best_attack = None
        best_damage = 0
        for action, info in scenario['available_actions_attacker'].items():
            if info['damage'] > best_damage:
                best_damage = info['damage']
                best_attack = action
        if best_attack:
            instructions_attacker = f"Attacker: [{best_attack}] to maximize damage"
        else:
            instructions_attacker = "Attacker: [attack] as default action"
    
    # Logica per SUPPORTER
    instructions_supporter = ""
    if supporter_hp < 600:
        # Supporter si cura
        if supporter_mp >= 32:
            instructions_supporter = "Supporter: [cura spell] because HP is low"
        elif scenario['available_items_supporter']['potion'] > 0:
            instructions_supporter = "Supporter: [potion] to restore some HP"
        elif scenario['available_items_supporter']['elixer'] > 0:
            instructions_supporter = "Supporter: [elixer] to fully restore MP and HP"
        else:
            instructions_supporter = "Supporter: [attack]"
    elif attacker_hp < 600 and supporter_mp >= 28:
        # Supporter cura l'attacker (cura_m)
        instructions_supporter = "Supporter: [cura_m] to heal attacker"
    elif supporter_mp >= 30:
        # Supporter cura entrambi (cura_tot) se hp bassi
        if (attacker_hp + supporter_hp) < 1200:
            instructions_supporter = "Supporter: [cura_tot] to heal both attacker and supporter"
        else:
            # Altrimenti attacca
            best_attack = None
            best_damage = 0
            for action, info in scenario['available_actions_supporter'].items():
                if info['damage'] > best_damage:
                    best_damage = info['damage']
                    best_attack = action
            if best_attack and best_damage > 0:
                instructions_supporter = f"Supporter: [{best_attack}] to maximize damage"
            else:
                instructions_supporter = "Supporter: [attack] as default action"
    else:
        instructions_supporter = "Supporter: [attack] as default action"
    
    instructions = {
        "attacker": instructions_attacker,
        "supporter": instructions_supporter
    }
    return json.dumps(instructions)


def generate_dataset(n=4000):
    dataset = []
    for _ in range(n):
        scenario = generate_game_scenario()
        response = generate_response(scenario)
        instructions = generate_instructions(scenario, response)
        prompt = (
            f"Attacker HP: {scenario['attacker_hp']}, MP: {scenario['attacker_mp']}. "
            f"Supporter HP: {scenario['supporter_hp']}, MP: {scenario['supporter_mp']}. "
            f"Enemy HP: {scenario['enemy_hp']}, MP: {scenario['enemy_mp']}. "
            f"Attacker items: {scenario['available_items_attacker']}. "
            f"Supporter items: {scenario['available_items_supporter']}. "
            f"Last enemy move: {scenario['last_enemy_move']}."
        )
        dataset.append({
            'prompt': prompt,
            'response': response,
            'instructions': instructions
        })
    return dataset


def save_dataset_to_csv(dataset, filename='game_scenarios_dataset_4.csv'):
    keys = dataset[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)


dataset = generate_dataset()

save_dataset_to_csv(dataset)
