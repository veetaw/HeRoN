from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item

PLAYER_1_NAME = "Giocatore1"
PLAYER_1_HEALTH = 2600
PLAYER_1_MP = 120
PLAYER_1_ATK = 300
PLAYER_1_DIF = 34


PLAYER_2_NAME = "Giocatore2"
PLAYER_2_HEALTH = 2300
PLAYER_2_MP = 180
PLAYER_2_ATK = 100
PLAYER_2_DIF = 50

ENEMY_NAME = "Nemico"
ENEMY_HEALTH = 6500
ENEMY_MP = 220
ENEMY_ATK = 525
ENEMY_DIF = 25

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


attacker_spells = [fire, thunder, blizzard, meteor, cura]
support_spells = [fire, cura_support, cura_tot, splash, cura_m, cura_totm]

player_items = [
        {"item": potion, "quantity": 3}, 
        {"item": grenade, "quantity": 2},
        {"item": hielixer, "quantity": 1}
    ]

player1 = Person(PLAYER_1_NAME, PLAYER_1_HEALTH, PLAYER_1_MP, PLAYER_1_ATK, PLAYER_1_DIF, attacker_spells, player_items)
player2 = Person(PLAYER_2_NAME, PLAYER_2_HEALTH, PLAYER_2_MP, PLAYER_2_ATK, PLAYER_2_DIF, support_spells, player_items)
enemy1 = Person(ENEMY_NAME, ENEMY_HEALTH, ENEMY_MP, ENEMY_ATK, ENEMY_DIF, [fire, cura], [])

def map_llm_action_to_attacker_action(llm_response):
    action = llm_response.strip().lower()
    if action == "attack":
        return 0
    elif action == "fire spell" or action == "fire_spell":
        return 1
    elif action == "thunder spell" or action == "thunder_spell":
        return 2
    elif action == "blizzard spell" or action == "blizzard_spell":
        return 3
    elif action == "meteor spell" or action == "meteor_spell":
        return 4
    elif action == "cura spell" or action == "cura_spell":
        return 5
    elif action == "potion":
        return 6
    elif action == "grenade":
        return 7
    elif action == "elixir" or action == "elixer":
        return 8
    elif action == "no_action":
        return -1
    return None

def map_llm_action_to_supporter_action(llm_response):
    action = llm_response.strip().lower()
    if action == "attack":
        return 0
    elif action == "fire spell":
        return 1
    elif action == "cura spell" or action == "cura_spell":
        return 2
    elif action == "cura_tot" or action == "cura tot":
        return 3
    elif action == "splash":
        return 4
    elif action == "cura_m" or action == "cura m":
        return 5
    elif action == "cura_totm" or action == "cura totm":
        return 6
    elif action == "potion":
        return 7
    elif action == "grenade":
        return 8
    elif action == "elixir" or action == "elixer":
        return 9
    elif action == "no_action":
        return -1
    return None

def map_action_attack(action):
    """Mappa l'indice azione al nome per ATTACKER"""
    actions_map = {
        -1: 'no_action',
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


def map_action_support(action):
    """Mappa l'indice azione al nome per SUPPORT"""
    actions_map = {
        -1: 'no_action',
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

