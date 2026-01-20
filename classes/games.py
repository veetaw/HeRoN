from classes.game import Person
from classes.magic import Spell
from classes.inventory import Item

PLAYER_1_NAME = "Giocatore1"
PLAYER_2_NAME = "Giocatore2"

ENEMY_NAME = "Nemico"

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

player1 = Person(PLAYER_1_NAME, 2600, 120, 300, 34, attacker_spells, player_items)
player2 = Person(PLAYER_2_NAME, 2300, 180, 100, 50, support_spells, player_items)
enemy1 = Person(ENEMY_NAME, 6500, 220, 525, 25, [fire, cura], [])

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
