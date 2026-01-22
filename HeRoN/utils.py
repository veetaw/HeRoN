SUPPORTER_MAX_HP = 2300
SUPPORTER_MAX_MP = 180

ATTACKER_MAX_HP = 2600
ATTACKER_MAX_MP = 120

ENEMY_MAX_HP = 6000
ENEMY_MAX_MP = 400

def build_prompt(game_description_attacker, game_description_supporter, last_enemy_move):
    return f"""You are a game assistant coordinating 2 players (attacker and supporter) in battle against an enemy.

            GAME SETUP:
            Supporter: max {SUPPORTER_MAX_HP} HP, {SUPPORTER_MAX_MP} MP
            Attacker: max {ATTACKER_MAX_HP} HP, {ATTACKER_MAX_MP} MP
            Enemy: {ENEMY_MAX_HP} HP, {ENEMY_MAX_MP} MP

            CURRENT STATE:
            Attacker: {game_description_attacker}
            Supporter: {game_description_supporter}
            Enemy last move: {last_enemy_move}

            ROLES:
            Attacker: Focuses on dealing damage to enemy
            Supporter: Can attack OR heal (self/mate/both) based on necessity

            JSON schema (strict):
            {{
              "attacker": "string, one of allowed actions",
              "supporter": "string, one of allowed actions",
              "reason_action_attacker": "string, max 40 words",
              "reason_action_supporter": "string, max 40 words"
            }}

            Respond ONLY with the JSON object, no additional text."""
