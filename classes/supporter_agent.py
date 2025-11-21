class SupporterAgent:
    def __init__(self, player_index=1):
        self.player_index = player_index

    def _find_spell_index(self, player, key):
        for i, s in enumerate(player.magic):
            if key.lower() in s.name.lower():
                return i
        return None

    def act(self, env):
        # Rule-based support: heal ally/self when low, otherwise small fire or attack
        if self.player_index >= len(env.players):
            return None

        supporter = env.players[self.player_index]
        ally = env.players[0]

        ally_pct = ally.get_hp() / ally.get_max_hp() if ally.get_max_hp() > 0 else 0
        self_pct = supporter.get_hp() / supporter.get_max_hp() if supporter.get_max_hp() > 0 else 0

        # Prefer strong heal for ally if very low
        if ally_pct < 0.4:
            idx = self._find_spell_index(supporter, 'curatotm')
            if idx is None:
                idx = self._find_spell_index(supporter, 'curam')
            if idx is None:
                idx = self._find_spell_index(supporter, 'cura')
            if idx is not None and supporter.get_mp() >= supporter.magic[idx].cost:
                return idx + 1

        # Heal self if low
        if self_pct < 0.4:
            idx = self._find_spell_index(supporter, 'curatot')
            if idx is None:
                idx = self._find_spell_index(supporter, 'cura')
            if idx is not None and supporter.get_mp() >= supporter.magic[idx].cost:
                return idx + 1

        # Minor heal for ally if moderately low
        if ally_pct < 0.7:
            idx = self._find_spell_index(supporter, 'curam')
            if idx is None:
                idx = self._find_spell_index(supporter, 'cura')
            if idx is not None and supporter.get_mp() >= supporter.magic[idx].cost:
                return idx + 1

        # If nothing to heal, try small offensive spell 'fire'
        idx = self._find_spell_index(supporter, 'fire')
        if idx is not None and supporter.get_mp() >= supporter.magic[idx].cost:
            return idx + 1

        # Fallback to basic attack
        return 0
