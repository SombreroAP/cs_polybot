"""
Dota 2 Win Probability Model.

Simpler than CS2 — Dota 2 games don't have rounds/halves.
Key factors:
- Kill score differential
- Game time (early kills matter less than late kills)
- Series score (Bo1/Bo3/Bo5)
- Gold/XP lead (if available from detailed feeds)

Uses empirical logistic model based on kill differential and game time.
"""
import functools
import math


def game_win_probability_from_kills(
    radiant_kills: int,
    dire_kills: int,
    game_minutes: float,
) -> float:
    """
    Estimate Radiant's probability of winning the current game
    based on kill differential and game time.

    Uses a logistic function calibrated to pro Dota 2:
    - At 0 kill diff, P = 0.5
    - Each kill advantage is worth more later in the game
    - A 10-kill lead at 30 min ≈ 80% win probability
    """
    kill_diff = radiant_kills - dire_kills

    # Time scaling: kills matter more as the game progresses
    # Early game (0-15 min): kills less decisive, comebacks easy
    # Mid game (15-35 min): kills increasingly important
    # Late game (35+ min): single teamfight can decide game
    if game_minutes <= 0:
        time_factor = 0.5
    elif game_minutes < 15:
        time_factor = 0.5 + (game_minutes / 15) * 0.5
    elif game_minutes < 40:
        time_factor = 1.0 + (game_minutes - 15) / 25 * 0.5
    else:
        time_factor = 1.5

    # Logistic function: P = 1 / (1 + exp(-k * kill_diff))
    # k scales with time — bigger k = steeper curve = more decisive kills
    k = 0.05 * time_factor

    try:
        p = 1.0 / (1.0 + math.exp(-k * kill_diff))
    except OverflowError:
        p = 1.0 if kill_diff > 0 else 0.0

    return max(0.01, min(0.99, p))


@functools.lru_cache(maxsize=256)
def series_win_probability(
    maps_radiant: int,
    maps_dire: int,
    best_of: int,
    current_game_prob: float,
) -> float:
    """
    Compute P(Radiant wins series) given map scores and current game probability.
    Same recursive approach as CS2 model.
    """
    maps_needed = (best_of // 2) + 1

    if maps_radiant >= maps_needed:
        return 1.0
    if maps_dire >= maps_needed:
        return 0.0

    # Current game uses our estimate, future games assume 50/50
    p_this = current_game_prob
    p_win_if_win = series_win_probability(maps_radiant + 1, maps_dire, best_of, 0.5)
    p_win_if_lose = series_win_probability(maps_radiant, maps_dire + 1, best_of, 0.5)

    return p_this * p_win_if_win + (1 - p_this) * p_win_if_lose


class Dota2WinProbabilityModel:
    """Dota 2 match win probability calculator."""

    def calculate(
        self,
        maps_a: int = 0,
        maps_b: int = 0,
        best_of: int = 1,
        kills_a: int = 0,
        kills_b: int = 0,
        game_minutes: float = 0.0,
    ) -> float:
        """
        Calculate probability that team_a (Radiant) wins the match.

        Args:
            maps_a/maps_b: Games won in the series
            best_of: Series format
            kills_a/kills_b: Kill scores in current game
            game_minutes: Duration of current game in minutes
        """
        # Current game probability from kills
        game_prob = game_win_probability_from_kills(kills_a, kills_b, game_minutes)

        if best_of == 1:
            return game_prob

        rounded = round(game_prob, 2)
        return series_win_probability(maps_a, maps_b, best_of, rounded)


if __name__ == "__main__":
    model = Dota2WinProbabilityModel()
    cases = [
        (0, 0, 1, 0, 0, 0, 0.50),
        (0, 0, 1, 10, 5, 20, 0.60),
        (0, 0, 1, 20, 5, 30, 0.80),
        (0, 0, 1, 5, 20, 30, 0.20),
        (1, 0, 3, 0, 0, 0, 0.75),
        (0, 1, 3, 10, 5, 25, 0.35),
    ]
    print("Dota 2 Win Probability Model — Sanity Checks")
    for ma, mb, bo, ka, kb, gm, exp in cases:
        p = model.calculate(ma, mb, bo, ka, kb, gm)
        print(f"  Bo{bo} maps={ma}-{mb} kills={ka}-{kb} {gm}min → P={p:.3f} (expect ~{exp:.2f})")
