"""
CS2 Win Probability Model.

Pure math module — no I/O, no side effects, independently testable.

Calculates the probability of team_a winning a CS2 match based on:
- Current map score in the series (Bo1/Bo3/Bo5)
- Current round score within the active map (MR12, first to 13)
- Which side (CT/T) team_a is currently playing
- Team economy state (eco/force/full buy)
- Overtime rules (MR3 format)

Uses recursive probability with memoization.
"""
import functools
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class CS2MapState:
    """State of the current map being played."""
    score_a: int = 0            # rounds won by team_a this map
    score_b: int = 0            # rounds won by team_b this map
    team_a_side: str = "ct"     # "ct" or "t" — team_a's current side
    team_a_money: int = 0       # total team economy (sum of 5 players)
    team_b_money: int = 0
    is_overtime: bool = False


def classify_buy(team_money: int) -> str:
    """
    Classify team buy level based on total team money (5 players).

    Returns: "eco", "force", or "full"
    """
    if team_money <= 0:
        return "unknown"
    if team_money < config.CS2_ECO_THRESHOLD:
        return "eco"
    elif team_money < config.CS2_FORCE_THRESHOLD:
        return "force"
    else:
        return "full"


def get_round_win_rate(
    team_a_side: str,
    team_a_buy: str,
    team_b_buy: str,
    round_number: int,
) -> float:
    """
    Calculate team_a's probability of winning the current round.

    Combines side advantage with economy matchup modifiers.

    Args:
        team_a_side: "ct" or "t"
        team_a_buy: "eco", "force", "full", or "unknown"
        team_b_buy: same
        round_number: 1-based round number (1 and 13 are pistol rounds)
    """
    # Pistol rounds (1 and 13) are approximately 50/50
    if round_number in (1, 13):
        return config.CS2_PISTOL_WIN_RATE

    # Base win rate from side advantage
    if team_a_side == "ct":
        base_p = config.CS2_CT_WIN_RATE
    else:
        base_p = 1.0 - config.CS2_CT_WIN_RATE

    # Economy modifiers — only apply when both teams' economy is known
    if team_a_buy == "unknown" or team_b_buy == "unknown":
        return base_p

    # Economy matchup matrix
    # Modifier is added to team_a's win rate
    ECONOMY_MODIFIERS = {
        ("full", "eco"):    0.30,   # team_a has rifles, team_b saving
        ("full", "force"):  0.15,   # team_a full buy vs partial buy
        ("full", "full"):   0.0,    # even buy, pure side advantage
        ("force", "eco"):   0.15,
        ("force", "force"): 0.0,
        ("force", "full"):  -0.15,
        ("eco", "eco"):     0.0,    # both saving, chaotic
        ("eco", "force"):   -0.15,
        ("eco", "full"):    -0.30,  # team_a saving vs full buy
    }

    modifier = ECONOMY_MODIFIERS.get((team_a_buy, team_b_buy), 0.0)
    p = base_p + modifier

    # Clamp to reasonable range
    return max(0.05, min(0.95, p))


def get_side_for_round(start_side: str, round_number: int) -> str:
    """
    Determine team_a's side for a given round number.

    CS2 MR12: rounds 1-12 = first half, 13-24 = second half (sides swap).
    Overtime MR3: sides swap every 3 rounds.

    Args:
        start_side: team_a's side at the start of this map ("ct" or "t")
        round_number: 1-based round number
    """
    if round_number <= 12:
        # First half
        return start_side
    elif round_number <= 24:
        # Second half — sides swapped
        return "t" if start_side == "ct" else "ct"
    else:
        # Overtime — sides swap every 3 rounds
        # OT starts at round 25, first OT half is rounds 25-27, second is 28-30, etc.
        ot_round = round_number - 24  # 1-based within OT
        ot_half = (ot_round - 1) // 3  # 0-based half index
        if ot_half % 2 == 0:
            return start_side
        else:
            return "t" if start_side == "ct" else "ct"


@functools.lru_cache(maxsize=2048)
def _map_win_prob_recursive(
    score_a: int,
    score_b: int,
    team_a_side: str,
    team_a_start_side: str,
    team_a_buy: str,
    team_b_buy: str,
) -> float:
    """
    Recursively compute P(team_a wins this map) from current score.

    Uses economy data for the immediate next round only, then assumes
    side-advantage-only for deeper recursion (economy is unknown for future rounds).

    Base cases:
        score_a >= 13 → team_a wins (1.0)
        score_b >= 13 → team_b wins (0.0)
        Both at 12 → overtime

    For overtime (MR3): first to 16 wins (4 OT rounds needed from 12-12).
    Simplified: model OT as 50/50 per round with slight CT lean.
    """
    # Base cases — regulation
    if score_a >= 13 and score_b < 13:
        return 1.0
    if score_b >= 13 and score_a < 13:
        return 0.0

    # Overtime base cases (first to 16 in standard MR3 OT)
    if score_a >= 13 and score_b >= 13:
        # In overtime — simplified to first to +4 above 12
        if score_a >= 16 and score_a > score_b:
            return 1.0
        if score_b >= 16 and score_b > score_a:
            return 0.0
        # Deep overtime — approximate as 50/50
        if score_a > 19 or score_b > 19:
            return 0.5

    current_round = score_a + score_b + 1
    current_side = get_side_for_round(team_a_start_side, current_round)

    # For the immediate next round, use economy data
    # For deeper recursion, use "unknown" economy (side-only)
    is_immediate = (current_side == team_a_side and
                    team_a_buy != "unknown" and team_b_buy != "unknown")

    if is_immediate:
        p = get_round_win_rate(current_side, team_a_buy, team_b_buy, current_round)
    else:
        p = get_round_win_rate(current_side, "unknown", "unknown", current_round)

    # Recursive: P(win) = p * P(win | we win this round) + (1-p) * P(win | we lose)
    p_win = p * _map_win_prob_recursive(
        score_a + 1, score_b, current_side, team_a_start_side, "unknown", "unknown"
    ) + (1 - p) * _map_win_prob_recursive(
        score_a, score_b + 1, current_side, team_a_start_side, "unknown", "unknown"
    )

    return p_win


@functools.lru_cache(maxsize=256)
def _series_win_prob(maps_a: int, maps_b: int, best_of: int, current_map_prob: float) -> float:
    """
    Compute P(team_a wins series) given current map scores.

    Uses current_map_prob for the active map, and 0.5 for future maps
    (no side/economy info for maps not yet started).
    """
    maps_needed = (best_of // 2) + 1

    if maps_a >= maps_needed:
        return 1.0
    if maps_b >= maps_needed:
        return 0.0

    # Current map uses our detailed probability
    # Future maps assume 50/50
    p_this_map = current_map_prob

    # If team_a wins this map
    p_win_series_if_win = _series_win_prob(maps_a + 1, maps_b, best_of, 0.5)
    # If team_a loses this map
    p_win_series_if_lose = _series_win_prob(maps_a, maps_b + 1, best_of, 0.5)

    return p_this_map * p_win_series_if_win + (1 - p_this_map) * p_win_series_if_lose


class CS2WinProbabilityModel:
    """
    CS2-specific win probability calculator.

    Combines map-level and series-level recursive probability models
    with CS2 mechanics: MR12, side advantage, economy cycles.
    """

    def calculate(
        self,
        maps_a: int = 0,
        maps_b: int = 0,
        best_of: int = 1,
        round_score_a: int = 0,
        round_score_b: int = 0,
        team_a_side: str = "ct",
        team_a_start_side: Optional[str] = None,
        team_a_money: int = 0,
        team_b_money: int = 0,
    ) -> float:
        """
        Calculate probability that team_a wins the match.

        Args:
            maps_a: Maps won by team_a in the series
            maps_b: Maps won by team_b
            best_of: Series format (1, 3, or 5)
            round_score_a: Rounds won by team_a on current map
            round_score_b: Rounds won by team_b on current map
            team_a_side: team_a's current side ("ct" or "t")
            team_a_start_side: team_a's side at map start (defaults to current)
            team_a_money: Total team economy for team_a (0 = unknown)
            team_b_money: Total team economy for team_b (0 = unknown)

        Returns:
            float: Probability of team_a winning the match (0.0 to 1.0)
        """
        if team_a_start_side is None:
            team_a_start_side = team_a_side

        # Classify economy
        team_a_buy = classify_buy(team_a_money)
        team_b_buy = classify_buy(team_b_money)

        # Calculate current map win probability
        map_prob = _map_win_prob_recursive(
            round_score_a,
            round_score_b,
            team_a_side,
            team_a_start_side,
            team_a_buy,
            team_b_buy,
        )

        # For Bo1, map probability IS match probability
        if best_of == 1:
            return max(0.01, min(0.99, map_prob))

        # For series, combine map probability with series state
        # Round current_map_prob to 2 decimals for cache efficiency
        rounded_map_prob = round(map_prob, 2)
        series_prob = _series_win_prob(maps_a, maps_b, best_of, rounded_map_prob)

        return max(0.01, min(0.99, series_prob))


# ─── Convenience for testing ─────────────────────────────────────────────────

def _test_model():
    """Quick sanity checks for the probability model."""
    model = CS2WinProbabilityModel()

    cases = [
        # (maps_a, maps_b, bo, round_a, round_b, side, money_a, money_b, expected_approx)
        (0, 0, 1, 0, 0, "ct", 0, 0, 0.50),      # match start, CT side
        (0, 0, 1, 12, 0, "ct", 0, 0, 1.0),       # 12-0 dominant
        (0, 0, 1, 0, 12, "ct", 0, 0, 0.0),       # 0-12 getting crushed
        (0, 0, 1, 6, 6, "ct", 0, 0, 0.53),       # halftime tied, CT side
        (0, 0, 1, 12, 12, "ct", 0, 0, 0.50),     # OT, should be ~50/50
        (1, 0, 3, 0, 0, "ct", 0, 0, 0.75),       # up 1-0 in Bo3
        (0, 1, 3, 0, 0, "ct", 0, 0, 0.25),       # down 0-1 in Bo3
        (0, 0, 1, 6, 5, "ct", 50000, 5000, 0.70), # ahead + full buy vs eco
    ]

    print("CS2 Win Probability Model — Sanity Checks")
    print("-" * 70)
    for maps_a, maps_b, bo, r_a, r_b, side, m_a, m_b, expected in cases:
        prob = model.calculate(
            maps_a=maps_a, maps_b=maps_b, best_of=bo,
            round_score_a=r_a, round_score_b=r_b,
            team_a_side=side, team_a_money=m_a, team_b_money=m_b,
        )
        status = "OK" if abs(prob - expected) < 0.15 else "CHECK"
        print(
            f"  Bo{bo} maps={maps_a}-{maps_b} rounds={r_a}-{r_b} "
            f"side={side} eco={classify_buy(m_a)}/{classify_buy(m_b)} "
            f"→ P={prob:.3f} (expect ~{expected:.2f}) [{status}]"
        )
    print()


if __name__ == "__main__":
    _test_model()
