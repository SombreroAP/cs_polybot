"""
Team intelligence module — fetches team rankings, recent form, and H2H records
from free esports data sources to produce informed pre-match priors.

Sources:
- OpenDota API (Dota2): team ratings, recent matches, rosters
- HLTV (CS2): world rankings (via scraping)
- VLR.gg (Valorant): team pages with recent results

All data is cached in SQLite to avoid repeated API calls.
"""
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from prior import combine_priors

logger = logging.getLogger(__name__)


@dataclass
class TeamProfile:
    """Aggregated team intelligence from one or more sources."""
    name: str
    game: str
    team_id: str = ""
    ranking: int = 0           # 1 = best, 0 = unranked
    rating: float = 0.0        # source-specific ELO/rating
    recent_form: float = 0.5   # win rate in last N matches (0-1)
    recent_matches: int = 0    # how many recent matches we have data for
    roster_size: int = 0       # 0 = unknown
    last_updated: float = 0.0
    # New intel fields
    win_streak: int = 0        # positive = winning, negative = losing
    schedule_strength: float = 0.0  # avg opponent rating in recent matches
    hero_pool: list = field(default_factory=list)  # [{"hero": "Void", "wr": 0.78, "picks": 23}]


@dataclass
class MatchIntel:
    """Pre-match intelligence for a specific matchup."""
    team_a: TeamProfile
    team_b: TeamProfile
    prior_prob_a: float = 0.5
    confidence: float = 0.0
    sources: list = field(default_factory=list)
    # H2H
    h2h_wins_a: int = 0       # team_a wins vs team_b
    h2h_wins_b: int = 0       # team_b wins vs team_a
    h2h_last_played_days: int = 0  # days since last meeting


class TeamIntelligence:
    """Fetches and caches team data from free esports APIs."""

    # OpenDota
    OPENDOTA_API = "https://api.opendota.com/api"
    CACHE_TTL = 6 * 3600  # 6 hours for team profiles
    TEAMS_LIST_TTL = 24 * 3600  # 24 hours for full team list

    def __init__(self, db=None):
        self.db = db
        self._session: Optional[aiohttp.ClientSession] = None
        # In-memory caches
        self._dota2_teams: dict = {}  # name/tag -> team data
        self._dota2_teams_fetched: float = 0
        self._dota2_heroes: dict = {}  # hero_id -> hero_name
        self._dota2_heroes_fetched: float = 0
        self._hltv_rankings: dict = {}  # team_name -> rank
        self._hltv_rankings_fetched: float = 0

    async def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(headers={
                "User-Agent": "Mozilla/5.0 (Macintosh)",
                "Accept": "application/json",
            })

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ─── Main Entry Point ────────────────────────────────────────────────────

    async def get_match_intel(self, game: str, team_a: str, team_b: str) -> Optional[MatchIntel]:
        """Get pre-match intelligence for a matchup. Returns None if no data available."""
        try:
            if game == "dota2":
                return await self._get_dota2_intel(team_a, team_b)
            elif game == "cs2":
                return await self._get_cs2_intel(team_a, team_b)
            elif game == "valorant":
                return await self._get_valorant_intel(team_a, team_b)
        except Exception as e:
            logger.error(f"Team intel error ({game}): {e}")
        return None

    # ─── Dota 2 (OpenDota API) ───────────────────────────────────────────────

    async def _ensure_dota2_teams(self):
        """Fetch full team list from OpenDota (cached 24h)."""
        now = time.time()
        if self._dota2_teams and (now - self._dota2_teams_fetched) < self.TEAMS_LIST_TTL:
            return

        await self._ensure_session()
        try:
            async with self._session.get(
                f"{self.OPENDOTA_API}/teams",
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"OpenDota /teams returned {resp.status}")
                    return
                teams = await resp.json()

            # Index by lowercase name and tag for fuzzy lookup
            self._dota2_teams = {}
            for t in teams:
                name = (t.get("name") or "").lower().strip()
                tag = (t.get("tag") or "").lower().strip()
                if name:
                    self._dota2_teams[name] = t
                if tag and tag != name:
                    self._dota2_teams[tag] = t

            self._dota2_teams_fetched = now
            logger.info(f"OpenDota: cached {len(teams)} teams")
        except Exception as e:
            logger.error(f"OpenDota teams fetch failed: {e}")

    def _find_dota2_team(self, name: str) -> Optional[dict]:
        """Fuzzy-match a team name to the OpenDota teams list."""
        key = name.lower().strip()
        # Exact match
        if key in self._dota2_teams:
            return self._dota2_teams[key]
        # Substring match
        for k, v in self._dota2_teams.items():
            if key in k or k in key:
                return v
        # No spaces match
        nospace = key.replace(" ", "")
        for k, v in self._dota2_teams.items():
            if nospace == k.replace(" ", ""):
                return v
        return None

    async def _ensure_dota2_hero_names(self):
        """Fetch hero ID → name mapping. Cached 24h."""
        now = time.time()
        if self._dota2_heroes and (now - self._dota2_heroes_fetched) < self.TEAMS_LIST_TTL:
            return
        await self._ensure_session()
        try:
            async with self._session.get(
                f"{self.OPENDOTA_API}/heroes",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    heroes = await resp.json()
                    self._dota2_heroes = {h["id"]: h["localized_name"] for h in heroes}
                    self._dota2_heroes_fetched = now
        except Exception:
            pass

    async def _get_dota2_matches(self, team_id: int, limit: int = 30) -> list:
        """Fetch recent matches for a team. Returns list of match dicts."""
        await self._ensure_session()
        try:
            async with self._session.get(
                f"{self.OPENDOTA_API}/teams/{team_id}/matches",
                params={"limit": limit},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return []
                return await resp.json()
        except Exception:
            return []

    def _analyze_matches(self, matches: list, team_id: int) -> dict:
        """Analyze a team's recent matches for form, streak, schedule strength."""
        wins, total, streak, results = 0, 0, 0, []
        opp_ratings = []

        for m in matches:
            radiant_win = m.get("radiant_win")
            is_radiant = m.get("radiant")
            if radiant_win is None or is_radiant is None:
                continue
            won = (radiant_win and is_radiant) or (not radiant_win and not is_radiant)
            results.append(won)
            if won:
                wins += 1
            total += 1
            # Track opponent for schedule strength
            opp_name = m.get("opposing_team_name", "")
            opp_data = self._find_dota2_team(opp_name) if opp_name else None
            if opp_data:
                opp_ratings.append(opp_data.get("rating", 0))

        # Calculate streak from most recent
        streak = 0
        for won in results:
            if won and streak >= 0:
                streak += 1
            elif not won and streak <= 0:
                streak -= 1
            else:
                break

        form = wins / total if total > 0 else 0.5
        sched = sum(opp_ratings) / len(opp_ratings) if opp_ratings else 0.0

        return {"form": form, "total": total, "streak": streak, "schedule_strength": sched}

    async def _get_dota2_hero_pool(self, team_id: int) -> list:
        """Fetch top heroes for a team from OpenDota."""
        await self._ensure_session()
        await self._ensure_dota2_hero_names()
        try:
            async with self._session.get(
                f"{self.OPENDOTA_API}/teams/{team_id}/heroes",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return []
                heroes = await resp.json()

            pool = []
            for h in sorted(heroes, key=lambda x: -x.get("games_played", 0))[:8]:
                gp = h.get("games_played", 0)
                w = h.get("wins", 0)
                if gp < 3:
                    continue
                name = self._dota2_heroes.get(h.get("hero_id", 0), f"Hero#{h.get('hero_id', 0)}")
                pool.append({"hero": name, "wr": round(w / gp, 2) if gp > 0 else 0, "picks": gp})
            return pool
        except Exception:
            return []

    async def _get_dota2_h2h(self, matches_a: list, matches_b: list) -> tuple[int, int, int]:
        """Find head-to-head record from match histories. Returns (a_wins, b_wins, days_since_last)."""
        b_match_ids = {m.get("match_id") for m in matches_b}
        a_wins, b_wins = 0, 0
        last_time = 0

        for m in matches_a:
            if m.get("match_id") in b_match_ids:
                radiant_win = m.get("radiant_win")
                is_radiant = m.get("radiant")
                if radiant_win is None or is_radiant is None:
                    continue
                won = (radiant_win and is_radiant) or (not radiant_win and not is_radiant)
                if won:
                    a_wins += 1
                else:
                    b_wins += 1
                start = m.get("start_time", 0)
                if start > last_time:
                    last_time = start

        days_since = int((time.time() - last_time) / 86400) if last_time > 0 else 999
        return a_wins, b_wins, days_since

    async def _get_dota2_intel(self, team_a: str, team_b: str) -> Optional[MatchIntel]:
        """Fetch comprehensive Dota2 team intelligence from OpenDota."""
        await self._ensure_dota2_teams()

        ta_data = self._find_dota2_team(team_a)
        tb_data = self._find_dota2_team(team_b)

        if not ta_data and not tb_data:
            return None

        prof_a = TeamProfile(name=team_a, game="dota2")
        prof_b = TeamProfile(name=team_b, game="dota2")
        sources = []
        matches_a, matches_b = [], []

        if ta_data:
            tid_a = ta_data["team_id"]
            prof_a.team_id = str(tid_a)
            prof_a.rating = ta_data.get("rating", 0)
            matches_a = await self._get_dota2_matches(tid_a, 30)
            stats_a = self._analyze_matches(matches_a, tid_a)
            prof_a.recent_form = stats_a["form"]
            prof_a.recent_matches = stats_a["total"]
            prof_a.win_streak = stats_a["streak"]
            prof_a.schedule_strength = stats_a["schedule_strength"]
            prof_a.hero_pool = await self._get_dota2_hero_pool(tid_a)
            prof_a.last_updated = time.time()
            sources.append(f"opendota:{prof_a.team_id}")

        if tb_data:
            tid_b = tb_data["team_id"]
            prof_b.team_id = str(tid_b)
            prof_b.rating = tb_data.get("rating", 0)
            matches_b = await self._get_dota2_matches(tid_b, 30)
            stats_b = self._analyze_matches(matches_b, tid_b)
            prof_b.recent_form = stats_b["form"]
            prof_b.recent_matches = stats_b["total"]
            prof_b.win_streak = stats_b["streak"]
            prof_b.schedule_strength = stats_b["schedule_strength"]
            prof_b.hero_pool = await self._get_dota2_hero_pool(tid_b)
            prof_b.last_updated = time.time()
            sources.append(f"opendota:{prof_b.team_id}")

        # H2H from cross-referencing match histories
        h2h_a, h2h_b, h2h_days = 0, 0, 999
        if matches_a and matches_b:
            h2h_a, h2h_b, h2h_days = await self._get_dota2_h2h(matches_a, matches_b)

        prior, confidence = self._calculate_prior(prof_a, prof_b)

        # H2H adjusts prior
        if h2h_a + h2h_b >= 2 and h2h_days < 180:
            h2h_rate = h2h_a / (h2h_a + h2h_b)
            h2h_prior = max(0.2, min(0.8, h2h_rate))
            # Blend: 70% base prior + 30% H2H
            prior = prior * 0.7 + h2h_prior * 0.3
            confidence = min(0.9, confidence + 0.1)

        intel = MatchIntel(
            team_a=prof_a, team_b=prof_b,
            prior_prob_a=max(0.15, min(0.85, prior)),
            confidence=confidence,
            sources=sources,
            h2h_wins_a=h2h_a, h2h_wins_b=h2h_b,
            h2h_last_played_days=h2h_days,
        )

        streak_a = f"+{prof_a.win_streak}" if prof_a.win_streak > 0 else str(prof_a.win_streak)
        streak_b = f"+{prof_b.win_streak}" if prof_b.win_streak > 0 else str(prof_b.win_streak)
        h2h_str = f"H2H: {h2h_a}-{h2h_b} ({h2h_days}d ago)" if h2h_a + h2h_b > 0 else "H2H: none"

        logger.info(
            f"[INTEL] Dota2 {team_a} vs {team_b} | "
            f"rating={prof_a.rating:.0f} vs {prof_b.rating:.0f} | "
            f"form={prof_a.recent_form:.2f}({streak_a}) vs {prof_b.recent_form:.2f}({streak_b}) | "
            f"{h2h_str} | prior={intel.prior_prob_a:.3f}"
        )
        return intel

    # ─── CS2 (HLTV Rankings) ─────────────────────────────────────────────────

    async def _fetch_hltv_rankings(self):
        """HLTV rankings disabled — always returns 403."""
        return

        await self._ensure_session()
        try:
            # Use HLTV ranking page
            from datetime import datetime
            today = datetime.now()
            # HLTV publishes rankings weekly on Mondays
            # Use most recent Monday
            import calendar
            days_since_monday = today.weekday()
            last_monday = today.replace(hour=0, minute=0, second=0, microsecond=0)
            if days_since_monday > 0:
                from datetime import timedelta
                last_monday = last_monday - timedelta(days=days_since_monday)

            url = f"https://www.hltv.org/ranking/teams/{last_monday.year}/{last_monday.strftime('%B').lower()}/{last_monday.day}"

            async with self._session.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"HLTV rankings returned {resp.status}")
                    return
                html = await resp.text()

            # Parse rankings from HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            rankings = {}
            for item in soup.find_all(class_="ranked-team"):
                rank_el = item.find(class_="position")
                name_el = item.find(class_="name")
                if rank_el and name_el:
                    rank_text = rank_el.get_text(strip=True).replace("#", "")
                    name = name_el.get_text(strip=True)
                    try:
                        rankings[name.lower()] = int(rank_text)
                    except ValueError:
                        pass

            if rankings:
                self._hltv_rankings = rankings
                self._hltv_rankings_fetched = now
                logger.info(f"HLTV: cached {len(rankings)} team rankings")
            else:
                logger.warning("HLTV: no rankings parsed from page")

        except Exception as e:
            logger.error(f"HLTV rankings fetch failed: {e}")

    def _find_hltv_rank(self, name: str) -> int:
        """Find a team's HLTV rank. Returns 0 if not ranked."""
        key = name.lower().strip()
        if key in self._hltv_rankings:
            return self._hltv_rankings[key]
        # Fuzzy match
        for k, v in self._hltv_rankings.items():
            if key in k or k in key:
                return v
        return 0

    async def _get_cs2_intel(self, team_a: str, team_b: str) -> Optional[MatchIntel]:
        """Fetch CS2 team intelligence from HLTV rankings."""
        await self._fetch_hltv_rankings()

        rank_a = self._find_hltv_rank(team_a)
        rank_b = self._find_hltv_rank(team_b)

        prof_a = TeamProfile(name=team_a, game="cs2", ranking=rank_a, last_updated=time.time())
        prof_b = TeamProfile(name=team_b, game="cs2", ranking=rank_b, last_updated=time.time())
        sources = []

        if rank_a > 0 or rank_b > 0:
            sources.append("hltv_ranking")

        # Prior from rankings
        if rank_a > 0 and rank_b > 0:
            # Logistic on rank differential: rank 1 vs 30 ≈ 80/20
            rank_diff = rank_b - rank_a  # positive = A is better
            prior = 1.0 / (1.0 + math.exp(-rank_diff / 8.0))
            confidence = 0.6  # rankings are decent signal
        elif rank_a > 0:
            # A is ranked, B is not — A is likely stronger
            prior = 0.65 + min(0.15, (30 - rank_a) / 100)  # rank 1 → 0.80, rank 30 → 0.65
            confidence = 0.4
        elif rank_b > 0:
            prior = 0.35 - min(0.15, (30 - rank_b) / 100)
            confidence = 0.4
        else:
            prior = 0.5
            confidence = 0.0

        prior = max(0.15, min(0.85, prior))

        intel = MatchIntel(
            team_a=prof_a, team_b=prof_b,
            prior_prob_a=prior, confidence=confidence,
            sources=sources,
        )

        if rank_a > 0 or rank_b > 0:
            logger.info(
                f"[INTEL] CS2 {team_a} (#{rank_a}) vs {team_b} (#{rank_b}) | "
                f"prior={prior:.3f} conf={confidence:.2f}"
            )
        return intel

    # ─── Valorant (VLR.gg) ───────────────────────────────────────────────────

    async def _get_valorant_intel(self, team_a: str, team_b: str) -> Optional[MatchIntel]:
        """Fetch Valorant team intelligence from VLR.gg team pages."""
        # VLR.gg scraping is rate-limited — just return empty for now
        # The market price prior from Phase 1 covers most of the value
        prof_a = TeamProfile(name=team_a, game="valorant")
        prof_b = TeamProfile(name=team_b, game="valorant")
        return MatchIntel(team_a=prof_a, team_b=prof_b, prior_prob_a=0.5, confidence=0.0)

    # ─── Prior Calculation ───────────────────────────────────────────────────

    def _calculate_prior(self, a: TeamProfile, b: TeamProfile) -> tuple[float, float]:
        """
        Calculate prior P(A wins) from team profiles.
        Returns (prior, confidence).

        Uses weighted combination of:
        - Rating differential (weight 3.0)
        - Recent form differential (weight 2.0)
        """
        signals = []
        total_weight = 0.0

        # Rating-based prior (OpenDota ELO-like rating)
        if a.rating > 0 and b.rating > 0:
            rating_diff = a.rating - b.rating
            # Logistic with scale ~200 (typical rating spread)
            rating_prior = 1.0 / (1.0 + math.exp(-rating_diff / 200.0))
            signals.append((rating_prior, 3.0))
            total_weight += 3.0

        # Form-based prior (recent win rate)
        if a.recent_matches >= 3 and b.recent_matches >= 3:
            form_diff = a.recent_form - b.recent_form
            form_prior = 0.5 + form_diff * 0.5  # scale to 0-1 range
            form_prior = max(0.2, min(0.8, form_prior))
            weight = min(2.0, (a.recent_matches + b.recent_matches) / 15)  # more data = more weight
            signals.append((form_prior, weight))
            total_weight += weight
        elif a.recent_matches >= 3:
            # Only have form for team A
            form_prior = 0.5 + (a.recent_form - 0.5) * 0.3
            signals.append((form_prior, 1.0))
            total_weight += 1.0
        elif b.recent_matches >= 3:
            form_prior = 0.5 - (b.recent_form - 0.5) * 0.3
            signals.append((form_prior, 1.0))
            total_weight += 1.0

        if not signals:
            return 0.5, 0.0

        # Weighted average
        weighted_sum = sum(p * w for p, w in signals)
        prior = weighted_sum / total_weight

        # Confidence based on data quality
        confidence = min(0.8, total_weight / 5.0)

        return max(0.15, min(0.85, prior)), confidence
