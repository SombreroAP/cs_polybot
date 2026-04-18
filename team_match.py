"""
Fuzzy team name matching for esports.

Handles common naming differences between data sources:
- "The MongolZ" vs "TheMongolz" (spaces)
- "Natus Vincere" vs "NAVI" (abbreviations)
- "Team Spirit" vs "Spirit" (prefix stripping)
- "FURIA Esports" vs "FURIA" (suffix stripping)
- "G2 Esports" vs "G2" (org name vs tag)
- "Virtus.pro" vs "VP" (known aliases)
"""
import re

# Known aliases: feed name → market name (and reverse)
ALIASES = {
    "natus vincere": ["navi", "na'vi"],
    "navi": ["natus vincere", "na'vi"],
    "virtus.pro": ["vp", "virtuspro"],
    "vp": ["virtus.pro", "virtuspro"],
    "faze clan": ["faze"],
    "faze": ["faze clan"],
    "betboom team": ["betboom"],
    "betboom": ["betboom team"],
    "og esports": ["og"],
    "team spirit": ["spirit"],
    "team liquid": ["liquid"],
    "team vitality": ["vitality"],
    "team falcons": ["falcons"],
    "team heretics": ["heretics"],
    "cloud9": ["c9"],
    "c9": ["cloud9"],
    "counter logic gaming": ["clg"],
    "evil geniuses": ["eg"],
    "ninjas in pyjamas": ["nip"],
    "fnatic": ["fnc"],
    "100 thieves": ["100t"],
    "t1": ["skt", "sk telecom"],
    "gen.g": ["geng", "gen g"],
}

# Common suffixes/prefixes that sources add or remove
STRIP_SUFFIXES = [
    " esports", " gaming", " esport", " e-sports", " e-sport",
    " clan", " team", " academy", " junior", " jr", " rising",
]
STRIP_PREFIXES = [
    "team ", "the ",
]


def normalize(name: str) -> str:
    """Normalize a team name for comparison."""
    n = name.lower().strip()
    # Remove common suffixes — require at least 1 char remaining (so we don't strip
    # the entire name to empty). Previously +2 wrongly preserved "G2 Esports" because
    # "G2" is only 2 chars, causing dedup misses.
    for suffix in STRIP_SUFFIXES:
        if n.endswith(suffix) and len(n) > len(suffix):
            n = n[:-len(suffix)].strip()
    # Remove common prefixes
    for prefix in STRIP_PREFIXES:
        if n.startswith(prefix) and len(n) > len(prefix):
            n = n[len(prefix):].strip()
    return n


def teams_match(name_a: str, name_b: str) -> bool:
    """Check if two team names refer to the same team."""
    a = name_a.lower().strip()
    b = name_b.lower().strip()

    # Exact match
    if a == b:
        return True

    # Normalized match (strip suffixes/prefixes)
    na = normalize(a)
    nb = normalize(b)
    if na == nb:
        return True

    # Space-stripped match ("The MongolZ" vs "TheMongolz")
    a_ns = a.replace(" ", "").replace(".", "").replace("-", "")
    b_ns = b.replace(" ", "").replace(".", "").replace("-", "")
    if a_ns == b_ns:
        return True

    # Normalized + space-stripped
    na_ns = na.replace(" ", "").replace(".", "").replace("-", "")
    nb_ns = nb.replace(" ", "").replace(".", "").replace("-", "")
    if na_ns == nb_ns:
        return True

    # Substring match (one contains the other)
    if len(a) >= 2 and len(b) >= 2:
        if a in b or b in a:
            return True
        if na in nb or nb in na:
            return True
        if na_ns in nb_ns or nb_ns in na_ns:
            return True

    # Known aliases
    a_aliases = ALIASES.get(a, []) + ALIASES.get(na, [])
    if b in a_aliases or nb in a_aliases:
        return True
    b_aliases = ALIASES.get(b, []) + ALIASES.get(nb, [])
    if a in b_aliases or na in b_aliases:
        return True

    return False


def team_in_text(team_name: str, text: str) -> bool:
    """Check if a team name appears in a text string (e.g., market question)."""
    t = team_name.lower().strip()
    q = text.lower()

    if t in q:
        return True

    # Normalized
    nt = normalize(t)
    if nt in q:
        return True

    # Space-stripped
    t_ns = t.replace(" ", "")
    q_ns = q.replace(" ", "")
    if t_ns in q_ns:
        return True

    # Aliases
    for alias in ALIASES.get(t, []) + ALIASES.get(nt, []):
        if alias in q:
            return True

    return False
