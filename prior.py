"""
Bayesian prior module for combining pre-match team intelligence
with in-game probability models.

The key insight: our in-game models (CS2 economy, Dota2 kills) assume
equally-skilled teams (base = 0.50). If we know from rankings/form that
team A is stronger (prior = 0.65), we shift the model output accordingly
using log-odds addition.

As in-game evidence accumulates, it naturally dominates the prior.
"""
import math
import logging

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))


def _logit(p: float) -> float:
    """Log-odds: log(p / (1-p))"""
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Inverse logit."""
    return 1.0 / (1.0 + math.exp(-x))


def apply_prior(model_prob: float, prior: float, base: float = 0.5) -> float:
    """
    Shift a model probability by a prior using log-odds addition.

    If the model outputs 0.60 assuming equal teams (base=0.5),
    and the prior says team A has 0.70 chance, the combined output
    is higher than either alone — the evidence reinforces.

    If model says 0.40 (team A losing) but prior is 0.70 (team A stronger),
    the combined is somewhere in between — pulled toward the prior early,
    dominated by evidence as the game progresses.

    Args:
        model_prob: In-game model's P(A wins), assuming equal teams
        prior: Pre-match P(A wins) from rankings, market, etc.
        base: The model's assumed baseline (always 0.5 for our models)

    Returns:
        Combined probability incorporating both prior and game state.
    """
    lo_model = _logit(model_prob)
    lo_prior = _logit(prior)
    lo_base = _logit(base)

    # Shift: add the prior's deviation from base to the model output
    combined = lo_model + (lo_prior - lo_base)
    return _clamp(_sigmoid(combined))


def combine_priors(market_prior: float, intel_prior: float,
                   intel_confidence: float = 0.5) -> float:
    """
    Combine a market-derived prior with team intelligence prior.

    Market prior gets base weight 2.0 (Polymarket encodes lots of info).
    Intel prior weight scales with confidence (0 to 2.0).

    Args:
        market_prior: P(A) from Polymarket pre-match price
        intel_prior: P(A) from team rankings/form/H2H
        intel_confidence: 0-1, how much to trust intel (data quality)

    Returns:
        Combined prior probability.
    """
    if intel_confidence < 0.1:
        return market_prior

    lo_market = _logit(market_prior)
    lo_intel = _logit(intel_prior)

    market_weight = 2.0
    intel_weight = intel_confidence * 2.0
    total_weight = market_weight + intel_weight

    lo_combined = (lo_market * market_weight + lo_intel * intel_weight) / total_weight

    combined = _clamp(_sigmoid(lo_combined))

    # Log large disagreements — could signal roster change or stale data
    if abs(market_prior - intel_prior) > 0.15:
        logger.warning(
            f"Prior disagreement: market={market_prior:.3f} intel={intel_prior:.3f} "
            f"(conf={intel_confidence:.2f}) → combined={combined:.3f}"
        )

    return combined
