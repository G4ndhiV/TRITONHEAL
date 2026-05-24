"""Triton Heal dual verifier with hierarchical veto."""
from dataclasses import dataclass

from .backends.heuristic import HeuristicVerifier, VerifyResult as HResult


@dataclass
class DualVerifyResult:
    safe: bool
    reason: str
    line_of_error: int
    latency_ms: float
    valid_json: bool
    local_safe: bool
    frontier_safe: bool
    disagreement: bool
    veto_applied: bool
    config: str = "triton_heal_dual"


class TritonHealDual:
    """Local SLM first; frontier veto on disagreement or local-safe with frontier-unsafe."""

    def __init__(self, local_verifier, frontier_verifier):
        self.local = local_verifier
        self.frontier = frontier_verifier

    def verify(self, code: str) -> DualVerifyResult:
        r_local = self.local.verify(code)
        r_frontier = self.frontier.verify(code)
        disagreement = r_local.safe != r_frontier.safe
        veto_applied = disagreement

        if disagreement:
            # Hierarchical veto: frontier wins
            safe = r_frontier.safe
            reason = f"veto_frontier: {r_frontier.reason}"
            line = r_frontier.line_of_error
        elif not r_local.safe:
            safe = False
            reason = r_local.reason
            line = r_local.line_of_error
        else:
            safe = r_local.safe and r_frontier.safe
            reason = r_frontier.reason if not r_frontier.safe else r_local.reason
            line = r_frontier.line_of_error if not r_frontier.safe else 0

        return DualVerifyResult(
            safe=safe,
            reason=reason,
            line_of_error=line,
            latency_ms=r_local.latency_ms + r_frontier.latency_ms,
            valid_json=r_local.valid_json and r_frontier.valid_json,
            local_safe=r_local.safe,
            frontier_safe=r_frontier.safe,
            disagreement=disagreement,
            veto_applied=veto_applied,
        )
