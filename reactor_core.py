# reactor_core.py
"""
Information Reactor (reactor_core.py) - integracja z drm_module5_improved.py

Integracja:
  from drm_module5_improved import DRMSystem, entropy_from_dist
  from reactor_core import InformationReactor, Reaction

  ds = DRMSystem()
  reactor = InformationReactor(drm=ds, budget_per_cycle=1.0)
  ds.register_reactor(reactor)
  reactor.register_reaction(Reaction(...))
  ds.run_cycle(evaluator, context=...)
"""
from __future__ import annotations
import time
import math
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Set

logger = logging.getLogger("reactor")
logger.setLevel(logging.INFO)

# Try to import helper from DRM; fallback simple implementations
try:
    from drm_module5_improved import entropy_from_dist
except Exception:
    def entropy_from_dist(dist: Dict[str, float]) -> float:
        return -sum(p * math.log(max(p, 1e-12)) for p in dist.values())

# ----------------- Reaction -----------------
@dataclass
class Reaction:
    id: str
    name: str
    triggers: Set[str] = field(default_factory=set)  # event keys or rule ids
    action: Optional[Callable[['DRMSystem', Dict[str, Any]], Dict[str, Any]]] = None
    action_rule_id: Optional[str] = None
    priority: int = 1
    cost_val: float = 0.1
    max_chain_depth: int = 3
    probabilistic: bool = True
    cooldown: int = 3
    last_executed_cycle: int = -9999
    max_runtime: float = 2.0   # seconds
    created_by: str = "reactor"

    def cost(self, drm=None) -> float:
        base = float(self.cost_val)
        if drm is not None:
            try:
                dist = drm.get_distribution()
                ent = entropy_from_dist(dist)
                # small entropy cost factor
                base += 0.01 * ent
            except Exception:
                pass
        return base

    def rate(self, drm, context: Dict[str, Any]) -> float:
        """
        rate = sigmoid(alpha*strength + beta*FRZ + gamma*curiosity*emergence - lambda*cost)
        Safe access to drm methods with fallbacks.
        """
        try:
            # try compute_scores (more bounded) then compute_strengths
            scores = drm.compute_scores() if hasattr(drm, "compute_scores") else {}
            strengths = drm.compute_strengths() if hasattr(drm, "compute_strengths") else {}
        except Exception:
            scores, strengths = {}, {}

        # strength proxy: average of relevant triggered names from scores then strengths
        svals = []
        for t in self.triggers:
            svals.append(scores.get(t, strengths.get(t, 0.0)))
        s = sum(svals) / max(1, len(svals)) if svals else (sum(strengths.values()) / max(1, len(strengths)) if strengths else 0.0)

        frz = getattr(drm, "_last_frz", 1.0)
        stats = drm.get_stats() if hasattr(drm, "get_stats") else {}
        total_rules = stats.get("total_rules", 0)
        active_rules = stats.get("active_rules", total_rules)
        novelty = 1.0 - (active_rules / (total_rules + 1)) if total_rules > 0 else 1.0
        diversity = stats.get("diversity_score", 0.5)
        emergence = max(0.1, 1.0 + (0.5 - diversity))  # more pressure when diversity low

        # normalize s to [-1,1]
        norm_s = math.tanh(s)
        alpha, beta, gamma, lam = 1.2, 0.5, 0.9, 2.0
        val = alpha * norm_s + beta * frz + gamma * novelty * emergence - lam * self.cost(drm)
        try:
            return 1.0 / (1.0 + math.exp(-val))
        except OverflowError:
            return 1.0 if val > 0 else 0.0

    def can_execute(self, current_cycle: int) -> bool:
        return (current_cycle - self.last_executed_cycle) >= self.cooldown

    def execute(self, drm, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes safely with timeout; supports callable actions or action_rule_id to update a rule.
        Returns dict: {ok, error, detail, duration}.
        """
        start = time.time()
        detail = None
        ok = False
        error = None

        def _call_action():
            nonlocal detail
            if callable(self.action):
                return self.action(drm, context)
            if self.action_rule_id and self.action_rule_id in getattr(drm, "rules", {}):
                # Update rule lightly: add replay and update posterior
                rule = drm.rules[self.action_rule_id]
                reward = float(context.get("simulated_reward", 0.5))
                try:
                    drm.replay.add((rule.id, reward, context))
                    rule.update_posterior(reward, context=context)
                    return {"updated_rule": rule.id, "reward": reward}
                except Exception as e:
                    return {"error": str(e)}
            return {"note": "no-op"}

        # Execute with ThreadPool to enforce timeout and avoid blocking
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call_action)
            try:
                detail = fut.result(timeout=self.max_runtime)
                ok = True
            except TimeoutError:
                fut.cancel()
                error = "timeout"
                ok = False
            except Exception as e:
                error = str(e)
                ok = False

        duration = time.time() - start
        # Post-execution safety checks: if detail indicates high uncertainty flag -> quarantine
        try:
            if isinstance(detail, dict):
                # heuristic: if action returned 'quarantine' signal or 'high_uncertainty'
                if detail.get("quarantine") or detail.get("high_uncertainty"):
                    # mark affected rule(s) quarantined if present
                    rid = detail.get("rule_id")
                    if rid and hasattr(drm, "quarantine_rule"):
                        drm.quarantine_rule(rid, reason="reaction_triggered")
                        drm.audit("reactor_quarantine", rid, {"reaction": self.id})
        except Exception:
            pass

        # Audit the reaction run on DRM if available
        try:
            if hasattr(drm, "audit"):
                drm.audit("reaction_execute", self.id, {
                    "ok": ok, "error": error, "duration": duration, "detail": detail
                })
        except Exception:
            pass

        return {"ok": ok, "error": error, "detail": detail, "duration": duration}

# ----------------- EnergyAccountant -----------------
@dataclass
class EnergyAccountant:
    budget_per_cycle: float = 1.0
    used: float = 0.0
    throughput_history: List[float] = field(default_factory=list)
    entropy_cost_k: float = 0.01  # multiplier for entropy cost

    def reset_cycle(self):
        self.throughput_history.append(self.used)
        if len(self.throughput_history) > 1000:
            self.throughput_history = self.throughput_history[-500:]
        self.used = 0.0

    def can_afford(self, cost: float) -> bool:
        return (self.used + cost) <= self.budget_per_cycle + 1e-12

    def consume(self, cost: float) -> bool:
        if self.can_afford(cost):
            self.used += cost
            return True
        return False

    def refund(self, cost: float):
        self.used = max(0.0, self.used - cost)

    def add_entropy_cost(self, dist: Dict[str, float]) -> float:
        ent = entropy_from_dist(dist)
        cost = self.entropy_cost_k * ent
        return cost

# ----------------- ReactionRegistry -----------------
class ReactionRegistry:
    def __init__(self):
        self._by_id: Dict[str, Reaction] = {}
        self._triggers_index: Dict[str, Set[str]] = {}

    def register(self, reaction: Reaction):
        self._by_id[reaction.id] = reaction
        for t in reaction.triggers:
            self._triggers_index.setdefault(t, set()).add(reaction.id)

    def get(self, rid: str) -> Optional[Reaction]:
        return self._by_id.get(rid)

    def find_applicable(self, context: Dict[str, Any], drm) -> List[Reaction]:
        """
        context contains event flags. Match exact triggers or prefix triggers 'triggered_by_X'.
        Always include reactions with empty triggers as background candidates.
        """
        ids = set()
        for key, val in context.items():
            if not val:
                continue
            if key in self._triggers_index:
                ids.update(self._triggers_index[key])
            # match any prefix-based triggered_by_*
            if key.startswith("triggered_by_") and key in self._triggers_index:
                ids.update(self._triggers_index[key])
        # convert to Reaction list
        candidates = [self._by_id[i] for i in ids]
        # include background reactions
        for rid, r in self._by_id.items():
            if not r.triggers:
                candidates.append(r)
        return candidates

    def all(self) -> List[Reaction]:
        return list(self._by_id.values())

# ----------------- ChainReactionLimiter -----------------
@dataclass
class ChainReactionLimiter:
    max_depth: int = 4
    max_activations_per_cycle: int = 12
    max_chain_per_reaction: int = 3

# ----------------- ReactionScheduler -----------------
class ReactionScheduler:
    def __init__(self, drm, registry: ReactionRegistry, accountant: EnergyAccountant,
                 limiter: ChainReactionLimiter):
        self.drm = drm
        self.registry = registry
        self.accountant = accountant
        self.limiter = limiter
        self.executions_this_cycle = 0

    def schedule_and_run(self, context: Dict[str, Any], current_cycle: int, depth: int = 0) -> List[Dict[str, Any]]:
        """
        Schedule and execute reactions respecting budget and limits.
        Supports single-level probabilistic chaining (depth control).
        """
        results = []
        if depth > self.limiter.max_depth:
            return results

        candidates = self.registry.find_applicable(context, self.drm)
        # compute score = priority * rate
        scored = []
        for r in candidates:
            if not r.can_execute(current_cycle):
                continue
            rate = r.rate(self.drm, context)
            score = (1.0 + 0.1 * r.priority) * rate
            scored.append((score, r))
        # sort high->low
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, reaction in scored:
            if self.executions_this_cycle >= self.limiter.max_activations_per_cycle:
                break
            c = reaction.cost(self.drm)
            # include entropy cost
            try:
                dist = self.drm.get_distribution() if hasattr(self.drm, "get_distribution") else {}
                c += self.accountant.add_entropy_cost(dist)
            except Exception:
                pass
            if not self.accountant.can_afford(c):
                continue
            # consume budget and execute
            consumed = self.accountant.consume(c)
            res = reaction.execute(self.drm, context)
            reaction.last_executed_cycle = current_cycle
            self.executions_this_cycle += 1
            results.append({"reaction_id": reaction.id, "score": score, "cost": c, "result": res})
            # chain: if probabilistic and rate high and depth < max_depth, create chained context and recurse
            if reaction.probabilistic and (self.executions_this_cycle < self.limiter.max_activations_per_cycle):
                chain_prob = min(1.0, score) * 0.6
                if chain_prob > 0.5 and depth + 1 <= self.limiter.max_depth:
                    # signal new trigger
                    chain_ctx = dict(context)
                    chain_ctx[f"triggered_by_{reaction.id}"] = True
                    # recursive scheduling but do not reset executions_this_cycle (global per cycle)
                    results += self.schedule_and_run(chain_ctx, current_cycle, depth=depth + 1)
        return results

# ----------------- InformationReactor -----------------
class InformationReactor:
    def __init__(self, drm=None, budget_per_cycle: float = 1.0,
                 max_activations: int = 12, max_chain_depth: int = 4):
        self.drm = drm
        self.registry = ReactionRegistry()
        self.accountant = EnergyAccountant(budget_per_cycle=budget_per_cycle)
        self.limiter = ChainReactionLimiter(max_depth=max_chain_depth, max_activations_per_cycle=max_activations)
        self.scheduler = ReactionScheduler(self.drm, self.registry, self.accountant, self.limiter)
        self.current_cycle = 0
        self.lock = threading.Lock()
        # hooks - override as needed
        self.on_stagnation = lambda drm, info: drm.emergency_revival(num_new=3) if hasattr(drm, "emergency_revival") else None
        self.on_high_throughput = lambda drm, info: None

    def register_reaction(self, reaction: Reaction):
        self.registry.register(reaction)

    def inject(self, reaction_or_callable) -> Optional[str]:
        if isinstance(reaction_or_callable, Reaction):
            self.register_reaction(reaction_or_callable)
            return reaction_or_callable.id
        if callable(reaction_or_callable):
            rid = f"anon_{int(time.time()*1000)}"
            r = Reaction(id=rid, name=rid, action=reaction_or_callable)
            self.register_reaction(r)
            return rid
        return None

    def schedule_and_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.current_cycle += 1
            self.accountant.reset_cycle()
            self.scheduler.executions_this_cycle = 0

            # Pre-hooks
            if context.get("stagnation_detected"):
                try:
                    self.on_stagnation(self.drm, context)
                except Exception as e:
                    logger.warning("on_stagnation hook failed: %s", e)

            # Run scheduler
            results = self.scheduler.schedule_and_run(context, current_cycle=self.current_cycle, depth=0)

            # Metrics & post processing
            total_duration = sum([r["result"].get("duration", 0.0) or 0.0 for r in results])
            avg_cost = (self.accountant.used / max(1, len(results))) if results else 0.0
            report = {
                "cycle": self.current_cycle,
                "executions": len(results),
                "budget_used": self.accountant.used,
                "throughput": total_duration,
                "avg_cost": avg_cost,
                "results": results
            }

            # throttle/generate hook
            if total_duration > 1.0:
                try:
                    self.on_high_throughput(self.drm, report)
                except Exception as e:
                    logger.warning("on_high_throughput hook failed: %s", e)

            # audit on DRM
            try:
                if hasattr(self.drm, "audit"):
                    self.drm.audit("reactor_cycle", None, report)
            except Exception:
                pass

            return report

    def shutdown(self):
        # graceful shutdown placeholder
        pass

# ----------------- Demo (minimal) -----------------
if __name__ == "__main__":
    # Simple smoke-demo that does not require full drm module.
    def sample_action(drm, ctx):
        # dummy action
        return {"note": "action-run", "time": time.time()}

    r1 = Reaction(id="r_revival", name="revival_on_stagnation", triggers={"stagnation_detected"},
                  action=lambda drm, ctx: (drm.emergency_revival(num_new=2) or {"revival": True}) if hasattr(drm, "emergency_revival") else {"revival": "no-drm"},
                  priority=5, cost_val=0.2, probabilistic=False)

    r2 = Reaction(id="r_promote", name="promote_uncertain", triggers=set(),
                  action=sample_action, priority=1, cost_val=0.05)

    # Standalone registry/accountant usage
    reg = ReactionRegistry()
    acct = EnergyAccountant(budget_per_cycle=0.5)
    lim = ChainReactionLimiter(max_depth=2, max_activations_per_cycle=5)
    sched = ReactionScheduler(drm=None, registry=reg, accountant=acct, limiter=lim)
    reg.register(r1); reg.register(r2)
    ctx = {"stagnation_detected": True}
    report = sched.schedule_and_run(ctx, current_cycle=1)
    print("Standalone report:", report)
