"""drm_module5_improved.py
Unified Dynamic Rule Matrix (DRM) module with key improvements for neural network integration.

Key improvements:
- Differentiable parameters for HYBRID rules
- Conflict detection and resolution
- Better gradient support preparation
- Enhanced semantic validation
- Improved error handling and logging

Features:
 - Rule types: LOGICAL, HEURISTIC, HYBRID with semantic fields (pre/post/params/tests/provenance)
 - Bayesian posterior (post_mean, post_var) for rule performance
 - Strength function S_i combining weight, usage, novelty, curiosity, emergence pressure
 - ReplayBuffer, RuleGenerator (latent), and generator training hook
 - Multiplicative update with adaptive eta and KL trust-region
 - StagnationDetector, EmergencyRevival, DiversityEnforcement
 - Conflict detection and resolution system
 - Semantic validator: validate_rule(context) with strict logic for LOGICAL rules
 - Serialization (to_dict/from_dict), explainability, mutate, compose
 - Lightweight APIs: add_rule, generate_rules, run_cycle, save/load JSON, export
Dependencies: numpy optional for acceleration
"""
from __future__ import annotations
import math, random, json, time, warnings
from typing import Dict, Any, List, Optional, Callable, Tuple, Set, Union
from collections import deque, defaultdict
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None
    HAS_NUMPY = False

EPS = 1e-9

# Rule types
LOGICAL = "LOGICAL"
HEURISTIC = "HEURISTIC" 
HYBRID = "HYBRID"

# Parameter types for better type safety
@dataclass
class RuleParameter:
    """Type-safe rule parameter with constraints and differentiation info"""
    value: float
    param_type: str = "float"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    requires_grad: bool = False
    constraint_fn: Optional[Callable[[float], bool]] = None
    
    def __post_init__(self):
        self.validate()
    
    def validate(self) -> bool:
        """Validate parameter constraints"""
        if self.min_val is not None and self.value < self.min_val - EPS:
            return False
        if self.max_val is not None and self.value > self.max_val + EPS:
            return False
        if self.constraint_fn and not self.constraint_fn(self.value):
            return False
        return True
    
    def clamp(self):
        """Clamp value to constraints"""
        if self.min_val is not None:
            self.value = max(self.min_val, self.value)
        if self.max_val is not None:
            self.value = min(self.max_val, self.value)

# ------------------ Enhanced Rule ------------------
class Rule:
    def __init__(self,
                 id: str,
                 name: Optional[str] = None,
                 rtype: str = HEURISTIC,
                 init_weight: float = 1.0,
                 init_mean: float = 0.5,
                 init_var: float = 0.25,
                 latent_z: Optional[List[float]] = None,
                 pre_conditions: Optional[List[str]] = None,
                 post_conditions: Optional[List[str]] = None,
                 params: Optional[Dict[str, Union[Dict[str, Any], RuleParameter]]] = None,
                 tests: Optional[List[Callable[['Rule', Dict[str, Any]], bool]]] = None,
                 provenance: Optional[Dict[str, Any]] = None,
                 priority: int = 0,
                 conflict_groups: Optional[Set[str]] = None):
        self.id = id
        self.name = name or id
        self.type = rtype
        self.priority = priority  # Higher priority rules win conflicts
        self.conflict_groups = conflict_groups or set()
        
        # probabilistic
        self.weight = float(max(EPS, init_weight))
        self.post_mean = float(init_mean)
        self.post_var = float(init_var)
        self.observations = 0
        self.usage_count = 0
        self.is_new = True
        self.quarantined = False
        self.quarantine_reason = None
        self.latent_z = list(latent_z) if latent_z is not None else None
        
        # semantic
        self.pre_conditions = list(pre_conditions) if pre_conditions else []
        self.post_conditions = list(post_conditions) if post_conditions else []
        
        # Enhanced parameters handling
        self.params: Dict[str, RuleParameter] = {}
        if params:
            for k, v in params.items():
                if isinstance(v, RuleParameter):
                    self.params[k] = v
                elif isinstance(v, dict):
                    # Convert legacy dict format
                    self.params[k] = RuleParameter(
                        value=v.get("value", 0.0),
                        param_type=v.get("type", "float"),
                        min_val=v.get("min"),
                        max_val=v.get("max"),
                        requires_grad=v.get("requires_grad", self.type == HYBRID)
                    )
        
        self.tests = list(tests) if tests else []
        self.provenance = dict(provenance) if provenance else {}
        self.created_at = time.time()
        
        # Enhanced diagnostics
        self.history = deque(maxlen=200)
        self.activation_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_activated = None
        
        # Validate on creation
        self._validate_construction()

    def _validate_construction(self):
        """Validate rule construction"""
        if self.type == LOGICAL and not self.tests:
            warnings.warn(f"LOGICAL rule {self.id} has no tests - may not be properly validated")
        
        # Validate all parameters
        for name, param in self.params.items():
            if not param.validate():
                logger.warning(f"Parameter {name} in rule {self.id} violates constraints")

    def get_differentiable_params(self) -> Dict[str, RuleParameter]:
        """Get parameters that require gradients (for PyTorch integration)"""
        return {k: v for k, v in self.params.items() if v.requires_grad}

    def update_param(self, param_name: str, new_value: float) -> bool:
        """Update parameter with validation"""
        if param_name not in self.params:
            return False
        
        old_value = self.params[param_name].value
        self.params[param_name].value = new_value
        
        if not self.params[param_name].validate():
            self.params[param_name].value = old_value  # Rollback
            return False
        
        self.params[param_name].clamp()
        return True

    # Bayesian posterior update (Gaussian conjugate) - enhanced
    def update_posterior(self, reward: Optional[float], obs_var: float = 0.05, context: Optional[Dict] = None):
        if reward is None:
            return
            
        self.observations += 1
        self.usage_count += 1
        self.is_new = False
        
        # Track success/failure
        if reward > 0.5:  # Threshold for success
            self.success_count += 1
        else:
            self.failure_count += 1
        
        prior_prec = 1.0 / max(EPS, self.post_var)
        like_prec = 1.0 / max(EPS, obs_var)
        post_var = 1.0 / (prior_prec + like_prec)
        post_mean = post_var * (self.post_mean * prior_prec + reward * like_prec)
        
        self.post_mean = float(post_mean)
        self.post_var = float(post_var)
        
        # Enhanced history with context info
        self.history.append(("update", reward, self.post_mean, self.post_var, context))

    def sample_posterior(self) -> float:
        sigma = math.sqrt(max(EPS, self.post_var))
        return random.gauss(self.post_mean, sigma)

    def post_prob_below(self, thr: float) -> float:
        if self.post_var <= 0:
            return 1.0 if self.post_mean < thr else 0.0
        z = (thr - self.post_mean) / math.sqrt(self.post_var)
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        return float(min(1.0, max(0.0, cdf)))

    def get_success_rate(self) -> float:
        """Get empirical success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / max(1, total)

    def get_confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Get confidence interval for posterior mean"""
        if self.post_var <= 0:
            return (self.post_mean, self.post_mean)
        
        z_score = 1.96  # For 95% CI
        margin = z_score * math.sqrt(self.post_var)
        return (self.post_mean - margin, self.post_mean + margin)

    # Enhanced semantics evaluation
    def evaluate_semantics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        res = {
            "syntactic": True, 
            "pre_ok": True, 
            "tests_pass": True, 
            "param_ok": True, 
            "statistical_ok": True, 
            "score": 0.0,
            "errors": [],
            "warnings": []
        }
        
        # syntactic: required context keys
        missing_keys = []
        for condition in self.pre_conditions:
            if condition not in context:
                res["syntactic"] = False
                res["pre_ok"] = False
                missing_keys.append(condition)
        
        if missing_keys:
            res["errors"].append(f"Missing context keys: {missing_keys}")
        
        # run tests with better error handling
        test_failures = []
        for i, test_fn in enumerate(self.tests):
            try:
                ok = bool(test_fn(self, context))
                if not ok:
                    test_failures.append(i)
            except Exception as e:
                test_failures.append(i)
                res["errors"].append(f"Test {i} failed with exception: {str(e)}")
        
        if test_failures:
            res["tests_pass"] = False
            res["warnings"].append(f"Failed tests: {test_failures}")
        
        # Enhanced parameter validation
        param_errors = []
        for name, param in self.params.items():
            if not param.validate():
                param_errors.append(name)
                res["param_ok"] = False
        
        if param_errors:
            res["errors"].append(f"Invalid parameters: {param_errors}")
        
        # Enhanced statistical check
        min_observations = 3 if self.type == LOGICAL else 1
        if self.observations < min_observations:
            res["warnings"].append(f"Insufficient observations: {self.observations} < {min_observations}")
        
        res["statistical_ok"] = (self.post_mean >= 0.05 and 
                                self.observations >= min_observations)
        
        # Enhanced scoring
        base_score = (
            (1.0 if res["syntactic"] else 0.0) +
            (1.0 if res["pre_ok"] else 0.0) +
            (1.0 if res["tests_pass"] else 0.0) +
            (1.0 if res["param_ok"] else 0.0) +
            self.post_mean
        )
        
        # Penalty for errors
        error_penalty = len(res["errors"]) * 0.1
        warning_penalty = len(res["warnings"]) * 0.05
        
        res["score"] = max(0.0, (base_score / 5.0) - error_penalty - warning_penalty)
        return res

    def explain(self) -> Dict[str, Any]:
        """Enhanced explanation with more diagnostics"""
        ci_low, ci_high = self.get_confidence_interval()
        
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "priority": self.priority,
            "conflict_groups": list(self.conflict_groups),
            "weight": self.weight,
            "post_mean": self.post_mean,
            "post_var": self.post_var,
            "confidence_interval": [ci_low, ci_high],
            "observations": self.observations,
            "usage_count": self.usage_count,
            "success_rate": self.get_success_rate(),
            "activation_count": self.activation_count,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "pre_conditions": list(self.pre_conditions),
            "post_conditions": list(self.post_conditions),
            "params": {k: {"value": v.value, "requires_grad": v.requires_grad, 
                          "min": v.min_val, "max": v.max_val} 
                      for k, v in self.params.items()},
            "provenance": dict(self.provenance),
            "created_at": self.created_at,
            "last_activated": self.last_activated
        }

    def mutate(self, op: str = "tweak_param", magnitude: float = 0.1) -> bool:
        """Enhanced mutation with better parameter handling"""
        if self.type == LOGICAL:
            return False  # Logical rules are immutable
        
        if op == "tweak_param" and self.params:
            mutated = False
            for name, param in self.params.items():
                if param.param_type == "float" and param.min_val is not None and param.max_val is not None:
                    old_value = param.value
                    span = param.max_val - param.min_val
                    delta = random.uniform(-magnitude, magnitude) * span
                    
                    if self.update_param(name, old_value + delta):
                        mutated = True
            return mutated
        
        elif op == "mutate_weight":
            factor = 1.0 + random.uniform(-magnitude, magnitude)
            self.weight = max(EPS, self.weight * factor)
            return True
        
        elif op == "perturb_latent" and self.latent_z is not None:
            self.latent_z = [z + random.gauss(0, magnitude) for z in self.latent_z]
            return True
        
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Enhanced serialization"""
        return {
            "id": self.id, 
            "name": self.name, 
            "type": self.type,
            "priority": self.priority,
            "conflict_groups": list(self.conflict_groups),
            "weight": self.weight, 
            "post_mean": self.post_mean, 
            "post_var": self.post_var,
            "observations": self.observations, 
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "latent_z": list(self.latent_z) if self.latent_z is not None else None,
            "pre_conditions": list(self.pre_conditions), 
            "post_conditions": list(self.post_conditions),
            "params": {k: {
                "value": v.value,
                "type": v.param_type,
                "min": v.min_val,
                "max": v.max_val,
                "requires_grad": v.requires_grad
            } for k, v in self.params.items()},
            "provenance": dict(self.provenance),
            "created_at": self.created_at
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Rule':
        """Enhanced deserialization"""
        # Convert params back to RuleParameter objects
        params = {}
        if "params" in d and d["params"]:
            for k, v in d["params"].items():
                if isinstance(v, dict):
                    params[k] = RuleParameter(
                        value=v.get("value", 0.0),
                        param_type=v.get("type", "float"),
                        min_val=v.get("min"),
                        max_val=v.get("max"),
                        requires_grad=v.get("requires_grad", False)
                    )
        
        rule = Rule(
            id=d["id"], 
            name=d.get("name"), 
            rtype=d.get("type", HEURISTIC),
            init_weight=d.get("weight", 1.0), 
            init_mean=d.get("post_mean", 0.5), 
            init_var=d.get("post_var", 0.25),
            latent_z=d.get("latent_z"), 
            pre_conditions=d.get("pre_conditions"),
            post_conditions=d.get("post_conditions"), 
            params=params,
            provenance=d.get("provenance"),
            priority=d.get("priority", 0),
            conflict_groups=set(d.get("conflict_groups", []))
        )
        
        # Restore additional state
        rule.observations = d.get("observations", 0)
        rule.usage_count = d.get("usage_count", 0)
        rule.success_count = d.get("success_count", 0)
        rule.failure_count = d.get("failure_count", 0)
        rule.quarantined = d.get("quarantined", False)
        rule.quarantine_reason = d.get("quarantine_reason")
        rule.created_at = d.get("created_at", time.time())
        
        return rule

# ------------------ Conflict Resolution System ------------------
class ConflictResolver:
    """Detects and resolves conflicts between rules"""
    
    def __init__(self):
        self.resolution_strategies = {
            "priority": self._resolve_by_priority,
            "performance": self._resolve_by_performance,
            "consensus": self._resolve_by_consensus
        }
    
    def detect_conflicts(self, rules: Dict[str, Rule], context: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Detect conflicts between rules. Returns list of (rule1_id, rule2_id, conflict_type)"""
        conflicts = []
        rule_list = list(rules.values())
        
        for i in range(len(rule_list)):
            for j in range(i + 1, len(rule_list)):
                rule_a, rule_b = rule_list[i], rule_list[j]
                conflict_type = self._check_conflict(rule_a, rule_b, context)
                if conflict_type:
                    conflicts.append((rule_a.id, rule_b.id, conflict_type))
        
        return conflicts
    
    def _check_conflict(self, rule_a: Rule, rule_b: Rule, context: Dict[str, Any]) -> Optional[str]:
        """Check if two rules conflict"""
        # Conflict group conflicts
        if rule_a.conflict_groups & rule_b.conflict_groups:
            return "group_conflict"
        
        # Post-condition conflicts (contradictory outputs)
        for post_a in rule_a.post_conditions:
            for post_b in rule_b.post_conditions:
                if self._contradictory_conditions(post_a, post_b):
                    return "post_condition_conflict"
        
        # Parameter conflicts (same parameter, different constraints)
        common_params = set(rule_a.params.keys()) & set(rule_b.params.keys())
        for param in common_params:
            if self._conflicting_param_constraints(rule_a.params[param], rule_b.params[param]):
                return "parameter_conflict"
        
        return None
    
    def _contradictory_conditions(self, cond_a: str, cond_b: str) -> bool:
        """Simple heuristic to detect contradictory conditions"""
        # Look for negation patterns like "X > 0.5" vs "X < 0.3"
        if ">" in cond_a and "<" in cond_b:
            return True
        if "<" in cond_a and ">" in cond_b:
            return True
        # Look for explicit negations
        if cond_a.startswith("not_") and cond_b == cond_a[4:]:
            return True
        if cond_b.startswith("not_") and cond_a == cond_b[4:]:
            return True
        return False
    
    def _conflicting_param_constraints(self, param_a: RuleParameter, param_b: RuleParameter) -> bool:
        """Check if parameter constraints conflict"""
        if param_a.min_val is not None and param_b.max_val is not None:
            if param_a.min_val > param_b.max_val:
                return True
        if param_b.min_val is not None and param_a.max_val is not None:
            if param_b.min_val > param_a.max_val:
                return True
        return False
    
    def resolve_conflicts(self, conflicts: List[Tuple[str, str, str]], rules: Dict[str, Rule], 
                         strategy: str = "priority") -> List[str]:
        """Resolve conflicts and return list of rule IDs to quarantine"""
        if strategy not in self.resolution_strategies:
            logger.warning(f"Unknown conflict resolution strategy: {strategy}")
            strategy = "priority"
        
        return self.resolution_strategies[strategy](conflicts, rules)
    
    def _resolve_by_priority(self, conflicts: List[Tuple[str, str, str]], rules: Dict[str, Rule]) -> List[str]:
        """Resolve conflicts by rule priority (higher priority wins)"""
        to_quarantine = set()
        
        for rule_a_id, rule_b_id, conflict_type in conflicts:
            if rule_a_id not in rules or rule_b_id not in rules:
                continue
                
            rule_a, rule_b = rules[rule_a_id], rules[rule_b_id]
            
            if rule_a.priority > rule_b.priority:
                to_quarantine.add(rule_b_id)
            elif rule_b.priority > rule_a.priority:
                to_quarantine.add(rule_a_id)
            else:
                # Equal priority - quarantine lower performing rule
                if rule_a.post_mean < rule_b.post_mean:
                    to_quarantine.add(rule_a_id)
                else:
                    to_quarantine.add(rule_b_id)
        
        return list(to_quarantine)
    
    def _resolve_by_performance(self, conflicts: List[Tuple[str, str, str]], rules: Dict[str, Rule]) -> List[str]:
        """Resolve conflicts by rule performance"""
        to_quarantine = set()
        
        for rule_a_id, rule_b_id, conflict_type in conflicts:
            if rule_a_id not in rules or rule_b_id not in rules:
                continue
                
            rule_a, rule_b = rules[rule_a_id], rules[rule_b_id]
            
            if rule_a.post_mean < rule_b.post_mean:
                to_quarantine.add(rule_a_id)
            else:
                to_quarantine.add(rule_b_id)
        
        return list(to_quarantine)
    
    def _resolve_by_consensus(self, conflicts: List[Tuple[str, str, str]], rules: Dict[str, Rule]) -> List[str]:
        """Resolve conflicts by considering multiple factors"""
        to_quarantine = set()
        
        for rule_a_id, rule_b_id, conflict_type in conflicts:
            if rule_a_id not in rules or rule_b_id not in rules:
                continue
                
            rule_a, rule_b = rules[rule_a_id], rules[rule_b_id]
            
            # Score based on multiple factors
            score_a = (rule_a.post_mean * 0.4 + 
                      rule_a.priority * 0.2 + 
                      rule_a.get_success_rate() * 0.3 +
                      (1.0 / (1.0 + rule_a.usage_count)) * 0.1)  # Novelty bonus
            
            score_b = (rule_b.post_mean * 0.4 + 
                      rule_b.priority * 0.2 + 
                      rule_b.get_success_rate() * 0.3 +
                      (1.0 / (1.0 + rule_b.usage_count)) * 0.1)
            
            if score_a < score_b:
                to_quarantine.add(rule_a_id)
            else:
                to_quarantine.add(rule_b_id)
        
        return list(to_quarantine)

# Keep existing classes with minor enhancements...
# (ReplayBuffer, RuleGenerator, StagnationDetector, DiversityEnforcer remain largely the same)

class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.buf = deque(maxlen=capacity)
        self.rule_performance = defaultdict(list)  # Track per-rule performance
    
    def add(self, entry: Tuple[str, float, Optional[Dict]]):
        """Enhanced to store context"""
        rule_id, reward, context = entry
        self.buf.append(entry)
        self.rule_performance[rule_id].append(reward)
    
    def get_rule_history(self, rule_id: str, k: int = 10) -> List[float]:
        """Get recent performance history for a rule"""
        return self.rule_performance[rule_id][-k:]
    
    def sample(self, k: int) -> List[Tuple[str, float, Optional[Dict]]]:
        if not self.buf:
            return []
        k = min(k, len(self.buf))
        return random.sample(list(self.buf), k)
    
    def __len__(self):
        return len(self.buf)

class RuleGenerator:
    def __init__(self, latent_dim: int = 8, seed: Optional[int] = None):
        self.latent_dim = int(latent_dim)
        if seed is not None:
            random.seed(seed)
        self.registry: Dict[str, List[float]] = {}
    
    def sample_latent(self):
        if HAS_NUMPY:
            return np.random.normal(size=(self.latent_dim,)).tolist()
        else:
            return [random.gauss(0, 1) for _ in range(self.latent_dim)]
    
    def generate(self, prefix="gen", idx: Optional[int] = None, 
                target_performance: Optional[float] = None) -> Rule:
        """Enhanced generation with target performance"""
        z = self.sample_latent()
        
        if target_performance is not None:
            # Bias latent toward target performance
            performance_bias = math.log(target_performance / (1 - target_performance + EPS))
            raw = performance_bias + sum(random.gauss(0, 0.3) * zi for zi in z)
        else:
            raw = sum(random.gauss(0, 0.5) * zi for zi in z) + random.gauss(0, 0.1)
        
        mean = 1.0 / (1.0 + math.exp(-raw))
        mean = max(0.01, min(0.99, mean))
        
        rid = f"{prefix}_{idx}" if idx is not None else f"{prefix}_{random.randint(0, 10**9)}"
        
        # Create enhanced rule with proper RuleParameter
        alpha_param = RuleParameter(
            value=random.uniform(0.0, 1.0),
            param_type="float",
            min_val=0.0,
            max_val=1.0,
            requires_grad=True
        )
        
        r = Rule(
            id=rid, 
            name=rid, 
            rtype=HEURISTIC, 
            init_weight=1.0, 
            init_mean=mean, 
            init_var=0.05, 
            latent_z=z,
            params={"alpha": alpha_param}, 
            provenance={"created_by": "generator", "target_performance": target_performance}
        )
        
        self.registry[r.id] = z
        return r
    
    def train_on_replay(self, replay: ReplayBuffer, rules: Dict[str, Rule], 
                       lr: float = 0.01, epochs: int = 1):
        """Enhanced training placeholder"""
        # This would be implemented with actual neural networks in torch_integration.py
        logger.info(f"Generator training placeholder - {len(replay)} samples, {lr} lr, {epochs} epochs")

# Enhanced remaining classes...
class StagnationDetector:
    def __init__(self, window: int = 20, entropy_drop_thresh: float = 0.3, no_improve_steps: int = 50):
        self.window = window
        self.entropy_history = deque(maxlen=window)
        self.performance_history = deque(maxlen=window)
        self.no_improve_steps = no_improve_steps
        self.best_score = -1e9
        self.steps_since_improve = 0
        self.entropy_drop_thresh = entropy_drop_thresh
    
    def observe(self, entropy: float, global_score: float, rule_count: int):
        """Enhanced observation with rule count"""
        self.entropy_history.append(entropy)
        self.performance_history.append(global_score)
        
        if global_score > self.best_score + 1e-9:
            self.best_score = global_score
            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1
    
    def is_stagnant(self) -> Tuple[bool, str]:
        """Returns (is_stagnant, reason)"""
        if len(self.entropy_history) < self.window:
            return False, "insufficient_data"
        
        e0 = self.entropy_history[0]
        e1 = self.entropy_history[-1]
        
        if e0 - e1 > self.entropy_drop_thresh:
            return True, "entropy_drop"
        
        if self.steps_since_improve >= self.no_improve_steps:
            return True, "no_improvement"
        
        # Check for performance degradation
        if len(self.performance_history) >= self.window:
            recent_avg = sum(list(self.performance_history)[-5:]) / 5
            old_avg = sum(list(self.performance_history)[:5]) / 5
            if old_avg - recent_avg > 0.1:  # 10% degradation
                return True, "performance_degradation"
        
        return False, "not_stagnant"

class DiversityEnforcer:
    def __init__(self, sim_threshold: float = 0.95, min_diversity: float = 0.1):
        self.sim_threshold = sim_threshold
        self.min_diversity = min_diversity
    
    def rule_similarity(self, a: Rule, b: Rule) -> float:
        """Enhanced similarity calculation"""
        if a.latent_z is None or b.latent_z is None:
            return 0.0
        
        if HAS_NUMPY:
            va = np.array(a.latent_z)
            vb = np.array(b.latent_z)
            ca = np.linalg.norm(va)
            cb = np.linalg.norm(vb)
            if ca < EPS or cb < EPS:
                return 0.0
            return float((va.dot(vb)) / (ca * cb))
        else:
            # cosine similarity
            dot = sum(x * y for x, y in zip(a.latent_z, b.latent_z))
            na = math.sqrt(sum(x * x for x in a.latent_z))
            nb = math.sqrt(sum(x * x for x in b.latent_z))
            if na < EPS or nb < EPS:
                return 0.0
            return dot / (na * nb)
    
    def compute_diversity_score(self, rules: Dict[str, Rule]) -> float:
        """Compute overall diversity score"""
        if len(rules) < 2:
            return 1.0
        
        total_sim = 0.0
        count = 0
        rule_list = list(rules.values())
        
        for i in range(len(rule_list)):
            for j in range(i + 1, len(rule_list)):
                total_sim += self.rule_similarity(rule_list[i], rule_list[j])
                count += 1
        
        avg_sim = total_sim / max(1, count)
        return 1.0 - avg_sim  # Higher diversity = lower average similarity
    
    def enforce(self, rules: Dict[str, Rule]) -> List[Tuple[str, str, float]]:
        """Enhanced diversity enforcement"""
        pairs = []
        rule_list = list(rules.values())
        
        for i in range(len(rule_list)):
            for j in range(i + 1, len(rule_list)):
                a = rule_list[i]
                b = rule_list[j]
                sim = self.rule_similarity(a, b)
                if sim > self.sim_threshold:
                    pairs.append((a.id, b.id, sim))
        
        return pairs

# ------------------ Enhanced DRMSystem ------------------
class DRMSystem:
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.replay = ReplayBuffer()
        self.generators: List[RuleGenerator] = []
        self.stagnation = StagnationDetector()
        self.diversity = DiversityEnforcer()
        self.conflict_resolver = ConflictResolver()
        self.audit_log: List[Dict[str, Any]] = []
        self.archived: Dict[str, Rule] = {}
        self.cycle_count = 0
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.diversity_history = deque(maxlen=1000)
    
    # Enhanced CRUD operations
    def add_rule(self, rule: Rule, check_conflicts: bool = True) -> Dict[str, Any]:
        """Enhanced rule addition with conflict checking"""
        result = {"success": True, "conflicts": [], "warnings": []}
        
        if rule.id in self.rules:
            result["success"] = False
            result["warnings"].append(f"Rule {rule.id} already exists")
            return result
        
        # Check for conflicts if requested
        if check_conflicts and self.rules:
            temp_rules = dict(self.rules)
            temp_rules[rule.id] = rule
            conflicts = self.conflict_resolver.detect_conflicts(temp_rules, {})
            
            rule_conflicts = [c for c in conflicts if rule.id in (c[0], c[1])]
            if rule_conflicts:
                result["conflicts"] = rule_conflicts
                result["warnings"].append(f"Rule has {len(rule_conflicts)} conflicts")
        
        self.rules[rule.id] = rule
        rule.weight = max(EPS, rule.weight)
        self.audit("add", rule.id, {
            "type": rule.type, 
            "priority": rule.priority,
            "conflicts": result["conflicts"]
        })
        
        return result
    
    def remove_rule(self, rule_id: str, archive: bool = True, reason: str = "manual") -> bool:
        """Enhanced rule removal"""
        if rule_id not in self.rules:
            return False
        
        r = self.rules.pop(rule_id)
        if archive:
            self.archived[rule_id] = r
        
        self.audit("remove", rule_id, {"reason": reason, "archived": archive})
        return True
    
    def quarantine_rule(self, rule_id: str, reason: str = "manual") -> bool:
        """Quarantine a rule without removing it"""
        if rule_id not in self.rules:
            return False
        
        self.rules[rule_id].quarantined = True
        self.rules[rule_id].quarantine_reason = reason
        self.audit("quarantine", rule_id, {"reason": reason})
        return True
    
    def unquarantine_rule(self, rule_id: str) -> bool:
        """Remove rule from quarantine"""
        if rule_id not in self.rules:
            return False
        
        self.rules[rule_id].quarantined = False
        self.rules[rule_id].quarantine_reason = None
        self.audit("unquarantine", rule_id, {})
        return True
    
    def register_generator(self, gen: RuleGenerator):
        self.generators.append(gen)
    
    def generate_rules(self, gen: RuleGenerator, count: int = 1, 
                      prefix: str = "g", target_performance: Optional[float] = None):
        """Enhanced rule generation"""
        for i in range(count):
            r = gen.generate(prefix, i, target_performance=target_performance)
            self.add_rule(r)
    
    def get_distribution(self, include_quarantined: bool = False) -> Dict[str, float]:
        """Get rule weight distribution"""
        if include_quarantined:
            active_rules = self.rules
        else:
            active_rules = {k: v for k, v in self.rules.items() if not v.quarantined}
        
        names = list(active_rules.keys())
        weights = [max(EPS, active_rules[n].weight) for n in names]
        s = sum(weights) or EPS
        return {n: w / s for n, w in zip(names, weights)}
    
    def get_active_rules(self) -> Dict[str, Rule]:
        """Get non-quarantined rules"""
        return {k: v for k, v in self.rules.items() if not v.quarantined}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        active_rules = self.get_active_rules()
        
        return {
            "total_rules": len(self.rules),
            "active_rules": len(active_rules),
            "quarantined_rules": len(self.rules) - len(active_rules),
            "archived_rules": len(self.archived),
            "cycle_count": self.cycle_count,
            "diversity_score": self.diversity.compute_diversity_score(active_rules),
            "avg_performance": sum(r.post_mean for r in active_rules.values()) / max(1, len(active_rules)),
            "total_observations": sum(r.observations for r in self.rules.values()),
            "rule_types": {
                LOGICAL: len([r for r in self.rules.values() if r.type == LOGICAL]),
                HEURISTIC: len([r for r in self.rules.values() if r.type == HEURISTIC]),
                HYBRID: len([r for r in self.rules.values() if r.type == HYBRID])
            },
            "replay_buffer_size": len(self.replay)
        }
    
    # Enhanced strength and scoring functions
    def compute_strengths(self, curiosity: float = 1.0, frz: float = 1.0, 
                         include_quarantined: bool = False) -> Dict[str, float]:
        """Enhanced strength computation"""
        strengths = {}
        rules_to_consider = self.rules if include_quarantined else self.get_active_rules()
        
        for rid, r in rules_to_consider.items():
            if r.quarantined and not include_quarantined:
                continue
                
            w = r.weight
            usage = r.usage_count
            novelty = 1.0 / (1.0 + usage)
            curiosity_term = curiosity * novelty if r.is_new else 0.0
            
            # Enhanced strength with success rate
            success_rate = r.get_success_rate()
            base = w * math.log(1.0 + usage + EPS)
            
            score = (base * 
                    (1.0 + curiosity_term) * 
                    (r.post_mean + 1e-3) * 
                    (success_rate + 0.1) *  # Success rate bonus
                    frz)
            
            strengths[rid] = float(score)
        
        return strengths
    
    def compute_scores(self, beta: float = 0.5, lam: float = 0.5, 
                      exploration_bonus: float = 0.02) -> Dict[str, float]:
        """Enhanced scoring with more factors"""
        scores = {}
        
        for name, r in self.rules.items():
            if r.quarantined:
                scores[name] = -1e9
                continue
            
            novelty = 1.0 / (1.0 + r.usage_count)
            risk = r.post_var
            exploration = exploration_bonus if r.is_new else 0.0
            
            # Add success rate and priority factors
            success_rate = r.get_success_rate()
            priority_bonus = r.priority * 0.1
            
            score = (r.post_mean + 
                    beta * novelty - 
                    lam * risk + 
                    exploration +
                    0.2 * success_rate +  # Success rate contribution
                    priority_bonus)
            
            scores[name] = float(score)
        
        return scores
    
    # Enhanced multiplicative update
    def multiplicative_update(self, eta: float = 0.05, beta: float = 0.5, 
                            lam: float = 0.5, kl_max: float = 0.5) -> float:
        """Enhanced multiplicative update with better numerical stability"""
        old = self.get_distribution()
        scores = self.compute_scores(beta=beta, lam=lam)
        
        active_rules = self.get_active_rules()
        names = list(active_rules.keys())
        
        if not names:
            return 0.0
        
        weights = [max(EPS, active_rules[n].weight) for n in names]
        
        # Compute tentative weights
        tentative = []
        for w, n in zip(weights, names):
            score = scores.get(n, 0.0)
            # Clip extreme scores for numerical stability
            score = max(-10.0, min(10.0, score))
            tentative.append(w * math.exp(eta * score))
        
        s = sum(tentative) or EPS
        tentative = [v / s for v in tentative]
        
        # Compute KL divergence
        kl = kl_divergence({n: tentative[i] for i, n in enumerate(names)}, old)
        
        # Adaptive eta if KL is too large
        if kl > kl_max and kl > 0:
            eta_scaled = max(1e-6, eta * (kl_max / kl))
            tentative = []
            for w, n in zip(weights, names):
                score = scores.get(n, 0.0)
                score = max(-10.0, min(10.0, score))
                tentative.append(w * math.exp(eta_scaled * score))
            
            s2 = sum(tentative) or EPS
            tentative = [v / s2 for v in tentative]
        
        # Update weights
        avg_scale = len(names) or 1
        for i, n in enumerate(names):
            self.rules[n].weight = max(EPS, tentative[i] * avg_scale)
        
        return kl
    
    # Enhanced validation
    def validate_rule(self, rule_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced rule validation"""
        if rule_id not in self.rules:
            return {"error": "not_found"}
        
        r = self.rules[rule_id]
        res = r.evaluate_semantics(context)
        
        decision = {
            "activate": False, 
            "quarantine": False, 
            "reason": None, 
            "score": res["score"], 
            "details": res,
            "confidence": r.post_mean,
            "uncertainty": math.sqrt(r.post_var)
        }
        
        # Enhanced decision logic based on rule type
        if r.type == LOGICAL:
            if res["tests_pass"] and res["syntactic"] and res["param_ok"]:
                decision["activate"] = True
                decision["reason"] = "logical_validated"
            else:
                decision["quarantine"] = True
                decision["reason"] = "logical_failed"
        
        elif r.type == HYBRID:
            # HYBRID rules need either tests to pass OR high confidence
            tests_ok = res["tests_pass"] and res["param_ok"]
            high_confidence = r.post_mean > 0.5 and r.observations >= 3
            
            if tests_ok or high_confidence:
                decision["activate"] = True
                decision["reason"] = "hybrid_validated"
            else:
                decision["quarantine"] = True
                decision["reason"] = "hybrid_low_conf"
        
        else:  # HEURISTIC
            # HEURISTIC rules are more forgiving
            if res["tests_pass"] or r.post_mean > 0.3:
                decision["activate"] = True
                decision["reason"] = "heuristic_accepted"
            else:
                if not res["tests_pass"] and r.post_mean < 0.2:
                    decision["quarantine"] = True
                    decision["reason"] = "heuristic_failing"
                else:
                    decision["activate"] = True
                    decision["reason"] = "heuristic_uncertain"
        
        # Apply quarantine if decided
        if decision["quarantine"]:
            r.quarantined = True
            r.quarantine_reason = decision["reason"]
            self.audit("quarantine", r.id, {"reason": decision["reason"]})
        
        # Track activation
        if decision["activate"]:
            r.activation_count += 1
            r.last_activated = time.time()
        
        return decision
    
    def explain_rule(self, rule_id: str) -> Dict[str, Any]:
        if rule_id not in self.rules:
            return {"error": "not_found"}
        return self.rules[rule_id].explain()
    
    def mutate_rule(self, rule_id: str, op: str = "tweak_param", 
                   magnitude: float = 0.1) -> Dict[str, Any]:
        """Enhanced mutation with better tracking"""
        if rule_id not in self.rules:
            return {"error": "not_found"}
        
        r = self.rules[rule_id]
        
        if r.type == LOGICAL:
            return {"mutated": False, "reason": "logical_protected"}
        
        old_params = {k: v.value for k, v in r.params.items()}
        old_weight = r.weight
        
        ok = r.mutate(op=op, magnitude=magnitude)
        
        new_params = {k: v.value for k, v in r.params.items()}
        
        self.audit("mutate", r.id, {
            "op": op, 
            "magnitude": magnitude, 
            "success": ok,
            "old_params": old_params,
            "new_params": new_params,
            "old_weight": old_weight,
            "new_weight": r.weight
        })
        
        return {
            "mutated": ok, 
            "new_params": new_params, 
            "new_weight": r.weight,
            "param_changes": {k: new_params[k] - old_params.get(k, 0) 
                            for k in new_params.keys()}
        }
    
    # Enhanced dezinformation detection
    def detect_dezinformation(self, mu_min: float = 0.1, tau: float = 0.95) -> List[str]:
        """Enhanced dezinformation detection"""
        quarantined = []
        
        for rid, r in list(self.rules.items()):
            if r.quarantined:
                continue
            
            # Multiple criteria for quarantine
            low_posterior = r.post_prob_below(mu_min) > tau
            low_success = r.get_success_rate() < 0.1 and r.observations >= 10
            high_uncertainty = r.post_var > 0.5 and r.observations >= 5
            
            if low_posterior or low_success or high_uncertainty:
                reason = []
                if low_posterior:
                    reason.append("low_posterior")
                if low_success:
                    reason.append("low_success_rate")
                if high_uncertainty:
                    reason.append("high_uncertainty")
                
                r.quarantined = True
                r.quarantine_reason = "_".join(reason)
                quarantined.append(rid)
                self.audit("quarantine_dezinfo", rid, {"reasons": reason})
        
        return quarantined
    
    # Enhanced diversity and revival mechanisms
    def enforce_diversity(self):
        """Enhanced diversity enforcement"""
        pairs = self.diversity.enforce(self.rules)
        
        for a_id, b_id, sim in pairs:
            # Choose which rule to perturb (prefer lower performing)
            if a_id in self.rules and b_id in self.rules:
                rule_a, rule_b = self.rules[a_id], self.rules[b_id]
                
                if rule_a.post_mean < rule_b.post_mean:
                    target_rule = rule_a
                else:
                    target_rule = rule_b
                
                # Try different perturbation strategies
                if target_rule.mutate(op="perturb_latent", magnitude=0.2):
                    self.audit("diversity_perturb_latent", target_rule.id, {"sim": sim})
                elif target_rule.mutate(op="tweak_param", magnitude=0.3):
                    self.audit("diversity_perturb_param", target_rule.id, {"sim": sim})
    
    def emergency_revival(self, num_new: int = 3):
        """Enhanced emergency revival"""
        # Generate new rules with diverse performance targets
        for gen in self.generators:
            for i in range(num_new):
                # Target different performance levels
                target_perf = 0.3 + (i / num_new) * 0.4  # 0.3 to 0.7 range
                r = gen.generate(prefix="revival", idx=None, target_performance=target_perf)
                r.priority = 1  # Give revival rules slight priority
                self.add_rule(r)
                self.audit("revival_generate", r.id, {"target_performance": target_perf})
        
        # Boost struggling but promising rules
        active_rules = self.get_active_rules()
        struggling_rules = [
            r for r in active_rules.values() 
            if (r.type == HEURISTIC and 
                r.post_mean < 0.4 and 
                r.post_var > 0.1 and 
                r.observations >= 3)
        ]
        
        struggling_rules.sort(key=lambda x: x.post_var, reverse=True)  # Most uncertain first
        
        for r in struggling_rules[:max(1, len(struggling_rules) // 3)]:
            r.weight *= 1.8  # Bigger boost
            r.post_var *= 0.8  # Reduce uncertainty slightly
            self.audit("revival_boost", r.id, {"new_weight": r.weight})
    
    def resolve_conflicts(self, strategy: str = "consensus") -> Dict[str, Any]:
        """Resolve conflicts between rules"""
        conflicts = self.conflict_resolver.detect_conflicts(self.rules, {})
        
        if not conflicts:
            return {"conflicts_found": 0, "quarantined": []}
        
        to_quarantine = self.conflict_resolver.resolve_conflicts(conflicts, self.rules, strategy)
        
        for rule_id in to_quarantine:
            if rule_id in self.rules:
                self.quarantine_rule(rule_id, f"conflict_resolution_{strategy}")
        
        return {
            "conflicts_found": len(conflicts),
            "quarantined": to_quarantine,
            "conflicts": conflicts
        }
    
    # Enhanced lifecycle run_cycle
    def run_cycle(self, evaluator: Callable[[Rule], Optional[float]],
                  eta: float = 0.05, beta: float = 0.5, lam: float = 0.5,
                  kl_max: float = 0.5, mu_min: float = 0.1, tau: float = 0.95,
                  replay_batch: int = 0, generator_train: bool = False,
                  resolve_conflicts: bool = True, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced run cycle with comprehensive tracking"""
        
        self.cycle_count += 1
        context = context or {}
        
        active_rules = self.get_active_rules()
        if not active_rules:
            return {"status": "no_active_rules", "cycle": self.cycle_count}
        
        # Evaluation phase
        sampled = []
        for rule_id, rule in active_rules.items():
            try:
                reward = evaluator(rule)
                if reward is not None:
                    self.replay.add((rule_id, float(reward), context))
                    rule.update_posterior(reward, context=context)
                sampled.append((rule_id, reward))
            except Exception as e:
                logger.warning(f"Evaluator failed for rule {rule_id}: {str(e)}")
                sampled.append((rule_id, None))
        
        # Replay stabilization
        if replay_batch and len(self.replay) > 0:
            for rule_id, reward, _ in self.replay.sample(replay_batch):
                if rule_id in self.rules:
                    self.rules[rule_id].update_posterior(reward)
        
        # Generator training
        if generator_train and self.generators and len(self.replay) > 0:
            for gen in self.generators:
                gen.train_on_replay(self.replay, self.rules)
        
        # Conflict resolution
        conflict_result = {}
        if resolve_conflicts:
            conflict_result = self.resolve_conflicts()
        
        # Weight updates
        kl = self.multiplicative_update(eta=eta, beta=beta, lam=lam, kl_max=kl_max)
        
        # Dezinformation detection
        quarantined = self.detect_dezinformation(mu_min=mu_min, tau=tau)
        
        # System metrics
        dist = self.get_distribution()
        ent = entropy_from_dist(dist)
        diversity_score = self.diversity.compute_diversity_score(active_rules)
        
        # Global performance
        global_score = sum(r.post_mean for r in active_rules.values()) / max(1, len(active_rules))
        
        # Track history
        self.performance_history.append(global_score)
        self.diversity_history.append(diversity_score)
        
        # Stagnation detection
        self.stagnation.observe(ent, global_score, len(active_rules))
        is_stagnant, stagnation_reason = self.stagnation.is_stagnant()
        
        if is_stagnant:
            self.audit("stagnation_detected", None, {
                "reason": stagnation_reason,
                "entropy": ent, 
                "global_score": global_score,
                "diversity": diversity_score
            })
            
            self.enforce_diversity()
            self.emergency_revival(num_new=3)
        
        # Comprehensive result
        result = {
            "cycle": self.cycle_count,
            "status": "completed",
            "evaluations": sampled,
            "quarantined": quarantined,
            "conflicts": conflict_result,
            "entropy": ent,
            "diversity_score": diversity_score,
            "distribution": dist,
            "kl_divergence": kl,
            "global_score": global_score,
            "stagnation": {
                "detected": is_stagnant,
                "reason": stagnation_reason
            },
            "stats": self.get_stats()
        }
        
        return result
    
    # Enhanced serialization and utilities
    def audit(self, action: str, rule_id: Optional[str], info: Dict[str, Any]):
        """Enhanced audit logging"""
        entry = {
            "timestamp": time.time(),
            "cycle": self.cycle_count,
            "action": action,
            "rule_id": rule_id,
            "info": info
        }
        self.audit_log.append(entry)
        
        # Keep audit log manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep last 5000 entries
    
    def get_audit_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get audit log summary"""
        logs = self.audit_log[-last_n:] if last_n else self.audit_log
        
        action_counts = defaultdict(int)
        rule_activity = defaultdict(int)
        
        for entry in logs:
            action_counts[entry["action"]] += 1
            if entry["rule_id"]:
                rule_activity[entry["rule_id"]] += 1
        
        return {
            "total_entries": len(logs),
            "action_counts": dict(action_counts),
            "most_active_rules": dict(sorted(rule_activity.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]),
            "time_range": (logs[0]["timestamp"], logs[-1]["timestamp"]) if logs else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Enhanced serialization"""
        return {
            "rules": [r.to_dict() for r in self.rules.values()],
            "archived": [r.to_dict() for r in self.archived.values()],
            "audit": list(self.audit_log),
            "cycle_count": self.cycle_count,
            "stats": self.get_stats()
        }
    
    def save_json(self, path: str, include_audit: bool = True):
        """Enhanced JSON saving with options"""
        data = self.to_dict()
        
        if not include_audit:
            data.pop("audit", None)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_json(self, path: str):
        """Enhanced JSON loading"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        
        # Load rules
        self.rules.clear()
        for rd in d.get("rules", []):
            r = Rule.from_dict(rd)
            self.rules[r.id] = r
        
        # Load archived rules
        self.archived.clear()
        for rd in d.get("archived", []):
            r = Rule.from_dict(rd)
            self.archived[r.id] = r
        
        # Load audit log
        self.audit_log = d.get("audit", [])
        self.cycle_count = d.get("cycle_count", 0)
        
        logger.info(f"Loaded {len(self.rules)} rules and {len(self.archived)} archived rules")


# Utility functions
def entropy_from_dist(dist: Dict[str, float]) -> float:
    return -sum(p * math.log(p + EPS) for p in dist.values())

def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    kl = 0.0
    for k in keys:
        pk = p.get(k, EPS)
        qk = q.get(k, EPS)
        kl += pk * math.log((pk + EPS) / (qk + EPS))
    return kl


# ------------------ Enhanced demonstration ------------------
if __name__ == "__main__":
    # Enhanced demo with more comprehensive testing
    logging.basicConfig(level=logging.INFO)
    
    ds = DRMSystem()
    gen = RuleGenerator(latent_dim=6, seed=42)
    ds.register_generator(gen)
    
    # Create a LOGICAL rule with enhanced parameters
    logical_param = RuleParameter(
        value=0.0,
        param_type="float", 
        min_val=0.0,
        max_val=1.0,
        requires_grad=False
    )
    
    r1 = Rule(
        id="mass_conservation", 
        name="Mass Conservation Law", 
        rtype=LOGICAL,
        priority=10,  # High priority
        pre_conditions=["domain_advec1d", "mass_initial"],
        post_conditions=["mass_change<1e-3"],
        params={"tolerance": logical_param},
        tests=[lambda rule, ctx: (
            "mass_change" in ctx and 
            abs(ctx.get("mass_change", 1.0)) < rule.params["tolerance"].value + 1e-3
        )],
        provenance={"created_by": "physics_expert", "domain": "conservation_laws"},
        conflict_groups={"conservation_laws"}
    )
    
    result = ds.add_rule(r1)
    print(f"Added logical rule: {result}")
    
    # Generate some heuristic rules
    ds.generate_rules(gen, count=3, prefix="heuristic", target_performance=0.6)
    
    # Create a HYBRID rule with differentiable parameters
    hybrid_alpha = RuleParameter(
        value=0.5,
        param_type="float",
        min_val=0.0,
        max_val=1.0,
        requires_grad=True  # This can be used with PyTorch
    )
    
    hybrid_beta = RuleParameter(
        value=0.3,
        param_type="float",
        min_val=0.0,
        max_val=1.0,
        requires_grad=True
    )
    
    r_hybrid = Rule(
        id="adaptive_diffusion",
        name="Adaptive Diffusion Control",
        rtype=HYBRID,
        priority=5,
        pre_conditions=["temperature_field", "boundary_conditions"],
        post_conditions=["diffusion_stable"],
        params={
            "alpha": hybrid_alpha,
            "beta": hybrid_beta
        },
        tests=[lambda rule, ctx: ctx.get("temperature_gradient", 0) < 10.0],
        provenance={"created_by": "ml_optimizer", "domain": "heat_transfer"},
        conflict_groups={"diffusion_control"}
    )
    
    result = ds.add_rule(r_hybrid)
    print(f"Added hybrid rule: {result}")
    
    # Enhanced evaluator that considers rule types and context
    def enhanced_evaluator(rule: Rule) -> Optional[float]:
        if rule.type == LOGICAL:
            # Logical rules get perfect score if mass is conserved
            mass_change = abs(demo_ctx.get("mass_change", 1.0))
            tolerance = rule.params.get("tolerance", RuleParameter(1e-3)).value
            return 1.0 if mass_change <= tolerance else 0.1
        
        elif rule.type == HYBRID:
            # HYBRID rules performance depends on parameters and context
            alpha = rule.params.get("alpha", RuleParameter(0.5)).value
            beta = rule.params.get("beta", RuleParameter(0.3)).value
            
            # Simulate performance based on parameters and context
            temp_gradient = demo_ctx.get("temperature_gradient", 5.0)
            performance = max(0.0, min(1.0, 
                0.5 + 0.3 * alpha - 0.2 * beta - 0.1 * (temp_gradient / 10.0)))
            
            # Add some noise
            performance += random.gauss(0, 0.05)
            return max(0.0, min(1.0, performance))
        
        else:  # HEURISTIC
            # Heuristic performance based on posterior with noise
            base_performance = rule.post_mean
            noise = random.gauss(0, 0.1)
            return max(0.0, min(1.0, base_performance + noise))
    
    # Enhanced demo context
    demo_ctx = {
        "domain_advec1d": True,
        "mass_initial": 1.0,
        "mass_change": 5e-4,  # Small change - should satisfy conservation
        "temperature_field": True,
        "boundary_conditions": True,
        "temperature_gradient": 7.5
    }
    
    print("\n=== Initial System Stats ===")
    print(json.dumps(ds.get_stats(), indent=2))
    
    # Test rule validation
    print("\n=== Rule Validation Tests ===")
    for rule_id in ds.rules.keys():
        validation = ds.validate_rule(rule_id, demo_ctx)
        print(f"{rule_id}: {validation['reason']} (score: {validation['score']:.3f})")
    
    # Run multiple cycles to see evolution
    print("\n=== Running Evolution Cycles ===")
    for cycle in range(5):
        result = ds.run_cycle(
            evaluator=enhanced_evaluator,
            eta=0.1,
            beta=0.6,
            lam=0.3,
            replay_batch=3,
            context=demo_ctx
        )
        
        print(f"Cycle {cycle + 1}:")
        print(f"  Global Score: {result['global_score']:.3f}")
        print(f"  Diversity: {result['diversity_score']:.3f}")
        print(f"  Entropy: {result['entropy']:.3f}")
        print(f"  Stagnation: {result['stagnation']['detected']} ({result['stagnation']['reason']})")
        print(f"  Active Rules: {result['stats']['active_rules']}")
        
        if result['quarantined']:
            print(f"  Quarantined: {result['quarantined']}")
        
        # Show top performing rules
        active_rules = ds.get_active_rules()
        if active_rules:
            sorted_rules = sorted(active_rules.items(), 
                                key=lambda x: x[1].post_mean, reverse=True)
            print("  Top rules:")
            for rule_id, rule in sorted_rules[:3]:
                print(f"    {rule_id}: {rule.post_mean:.3f} (±{math.sqrt(rule.post_var):.3f})")
    
    # Test mutations
    print("\n=== Testing Mutations ===")
    for rule_id in list(ds.get_active_rules().keys())[:2]:
        mutation_result = ds.mutate_rule(rule_id, "tweak_param", 0.2)
        print(f"Mutated {rule_id}: {mutation_result}")
    
    # Test conflict detection
    print("\n=== Conflict Detection ===")
    conflict_result = ds.resolve_conflicts("consensus")
    print(f"Conflicts found: {conflict_result['conflicts_found']}")
    if conflict_result['conflicts']:
        for conflict in conflict_result['conflicts']:
            print(f"  {conflict[0]} vs {conflict[1]}: {conflict[2]}")
    
    # Show final system state
    print("\n=== Final System Summary ===")
    print(json.dumps(ds.get_stats(), indent=2))
    
    # Test serialization
    print("\n=== Testing Serialization ===")
    ds.save_json("drm_test_save.json")
    
    # Create new system and load
    ds2 = DRMSystem()
    ds2.load_json("drm_test_save.json")
    print(f"Loaded system has {len(ds2.rules)} rules")
    
    # Show audit summary
    print("\n=== Audit Summary ===")
    audit_summary = ds.get_audit_summary(last_n=20)
    print(f"Recent audit entries: {audit_summary['total_entries']}")
    print("Action counts:", audit_summary['action_counts'])
    
    # Show rule explanations
    print("\n=== Rule Explanations ===")
    for rule_id in list(ds.rules.keys())[:2]:
        explanation = ds.explain_rule(rule_id)
        print(f"\nRule: {rule_id}")
        print(f"  Type: {explanation['type']}")
        print(f"  Performance: {explanation['post_mean']:.3f} ± {math.sqrt(explanation['post_var']):.3f}")
        print(f"  Success Rate: {explanation['success_rate']:.3f}")
        print(f"  Usage Count: {explanation['usage_count']}")
        if explanation.get('params'):
            print("  Parameters:")
            for param_name, param_info in explanation['params'].items():
                grad_str = " (grad)" if param_info['requires_grad'] else ""
                print(f"    {param_name}: {param_info['value']:.3f}{grad_str}")
    
    print("\n=== Demo Complete ===")
    print("The enhanced DRM system is ready for PyTorch integration!")