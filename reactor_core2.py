"""reactor_core_improved.py
Enhanced Information Reactor with tight DRM integration and thermodynamic conservation.

Key improvements over reactor_core.py:
- InformationState unifying (S,I,E) framework
- ReactiveRule integrating DRM rules with reaction mechanisms
- ConservationValidator enforcing information/energy conservation laws
- DRM 3.0 formula with anti-stagnation mechanisms
- Tight coupling with drm_module5_improved.py
- Thermodynamic reaction rates and equilibrium
- Emergent information dynamics

Usage:
    from drm_module5_improved import DRMSystem
    from reactor_core_improved import InformationReactor, InformationState
    
    drm = DRMSystem()
    reactor = InformationReactor(drm_system=drm)
    
    initial_state = InformationState(...)
    new_state = reactor.catalytic_cycle(initial_state, stimulus)
"""

from __future__ import annotations
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from collections import deque, defaultdict
import threading

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Import DRM components
try:
    from drm_module5_improved import (
        DRMSystem, Rule, RuleParameter, LOGICAL, HEURISTIC, HYBRID,
        entropy_from_dist, kl_divergence
    )
    HAS_DRM = True
except ImportError:
    HAS_DRM = False
    # Fallback definitions
    LOGICAL, HEURISTIC, HYBRID = "LOGICAL", "HEURISTIC", "HYBRID"

logger = logging.getLogger(__name__)
EPS = 1e-9

# =============================================================================
# Information State - Core (S,I,E) Framework
# =============================================================================

@dataclass
class InformationState:
    """Unified (Structure, Information, Energy) state representation"""
    
    # S - Structural state (explicit observable state)
    structural: Dict[str, Any] = field(default_factory=dict)
    
    # I - Information content (implicit/latent information)
    latent_info: List[float] = field(default_factory=list)
    information_entropy: float = 0.5
    
    # E - Energy budget and flows
    available_energy: float = 1.0
    energy_flow_rate: float = 0.1
    dissipated_energy: float = 0.0
    
    # Additional state tracking
    creation_time: float = field(default_factory=time.time)
    transformation_count: int = 0
    complexity_measure: float = 0.0
    
    def total_information(self) -> float:
        """Calculate total information content"""
        latent_norm = math.sqrt(sum(x*x for x in self.latent_info)) if self.latent_info else 0.0
        structural_info = len(self.structural) * 0.1  # Simple proxy
        return self.information_entropy + latent_norm + structural_info
    
    def information_density(self) -> float:
        """Information per unit energy"""
        return self.total_information() / max(self.available_energy, EPS)
    
    def copy(self) -> 'InformationState':
        """Create deep copy of state"""
        return InformationState(
            structural=dict(self.structural),
            latent_info=list(self.latent_info),
            information_entropy=self.information_entropy,
            available_energy=self.available_energy,
            energy_flow_rate=self.energy_flow_rate,
            dissipated_energy=self.dissipated_energy,
            transformation_count=self.transformation_count,
            complexity_measure=self.complexity_measure
        )
    
    def merge_with(self, other: 'InformationState', weight: float = 0.5) -> 'InformationState':
        """Merge two information states"""
        merged = self.copy()
        
        # Merge structural (union of keys, weighted values for conflicts)
        for key, value in other.structural.items():
            if key in merged.structural:
                if isinstance(merged.structural[key], (int, float)) and isinstance(value, (int, float)):
                    merged.structural[key] = weight * merged.structural[key] + (1-weight) * value
            else:
                merged.structural[key] = value
        
        # Merge latent information
        if other.latent_info:
            if merged.latent_info:
                # Weighted combination
                min_len = min(len(merged.latent_info), len(other.latent_info))
                for i in range(min_len):
                    merged.latent_info[i] = weight * merged.latent_info[i] + (1-weight) * other.latent_info[i]
                # Append remaining
                if len(other.latent_info) > min_len:
                    merged.latent_info.extend(other.latent_info[min_len:])
            else:
                merged.latent_info = list(other.latent_info)
        
        # Merge energetics
        merged.information_entropy = weight * merged.information_entropy + (1-weight) * other.information_entropy
        merged.available_energy = merged.available_energy + other.available_energy
        merged.transformation_count = max(merged.transformation_count, other.transformation_count) + 1
        
        return merged

# =============================================================================
# Conservation Validator
# =============================================================================

class ConservationValidator:
    """Validates conservation laws for information reactions"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.violations = deque(maxlen=100)
    
    def validate_reaction(self, 
                         before: InformationState, 
                         after: InformationState,
                         energy_input: float = 0.0,
                         allow_information_creation: bool = True) -> Dict[str, Any]:
        """Validate conservation laws for a reaction"""
        
        result = {
            "valid": True,
            "violations": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Energy conservation: E_before + E_input >= E_after + E_dissipated
        energy_before = before.available_energy + energy_input
        energy_after = after.available_energy + after.dissipated_energy
        energy_violation = energy_after - energy_before
        
        result["metrics"]["energy_violation"] = energy_violation
        
        if energy_violation > self.tolerance:
            result["valid"] = False
            result["violations"].append(f"Energy not conserved: {energy_violation:.6f}")
        
        # Information bounds: I_after <= I_before + energy_input (information can't exceed energy input)
        info_before = before.total_information()
        info_after = after.total_information()
        info_increase = info_after - info_before
        
        result["metrics"]["information_change"] = info_increase
        
        if not allow_information_creation and info_increase > energy_input + self.tolerance:
            result["valid"] = False
            result["violations"].append(f"Information increase exceeds energy input: {info_increase:.6f} > {energy_input:.6f}")
        
        # Entropy should not decrease without energy input (2nd law analogy)
        entropy_change = after.information_entropy - before.information_entropy
        result["metrics"]["entropy_change"] = entropy_change
        
        if entropy_change < -self.tolerance and energy_input < self.tolerance:
            result["warnings"].append(f"Entropy decreased without energy input: {entropy_change:.6f}")
        
        # Complexity bounds
        complexity_change = after.complexity_measure - before.complexity_measure
        result["metrics"]["complexity_change"] = complexity_change
        
        if complexity_change > 2.0 * energy_input + self.tolerance:
            result["warnings"].append(f"Complexity increase seems too large: {complexity_change:.6f}")
        
        # Record violations for analysis
        if not result["valid"]:
            self.violations.append({
                "timestamp": time.time(),
                "violations": result["violations"],
                "metrics": result["metrics"]
            })
        
        return result

# =============================================================================
# Reactive Rule - Unified Rule + Reaction
# =============================================================================

class ReactiveRule:
    """Unified rule that can both evaluate and react"""
    
    def __init__(self, 
                 base_rule: Rule,
                 reaction_triggers: Set[str] = None,
                 energy_cost: float = 0.1,
                 information_transform: Optional[Callable] = None,
                 reaction_priority: int = 1,
                 max_runtime: float = 1.0):
        
        self.base_rule = base_rule
        self.reaction_triggers = reaction_triggers or set()
        self.energy_cost = energy_cost
        self.information_transform = information_transform
        self.reaction_priority = reaction_priority
        self.max_runtime = max_runtime
        
        # Reaction-specific tracking
        self.reaction_count = 0
        self.total_energy_consumed = 0.0
        self.average_reaction_time = 0.0
        self.last_reaction_cycle = -1
        
        # DRM 3.0 components
        self.stagnation_resistance = 1.0
        self.curiosity_level = 1.0
        self.emergence_pressure = 1.0
    
    def can_react(self, state: InformationState, context: Dict[str, Any], current_cycle: int) -> bool:
        """Check if reaction can be triggered"""
        # Energy check
        if state.available_energy < self.energy_cost:
            return False
        
        # Cooldown check (prevent too frequent reactions)
        if current_cycle - self.last_reaction_cycle < 3:
            return False
        
        # Trigger check
        if self.reaction_triggers:
            triggered = any(context.get(trigger, False) for trigger in self.reaction_triggers)
            if not triggered:
                return False
        
        # Semantic validation through base rule
        validation = self.base_rule.evaluate_semantics(context)
        return validation["score"] > 0.3
    
    def compute_reaction_rate(self, state: InformationState, context: Dict[str, Any]) -> float:
        """DRM 3.0 reaction rate formula with anti-stagnation"""
        
        # Base strength from DRM rule
        base_strength = (self.base_rule.weight * 
                        math.log(self.base_rule.usage_count + 1) * 
                        (1 + self.base_rule.observations / 100.0) * 
                        self.base_rule.post_mean)
        
        # FRZ_Adaptive - boost when stagnating, dampen when chaotic
        system_activity = context.get("system_activity", 0.5)
        if system_activity < 0.2:  # Stagnation detected
            frz_adaptive = 1.5 + (0.2 - system_activity) * 2.0  # BOOST
        elif system_activity > 0.8:  # Too chaotic
            frz_adaptive = 0.5
        else:
            frz_adaptive = 1.0
        
        # Emergence pressure
        diversity_score = context.get("diversity_score", 0.5)
        emergence_pressure = max(1.0, 2.0 - diversity_score * 2.0)  # More pressure when low diversity
        
        # Curiosity drive - hunger for new information
        novelty = 1.0 / (1.0 + self.reaction_count)
        information_density = state.information_density()
        curiosity_drive = 1.0 + 0.5 * novelty + 0.3 * math.tanh(information_density - 1.0)
        
        # Anti-stagnation factor
        stagnation_detected = context.get("stagnation_detected", False)
        if stagnation_detected:
            anti_stagnation = 2.0 + 0.5 * self.stagnation_resistance
        else:
            anti_stagnation = 1.0
        
        # Energy efficiency factor
        energy_efficiency = min(2.0, state.available_energy / max(self.energy_cost, EPS))
        
        # Complete DRM 3.0 formula
        rate = (base_strength * 
                frz_adaptive * 
                emergence_pressure * 
                curiosity_drive * 
                anti_stagnation * 
                energy_efficiency)
        
        # Normalize to [0,1] with sigmoid
        return 1.0 / (1.0 + math.exp(-math.tanh(rate)))
    
    def apply_reaction(self, state: InformationState, context: Dict[str, Any]) -> InformationState:
        """Apply reaction transformation: (S,I,E) + B → (S',I',E')"""
        
        start_time = time.time()
        new_state = state.copy()
        
        try:
            # Energy consumption
            new_state.available_energy = max(0.0, new_state.available_energy - self.energy_cost)
            new_state.dissipated_energy += self.energy_cost * 0.2  # Some energy always dissipates
            
            # Information transformation
            if self.information_transform:
                transformed_info = self.information_transform(state, context)
                if isinstance(transformed_info, dict):
                    new_state.structural.update(transformed_info.get("structural", {}))
                    new_state.latent_info = transformed_info.get("latent_info", new_state.latent_info)
                    new_state.information_entropy = transformed_info.get("entropy", new_state.information_entropy)
            else:
                # Default transformation based on rule type
                if self.base_rule.type == LOGICAL:
                    # LOGICAL rules create precise information
                    new_state.complexity_measure += 0.1
                    new_state.information_entropy *= 0.95  # Slight reduction in entropy
                
                elif self.base_rule.type == HYBRID:
                    # HYBRID rules transform information
                    if new_state.latent_info:
                        # Apply rule parameters as transformation
                        for param_name, param in self.base_rule.params.items():
                            if len(new_state.latent_info) > 0:
                                new_state.latent_info[0] += param.value * 0.1
                    
                    new_state.complexity_measure += 0.05
                
                else:  # HEURISTIC
                    # HEURISTIC rules add uncertainty but can find patterns
                    new_state.information_entropy += 0.02
                    if len(new_state.latent_info) < 10:
                        new_state.latent_info.append(self.base_rule.post_mean)
            
            # Update transformation tracking
            new_state.transformation_count += 1
            
            # Update reaction tracking
            self.reaction_count += 1
            self.total_energy_consumed += self.energy_cost
            
            reaction_time = time.time() - start_time
            self.average_reaction_time = ((self.average_reaction_time * (self.reaction_count - 1) + 
                                         reaction_time) / self.reaction_count)
            
            # Update rule posterior based on successful reaction
            energy_efficiency = (self.energy_cost / max(reaction_time, EPS))
            reward = min(1.0, energy_efficiency * new_state.information_density())
            self.base_rule.update_posterior(reward, context=context)
            
        except Exception as e:
            logger.warning(f"Reaction {self.base_rule.id} failed: {e}")
            # Return original state on failure
            return state
        
        return new_state
    
    def explain_reaction(self) -> Dict[str, Any]:
        """Explain reaction capabilities and performance"""
        return {
            "rule_id": self.base_rule.id,
            "rule_type": self.base_rule.type,
            "triggers": list(self.reaction_triggers),
            "energy_cost": self.energy_cost,
            "reaction_count": self.reaction_count,
            "total_energy_consumed": self.total_energy_consumed,
            "average_reaction_time": self.average_reaction_time,
            "efficiency": self.total_energy_consumed / max(self.reaction_count, 1),
            "stagnation_resistance": self.stagnation_resistance,
            "curiosity_level": self.curiosity_level
        }

# =============================================================================
# System Vitals Monitor - DRM 3.0 Anti-Stagnation
# =============================================================================

class SystemVitalsMonitor:
    """Monitor system health and detect stagnation patterns"""
    
    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        
        # Activity tracking
        self.activity_history = deque(maxlen=history_window)
        self.diversity_history = deque(maxlen=history_window)
        self.information_flow_history = deque(maxlen=history_window)
        self.energy_efficiency_history = deque(maxlen=history_window)
        
        # Stagnation detection
        self.stagnation_threshold = 0.1
        self.death_spiral_threshold = 0.05
        self.boredom_level = 0.0
        
        # Emergence tracking
        self.emergence_events = deque(maxlen=100)
        self.last_emergence_cycle = -1
    
    def observe_cycle(self, 
                     activity_level: float,
                     diversity_score: float, 
                     information_flow: float,
                     energy_efficiency: float,
                     cycle_number: int):
        """Record observations for this cycle"""
        
        self.activity_history.append(activity_level)
        self.diversity_history.append(diversity_score)
        self.information_flow_history.append(information_flow)
        self.energy_efficiency_history.append(energy_efficiency)
        
        # Update boredom level
        self.boredom_level = self._calculate_boredom()
    
    def _calculate_boredom(self) -> float:
        """Calculate system boredom level (inverse of engagement)"""
        if len(self.activity_history) < 5:
            return 0.0
        
        recent_activity = sum(list(self.activity_history)[-5:]) / 5
        recent_diversity = sum(list(self.diversity_history)[-5:]) / 5
        
        # Boredom increases with low activity and low diversity
        boredom = 1.0 - (recent_activity * recent_diversity)
        return max(0.0, min(1.0, boredom))
    
    def detect_stagnation(self) -> Tuple[bool, str, float]:
        """Detect various forms of stagnation"""
        
        if len(self.activity_history) < self.history_window:
            return False, "insufficient_data", 0.0
        
        # Check for low activity
        avg_activity = sum(self.activity_history) / len(self.activity_history)
        if avg_activity < self.stagnation_threshold:
            return True, "low_activity", avg_activity
        
        # Check for diversity collapse
        avg_diversity = sum(self.diversity_history) / len(self.diversity_history)
        if avg_diversity < 0.2:
            return True, "diversity_collapse", avg_diversity
        
        # Check for information flow stagnation
        avg_info_flow = sum(self.information_flow_history) / len(self.information_flow_history)
        if avg_info_flow < 0.1:
            return True, "information_stagnation", avg_info_flow
        
        # Check for death spiral
        if (avg_activity < self.death_spiral_threshold and 
            avg_diversity < self.death_spiral_threshold):
            return True, "death_spiral", avg_activity
        
        return False, "healthy", avg_activity
    
    def calculate_emergence_pressure(self, cycle_number: int) -> float:
        """Calculate emergence pressure based on system state"""
        
        base_pressure = 1.0
        
        # Time since last emergence
        cycles_since_emergence = cycle_number - self.last_emergence_cycle
        time_pressure = min(2.0, cycles_since_emergence / 20.0)
        
        # Stagnation pressure
        stagnation_pressure = self.boredom_level * 2.0
        
        # Diversity pressure
        current_diversity = self.diversity_history[-1] if self.diversity_history else 0.5
        diversity_pressure = max(0.0, 1.0 - current_diversity) * 1.5
        
        total_pressure = base_pressure + time_pressure + stagnation_pressure + diversity_pressure
        
        return min(5.0, total_pressure)  # Cap at 5x
    
    def record_emergence(self, cycle_number: int, event_info: Dict[str, Any]):
        """Record an emergence event"""
        self.emergence_events.append({
            "cycle": cycle_number,
            "timestamp": time.time(),
            "info": event_info
        })
        self.last_emergence_cycle = cycle_number

# =============================================================================
# Enhanced Information Reactor
# =============================================================================

class InformationReactor:
    """Enhanced Information Reactor with tight DRM integration"""
    
    def __init__(self, 
                 drm_system: Optional['DRMSystem'] = None,
                 energy_budget: float = 1.0,
                 conservation_tolerance: float = 1e-6):
        
        self.drm = drm_system
        self.energy_budget = energy_budget
        
        # Core components
        self.reactive_rules: Dict[str, ReactiveRule] = {}
        self.conservation_validator = ConservationValidator(tolerance=conservation_tolerance)
        self.vitals_monitor = SystemVitalsMonitor()
        
        # State management
        self.current_state = InformationState(available_energy=energy_budget)
        self.state_history = deque(maxlen=100)
        
        # Cycle management
        self.cycle_count = 0
        self.reaction_log = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Integration hooks
        if self.drm:
            self._setup_drm_integration()
    
    def _setup_drm_integration(self):
        """Setup tight integration with DRM system"""
        # Convert existing DRM rules to reactive rules
        for rule_id, rule in self.drm.rules.items():
            self.add_reactive_rule(rule)
        
        # Setup hooks in DRM system (if supported)
        if hasattr(self.drm, 'cycle_hooks'):
            self.drm.cycle_hooks.append(self._on_drm_cycle)
    
    def add_reactive_rule(self, 
                         rule: Rule,
                         reaction_triggers: Set[str] = None,
                         energy_cost: float = None,
                         information_transform: Optional[Callable] = None) -> str:
        """Add a rule as a reactive rule"""
        
        # Auto-determine energy cost based on rule complexity
        if energy_cost is None:
            base_cost = 0.05
            if rule.type == LOGICAL:
                energy_cost = base_cost * 0.5  # Logical rules are efficient
            elif rule.type == HYBRID:
                energy_cost = base_cost * 1.5  # Hybrid rules cost more
            else:
                energy_cost = base_cost
        
        # Auto-determine triggers from rule conditions
        if reaction_triggers is None:
            reaction_triggers = set(rule.pre_conditions)
            
            # Add default triggers based on rule type
            if rule.type == LOGICAL:
                reaction_triggers.add("logical_validation_needed")
            elif rule.type == HYBRID:
                reaction_triggers.add("adaptive_control_needed")
            else:
                reaction_triggers.add("heuristic_guidance_needed")
        
        reactive_rule = ReactiveRule(
            base_rule=rule,
            reaction_triggers=reaction_triggers,
            energy_cost=energy_cost,
            information_transform=information_transform,
            reaction_priority=getattr(rule, 'priority', 1)
        )
        
        self.reactive_rules[rule.id] = reactive_rule
        
        logger.info(f"Added reactive rule {rule.id} with {len(reaction_triggers)} triggers")
        return rule.id
    
    def catalytic_cycle(self, 
                       stimulus: Dict[str, Any],
                       max_reactions: int = 10,
                       energy_input: float = 0.0) -> InformationState:
        """Main reactor cycle with thermodynamic validation"""
        
        with self.lock:
            self.cycle_count += 1
            cycle_start_time = time.time()
            
            # Add energy input to current state
            if energy_input > 0:
                self.current_state.available_energy += energy_input
                self.current_state.energy_flow_rate = energy_input
            
            # Store initial state for conservation checking
            initial_state = self.current_state.copy()
            
            # Find applicable reactions
            applicable_reactions = self._find_applicable_reactions(stimulus)
            
            # Sort by reaction rate (DRM 3.0 formula)
            reaction_rates = []
            for reactive_rule in applicable_reactions:
                rate = reactive_rule.compute_reaction_rate(self.current_state, stimulus)
                reaction_rates.append((rate, reactive_rule))
            
            reaction_rates.sort(key=lambda x: x[0], reverse=True)
            
            # Execute reactions with conservation validation
            executed_reactions = []
            total_information_created = 0.0
            
            for rate, reactive_rule in reaction_rates[:max_reactions]:
                if not reactive_rule.can_react(self.current_state, stimulus, self.cycle_count):
                    continue
                
                # Apply reaction
                candidate_state = reactive_rule.apply_reaction(self.current_state, stimulus)
                
                # Validate conservation laws
                conservation_result = self.conservation_validator.validate_reaction(
                    self.current_state, candidate_state, energy_input
                )
                
                if conservation_result["valid"]:
                    # Accept the transformation
                    self.current_state = candidate_state
                    reactive_rule.last_reaction_cycle = self.cycle_count
                    
                    executed_reactions.append({
                        "rule_id": reactive_rule.base_rule.id,
                        "rate": rate,
                        "energy_cost": reactive_rule.energy_cost,
                        "conservation": conservation_result
                    })
                    
                    # Track information creation
                    info_change = conservation_result["metrics"].get("information_change", 0.0)
                    total_information_created += max(0.0, info_change)
                    
                    # Update DRM rule performance
                    reward = min(1.0, rate * conservation_result["metrics"].get("energy_efficiency", 1.0))
                    if self.drm:
                        reactive_rule.base_rule.update_posterior(reward, context=stimulus)
                
                else:
                    logger.warning(f"Conservation violation in rule {reactive_rule.base_rule.id}: {conservation_result['violations']}")
            
            # Calculate cycle metrics
            cycle_duration = time.time() - cycle_start_time
            activity_level = len(executed_reactions) / max(max_reactions, 1)
            
            # Update system vitals
            diversity_score = self._calculate_diversity_score()
            information_flow = total_information_created / max(cycle_duration, EPS)
            energy_efficiency = total_information_created / max(energy_input + 0.1, EPS)
            
            self.vitals_monitor.observe_cycle(
                activity_level, diversity_score, information_flow, energy_efficiency, self.cycle_count
            )
            
            # Check for stagnation and apply anti-stagnation measures
            stagnation_detected, stagnation_reason, stagnation_level = self.vitals_monitor.detect_stagnation()
            
            if stagnation_detected:
                self._apply_anti_stagnation_measures(stagnation_reason, stagnation_level)
            
            # Log the cycle
            cycle_result = {
                "cycle": self.cycle_count,
                "executed_reactions": len(executed_reactions),
                "activity_level": activity_level,
                "diversity_score": diversity_score,
                "information_flow": information_flow,
                "energy_efficiency": energy_efficiency,
                "stagnation_detected": stagnation_detected,
                "stagnation_reason": stagnation_reason,
                "total_information": self.current_state.total_information(),
                "available_energy": self.current_state.available_energy,
                "reactions": executed_reactions
            }
            
            self.reaction_log.append(cycle_result)
            
            # Store state in history
            self.state_history.append(self.current_state.copy())
            
            # Update stimulus with computed metrics for next iteration
            stimulus.update({
                "system_activity": activity_level,
                "diversity_score": diversity_score,
                "stagnation_detected": stagnation_detected,
                "emergence_pressure": self.vitals_monitor.calculate_emergence_pressure(self.cycle_count)
            })
            
            return self.current_state
    
    def _find_applicable_reactions(self, stimulus: Dict[str, Any]) -> List[ReactiveRule]:
        """Find reactive rules that can be triggered by stimulus"""
        applicable = []
        
        for reactive_rule in self.reactive_rules.values():
            # Check triggers
            if reactive_rule.reaction_triggers:
                if any(stimulus.get(trigger, False) for trigger in reactive_rule.reaction_triggers):
                    applicable.append(reactive_rule)
            else:
                # Background reactions (always considered)
                applicable.append(reactive_rule)
        
        return applicable
    
    def _calculate_diversity_score(self) -> float:
        """Calculate current system diversity"""
        if not self.reactive_rules:
            return 0.0
        
        # Simple diversity based on rule type distribution
        type_counts = defaultdict(int)
        for reactive_rule in self.reactive_rules.values():
            type_counts[reactive_rule.base_rule.type] += 1
        
        total_rules = len(self.reactive_rules)
        type_probs = [count / total_rules for count in type_counts.values()]
        
        # Shannon entropy of type distribution
        diversity = -sum(p * math.log(p + EPS) for p in type_probs)
        return diversity / math.log(len(type_counts) + EPS)
    
    def _apply_anti_stagnation_measures(self, stagnation_reason: str, stagnation_level: float):
        """Apply DRM 3.0 anti-stagnation measures"""
        
        logger.info(f"Applying anti-stagnation measures: {stagnation_reason} (level: {stagnation_level:.3f})")
        
        if stagnation_reason == "death_spiral":
            # Emergency measures for death spiral
            self._emergency_system_revival()
        
        elif stagnation_reason == "low_activity":
            # Boost weak rules and inject energy
            self._boost_weak_rules(factor=1.5 + stagnation_level)
            self.current_state.available_energy += self.energy_budget * 0.5
        
        elif stagnation_reason == "diversity_collapse":
            # Force diversity through rule mutations
            self._force_diversity_mutations()
        
        elif stagnation_reason == "information_stagnation":
            # Inject new information sources
            self._inject_information_sources()
        
        # Update all reactive rules' stagnation resistance
        for reactive_rule in self.reactive_rules.values():
            reactive_rule.stagnation_resistance *= (1.0 + stagnation_level * 0.5)
    
    def _emergency_system_revival(self):
        """Emergency revival for death spiral"""
        logger.warning("EMERGENCY SYSTEM REVIVAL ACTIVATED")
        
        # Massive energy injection
        self.current_state.available_energy = self.energy_budget * 2.0
        
        # Reset all quarantined rules
        if self.drm:
            for rule in self.drm.rules.values():
                if rule.quarantined:
                    rule.quarantined = False
                    rule.quarantine_reason = None
                    rule.weight *= 2.0  # Boost weight
        
        # Force random mutations on all rules
        for reactive_rule in self.reactive_rules.values():
            reactive_rule.base_rule.mutate("tweak_param", magnitude=0.5)
            reactive_rule.curiosity_level = 2.0
        
        # Generate emergency rules if generators available
        if self.drm and hasattr(self.drm, 'generators'):
            for gen in self.drm.generators:
                for i in range(3):
                    emergency_rule = gen.generate(prefix="emergency", idx=i)
                    emergency_rule.priority = 10  # High priority
                    self.drm.add_rule(emergency_rule)
                    self.add_reactive_rule(emergency_rule)
    
    def _boost_weak_rules(self, factor: float = 1.5):
        """Boost performance of weak rules"""
        weak_rules = [
            rr for rr in self.reactive_rules.values()
            if rr.base_rule.post_mean < 0.3 and not rr.base_rule.quarantined
        ]
        
        for reactive_rule in weak_rules:
            reactive_rule.base_rule.weight *= factor
            reactive_rule.stagnation_resistance *= factor
            reactive_rule.curiosity_level = min(3.0, reactive_rule.curiosity_level * 1.2)
        
        logger.info(f"Boosted {len(weak_rules)} weak rules by factor {factor:.2f}")
    
    def _force_diversity_mutations(self):
        """Force mutations to increase diversity"""
        mutation_count = 0
        
        for reactive_rule in self.reactive_rules.values():
            if reactive_rule.base_rule.type != LOGICAL:  # Don't mutate logical rules
                if random.random() < 0.3:  # 30% chance to mutate
                    if reactive_rule.base_rule.mutate("tweak_param", magnitude=0.3):
                        mutation_count += 1
        
        logger.info(f"Applied {mutation_count} diversity mutations")
    
    def _inject_information_sources(self):
        """Inject new information to break stagnation"""
        # Add random information to latent space
        if len(self.current_state.latent_info) < 20:
            for _ in range(3):
                self.current_state.latent_info.append(random.gauss(0, 1))
        
        # Increase information entropy
        self.current_state.information_entropy += 0.1
        
        # Add new context signals
        new_signals = {
            f"injected_signal_{i}": random.random() > 0.5
            for i in range(3)
        }
        
        self.current_state.structural.update(new_signals)
        
        logger.info("Injected new information sources")
    
    def _on_drm_cycle(self, drm_result: Dict[str, Any]):
        """Hook called when DRM completes a cycle"""
        # Synchronize state with DRM
        if "entropy" in drm_result:
            self.current_state.information_entropy = drm_result["entropy"]
        
        # Check for DRM-detected stagnation
        if drm_result.get("stagnation", {}).get("detected", False):
            self.vitals_monitor.record_emergence(self.cycle_count, {
                "type": "drm_stagnation",
                "reason": drm_result["stagnation"]["reason"]
            })
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics"""
        stagnation_detected, stagnation_reason, stagnation_level = self.vitals_monitor.detect_stagnation()
        
        return {
            "cycle_count": self.cycle_count,
            "current_state": {
                "total_information": self.current_state.total_information(),
                "available_energy": self.current_state.available_energy,
                "information_density": self.current_state.information_density(),
                "complexity": self.current_state.complexity_measure,
                "transformation_count": self.current_state.transformation_count
            },
            "system_health": {
                "stagnation_detected": stagnation_detected,
                "stagnation_reason": stagnation_reason,
                "stagnation_level": stagnation_level,
                "boredom_level": self.vitals_monitor.boredom_level,
                "diversity_score": self._calculate_diversity_score(),
                "emergence_pressure": self.vitals_monitor.calculate_emergence_pressure(self.cycle_count)
            },
            "reactive_rules": {
                "total": len(self.reactive_rules),
                "active": len([rr for rr in self.reactive_rules.values() if not rr.base_rule.quarantined]),
                "average_energy_cost": sum(rr.energy_cost for rr in self.reactive_rules.values()) / max(len(self.reactive_rules), 1),
                "total_reactions": sum(rr.reaction_count for rr in self.reactive_rules.values())
            },
            "conservation": {
                "violations_detected": len(self.conservation_validator.violations),
                "recent_violations": list(self.conservation_validator.violations)[-5:] if self.conservation_validator.violations else []
            },
            "performance_metrics": {
                "reactions_per_cycle": len(self.reaction_log[-10:]) / 10 if len(self.reaction_log) >= 10 else 0,
                "average_information_flow": sum(log.get("information_flow", 0) for log in list(self.reaction_log)[-10:]) / 10 if self.reaction_log else 0,
                "energy_utilization": 1.0 - (self.current_state.available_energy / self.energy_budget) if self.energy_budget > 0 else 0
            }
        }
    
    def inject_stimulus(self, stimulus_type: str, stimulus_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject external stimulus and observe reaction"""
        
        # Prepare stimulus context
        stimulus_context = {
            "stimulus_type": stimulus_type,
            "timestamp": time.time(),
            **stimulus_data
        }
        
        # Add energy based on stimulus strength
        stimulus_energy = stimulus_data.get("energy", 0.1)
        
        # Run catalytic cycle
        previous_state = self.current_state.copy()
        new_state = self.catalytic_cycle(stimulus_context, energy_input=stimulus_energy)
        
        # Analyze reaction
        reaction_analysis = {
            "stimulus_type": stimulus_type,
            "state_change": {
                "information_delta": new_state.total_information() - previous_state.total_information(),
                "energy_delta": new_state.available_energy - previous_state.available_energy,
                "complexity_delta": new_state.complexity_measure - previous_state.complexity_measure
            },
            "emergence_detected": abs(new_state.total_information() - previous_state.total_information()) > 0.1,
            "system_response": "adaptive" if new_state.complexity_measure > previous_state.complexity_measure else "conservative"
        }
        
        # Record emergence if significant
        if reaction_analysis["emergence_detected"]:
            self.vitals_monitor.record_emergence(self.cycle_count, {
                "stimulus": stimulus_type,
                "information_delta": reaction_analysis["state_change"]["information_delta"]
            })
        
        return reaction_analysis
    
    def evolve_rules(self, evolution_pressure: float = 1.0):
        """Evolve reactive rules based on performance"""
        
        # Sort rules by performance
        rule_performance = []
        for rule_id, reactive_rule in self.reactive_rules.items():
            performance_score = (
                reactive_rule.base_rule.post_mean * 0.4 +
                reactive_rule.base_rule.get_success_rate() * 0.3 +
                (1.0 / (1.0 + reactive_rule.reaction_count)) * 0.2 +  # Novelty
                reactive_rule.stagnation_resistance * 0.1
            )
            rule_performance.append((performance_score, rule_id, reactive_rule))
        
        rule_performance.sort(key=lambda x: x[0], reverse=True)
        
        # Remove worst performers if too many rules
        if len(self.reactive_rules) > 20:
            worst_performers = rule_performance[-5:]
            for _, rule_id, reactive_rule in worst_performers:
                if reactive_rule.base_rule.type != LOGICAL:  # Protect logical rules
                    del self.reactive_rules[rule_id]
                    if self.drm:
                        self.drm.remove_rule(rule_id, reason="evolution_selection")
        
        # Boost top performers
        top_performers = rule_performance[:5]
        for _, rule_id, reactive_rule in top_performers:
            reactive_rule.base_rule.weight *= (1.0 + evolution_pressure * 0.1)
        
        logger.info(f"Evolved rules: boosted {len(top_performers)} top performers")
    
    def get_reaction_network_graph(self) -> Dict[str, Any]:
        """Get network representation of reactions for visualization"""
        nodes = []
        edges = []
        
        for rule_id, reactive_rule in self.reactive_rules.items():
            nodes.append({
                "id": rule_id,
                "type": reactive_rule.base_rule.type,
                "performance": reactive_rule.base_rule.post_mean,
                "energy_cost": reactive_rule.energy_cost,
                "reaction_count": reactive_rule.reaction_count,
                "quarantined": reactive_rule.base_rule.quarantined
            })
            
            # Create edges based on triggers and post-conditions
            for trigger in reactive_rule.reaction_triggers:
                for other_id, other_reactive_rule in self.reactive_rules.items():
                    if other_id != rule_id:
                        if trigger in other_reactive_rule.base_rule.post_conditions:
                            edges.append({
                                "source": other_id,
                                "target": rule_id,
                                "type": "trigger",
                                "weight": reactive_rule.base_rule.post_mean
                            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "connectivity": len(edges) / max(len(nodes), 1)
            }
        }
    
    def save_state(self, filepath: str):
        """Save complete reactor state"""
        state_data = {
            "current_state": {
                "structural": self.current_state.structural,
                "latent_info": self.current_state.latent_info,
                "information_entropy": self.current_state.information_entropy,
                "available_energy": self.current_state.available_energy,
                "energy_flow_rate": self.current_state.energy_flow_rate,
                "complexity_measure": self.current_state.complexity_measure,
                "transformation_count": self.current_state.transformation_count
            },
            "reactive_rules": {
                rule_id: {
                    "base_rule": reactive_rule.base_rule.to_dict(),
                    "reaction_triggers": list(reactive_rule.reaction_triggers),
                    "energy_cost": reactive_rule.energy_cost,
                    "reaction_count": reactive_rule.reaction_count,
                    "stagnation_resistance": reactive_rule.stagnation_resistance,
                    "curiosity_level": reactive_rule.curiosity_level
                }
                for rule_id, reactive_rule in self.reactive_rules.items()
            },
            "cycle_count": self.cycle_count,
            "diagnostics": self.get_system_diagnostics()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"Saved reactor state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load complete reactor state"""
        import json
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Restore current state
        state_dict = state_data["current_state"]
        self.current_state = InformationState(
            structural=state_dict["structural"],
            latent_info=state_dict["latent_info"],
            information_entropy=state_dict["information_entropy"],
            available_energy=state_dict["available_energy"],
            energy_flow_rate=state_dict["energy_flow_rate"],
            complexity_measure=state_dict["complexity_measure"],
            transformation_count=state_dict["transformation_count"]
        )
        
        # Restore reactive rules
        self.reactive_rules.clear()
        for rule_id, rule_data in state_data["reactive_rules"].items():
            from drm_module5_improved import Rule
            base_rule = Rule.from_dict(rule_data["base_rule"])
            
            reactive_rule = ReactiveRule(
                base_rule=base_rule,
                reaction_triggers=set(rule_data["reaction_triggers"]),
                energy_cost=rule_data["energy_cost"],
                reaction_priority=base_rule.priority
            )
            
            reactive_rule.reaction_count = rule_data["reaction_count"]
            reactive_rule.stagnation_resistance = rule_data["stagnation_resistance"]
            reactive_rule.curiosity_level = rule_data["curiosity_level"]
            
            self.reactive_rules[rule_id] = reactive_rule
        
        self.cycle_count = state_data["cycle_count"]
        
        logger.info(f"Loaded reactor state from {filepath}")

# =============================================================================
# DRM-Reactor Integration Bridge
# =============================================================================

class DRMReactorBridge:
    """Tight integration bridge between DRM and Reactor systems"""
    
    def __init__(self, drm_system: 'DRMSystem', reactor: InformationReactor):
        self.drm = drm_system
        self.reactor = reactor
        self.integration_active = True
        
        # Setup bidirectional hooks
        self._setup_integration_hooks()
    
    def _setup_integration_hooks(self):
        """Setup bidirectional communication"""
        
        # Hook into DRM cycle to update reactor state
        original_run_cycle = self.drm.run_cycle
        
        def enhanced_run_cycle(*args, **kwargs):
            # Run original DRM cycle
            result = original_run_cycle(*args, **kwargs)
            
            if self.integration_active:
                # Update reactor state based on DRM results
                self._sync_drm_to_reactor(result, kwargs.get("context", {}))
            
            return result
        
        self.drm.run_cycle = enhanced_run_cycle
    
    def _sync_drm_to_reactor(self, drm_result: Dict[str, Any], context: Dict[str, Any]):
        """Synchronize DRM state to reactor"""
        
        # Update reactor's information state
        if "entropy" in drm_result:
            self.reactor.current_state.information_entropy = drm_result["entropy"]
        
        if "global_score" in drm_result:
            # Use global score to update available energy
            performance_ratio = drm_result["global_score"]
            energy_bonus = performance_ratio * 0.2
            self.reactor.current_state.available_energy += energy_bonus
        
        # Add any new rules from DRM to reactor
        for rule_id, rule in self.drm.rules.items():
            if rule_id not in self.reactor.reactive_rules:
                self.reactor.add_reactive_rule(rule)
        
        # Remove rules that were removed from DRM
        to_remove = []
        for rule_id in self.reactor.reactive_rules:
            if rule_id not in self.drm.rules:
                to_remove.append(rule_id)
        
        for rule_id in to_remove:
            del self.reactor.reactive_rules[rule_id]
    
    def unified_cycle(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Run unified DRM + Reactor cycle"""
        
        # 1. Reactor catalytic cycle
        reactor_result = self.reactor.catalytic_cycle(stimulus)
        
        # 2. Update DRM context with reactor state
        enhanced_context = dict(stimulus)
        enhanced_context.update({
            "reactor_state": reactor_result.structural,
            "information_density": reactor_result.information_density(),
            "available_energy": reactor_result.available_energy,
            "system_activity": len(self.reactor.reaction_log[-1]["reactions"]) / 10 if self.reactor.reaction_log else 0
        })
        
        # 3. Run DRM cycle with enhanced context
        def reactor_aware_evaluator(rule: Rule) -> Optional[float]:
            # Use reactor state to evaluate rules
            if rule.id in self.reactor.reactive_rules:
                reactive_rule = self.reactor.reactive_rules[rule.id]
                return reactive_rule.compute_reaction_rate(reactor_result, enhanced_context)
            else:
                # Fallback evaluation
                return rule.sample_posterior()
        
        drm_result = self.drm.run_cycle(reactor_aware_evaluator, context=enhanced_context)
        
        # 4. Combine results
        unified_result = {
            "reactor": {
                "state": reactor_result,
                "diagnostics": self.reactor.get_system_diagnostics()
            },
            "drm": drm_result,
            "integration": {
                "total_information": reactor_result.total_information(),
                "information_flow": drm_result.get("entropy", 0),
                "energy_efficiency": reactor_result.information_density(),
                "system_coherence": self._calculate_system_coherence()
            }
        }
        
        return unified_result
    
    def _calculate_system_coherence(self) -> float:
        """Calculate coherence between DRM and Reactor states"""
        
        # Compare DRM distribution entropy with reactor information entropy
        drm_dist = self.drm.get_distribution()
        drm_entropy = entropy_from_dist(drm_dist)
        reactor_entropy = self.reactor.current_state.information_entropy
        
        # Coherence is higher when entropies are aligned
        entropy_alignment = 1.0 - abs(drm_entropy - reactor_entropy) / (drm_entropy + reactor_entropy + EPS)
        
        # Check rule activation alignment
        active_drm_rules = len(self.drm.get_active_rules())
        active_reactor_rules = len([rr for rr in self.reactor.reactive_rules.values() if not rr.base_rule.quarantined])
        
        rule_alignment = min(active_drm_rules, active_reactor_rules) / max(active_drm_rules, active_reactor_rules, 1)
        
        # Overall coherence
        coherence = (entropy_alignment * 0.6 + rule_alignment * 0.4)
        
        return coherence

# =============================================================================
# Factory Functions for Easy Setup
# =============================================================================

def create_physics_informed_reactor(domain: str = "fluid_dynamics") -> Tuple[DRMSystem, InformationReactor]:
    """Create reactor setup for physics-informed learning"""
    
    if not HAS_DRM:
        raise ImportError("drm_module5_improved not available")
    
    # Create DRM system
    drm = DRMSystem()
    
    # Create reactor
    reactor = InformationReactor(drm_system=drm, energy_budget=2.0)
    
    # Add domain-specific logical rules
    if domain == "fluid_dynamics":
        # Conservation of mass
        mass_conservation = Rule(
            id="mass_conservation",
            name="Mass Conservation Law",
            rtype=LOGICAL,
            priority=10,
            pre_conditions=["velocity_field", "density_field"],
            post_conditions=["mass_conserved"],
            tests=[lambda rule, ctx: abs(ctx.get("mass_change", 1.0)) < 1e-6],
            provenance={"domain": "fluid_dynamics", "law": "conservation"}
        )
        
        # Navier-Stokes momentum
        momentum_rule = Rule(
            id="momentum_conservation", 
            name="Momentum Conservation",
            rtype=LOGICAL,
            priority=9,
            pre_conditions=["velocity_field", "pressure_field"],
            post_conditions=["momentum_conserved"]
        )
        
        drm.add_rule(mass_conservation)
        drm.add_rule(momentum_rule)
        
        # Add reactive versions
        reactor.add_reactive_rule(
            mass_conservation,
            reaction_triggers={"mass_imbalance_detected", "density_change"},
            energy_cost=0.05
        )
        
        reactor.add_reactive_rule(
            momentum_rule,
            reaction_triggers={"velocity_anomaly", "pressure_gradient_change"},
            energy_cost=0.08
        )
    
    return drm, reactor

def create_adaptive_control_reactor() -> Tuple[DRMSystem, InformationReactor]:
    """Create reactor for adaptive control systems"""
    
    if not HAS_DRM:
        raise ImportError("drm_module5_improved not available")
    
    drm = DRMSystem()
    reactor = InformationReactor(drm_system=drm, energy_budget=1.5)
    
    # Add control-specific HYBRID rules
    pid_control = Rule(
        id="adaptive_pid",
        name="Adaptive PID Controller",
        rtype=HYBRID,
        priority=5,
        params={
            "kp": RuleParameter(value=1.0, min_val=0.0, max_val=10.0, requires_grad=True),
            "ki": RuleParameter(value=0.1, min_val=0.0, max_val=5.0, requires_grad=True),
            "kd": RuleParameter(value=0.01, min_val=0.0, max_val=2.0, requires_grad=True)
        },
        pre_conditions=["error_signal", "setpoint"],
        post_conditions=["control_output"]
    )
    
    drm.add_rule(pid_control)
    
    # Create reactive version with control-specific transformations
    def control_transform(state: InformationState, context: Dict[str, Any]) -> Dict[str, Any]:
        error = context.get("error_signal", 0.0)
        
        # Update control state
        new_structural = dict(state.structural)
        new_structural["control_error"] = error
        new_structural["control_active"] = abs(error) > 0.01
        
        return {
            "structural": new_structural,
            "latent_info": state.latent_info + [error * 0.1],
            "entropy": state.information_entropy + abs(error) * 0.01
        }
    
    reactor.add_reactive_rule(
        pid_control,
        reaction_triggers={"error_change", "setpoint_change"},
        energy_cost=0.1,
        information_transform=control_transform
    )
    
    return drm, reactor

# =============================================================================
# Demo and Testing
# =============================================================================

def demo_improved_reactor():
    """Comprehensive demo of improved reactor system"""
    
    print("=== Enhanced Information Reactor Demo ===")
    
    # Create physics-informed reactor
    drm, reactor = create_physics_informed_reactor("fluid_dynamics")
    
    # Create integration bridge
    bridge = DRMReactorBridge(drm, reactor)
    
    # Initial system state
    initial_state = InformationState(
        structural={"domain": "fluid_dynamics", "simulation_active": True},
        latent_info=[0.5, 0.3, 0.8],
        information_entropy=0.6,
        available_energy=2.0
    )
    
    reactor.current_state = initial_state
    
    print(f"Initial system state: {reactor.current_state.total_information():.3f} total info")
    
    # Run simulation cycles
    print("\n=== Running Simulation Cycles ===")
    
    for cycle in range(8):
        # Create stimulus based on physics simulation
        stimulus = {
            "velocity_field": True,
            "density_field": True,
            "mass_change": random.uniform(-1e-4, 1e-4),  # Small mass changes
            "velocity_anomaly": random.random() > 0.7,
            "pressure_gradient_change": random.random() > 0.8,
            "simulation_time": cycle * 0.1,
            "energy": 0.2
        }
        
        # Run unified cycle
        result = bridge.unified_cycle(stimulus)
        
        print(f"\nCycle {cycle + 1}:")
        print(f"  Reactor reactions: {len(result['reactor']['diagnostics']['performance_metrics'])}")
        print(f"  Information flow: {result['integration']['information_flow']:.3f}")
        print(f"  System coherence: {result['integration']['system_coherence']:.3f}")
        print(f"  Available energy: {result['reactor']['state'].available_energy:.3f}")
        
        # Check for stagnation
        health = result['reactor']['diagnostics']['system_health']
        if health['stagnation_detected']:
            print(f"  🚨 STAGNATION: {health['stagnation_reason']} (level: {health['stagnation_level']:.3f})")
        
        # Show top reactive rules
        diagnostics = result['reactor']['diagnostics']
        if reactor.reactive_rules:
            best_rule = max(reactor.reactive_rules.values(), 
                           key=lambda rr: rr.base_rule.post_mean)
            print(f"  Best rule: {best_rule.base_rule.id} (perf: {best_rule.base_rule.post_mean:.3f})")
    
    # Final diagnostics
    print("\n=== Final System Diagnostics ===")
    final_diagnostics = reactor.get_system_diagnostics()
    
    print(f"Total cycles: {final_diagnostics['cycle_count']}")
    print(f"Total information: {final_diagnostics['current_state']['total_information']:.3f}")
    print(f"Information density: {final_diagnostics['current_state']['information_density']:.3f}")
    print(f"System diversity: {final_diagnostics['system_health']['diversity_score']:.3f}")
    print(f"Conservation violations: {final_diagnostics['conservation']['violations_detected']}")
    
    # Test serialization
    print("\n=== Testing Serialization ===")
    reactor.save_state("reactor_test_state.json")
    
    # Create new reactor and load state
    new_reactor = InformationReactor(energy_budget=2.0)
    new_reactor.load_state("reactor_test_state.json")
    
    print(f"Loaded reactor has {len(new_reactor.reactive_rules)} reactive rules")
    
    print("\n=== Demo Complete ===")
    print("Enhanced Information Reactor with DRM 3.0 anti-stagnation is ready!")

if __name__ == "__main__":
    import random
    random.seed(42)
    demo_improved_reactor()