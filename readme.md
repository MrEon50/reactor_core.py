# reactor_core — Information Reactor (concept)

The `reactor_core.py` module implements a secure, budget-friendly Information Reactor for integration with `drm_module5_improved.py`.
The goal is to control the triggering of information reactions, manage execution costs, and support stagnation mechanisms.

## Key Features

* Reactions (`Reaction`) with triggers, cost, priority, and cooldown.
* Energy Accountant (`EnergyAccountant`) with per-cycle budget and entropy cost.
* Reaction Registry (`ReactionRegistry`) and contextual selection.
* Scheduler (`ReactionScheduler`) with chain depth control and activation limit.
* Reactor Core (`InformationReactor`) with hooks: `on_stagnation` and `on_high_throughput`.
* Security: action timeouts, per-reaction cooldown, rule quarantine, per-cycle budget.

## Integration with `drm_module5_improved.py`

* The architecture allows for the import and registration of the reactor in `DRMSystem`.
* `DRMSystem.register_reactor(reactor)` sets the binding and enables calls from `run_cycle(...)`.
* `Reaction.execute` uses `drm.replay.add`, `rule.update_posterior`, and `drm.audit` for consistent feedback.
* `reactor.schedule_and_run(context)` accepts the same `context` that DRM passes after evaluation.

> A quick reminder: `drm_module5_improved.py` brings innovations like `FRZ_Adaptive`, `StagnationDetector`, `EmergencyRevival`, `DiversityEnforcer`, latent `RuleGenerator`, advanced semantic validation, and KL-constrained weight updates. The reactor uses these mechanisms as fuel and safety.

## Quick start (example)

```python
from drm_module5_improved import DRMSystem
from reactor_core import InformationReactor, Reaction

ds = DRMSystem()
reactor = InformationReactor(drm=ds, budget_per_cycle=1.0)
ds.register_reactor(reactor)

# reaction: on stagnation -> emergency revival
r = Reaction( 
id="r_revival", 
name="revival_on_stagnation", 
triggers={"stagnation_detected"}, 
action=lambda drm, ctx: (drm.emergency_revival(num_new=3) or {"revival": True}), 
priority=5, 
cost_val=0.2, 
probabilistic=False
)
reactor.register_reaction(r)

# In DRM cycle after evaluation:
# ctx contains "stagnation_detected": True/False
# DRM will call reactor.schedule_and_run(ctx) and save the report in result["reactor_report"]
```

## API (shortcut)

* `Reaction(id, name, triggers, action, action_rule_id, priority, cost_val, ...)`

* `rate(drm, context)` -> float (activation probability)
* `execute(drm, context)` -> dict (safe execution with timeout)
* `EnergyAccountant(budget_per_cycle)`

* `can_afford(cost)`, `consume(cost)`, `reset_cycle()`
* `ReactionRegistry()`

* `register(reaction)`, `find_applicable(context, drm)`
* `ReactionScheduler(drm, registry, accountant, limiter)`

* `schedule_and_run(context, current_cycle)`
* `InformationReactor(drm, budget_per_cycle, max_activations, max_chain_depth)`

* `register_reaction(reaction)`, `inject(...)`, `schedule_and_run(context)`, `on_stagnation`, `on_high_throughput`

## Default parameters and safety limits

* `budget_per_cycle` defaults to `1.0`.
* `max_activations_per_cycle` defaults to `12`.
* `max_chain_depth` defaults to `4`.
* `per-reaction cooldown` defaults to `3` cycles.
* `timeout` for actions defaults to `~2s`.
Adjust these values ​​to your resources and production requirements.

## Practical Applications

* Automatic system revival upon detection of stagnation.
* Controlling the rule generator and managing exploration.
* Controlled orchestration of actions in the decision pipeline.
* Simulations of information chain reactions with cost control.
* Moderation and quarantine of suspicious/regulatory rules.

## Testing and Validation

* Run simulations: `death_spiral`, `throughput_stress`, `budget_limit` scenarios.
* Minimal unit test: register the reaction, trigger `stagnation_detected`, verify that `emergency_revival` adds rules.
* Recommended: Integration tests with real rule generators and a replay buffer.

## Implementation Notes

* The reactor is designed as a module (in the `reactor_core.py` file) for import.
* Alternatively, you can deploy the reactor as a separate process and communicate via RPC/HTTP. Direct integration has lower latency and immediate access to the replay/generator.
* Monitor and log metrics: budget_used, executions, throughput, avg_cost.

## License


* Default: None; include the appropriate project license (MIT/BSD/Apache) before commercial use.
