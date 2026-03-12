# Maxwell's Demon Simulation Toolkit

A complete computational framework for designing, testing, and optimizing Maxwell's Demon implementations before building the physical model.

## Overview

This toolkit provides:

- **Realistic 2D physics engine** - Velocity Verlet integration, elastic collisions, energy/momentum tracking
- **Thermodynamic analysis** - Temperature, entropy, energy distributions, particle separation metrics
- **Multiple demon strategies** - From simple velocity thresholding to measurement-aware control
- **Real-time visualization & post-processing** - Track separation, gate activity, efficiency
- **Parameter optimization** - Search parameter space to find best demon configuration

## Why This Approach?

Your physical model can't perfectly replicate thermodynamics (vibrating table ≠ true gas). This simulation serves as:
1. **Ground truth** for what *should* happen thermodynamically
2. **Validation platform** for mechanical designs before building
3. **Optimization tool** to maximize separation efficiency
4. **Documentation** of expected performance

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install numpy matplotlib scipy
```

### 2. Run the Notebook

```python
# In Jupyter, open MAXWELL_DEMON_NOTEBOOK.py
# Or convert to .ipynb format first:
# jupyter nbconvert --to notebook MAXWELL_DEMON_NOTEBOOK.py

# Then run cells in order
```

Or use the quick reference:

```python
from maxwell_demon_physics_core import SimulationEngine
from maxwell_demon_control import VelocityThresholdDemon
from maxwell_demon_thermo_analysis import ThermodynamicAnalyzer

# Create simulation
sim = SimulationEngine(box_size=10.0, num_particles=250, temperature=1.0)

# Equilibrate
sim.run(num_steps=2000, record_interval=100)

# Create demon
demon = VelocityThresholdDemon(sim, velocity_threshold=0.5)

# Run with demon for 5000 steps
for i in range(5000):
    demon.step()
    sim.step()
    sim._record_state()

# Analyze
analyzer = ThermodynamicAnalyzer(sim)
sep = analyzer.separation_by_temperature()
print(f"Temperature separation: ΔT = {sep['delta_T']:.4f}")
```

## Module Reference

### 1. **maxwell_demon_physics_core.py**

Core physics engine with particle dynamics.

**Main class:** `SimulationEngine`

**Key methods:**
- `run(num_steps, record_interval)` - Execute simulation
- `compute_temperature()` - Current temperature
- `compute_total_energy()` - Total kinetic energy
- `get_particles_in_region(x_min, x_max, y_min, y_max)` - Query particles
- `get_particles_approaching_from_side(side, x_threshold)` - For demon logic

**Key properties:**
- `particles` - List of Particle objects
- `history` - Time-series data (energy, temperature, momentum)

**Tuning parameters:**
- `dt` - Time step (smaller = more stable, slower). Default 0.001. Try 0.0005 if unstable.
- `restitution` - Collision elasticity (1.0 = perfect). Default 0.99.
- `gravity` - Gravitational acceleration. Keep at 0 for vibrating table analog.
- `temperature` - Initial kinetic energy. Controls initial velocity distribution.

---

### 2. **maxwell_demon_thermo_analysis.py**

Compute thermodynamic properties and separation metrics.

**Main class:** `ThermodynamicAnalyzer`

**Key methods for SEPARATION (most important):**
- `separation_by_temperature(left_region, right_region)` - Classic demon effect
- `separation_by_density(left_region, right_region)` - Number density difference
- `separation_by_mass(left_region, right_region)` - Mass separation (for multi-species)

**Key methods for DISTRIBUTIONS:**
- `speed_distribution(bins)` - Histogram of speeds (compare to Maxwell-Boltzmann)
- `velocity_distribution_2d(bins)` - 2D velocity space
- `kinetic_energy_distribution(bins)` - Energy histogram

**Key methods for DIAGNOSTICS:**
- `temperature()` - Overall temperature
- `temperature_regional(x_min, x_max, y_min, y_max)` - Regional temperature
- `entropy_from_speed_distribution(bins)` - Shannon entropy
- `energy_conservation_error(initial_energy)` - Check simulation stability
- `collision_rate()` - Interactions per particle per time

---

### 3. **maxwell_demon_control.py**

Demon strategies for selective particle admission.

**Available demons:**

1. **VelocityThresholdDemon** (simplest, start here)
   - Opens gate if particle speed > threshold
   - Best for quick exploration
   
2. **HysteresisDemon** (recommended for stability)
   - Hysteresis prevents gate jitter
   - Separate thresholds for opening/closing
   
3. **MeasurementDemon** (advanced)
   - Measures kinetic energy
   - Tracks measurement cost
   - Accounts for Szilard's paradox
   
4. **DensityDemon** (alternative physics)
   - Opens gate based on density gradients
   - Useful for pressure-based separation

**Common interface:**
```python
demon = SpecificDemon(sim, ...)
for step in range(N_STEPS):
    gate_open, particles_passed = demon.step()
    sim.step()
    
# Query results
demon.history['gate_open']
demon.history['particles_passed']
demon.gate_open_count / demon.step_count  # Duty cycle
```

---

## Workflow for Optimization

### Phase 1: Understand Your Baseline
```python
# Create and equilibrate (no demon)
sim = SimulationEngine(box_size=10.0, num_particles=250)
sim.run(num_steps=2000, record_interval=100)
analyzer = ThermodynamicAnalyzer(sim)
print(f"Baseline temperature: {analyzer.temperature():.4f}")
```

### Phase 2: Test Single Strategy
```python
# Run with one demon type
demon = VelocityThresholdDemon(sim, velocity_threshold=0.5)
for i in range(5000):
    demon.step()
    sim.step()
    sim._record_state()

# Measure separation
sep = analyzer.separation_by_temperature()
print(f"ΔT achieved: {sep['delta_T']:.4f}")
print(f"Gate duty cycle: {demon.gate_open_count/demon.step_count:.1%}")
```

### Phase 3: Scan Parameter Space
```python
import numpy as np

best_delta_T = 0
best_params = {}

for threshold in np.linspace(0.2, 1.0, 10):
    for gate_width in [0.5, 1.0, 1.5]:
        sim = SimulationEngine(num_particles=200)
        sim.run(num_steps=1000, record_interval=50)
        
        demon = VelocityThresholdDemon(
            sim,
            velocity_threshold=threshold,
            gate_width=gate_width
        )
        
        for i in range(3000):
            demon.step()
            sim.step()
            sim._record_state()
        
        sep = ThermodynamicAnalyzer(sim).separation_by_temperature()
        
        if sep['delta_T'] > best_delta_T:
            best_delta_T = sep['delta_T']
            best_params = {'threshold': threshold, 'gate_width': gate_width}

print(f"Best: {best_params} with ΔT = {best_delta_T:.4f}")
```

### Phase 4: Characterize Performance
```python
# With optimized parameters, run long simulation
sim = SimulationEngine(num_particles=300)
sim.run(num_steps=2000, record_interval=100)  # Equilibrate

demon = VelocityThresholdDemon(sim, **best_params)

# Run and record detailed data
for i in range(10000):
    demon.step()
    sim.step()
    if (i+1) % 50 == 0:
        sim._record_state()

# Analyze
analyzer = ThermodynamicAnalyzer(sim)

# Time series
delta_T_vs_time = []
for t in sim.history['time']:
    # Extract regional temperatures at each time
    pass

# Steady-state metrics
sep_final = analyzer.separation_by_temperature()
energy_error = analyzer.energy_conservation_error(initial_energy)
collision_rate = analyzer.collision_rate()

print(f"Separation rate: {sep_final['delta_T'] / sim.time:.4f} ΔT/time")
print(f"Energy conservation: {energy_error*100:.3f}%")
print(f"Collision frequency: {collision_rate:.4f} per particle per time")
```

## Physical Implementation Guidance

Once simulation is optimized, translate to physical model:

### From Simulation to Physical Design

| Simulation | Physical Interpretation |
|-----------|------------------------|
| `velocity_threshold` | Gate opening speed (tune spring tension, gap width) |
| `gate_width` | Size of mechanical gate opening zone |
| `gate_open_count / total_steps` | How often to actuate gate mechanism |
| `particles_passed` | Average particle flux (sets operating frequency) |
| `separation_metric` | Expected thermal gradient in physical model |

### Key Physical Constraints

1. **Gate Mechanism**
   - Velocity threshold in simulation → mechanical response time in reality
   - Simulation uses instant on/off; reality has finite response
   - Test with `HysteresisDemon` to model delayed response

2. **Vibration Table**
   - Simulation: elastic collisions with fixed walls
   - Reality: finite vibration frequency, amplitude
   - Simulation should match your actual vibration spectrum

3. **Particle Mass Variation**
   - Use different `mass` values in SimulationEngine
   - Test `separation_by_mass()` to verify multi-species separation
   - Physical model: different sized balls

4. **Thermal Efficiency**
   - Monitor `energy_conservation_error` - stays low → stable mechanism
   - High error suggests mechanical instability (turbulence, chaotic mixing)

## Troubleshooting

### Energy Not Conserved (error > 1%)
- Reduce `dt` (try 0.0005 or 0.0001)
- Reduce `num_particles` (fewer collision events)
- Increase `restitution` slightly

### Demon Not Creating Separation
- Check gate is being triggered: `print(demon.gate_open_count)`
- Visualize particle positions and velocities
- Try different threshold value
- Switch to `HysteresisDemon` for stability

### Simulation Too Slow
- Reduce `num_particles` (100-200 is usually enough)
- Increase `dt` (0.002 or 0.005 with care)
- Use smaller `box_size`

### Separation Decays Quickly
- **This is expected!** Entropy always increases
- Run longer to measure steady-state separation *rate*
- Demon must continuously operate to maintain gradient

## Files in This Package

```
maxwell_demon_physics_core.py        # Physics engine (core)
maxwell_demon_thermo_analysis.py     # Thermodynamic analysis
maxwell_demon_control.py             # Demon strategies
MAXWELL_DEMON_NOTEBOOK.py            # Complete Jupyter workflow
QUICK_REFERENCE.py                   # API quick reference
README.md                            # This file
```

## Next Steps

1. **Immediate** (today)
   - Install dependencies
   - Run QUICK_REFERENCE.py examples
   - Create one simple simulation

2. **Short-term** (this week)
   - Run MAXWELL_DEMON_NOTEBOOK.py end-to-end
   - Test different demon strategies
   - Scan parameter space for your geometry

3. **Medium-term** (before physical build)
   - Optimize for maximum ΔT or Δn
   - Characterize efficiency and power
   - Compare strategies on your specific setup
   - Document expected performance

4. **Physical build**
   - Use simulation predictions to guide mechanical design
   - Validate real model against simulation
   - Iterate: measure physical performance, refine simulation, rebuild

## Extending the Simulation

The toolkit is designed to be extended. Common additions:

### Add Custom Demon Strategy
```python
from maxwell_demon_control import MaxwellDemon

class MyCustomDemon(MaxwellDemon):
    def step(self):
        # Your logic here
        gate_open = ...
        particles_passed = ...
        self.record_state(gate_open, particles_passed)
        return gate_open, particles_passed
    
    def reset(self):
        # Reset state
        pass
```

### Add External Force
```python
# In SimulationEngine._compute_forces():
for particle in self.particles:
    particle.ax = -my_custom_force_x(particle.x, particle.y)
    particle.ay = -my_custom_force_y(particle.x, particle.y)
```

### Add Particle Types (by mass or property)
```python
sim = SimulationEngine(num_particles=250)
# Create mixed-mass particles
for i, p in enumerate(sim.particles):
    p.mass = 1.0 if i % 2 == 0 else 2.0  # Alternating light/heavy
```

## References

- Szilard, L. (1929). "On the decrease of entropy in a thermodynamic system..."
- Evans, D.J., Cohen, E.G.D., Searles, D.J. (2005). "Assay of transient dynamical response to viscous shear..."
- Maxwell, J.C. (1867). Theory of Heat (original demon thought experiment)

## Questions?

If the simulation doesn't match your expectations:

1. Check energy conservation (indicator of numerical stability)
2. Visualize particle positions and velocities
3. Compare your results to baseline (no demon) to isolate demon effect
4. Try the other demon strategies - they respond differently
5. Scan parameter space systematically rather than guessing

## License

This toolkit is provided as-is for your physical modeling research.

---

**Built for precise mechanical design before physical implementation.**
