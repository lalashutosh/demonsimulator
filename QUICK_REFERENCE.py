"""
MAXWELL'S DEMON SIMULATION TOOLKIT
Quick Reference Guide & API Documentation
===========================================

This guide shows you how to use the three core modules.
All code is ready to run in Jupyter. Copy-paste examples below to start.
"""

# ============================================================================
# MODULE 1: PHYSICS ENGINE (maxwell_demon_physics_core.py)
# ============================================================================

"""
Core class: SimulationEngine

Handles:
- Particle creation and properties
- 2D mechanics (Velocity Verlet integration)
- Collision detection and resolution
- Energy & momentum tracking

KEY PARAMETERS:
  box_size         : Size of simulation box (default: 10.0)
  num_particles    : Initial particles to create (default: 100)
  dt               : Time step (default: 0.001)
                     ⚠️ Smaller = more stable but slower
                     ✓ Use ~0.001 for typical setups
  
  gravity          : Gravitational acceleration (default: 0.0)
                     Set to 0 for vibrating table analog
  
  restitution      : Collision elasticity (default: 0.99)
                     1.0 = perfectly elastic
                     <1.0 = loses energy in collisions
  
  friction         : Tangential friction (default: 0.0)
  
  particle_radius  : Size of particles (default: 0.15)
  
  temperature      : Initial kinetic energy (default: 1.0)
                     Controls initial velocity distribution width

INITIALIZATION:
"""

from maxwell_demon_physics_core import SimulationEngine

# Create simulation
sim = SimulationEngine(
    box_size=10.0,
    num_particles=300,
    dt=0.001,
    gravity=0.0,
    restitution=0.99,
    temperature=1.5
)

print(f"Created: {sim}")
# Output: SimulationEngine(particles=300, box_size=10.0, time=0.0000)

"""
RUNNING SIMULATION:
"""

# Run for 5000 integration steps
# Record data every 50 steps
history = sim.run(num_steps=5000, record_interval=50)

# Access history
import numpy as np
times = np.array(history['time'])
energies = np.array(history['total_energy'])
temperatures = np.array(history['temperature'])

"""
QUERYING STATE:
"""

# Current observable properties
T = sim.compute_temperature()           # Temperature
E = sim.compute_total_energy()          # Total KE
p = sim.compute_total_momentum()        # Momentum vector
v_mean = sim.compute_mean_speed()       # Mean speed

# Distributions
speeds_hist, speed_bins = sim.compute_velocity_distribution(bins=30)
energies_hist, energy_bins = sim.compute_energy_distribution(bins=40)

# Get particle data
positions = sim.get_particle_positions()           # (N, 2) array
velocities = sim.get_particle_velocities()        # (N, 2) array
masses = sim.get_particle_masses()                 # (N,) array

# Get particles in region
particles = sim.get_particles_in_region(
    x_min=2.0, x_max=5.0,
    y_min=0.0, y_max=10.0
)

# Get approaching particles
approaching_left = sim.get_particles_approaching_from_side(
    side='left',      # 'left', 'right', 'bottom', 'top'
    x_threshold=3.0   # Custom threshold position
)

"""
MANUAL STEPPING (for demon control):
"""

# Single step
sim.step()

# Access particles directly
for particle in sim.particles:
    print(f"Particle: pos={particle.pos}, v={particle.speed:.3f}, KE={particle.kinetic_energy:.4f}")


# ============================================================================
# MODULE 2: THERMODYNAMIC ANALYSIS (maxwell_demon_thermo_analysis.py)
# ============================================================================

"""
Core class: ThermodynamicAnalyzer

Computes thermodynamic properties of the ensemble.
Initialize once, use for all analysis.
"""

from maxwell_demon_thermo_analysis import ThermodynamicAnalyzer

analyzer = ThermodynamicAnalyzer(sim)

"""
TEMPERATURE & ENERGY:
"""

T = analyzer.temperature()                    # Overall temperature

# Temperature by mass (if multiple species)
T_by_mass = analyzer.temperature_per_species([1.0, 2.0, 4.0])

# Temperature in specific region
T_left = analyzer.temperature_regional(
    x_min=0.0, x_max=5.0,
    y_min=0.0, y_max=10.0
)

mean_KE = analyzer.mean_kinetic_energy()      # Mean KE per particle

"""
DISTRIBUTIONS:
"""

# Speed histogram
hist, bin_centers = analyzer.speed_distribution(bins=50, weights=True)
# weights=True normalizes to probability density (compare to Maxwell-Boltzmann)

# Maxwell-Boltzmann reference
speeds = np.linspace(0, 3, 100)
pdf = analyzer.maxwell_boltzmann_speed(T=1.5, m=1.0, speeds=speeds)

# Kinetic energy histogram
KE_hist, KE_bins = analyzer.kinetic_energy_distribution(bins=40, weights=True)

# 2D velocity distribution
vel_hist, vx_edges, vy_edges = analyzer.velocity_distribution_2d(bins=20)

"""
ENTROPY:
"""

S_velocity = analyzer.entropy_from_speed_distribution(bins=50)
S_phase_space = analyzer.entropy_from_phase_space(n_bins_x=10, n_bins_v=10)

"""
PARTICLE SEPARATION METRICS (KEY FOR DEMON SUCCESS):
"""

# Separation by mass (if using different particle masses)
sep_mass = analyzer.separation_by_mass(
    left_region=3.0,    # x-position boundary
    right_region=7.0
)
print(f"Mass in left region: {sep_mass['mass_left']}")
print(f"Separation metric: {sep_mass['separation_metric']}")

# Separation by temperature (classic demon effect)
sep_temp = analyzer.separation_by_temperature(
    left_region=3.0,
    right_region=7.0
)
print(f"ΔT = {sep_temp['delta_T']}")
print(f"T_left = {sep_temp['T_left']:.4f}, T_right = {sep_temp['T_right']:.4f}")

# Separation by density
sep_density = analyzer.separation_by_density(
    left_region=3.0,
    right_region=7.0
)
print(f"Particles left: {sep_density['n_left']}, right: {sep_density['n_right']}")

"""
QUALITY CHECKS:
"""

initial_E = sim.compute_total_energy()  # Save at t=0
error = analyzer.energy_conservation_error(initial_E)
print(f"Energy error: {error*100:.3f}%")  # Should be << 1%

collision_rate = analyzer.collision_rate()
print(f"Collisions per particle per unit time: {collision_rate:.4f}")

# Pair correlation function (clustering measure)
g, r = analyzer.pair_correlation_function(max_distance=5.0, bins=40)


# ============================================================================
# MODULE 3: DEMON CONTROL (maxwell_demon_control.py)
# ============================================================================

"""
Demon Strategies:

1. VelocityThresholdDemon    - Opens gate if v > threshold (simplest)
2. MeasurementDemon          - Measures KE, admits high-energy particles
3. DensityDemon              - Responds to density gradients
4. HysteresisDemon           - Hysteresis prevents jitter

All have same interface:
  step() -> (gate_is_open, particles_that_passed)
"""

from maxwell_demon_control import (
    VelocityThresholdDemon,
    MeasurementDemon,
    DensityDemon,
    HysteresisDemon
)

"""
VELOCITY THRESHOLD DEMON (START HERE):
"""

demon = VelocityThresholdDemon(
    sim,
    side='left',                  # Separate from left
    velocity_threshold=0.5,       # Admit particles with v > 0.5
    gate_x=5.0,                  # Gate position
    gate_width=1.0               # Detection region width
)

# Run simulation with demon active
for step_idx in range(5000):
    gate_open, particles_passed = demon.step()
    sim.step()
    
    if (step_idx + 1) % 500 == 0:
        print(f"Step {step_idx+1}: Gate {'OPEN' if gate_open else 'CLOSED'}, "
              f"{len(particles_passed)} particles passed")

# Check demon statistics
print(f"Gate opened {demon.gate_open_count} times")
print(f"Total particles passed: {sum(demon.history['particles_passed'])}")

# Access history
gate_history = demon.history['gate_open']
particles_history = demon.history['particles_passed']

"""
HYSTERESIS DEMON (FOR STABLE GATING):
"""

demon_h = HysteresisDemon(
    sim,
    side='left',
    open_threshold=0.6,    # Speed needed to OPEN gate
    close_threshold=0.3,   # Speed to CLOSE gate
    gate_x=5.0,
    gate_width=1.0
)

# Use same way as above
for step_idx in range(5000):
    gate_open, particles_passed = demon_h.step()
    sim.step()

"""
MEASUREMENT DEMON (WITH ENERGY COST):
"""

demon_m = MeasurementDemon(
    sim,
    side='left',
    energy_threshold=0.8,       # Admit if KE > 0.8
    measurement_cost=0.05,      # Cost per measurement
    gate_x=5.0,
    gate_width=1.0
)

for step_idx in range(5000):
    gate_open, particles_passed = demon_m.step()
    sim.step()

# Check energy accounting
print(f"Total measurement cost: {demon_m.total_measurement_cost:.4f}")
print(f"Net work extracted: {demon_m.net_work():.4f}")

"""
DENSITY DEMON (PRESSURE-BASED):
"""

demon_d = DensityDemon(
    sim,
    side='left',
    gate_x=5.0,
    gate_width=1.0,
    density_threshold=0.5   # Open if this side is >0.5x denser
)

for step_idx in range(5000):
    gate_open, particles_passed = demon_d.step()
    sim.step()


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

"""
Full example: baseline -> demon -> analysis
"""

print("=" * 60)
print("COMPLETE WORKFLOW")
print("=" * 60)

# 1. Create and equilibrate
sim = SimulationEngine(box_size=10.0, num_particles=250, temperature=1.0)
initial_energy = sim.compute_total_energy()
sim.run(num_steps=2000, record_interval=100)

print(f"Equilibrated at T = {sim.compute_temperature():.4f}")

# 2. Create analyzer (for baseline)
analyzer = ThermodynamicAnalyzer(sim)
T_baseline = analyzer.temperature()
sep_baseline = analyzer.separation_by_temperature()

print(f"Baseline: T = {T_baseline:.4f}, ΔT = {sep_baseline['delta_T']:.4f}")

# 3. Add demon
demon = VelocityThresholdDemon(
    sim,
    velocity_threshold=0.5,
    gate_x=5.0,
    gate_width=1.0
)

print("\nRunning with demon...")
for step_idx in range(5000):
    demon.step()
    sim.step()
    sim._record_state()

# 4. Analyze results
sep_final = analyzer.separation_by_temperature()
energy_error = analyzer.energy_conservation_error(initial_energy)

print(f"\nFinal: T = {sim.compute_temperature():.4f}, ΔT = {sep_final['delta_T']:.4f}")
print(f"Energy error: {energy_error*100:.3f}%")
print(f"Demon duty cycle: {demon.gate_open_count / demon.step_count * 100:.1f}%")

# 5. Visualize (matplotlib not shown here, but standard)
# import matplotlib.pyplot as plt
# plt.plot(np.array(sim.history['time']), sim.history['temperature'])
# plt.show()


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
PROBLEM: Energy increasing or very unstable
CAUSE: Time step too large or collision resolution issues
FIX:
  - Reduce dt (try 0.0005 or 0.0001)
  - Reduce num_particles (fewer collisions to resolve)
  - Increase restitution slightly (<= 0.99)

PROBLEM: Demon not creating separation
CAUSE: Threshold/strategy not effective for your setup
FIX:
  - Visualize gate activity and particles passing
  - Try different thresholds (scan parameter space)
  - Switch to HysteresisDemon for more stable gating
  - Check that particles actually approach the gate

PROBLEM: Simulation too slow
CAUSE: Too many particles or too small dt
FIX:
  - Reduce num_particles (try 100-200)
  - Increase dt (0.002 or 0.005 with caution)
  - Run on GPU (if using numba-accelerated version)

PROBLEM: Separation not persistent (returns to equilibrium quickly)
CAUSE: This is EXPECTED! Demon needs continuous input
FIX:
  - Run longer simulation
  - Measure steady-state separation rate, not absolute separation
  - Consider work input needed to maintain separation (entropy cost)
"""


# ============================================================================
# PARAMETER OPTIMIZATION TEMPLATE
# ============================================================================

"""
Scan parameter space to find optimal demon settings:
"""

import numpy as np
from maxwell_demon_physics_core import SimulationEngine
from maxwell_demon_control import VelocityThresholdDemon
from maxwell_demon_thermo_analysis import ThermodynamicAnalyzer

# Parameter ranges
thresholds = np.linspace(0.3, 1.0, 8)
gate_widths = [0.5, 1.0, 1.5, 2.0]

results = []

for threshold in thresholds:
    for gate_width in gate_widths:
        sim = SimulationEngine(num_particles=200)
        sim.run(num_steps=1000, record_interval=50)  # Equilibrate
        
        demon = VelocityThresholdDemon(
            sim,
            velocity_threshold=threshold,
            gate_width=gate_width
        )
        
        # Run with demon
        for _ in range(3000):
            demon.step()
            sim.step()
            sim._record_state()
        
        # Measure separation
        analyzer = ThermodynamicAnalyzer(sim)
        sep = analyzer.separation_by_temperature()
        
        results.append({
            'threshold': threshold,
            'gate_width': gate_width,
            'delta_T': sep['delta_T'],
            'duty_cycle': demon.gate_open_count / demon.step_count,
        })

# Find best result
best = max(results, key=lambda x: x['delta_T'])
print(f"Best parameters: threshold={best['threshold']:.2f}, "
      f"gate_width={best['gate_width']:.1f}")
print(f"Achieved ΔT = {best['delta_T']:.4f}")
