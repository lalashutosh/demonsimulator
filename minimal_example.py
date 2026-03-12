"""
MAXWELL'S DEMON - MINIMAL WORKING EXAMPLE

Run this first to verify installation and understand the basic workflow.
~5 minutes to run, produces key results.

Usage:
    python minimal_example.py

Or in Jupyter:
    exec(open('minimal_example.py').read())
"""

import numpy as np
import matplotlib.pyplot as plt

# Import toolkit
from maxwell_demon_physics_core import SimulationEngine
from maxwell_demon_thermo_analysis import ThermodynamicAnalyzer
from maxwell_demon_control import VelocityThresholdDemon

print("\n" + "="*70)
print("MAXWELL'S DEMON - MINIMAL EXAMPLE")
print("="*70)

# ============================================================================
# STEP 1: Create and equilibrate baseline
# ============================================================================

print("\n[1] Creating and equilibrating baseline simulation...")

sim = SimulationEngine(
    box_size=10.0,
    num_particles=200,
    dt=0.001,
    temperature=1.0
)

initial_energy = sim.compute_total_energy()
print(f"    Initial: T={sim.compute_temperature():.4f}, E={initial_energy:.4f}")

# Equilibrate (no demon)
sim.run(num_steps=1000, record_interval=50)

print(f"    After equilibration: T={sim.compute_temperature():.4f}, E={sim.compute_total_energy():.4f}")

# ============================================================================
# STEP 2: Measure baseline separation
# ============================================================================

print("\n[2] Measuring baseline (no demon)...")

analyzer = ThermodynamicAnalyzer(sim)
sep_baseline = analyzer.separation_by_temperature()

print(f"    T_left  = {sep_baseline['T_left']:.4f}")
print(f"    T_right = {sep_baseline['T_right']:.4f}")
print(f"    ΔT      = {sep_baseline['delta_T']:.6f} (essentially zero)")

# ============================================================================
# STEP 3: Add demon and run
# ============================================================================

print("\n[3] Adding VelocityThresholdDemon and running...")

demon = VelocityThresholdDemon(
    sim,
    side='left',
    velocity_threshold=0.5,
    gate_x=5.0,
    gate_width=1.0
)

print(f"    {demon}")

# Run with demon
for step in range(3000):
    demon.step()
    sim.step()
    sim._record_state()

print(f"    Gate opened {demon.gate_open_count} times ({demon.gate_open_count/3000*100:.1f}%)")
print(f"    Total particles passed: {sum(demon.history['particles_passed'])}")

# ============================================================================
# STEP 4: Measure final separation
# ============================================================================

print("\n[4] Measuring final separation with demon...")

sep_final = analyzer.separation_by_temperature()

print(f"    T_left  = {sep_final['T_left']:.4f}")
print(f"    T_right = {sep_final['T_right']:.4f}")
print(f"    ΔT      = {sep_final['delta_T']:.6f}")

improvement = sep_final['delta_T'] / (sep_baseline['delta_T'] + 1e-10)
print(f"\n    ✓ Improvement: ΔT increased by factor of {improvement:.1f}x")

# ============================================================================
# STEP 5: Quality checks
# ============================================================================

print("\n[5] Verifying simulation quality...")

final_energy = sim.compute_total_energy()
energy_error = abs(final_energy - initial_energy) / initial_energy * 100

print(f"    Energy conservation error: {energy_error:.3f}%", end="")
if energy_error < 1.0:
    print(" ✓ (good)")
else:
    print(" ⚠ (warning)")

collision_rate = analyzer.collision_rate()
print(f"    Collision rate: {collision_rate:.4f} collisions/(particle·time)")

entropy = analyzer.entropy_from_speed_distribution()
print(f"    Entropy: {entropy:.4f} nats")

# ============================================================================
# STEP 6: Visualize
# ============================================================================

print("\n[6] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Temperature evolution
ax = axes[0, 0]
times = np.array(sim.history['time'])
temps = np.array(sim.history['temperature'])
ax.plot(times, temps, 'b-', linewidth=2)
ax.axhline(y=sim.compute_temperature(), color='g', linestyle='--', alpha=0.7, label='Final T')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature')
ax.set_title('Temperature Evolution')
ax.grid(True, alpha=0.3)
ax.legend()

# Gate activity
ax = axes[0, 1]
demon_steps = np.arange(len(demon.history['gate_open']))
ax.fill_between(demon_steps, np.array(demon.history['gate_open']).astype(float), alpha=0.3)
ax.set_xlabel('Demon Step')
ax.set_ylabel('Gate State')
ax.set_title('Gate Activity (1 = open, 0 = closed)')
ax.grid(True, alpha=0.3)

# Particle positions
ax = axes[1, 0]
positions = sim.get_particle_positions()
ax.scatter(positions[:, 0], positions[:, 1], s=20, alpha=0.5, edgecolors='blue')
ax.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Gate')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Final Particle Positions')
ax.legend()
ax.grid(True, alpha=0.2)

# Particle flux through gate
ax = axes[1, 1]
particles_passed = np.array(demon.history['particles_passed'])
ax.plot(demon_steps, particles_passed, 'g-', linewidth=1.5)
ax.fill_between(demon_steps, particles_passed, alpha=0.3)
ax.set_xlabel('Demon Step')
ax.set_ylabel('Particles Passed')
ax.set_title('Particle Flux Through Gate')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('minimal_example_results.png', dpi=150, bbox_inches='tight')
print("    ✓ Saved figure: minimal_example_results.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
Baseline (no demon):
  ΔT = {sep_baseline['delta_T']:.6f}

With VelocityThresholdDemon (v > 0.5):
  ΔT = {sep_final['delta_T']:.6f}
  
  Gate duty cycle: {demon.gate_open_count/3000*100:.1f}%
  Particles per opening: {sum(demon.history['particles_passed'])/max(demon.gate_open_count, 1):.2f}
  Energy error: {energy_error:.3f}%

Next steps:
  1. Try different threshold values (0.3, 0.4, 0.6, 0.7)
  2. Try HysteresisDemon or DensityDemon
  3. Optimize for maximum ΔT
  4. See QUICK_REFERENCE.py for parameter scanning template
  5. Run MAXWELL_DEMON_NOTEBOOK.py for complete analysis
""")

print("="*70)
print("✓ MINIMAL EXAMPLE COMPLETE")
print("="*70 + "\n")
