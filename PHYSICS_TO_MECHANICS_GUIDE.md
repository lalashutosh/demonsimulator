"""
SIMULATION TO PHYSICAL MODEL TRANSLATION GUIDE

How to use your validated computational model to design & build the physical apparatus.

This document bridges the gap between simulation parameters and real mechanical systems.
"""

# ============================================================================
# SECTION 1: SIMULATION OUTPUT → MECHANICAL REQUIREMENTS
# ============================================================================

"""
Your simulation produces key metrics that inform physical design:

SIMULATION METRIC          → PHYSICAL INTERPRETATION         → DESIGN IMPLICATION
─────────────────────────────────────────────────────────────────────────────

velocity_threshold         → Gate opening sensitivity        → Spring stiffness, gap size
  (particles with v > T     triggers at specific speed
   can pass)

gate_open_count / steps    → How often to actuate gate        → Actuation frequency
                             (duty cycle)                      Stepper motor speed

particles_passed / opening → Particle flux through gate       → Average operating rate

separation_metric (ΔT)     → Expected thermal gradient       → Temperature probe placement
                             in physical model

energy_conservation_error  → Mechanical stability check       → If >1%, expect turbulence
                             (low error = stable)

collision_rate             → Interaction frequency           → Vibration frequency should
                             (particles bump into                exceed this by 2-3x
                              each other this often)

separation_metric          → How much hotter/denser one      → Should be measurable with
                             side gets                         thermometer or particle counter

─────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# SECTION 2: VIBRATING TABLE DESIGN
# ============================================================================

"""
Your simulation uses elastic collisions with fixed walls.
Physical reality: elastic collisions with VIBRATING walls.

SIMULATION ←→ PHYSICAL MODEL EQUIVALENCE:

Simulation Aspect          Physical Interpretation
─────────────────────────────────────────────────────

Particle speed v           Kinetic energy from table vibration

Collision frequency        Vibration frequency × amplitude

"Temperature"              RMS velocity of particles
T ∝ <v²>                   ∝ (vibration_frequency × amplitude)²

Restitution coeff (0.99)   Elasticity of ball-table contact
                           (lower = energy loss per bounce)

─────────────────────────────────────────────────────────────────────────────

DESIGN RULE: Vibration Frequency

Your simulation runs at time step dt = 0.001.
This is equivalent to time resolution of your collision detection.

For physical model:
  - Vibration frequency should be F_vib >> collision_rate
  - Typical: collision_rate ≈ 1-10 collisions/(particle·time)
  - Set F_vib = 50-100 Hz minimum (300-500 Hz better)
  - This ensures many vibration cycles per particle transit through gate

Why? Simulation treats collisions as discrete events.
Physical vibration is continuous. Higher frequency → better simulation match.

DESIGN RULE: Amplitude Tuning

Simulation temperature T is controlled by initial velocity distribution:
  T = sqrt(2/pi) × v_typical²/2

Physical table vibration produces RMS velocity:
  v_rms ≈ 2π × f × A
  
  where f = vibration frequency (Hz), A = amplitude (mm)

To achieve target "temperature" in simulation:
  1. Run simulation with target T
  2. Measure mean_speed = sim.compute_mean_speed()
  3. For physical table:
     A = mean_speed / (2π × f)
     
Example:
  - Simulation: mean_speed = 2.0 (units)
  - Physical frequency: f = 100 Hz
  - Required amplitude: A = 2.0 / (2π × 100) ≈ 3.2 mm
  
ALTERNATIVE: Run parameter scan in simulation to find T vs v_mean relationship
  for your specific num_particles and box_size. Then directly calibrate table.

─────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# SECTION 3: GATE MECHANISM DESIGN
# ============================================================================

"""
Your demon decides: gate_open = f(particle_velocity, position, etc.)

VELOCITY THRESHOLD DEMON (simplest to implement physically):

Simulation:  gate_open if (particle.vx > v_threshold)

Physical:    Need mechanical sensor to measure/estimate particle velocity
             and trigger gate opening

IMPLEMENTATION OPTIONS:

Option A: Passive (Velocity-Tuned Spring Gate)
─────────────────────────────────────────────────
  
  Concept: Gate naturally opens when particles hit it hard enough
  
  Design:
    - Spring-loaded gate (mass m_gate, spring constant k)
    - Gate position: x_gate (center of box)
    - When particle approaches with speed v:
      - If v > v_threshold: spring compresses enough to open
      - If v < v_threshold: spring bounces particle back
  
  Physics:
    - Gate equilibrium: k × x_rest = m_gate × g (if vertical)
                       or x_rest = 0 (if horizontal)
    - Compression for opening: δ_open
    - Spring constant: k = F / δ_open
    - Threshold speed: v_threshold = sqrt(2 × E / m_particle)
                      where E = spring potential energy at δ_open
  
  Advantages: No electronics, purely mechanical
  Disadvantages: Less control, harder to tune precisely
  
  How to calibrate:
    1. From simulation: get velocity_threshold (e.g., 0.5 units)
    2. In physical model: measure v_typical for your particles & table
    3. Design spring to open at v = v_threshold × v_typical

Option B: Active (Sensor + Electronically Controlled)
────────────────────────────────────────────────────
  
  Concept: Sensor measures speed, electronics decide gate position
  
  Design:
    - Optical/capacitive sensor near gate region detects approaching particles
    - Estimate velocity from sensor signal rate
    - Servo motor or solenoid actuates gate based on estimate
    - Microcontroller (Arduino/Raspberry Pi) runs decision logic
  
  Code sketch:
    ```cpp
    float sensor_reading = analogRead(SENSOR_PIN);
    float velocity_estimate = filter(sensor_reading);
    
    if (velocity_estimate > VELOCITY_THRESHOLD) {
        digitalWrite(GATE_PIN, HIGH);  // Open gate
    } else {
        digitalWrite(GATE_PIN, LOW);   // Close gate
    }
    ```
  
  Advantages: Programmable, can implement complex logic
  Disadvantages: Requires electronics, measurement latency

Option C: Hybrid (Passive + Adjustment)
──────────────────────────────────────
  
  Concept: Passive spring gate + adjustable trigger point
  
  Design:
    - Base spring gate (Option A)
    - Mechanical adjustment: change spring pre-compression
      or move gate location to tune velocity_threshold
    - No electronics, but fully tunable
  
  How to adjust:
    - Increase pre-compression → higher velocity_threshold
    - Decrease pre-compression → lower velocity_threshold
    - This changes when gate opens

─────────────────────────────────────────────────────────────────────────────

GATE GEOMETRY (from your simulation):

Simulation parameter: gate_width = 1.0 (typical)
Physical interpretation:
  - This is the region where demon "sees" approaching particles
  - Too narrow: few particles trigger gate
  - Too wide: gate activates randomly
  
Physical design:
  - Gate width ≈ 2-3 × particle diameter
  - For ~5mm diameter balls: gate width ≈ 10-15 mm
  
Simulation parameter: gate_x = 5.0 (center)
Physical interpretation:
  - Position of gate dividing left/right chambers
  - Simulation box = 10 units; gate at 5 = center
  
Physical design:
  - Place gate at geometric center of your vibrating table
  - Ensures equal volume on each side (for fairness)
  
─────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# SECTION 4: PARTICLE SELECTION & PREPARATION
# ============================================================================

"""
Simulation particles: idealized spheres, perfectly elastic collisions

Physical particles: must be carefully selected

PARTICLE SPECIFICATIONS:

1. Material Choice
   ─────────────────
   Goal: Maximize restitution coefficient (energy conservation)
   Options:
     - Steel balls: e ≈ 0.95-0.99 (excellent, matches simulation)
     - Glass marbles: e ≈ 0.90-0.95 (good, some energy loss)
     - Rubber: e < 0.90 (dissipates energy)
     - Plastic: e ≈ 0.85-0.90 (moderate)
   
   Recommendation: Steel balls (ball bearings)
   - High restitution
   - Uniform mass/radius (important!)
   - Durable
   - Well-characterized properties

2. Size Selection
   ───────────────
   From simulation:
     - particle_radius = 0.15 (typical)
     - box_size = 10.0
     - ratio = 0.15/10.0 = 0.015 = 1.5%
   
   Physical implementation:
     - If your box is 200 mm:
       particle_diameter = 0.015 × 200 = 3 mm
     - If your box is 300 mm:
       particle_diameter = 0.015 × 300 = 4.5 mm
     
   Constraint: Not too large (few particles) or too small (friction dominates)
   Typical range: 3-8 mm diameter for tabletop demonstrations

3. Mass Uniformity
   ────────────────
   Critical: All particles same mass (or intentionally different for mass-species)
   
   For baseline simulation:
     - Use particles all same mass
     - Test separation by density first
   
   For advanced (multi-species) demon:
     - Use 2-3 different ball materials or sizes
     - Measure individual masses (scale ±0.01g)
     - Record mass ratios
     - Update simulation with actual masses:
       ```python
       sim.particles[0].mass = 1.0
       sim.particles[1].mass = 2.0  # Heavy particle
       sim.particles[2].mass = 1.0
       # etc...
       ```

4. Surface Finish
   ──────────────
   Goal: Minimize friction during collisions
   
   - Polish if possible
   - Clean before use (remove dust)
   - Avoid surfaces that attract dust
   - Re-polish if needed between runs

─────────────────────────────────────────────────────────────────────────────

PARTICLE NUMBER SELECTION:

Simulation: num_particles = 200-300 (typical)

Physical consideration:
  - Too few (<50): Statistical noise, low separation effect
  - Too many (>500): Difficult to control, hard to count, heating issues
  - Sweet spot: 100-300 particles
  
Example for 3mm steel balls:
  - Volume per particle ≈ 14 mm³
  - 200 particles ≈ 2,800 mm³ of solid
  - If box is 200×200×20 mm (thin layer):
    Volume = 800,000 mm³
    Packing fraction ≈ 2,800/800,000 ≈ 0.35%
    This is reasonable (low enough to avoid jamming)

─────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# SECTION 5: MEASUREMENT & VALIDATION
# ============================================================================

"""
Once physical model is built: Validate against simulation

KEY MEASUREMENTS TO PERFORM:

1. Temperature Validation
   ──────────────────────
   
   Simulation produces:
     - T = mean(v²) / 2 in reduced units
   
   Physical measurement:
     - Option A: Thermal imaging (infrared camera)
       Shows actual temperature gradient if demon works
     - Option B: Laser/photographic tracking
       Measure individual particle velocities → compute T
     - Option C: Analog: visual observation of separation
   
   Comparison:
     - Baseline (no demon): T_left ≈ T_right (flat)
     - With demon (after 1-2 minutes):
       T_left vs T_right should show gradient
     - Delta_T magnitude should be within factor of ~2
       of simulation prediction

2. Density Separation
   ──────────────────
   
   Simplest to measure!
   
   Method:
     1. Run with demon for 2-5 minutes
     2. Stop vibration
     3. Count particles on left vs right side
     4. Compute delta_n = |n_left - n_right|
     5. Compare to simulation prediction
   
   Expected: 10-20% difference in particle count over 5 min
   (More for optimized demon parameters)

3. Gate Statistics
   ────────────────
   
   Measurement:
     1. Install counter on gate mechanism
        (mechanical tally, optical sensor, etc.)
     2. Run for known time T
     3. Count total gate openings N
     4. Compute duty_cycle = N × t_open / T
     5. Compare to simulation: demon.gate_open_count / demo.step_count
   
   Should match within factor of ~1.5-2x
   (Physical friction and latency affect exact timing)

4. Energy Conservation Check
   ──────────────────────────
   
   Simulation: measures energy_error = |E_final - E_initial| / E_initial
   
   Physical equivalent:
     - Measure: Does system reach stable oscillation?
     - If energy error > 5%: excess damping (friction, air resistance)
     - If energy error < 2%: system is stable and matches simulation
   
   Diagnostic: Run baseline (no demon, just vibration) for 5 minutes
     - Particle speeds should remain roughly constant
     - If slowing down: too much friction (polish balls, table)
     - If staying constant: good, proceed to demon testing

─────────────────────────────────────────────────────────────────────────────

ITERATION LOOP (Physical vs Simulation):

1. Run simulation with best parameters from optimization
   → Record expected ΔT, duty_cycle, particles_passed
   
2. Build physical model to match simulation geometry
   
3. Test physical model
   → Measure ΔT, duty_cycle, particles_passed
   
4. Compare results:
     - If matches (within 50%): ✓ Good, demon works as designed
     - If ΔT too small: 
       * Reduce velocity_threshold (gate opens earlier)
       * Increase table vibration amplitude
       * Check gate mechanism (is it actually opening?)
     - If duty_cycle too high:
       * Increase velocity_threshold
       * Slow down table vibration
     
5. Adjust physical parameters and re-test
   
6. When physical matches simulation:
   → You've successfully validated the model
   → Proceed to characterization and optimization

─────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# SECTION 6: WORK EXTRACTION & EFFICIENCY
# ============================================================================

"""
Final goal: Can you extract useful work from the thermal gradient created?

THEORETICAL BACKGROUND:

Maxwell's Demon appears to violate 2nd law (entropy decrease).
Resolution: Demon must pay entropy cost for measurement/operation.

For your physical model:

Work Input:
  - Vibration table: electrical energy into table motor
  - Gate actuation: energy to open/close gate

Work Output (potential):
  - Thermal gradient: can be used in small heat engine
    (Though practical power is tiny)

Efficiency Analysis:

Define:
  - W_in = total energy input to vibration + gate actuation
  - W_out = useful work extracted (if any)
  - Efficiency = W_out / W_in
  
Measurement:
  1. Vibration power:
     P_vib = Force × velocity
           = (amplitude) × (frequency) × (mass × g)
           ≈ 0.003 × 100 × 0.1 × 10 ≈ 0.3 W (order of magnitude)
  
  2. Gate actuation power (if active):
     P_gate = Force × velocity_gate
     (measure with force sensor on gate)
  
  3. Useful work:
     This is harder. You could:
     - Place small peltier element on hot side of gradient
     - Measure power output
     - Compare to P_in
  
  Prediction: Efficiency << 1% (tiny)
  This is EXPECTED and CORRECT!
  
  Szilard proved: Entropy cost of measurement ≥ log(2) × k_B × T
  For physical parameters, this exceeds any practical work output.

PRACTICAL PERSPECTIVE:

The value of your demon is NOT efficiency (it's terrible by design).
The value is:
  1. **Demonstration of thermodynamic principle** (educational)
  2. **Validation of simulation** (you built a validated model)
  3. **Understanding of complex system dynamics** (research)
  4. **Beautiful physics** (yes, this matters)

Don't aim for energy-positive operation. It's impossible without
violating thermodynamics. Aim for a clean, well-characterized system
that matches simulation predictions.

─────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# SECTION 7: TROUBLESHOOTING PHYSICAL IMPLEMENTATION
# ============================================================================

"""
COMMON ISSUES & SOLUTIONS:

Issue: Physical model doesn't show separation
──────────────────────────────────────────
  Likely cause: Gate mechanism not working
  
  Diagnostics:
    1. Check gate moves when should open
       → If not: mechanism is jammed, re-examine pivot, friction
    2. Check particles reach gate region
       → If not: increase table vibration amplitude
    3. Check gate actually blocks/allows particle flow
       → Use high-speed camera to observe
    4. Verify velocity_threshold is appropriate
       → Lower threshold → more particles pass
  
  Solutions:
    - Lubricate gate pivot (light machine oil)
    - Increase gate opening distance
    - Adjust sensitivity (spring stiffness for passive gate)
    - Check gate isn't sticky due to dust


Issue: Vibration causes excessive heating/noise
──────────────────────────────────────────────
  Likely cause: Friction, impacts too violent
  
  Diagnostics:
    - Feel table: is it getting hot?
    - Listen: loud banging suggests high-energy collisions
    - Observe: do balls seem to be moving slower over time?
  
  Solutions:
    - Reduce vibration amplitude
    - Lower vibration frequency
    - Check particle material (less elastic material = more damping)
    - Verify table surface is smooth (sand if rough)
    - Check ball surface (polish, remove scratches)
    - Reduce number of balls temporarily to test


Issue: Gate opening too much/too little
────────────────────────────────────────
  Likely cause: velocity_threshold setting wrong
  
  Diagnostics:
    - Count gate openings over 1 minute
    - Compare to simulation: demon.gate_open_count / steps × (60 / time_per_step)
  
  Solutions:
    - If opening too much: increase threshold (stiffer spring, higher spring preload)
    - If opening too little: decrease threshold
    - Systematic: test 5-6 different settings, plot vs separation achieved


Issue: Separation appears and then reverses
────────────────────────────────────────────
  Likely cause: Entropy is winning (expected!)
  
  This IS expected behavior. Demon can create separation temporarily,
  but 2nd law of thermodynamics means it will decay over time.
  
  Solutions:
    - Measure separation RATE (ΔT per minute) rather than absolute
    - Run longer to see if steady state is reached
    - This may actually be more interesting: quantifying how long
      the demon can maintain separation


Issue: Particles aren't clearly separating
───────────────────────────────────────────
  Likely cause: Separation too subtle or particles mixing
  
  Diagnostics:
    - Mark particles (paint left/right with color)
    - After 5-10 min, are colors still roughly separated?
    - Or do they form a homogeneous mix immediately?
  
  Solutions:
    - Increase gate duty cycle (let more particles through)
    - Optimize velocity_threshold via systematic testing
    - Check gate position (move it, maybe 50-50 split isn't natural)
    - Run longer (separation is slow, needs patience)

─────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# SECTION 8: DOCUMENTATION & PUBLICATION
# ============================================================================

"""
Once you have results, document them:

SUGGESTED DATA TO RECORD:

1. Setup parameters:
   - Box dimensions (L × W × H mm)
   - Vibration frequency & amplitude
   - Particle diameter, material, mass
   - Number of particles
   - Gate position and width
   - Velocity threshold (if tuned)

2. Simulation predictions:
   - Expected ΔT at 5 min
   - Expected duty cycle (gate openings/min)
   - Expected particles_passed per opening
   - Energy conservation error

3. Physical measurements:
   - Actual ΔT vs time (temperature probes or estimation)
   - Gate openings/min (observed)
   - Separation rate (% particles per side vs time)
   - Energy/power input (if measured)

4. Comparison:
   - Predicted vs observed ΔT
   - Predicted vs observed duty cycle
   - Predicted vs observed particles_passed
   - Percent error in each

5. Photographs/Video:
   - Overview of setup
   - Close-up of gate mechanism
   - High-speed video of particles near gate
   - Time-lapse of separation developing

SUGGESTED PRESENTATION STRUCTURE:

"Maxwell's Demon in Silico and In Vitro: Computational Model Validation
 and Physical Implementation of a Velocity-Selective Gate"

1. Introduction
   - Historical context (Maxwell, Szilard, modern demons)
   - Purpose: validate computational model with physical experiment
   
2. Computational Model
   - Physics engine (Velocity Verlet, elastic collisions)
   - Demon strategy (velocity threshold)
   - Optimization results
   - Predicted separation metrics
   
3. Physical Implementation
   - Design decisions (vibration frequency, particle selection, gate mechanism)
   - Dimensioning from simulation
   - Calibration procedure
   
4. Validation Results
   - Comparison: simulation vs physical measurements
   - Quantitative agreement in ΔT, duty cycle, etc.
   - Sources of discrepancy and explanations
   
5. Characterization
   - Separation over time
   - Efficiency analysis
   - Scaling with parameters
   
6. Conclusions
   - Lessons learned
   - Thermodynamic implications
   - Future improvements

─────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# SUMMARY CHECKLIST
# ============================================================================

"""
BEFORE BUILDING PHYSICAL MODEL:

☐ Run simulation thoroughly and document results
☐ Optimize demon strategy (which algorithm? what parameters?)
☐ Record predicted ΔT, duty_cycle, particles_per_opening
☐ Calculate required vibration frequency & amplitude
☐ Calculate required gate spring constant (if passive)

DURING CONSTRUCTION:

☐ Select particles (material, size, mass uniformity)
☐ Build vibrating table (correct frequency & amplitude)
☐ Implement gate mechanism (test mechanical response)
☐ Calibrate velocity_threshold (adjust spring/preload)
☐ Set up measurement system (temperature, particle counting)

DURING TESTING:

☐ Verify baseline equilibration (no demon): T is flat
☐ Test demon activation: gate moves at right times
☐ Measure initial separation (at t=1 min, t=5 min)
☐ Compare to simulation prediction
☐ Iterate: adjust parameters if not matching
☐ Document everything: photos, video, data

ANALYSIS:

☐ Create comparison plots: simulation vs physical
☐ Quantify agreement (percent error in ΔT, etc.)
☐ Identify sources of discrepancy
☐ Validate any thermodynamic predictions
☐ Prepare summary for presentation/publication

─────────────────────────────────────────────────────────────────────────────
"""

print(__doc__)
