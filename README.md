# Evolution Simulation

A Rust-based evolution simulation that demonstrates predator-prey dynamics, group behavior, and natural selection in a 2D environment. Built with GGEZ game engine.

## Features

- **Dynamic Ecosystem**: Simulates predator-prey relationships with complex behaviors
- **Emergent Behaviors**:
  - Pack hunting for predators
  - Herd formation for prey
  - Food foraging
  - Stamina management
  - Energy consumption
  
- **Advanced AI Behaviors**:
  - Hunting and stalking
  - Fleeing and scattering
  - Grazing and resting
  - Group coordination
  - Collision avoidance

- **Visual Indicators**:
  - Energy levels
  - Stamina bars
  - Movement direction
  - Creature states
  - Size-based interactions

## Requirements

- Rust (latest stable version)
- Required dependencies:
  - GGEZ game engine
  - rand crate

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jobyfoster/evolution-sim-rs.git
cd evolution-sim-rs
```

2. Build and run the project:
```bash
cargo run --release
```

## How It Works

### Creatures
The simulation features two types of creatures:
- **Predators**: Hunt in packs and chase prey
- **Prey**: Form herds and forage for food

Each creature has:
- Energy levels that deplete over time
- Stamina for sprinting
- Various states (hunting, fleeing, resting, etc.)
- Size-based interactions
- Group coordination abilities

### Simulation Rules
- Predators hunt prey to gain energy
- Prey consume food particles to survive
- All creatures manage stamina and energy
- Population balance is maintained automatically
- Creatures exhibit complex group behaviors

### Environment
- Wraparound world boundaries
- Randomly distributed food particles
- Dynamic food replenishment
- Visual effects for better understanding of creature states

## Configuration

Key simulation parameters can be adjusted in `src/main.rs`:

```rust
const VISION_RANGE: f64 = 150.0;
const EATING_RANGE: f64 = 15.0;
const FOOD_COUNT: usize = 200;
const INITIAL_PREDATORS: usize = 12;
const INITIAL_PREY: usize = 80;
```