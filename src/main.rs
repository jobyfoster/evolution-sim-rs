use ggez::{
    event,
    graphics::{self, Color, DrawMode, Mesh, Rect},
    Context, GameResult,
};
use rand::Rng;
use std::vec::Vec;

//////////////////////////////////////////////////////////////////////////////
// Simulation Constants
//////////////////////////////////////////////////////////////////////////////

/// Range at which creatures can detect others
const VISION_RANGE: f64 = 150.0;
/// Range at which predators can catch prey
const EATING_RANGE: f64 = 15.0;
/// Energy gained by predators when eating prey
const EATING_GAIN: f64 = 50.0;
/// Number of food particles in the world
const FOOD_COUNT: usize = 200;
/// Energy gained by prey when eating food
const FOOD_ENERGY: f64 = 25.0;
/// Base energy consumption rate per update
const BASE_ENERGY_CONSUMPTION: f64 = 0.05;
/// Energy consumption multiplier based on speed
const SPEED_ENERGY_FACTOR: f64 = 0.1;
/// Absolute minimum separation multiplier based on combined sizes
const MIN_SEPARATION_FACTOR: f64 = 1.2; // Ensures entities never overlap based on their sizes
/// Preferred separation multiplier for same-type creatures
const PREFERRED_SEPARATION_FACTOR: f64 = 1.8;
/// Strong separation force for overlapping creatures
const SEPARATION_FORCE: f64 = 2.5;
/// Base collision avoidance force
const COLLISION_FORCE: f64 = 1.0;
/// Range for pack hunting coordination (smaller for tighter predator groups)
const PACK_RANGE: f64 = 40.0;
/// Range for prey herding (larger for bigger herds)
const HERD_RANGE: f64 = 150.0;
/// Factor for herd cohesion (adjusted for smoother movement)
const HERD_COHESION: f64 = 0.04;
/// Initial number of predators
const INITIAL_PREDATORS: usize = 20;
/// Initial number of prey
const INITIAL_PREY: usize = 150;
/// Minimum movement threshold to prevent stuttering
const MOVEMENT_THRESHOLD: f64 = 0.01;
/// How much prey align their movement
const HERD_ALIGNMENT_FACTOR: f64 = 0.3;
/// Range for coordinated attacks
const PACK_ATTACK_RANGE: f64 = 30.0;
/// Adjust constants for better chase dynamics
const PREDATOR_BASE_SPEED: f64 = 3.2;     // Even faster base speed
const CHASE_PREDICTION: f64 = 0.5;        // Better prediction
/// Adjust constants for more dynamic movement
const PREY_BASE_SPEED: f64 = 2.8;         // Decent base speed for prey
const PREDATOR_SPRINT: f64 = 4.8;         // Very fast sprint for short chases
const PREY_SPRINT: f64 = 4.2;             // Fast escape speed
const ACCELERATION: f64 = 0.15;           // How quickly speed changes
const INERTIA: f64 = 0.8;                 // Movement smoothing
const STAMINA_MAX: f64 = 100.0;
const STAMINA_RECOVERY: f64 = 0.2;
const SPRINT_COST: f64 = 0.5;

//////////////////////////////////////////////////////////////////////////////
// Food Implementation
//////////////////////////////////////////////////////////////////////////////

/// Represents a food particle in the simulation
#[derive(Clone, Debug)]
struct Food {
    x: f64,
    y: f64,
    energy: f64,
    eaten: bool,
}

impl Food {
    /// Creates a new food particle at random coordinates
    fn new(x: f64, y: f64) -> Self {
        Food {
            x,
            y,
            energy: FOOD_ENERGY,
            eaten: false,
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Creature Implementation
//////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, PartialEq)]
enum CreatureState {
    Hunting,
    Stalking,
    Fleeing,
    Grazing,
    Resting,
    Scattering,
    Grouping,
}

/// Represents a living creature in the simulation
#[derive(Clone, Debug)]
struct Creature {
    x: f64,
    y: f64,
    speed: f64,
    size: f64,
    energy: f64,
    stamina: f64,
    is_predator: bool,
    dx: f64,
    dy: f64,
    state: CreatureState,
    last_state_change: f64,    // Time tracking for state changes
    group_id: Option<usize>,   // For pack/herd coordination
    target_id: Option<usize>,  // Current chase/flee target
}

impl Creature {
    /// Creates a new creature with randomized traits
    fn new(x: f64, y: f64, is_predator: bool) -> Self {
        let mut rng = rand::thread_rng();
        Creature {
            x,
            y,
            speed: if is_predator {
                rng.gen_range(2.5..PREDATOR_BASE_SPEED)
            } else {
                rng.gen_range(2.2..PREY_BASE_SPEED)
            },
            size: if is_predator {
                rng.gen_range(1.8..2.8)
            } else {
                rng.gen_range(1.0..1.8)
            },
            energy: 100.0,
            stamina: STAMINA_MAX,
            is_predator,
            dx: 0.0,
            dy: 0.0,
            state: if is_predator { 
                CreatureState::Hunting 
            } else { 
                CreatureState::Grazing 
            },
            last_state_change: 0.0,
            group_id: None,
            target_id: None,
        }
    }

    /// Calculates distance to another creature
    fn distance_to(&self, other: &Creature) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculates distance to a food particle
    fn distance_to_food(&self, food: &Food) -> f64 {
        let dx = self.x - food.x;
        let dy = self.y - food.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate minimum separation distance between two creatures
    fn min_separation_distance(&self, other: &Creature) -> f64 {
        (self.size + other.size) * MIN_SEPARATION_FACTOR * 5.0 // Multiply by 5.0 to match rendering scale
    }

    /// Calculate preferred separation distance between two creatures
    fn preferred_separation_distance(&self, other: &Creature) -> f64 {
        (self.size + other.size) * PREFERRED_SEPARATION_FACTOR * 5.0
    }

    /// Enhanced collision avoidance with size-based distances
    fn calculate_collision_avoidance(&self, nearby: &[Creature]) -> (f64, f64) {
        let mut avoid_x = 0.0;
        let mut avoid_y = 0.0;
        let mut strong_separation_x = 0.0;
        let mut strong_separation_y = 0.0;
        let mut has_strong_separation = false;

        for other in nearby {
            let dist = self.distance_to(other);
            let min_dist = self.min_separation_distance(other);
            let preferred_dist = self.preferred_separation_distance(other);

            if dist < min_dist && dist > 0.0 {
                // Strong separation force when too close (based on sizes)
                let force = (1.0 - (dist / min_dist)).powf(2.0) * SEPARATION_FORCE;
                strong_separation_x += (self.x - other.x) / dist * force;
                strong_separation_y += (self.y - other.y) / dist * force;
                has_strong_separation = true;
            } else if dist < preferred_dist && dist > 0.0 {
                // Normal collision avoidance for preferred spacing
                let force = (1.0 - (dist / preferred_dist)).powf(2.0) * COLLISION_FORCE;
                avoid_x += (self.x - other.x) / dist * force;
                avoid_y += (self.y - other.y) / dist * force;
            }
        }

        if has_strong_separation {
            (strong_separation_x, strong_separation_y)
        } else {
            (avoid_x, avoid_y)
        }
    }

    /// Enhanced group center calculation with weighted distances
    fn calculate_group_center(&self, nearby: &[Creature]) -> Option<(f64, f64)> {
        let mut total_weight = 0.0;
        let mut center_x = 0.0;
        let mut center_y = 0.0;

        let range = if self.is_predator { PACK_RANGE } else { HERD_RANGE };

        for other in nearby {
            if other.is_predator == self.is_predator {
                let dist = self.distance_to(other);
                if dist < range && dist > 0.0 {
                    let weight = 1.0 - (dist / range);
                    center_x += other.x * weight;
                    center_y += other.y * weight;
                    total_weight += weight;
                }
            }
        }

        if total_weight > 0.0 {
            Some((center_x / total_weight, center_y / total_weight))
        } else {
            None
        }
    }

    /// Updates creature movement based on surroundings
    fn update_movement(&mut self, nearby: &[Creature], food: &[Food], width: f64, height: f64) {
        // Update stamina and energy
        self.stamina = (self.stamina + STAMINA_RECOVERY * 
            if self.state == CreatureState::Resting { 2.0 } else { 1.0 })
            .min(STAMINA_MAX);
        
        // Calculate base movement vector
        let (mut target_dx, mut target_dy, should_sprint) = match self.state {
            CreatureState::Hunting | CreatureState::Stalking => 
                self.calculate_hunting_movement(nearby),
            CreatureState::Fleeing | CreatureState::Scattering => 
                self.calculate_fleeing_movement(nearby),
            CreatureState::Grazing => 
                self.calculate_grazing_movement(nearby, food),
            CreatureState::Grouping => 
                self.calculate_group_movement(nearby),
            CreatureState::Resting => 
                (self.dx * 0.95, self.dy * 0.95, false),
        };

        // Apply collision avoidance
        let (avoid_x, avoid_y) = self.calculate_collision_avoidance(nearby);
        target_dx += avoid_x;
        target_dy += avoid_y;

        // Normalize movement vector
        let magnitude = (target_dx * target_dx + target_dy * target_dy).sqrt();
        if magnitude > MOVEMENT_THRESHOLD {
            target_dx /= magnitude;
            target_dy /= magnitude;

            // Apply speed and sprint
            let current_speed = if should_sprint && self.stamina > SPRINT_COST {
                self.stamina -= SPRINT_COST;
                if self.is_predator { PREDATOR_SPRINT } else { PREY_SPRINT }
            } else {
                self.speed
            };

            target_dx *= current_speed;
            target_dy *= current_speed;
        }

        // Apply smooth acceleration
        self.dx = self.dx * INERTIA + target_dx * ACCELERATION;
        self.dy = self.dy * INERTIA + target_dy * ACCELERATION;

        // Update position with boundary wrapping
        self.x = (self.x + self.dx + width) % width;
        self.y = (self.y + self.dy + height) % height;

        // Energy consumption
        let speed = (self.dx * self.dx + self.dy * self.dy).sqrt();
        self.energy -= BASE_ENERGY_CONSUMPTION + speed * SPEED_ENERGY_FACTOR;

        // Update state
        self.update_state(nearby);
    }

    /// Calculate predator movement with improved hunting behavior
    fn calculate_hunting_movement(&self, nearby: &[Creature]) -> (f64, f64, bool) {
        let pack_members = nearby.iter()
            .filter(|c| c.is_predator && self.distance_to(c) < PACK_RANGE)
            .count();

        if let Some(target) = self.find_best_target(nearby) {
            let dist = self.distance_to(target);
            if dist < VISION_RANGE {
                // Calculate intercept point
                let prediction_time = dist * CHASE_PREDICTION;
                let intercept_x = target.x + target.dx * prediction_time;
                let intercept_y = target.y + target.dy * prediction_time;

                // Pack hunting tactics
                if pack_members >= 2 {
                    // Surround prey
                    let angle_offset = 2.0 * std::f64::consts::PI / pack_members as f64;
                    let base_angle = (target.y - self.y).atan2(target.x - self.x);
                    let my_angle = base_angle + angle_offset * (pack_members as f64 / 2.0);

                    let surround_x = target.x + my_angle.cos() * PACK_ATTACK_RANGE;
                    let surround_y = target.y + my_angle.sin() * PACK_ATTACK_RANGE;

                    let dx = surround_x - self.x;
                    let dy = surround_y - self.y;
                    let chase_dist = (dx * dx + dy * dy).sqrt();

                    (dx / chase_dist, dy / chase_dist, true)
                } else {
                    // Direct chase
                    let dx = intercept_x - self.x;
                    let dy = intercept_y - self.y;
                    let chase_dist = (dx * dx + dy * dy).sqrt();
                    (dx / chase_dist, dy / chase_dist, dist < VISION_RANGE * 0.5)
                }
            } else {
                self.calculate_search_movement()
            }
        } else {
            self.calculate_search_movement()
        }
    }

    /// Calculate prey movement with improved evasion behavior
    fn calculate_fleeing_movement(&self, nearby: &[Creature]) -> (f64, f64, bool) {
        let mut escape_x = 0.0;
        let mut escape_y = 0.0;
        let mut threat_level = 0.0;
        let mut nearest_predator_dist = f64::MAX;

        for predator in nearby.iter().filter(|c| c.is_predator) {
            let dist = self.distance_to(predator);
            if dist < VISION_RANGE {
                let weight = (1.0 - (dist / VISION_RANGE)).powi(2);
                escape_x += (self.x - predator.x) * weight;
                escape_y += (self.y - predator.y) * weight;
                threat_level += weight;
                nearest_predator_dist = nearest_predator_dist.min(dist);
            }
        }

        if threat_level > 0.0 {
            let mag = (escape_x * escape_x + escape_y * escape_y).sqrt();
            if mag > 0.0 {
                // Sprint if predator is very close
                let should_sprint = nearest_predator_dist < VISION_RANGE * 0.4;
                (escape_x / mag, escape_y / mag, should_sprint)
            } else {
                (rand::random::<f64>() * 2.0 - 1.0, 
                 rand::random::<f64>() * 2.0 - 1.0, 
                 true)
            }
        } else {
            (self.dx, self.dy, false)
        }
    }

    /// Calculate prey movement with improved evasion behavior
    fn calculate_grazing_movement(&self, nearby: &[Creature], food: &[Food]) -> (f64, f64, bool) {
        // Check for predators first (early warning system)
        let predator_nearby = nearby.iter()
            .any(|c| c.is_predator && self.distance_to(c) < VISION_RANGE * 1.2);
        
        if predator_nearby {
            return self.calculate_fleeing_movement(nearby);
        }

        // Find nearest food and herd center
        let nearest_food = food.iter()
            .min_by_key(|f| (self.distance_to_food(f) * 1000.0) as i32);
        
        let herd_center = self.calculate_group_center(nearby);
        
        let mut target_dx = 0.0;
        let mut target_dy = 0.0;
        let mut priority = 0;

        // Balance between food seeking and herd cohesion
        if let Some(food) = nearest_food {
            let dist = self.distance_to_food(food);
            if dist < VISION_RANGE * 0.5 {
                let food_weight = 1.0 - (dist / (VISION_RANGE * 0.5));
                target_dx += (food.x - self.x) * food_weight;
                target_dy += (food.y - self.y) * food_weight;
                priority = 1;
            }
        }

        if let Some((herd_x, herd_y)) = herd_center {
            let herd_weight = if priority == 0 { 1.0 } else { 0.3 };
            target_dx += (herd_x - self.x) * HERD_COHESION * herd_weight;
            target_dy += (herd_y - self.y) * HERD_COHESION * herd_weight;
        }

        let mag = (target_dx * target_dx + target_dy * target_dy).sqrt();
        if mag > 0.0 {
            (target_dx / mag, target_dy / mag, false)
        } else {
            self.calculate_wander_movement()
        }
    }

    /// Calculate prey movement with improved evasion behavior
    fn calculate_wander_movement(&self) -> (f64, f64, bool) {
        let mut rng = rand::thread_rng();
        let current_angle = self.dy.atan2(self.dx);
        let angle_change = rng.gen_range(-0.5..0.5);
        let new_angle = current_angle + angle_change;
        (new_angle.cos(), new_angle.sin(), false)
    }

    /// Find the best prey target for hunting
    fn find_best_target<'a>(&self, nearby: &'a [Creature]) -> Option<&'a Creature> {
        nearby.iter()
            .filter(|c| !c.is_predator)
            .min_by_key(|prey| {
                let dist = self.distance_to(prey);
                let stamina_factor = (prey.stamina / 10.0) as i32;
                let isolation_factor = nearby.iter()
                    .filter(|c| !c.is_predator && 
                           prey.distance_to(c) < HERD_RANGE)
                    .count() as i32;
                
                ((dist * 10.0) as i32) + stamina_factor + isolation_factor * 5
            })
    }

    /// Update the state of the creature based on surroundings
    fn update_state(&mut self, nearby: &[Creature]) {
        self.state = if self.is_predator {
            if self.stamina < STAMINA_MAX * 0.2 {
                CreatureState::Resting
            } else if let Some(target) = self.find_best_target(nearby) {
                let dist = self.distance_to(target);
                if dist < VISION_RANGE * 0.3 {
                    CreatureState::Hunting
                } else if dist < VISION_RANGE {
                    CreatureState::Stalking
                } else {
                    CreatureState::Grouping
                }
            } else {
                CreatureState::Grouping
            }
        } else {
            let predator_count = nearby.iter()
                .filter(|c| c.is_predator && self.distance_to(c) < VISION_RANGE)
                .count();

            if predator_count > 1 {
                CreatureState::Scattering
            } else if predator_count == 1 {
                CreatureState::Fleeing
            } else if self.stamina < STAMINA_MAX * 0.3 {
                CreatureState::Resting
            } else {
                CreatureState::Grazing
            }
        };
    }

    fn calculate_group_movement(&self, nearby: &[Creature]) -> (f64, f64, bool) {
        let same_type = nearby.iter()
            .filter(|c| c.is_predator == self.is_predator)
            .collect::<Vec<_>>();

        let range = if self.is_predator { PACK_RANGE } else { HERD_RANGE };

        // Calculate separation
        let mut sep_x = 0.0;
        let mut sep_y = 0.0;
        let mut sep_count = 0;

        // Calculate alignment
        let mut align_x = 0.0;
        let mut align_y = 0.0;
        let mut align_count = 0;

        for other in &same_type {
            let dist = self.distance_to(other);
            if dist < range {
                // Separation
                if dist < range * 0.5 {
                    let factor = 1.0 - (dist / (range * 0.5));
                    sep_x += (self.x - other.x) * factor;
                    sep_y += (self.y - other.y) * factor;
                    sep_count += 1;
                }

                // Alignment
                align_x += other.dx;
                align_y += other.dy;
                align_count += 1;
            }
        }

        // Calculate final movement vector
        let mut target_dx = 0.0;
        let mut target_dy = 0.0;

        // Apply separation
        if sep_count > 0 {
            let sep_mag = (sep_x * sep_x + sep_y * sep_y).sqrt();
            if sep_mag > 0.0 {
                target_dx += sep_x / sep_mag;
                target_dy += sep_y / sep_mag;
            }
        }

        // Apply alignment
        if align_count > 0 {
            let align_mag = (align_x * align_x + align_y * align_y).sqrt();
            if align_mag > 0.0 {
                target_dx += (align_x / align_mag) * HERD_ALIGNMENT_FACTOR;
                target_dy += (align_y / align_mag) * HERD_ALIGNMENT_FACTOR;
            }
        }

        // Normalize final vector
        let mag = (target_dx * target_dx + target_dy * target_dy).sqrt();
        if mag > 0.0 {
            (target_dx / mag, target_dy / mag, false)
        } else {
            self.calculate_wander_movement()
        }
    }

    fn calculate_search_movement(&self) -> (f64, f64, bool) {
        let mut rng = rand::thread_rng();
        let angle = rng.gen_range(0.0..std::f64::consts::PI * 2.0);
        (angle.cos(), angle.sin(), false)
    }
}

//////////////////////////////////////////////////////////////////////////////
// World Implementation
//////////////////////////////////////////////////////////////////////////////

/// Represents the simulation world containing all entities
struct World {
    width: f64,
    height: f64,
    creatures: Vec<Creature>,
    food: Vec<Food>,
}

impl World {
    /// Creates a new world with initial populations
    fn new(width: f64, height: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut creatures = Vec::new();
        let mut food = Vec::new();

        // Initialize predators
        for _ in 0..INITIAL_PREDATORS {
            creatures.push(Creature::new(
                rng.gen_range(0.0..width),
                rng.gen_range(0.0..height),
                true,
            ));
        }

        // Initialize prey
        for _ in 0..INITIAL_PREY {
            creatures.push(Creature::new(
                rng.gen_range(0.0..width),
                rng.gen_range(0.0..height),
                false,
            ));
        }

        // Initialize food
        for _ in 0..FOOD_COUNT {
            food.push(Food::new(width, height));
        }

        World {
            width,
            height,
            creatures,
            food,
        }
    }

    /// Updates the world state for one time step
    fn update(&mut self) {
        let creatures_clone = self.creatures.clone();
        let food_clone = self.food.clone();
        
        // Update all creatures
        for creature in &mut self.creatures {
            creature.update_movement(&creatures_clone, &food_clone, self.width, self.height);
        }

        // Handle interactions and cleanup
        let mut dead_creatures = Vec::new();
        let new_creatures = Vec::new();
        let mut energy_transfers = Vec::new();

        // Process interactions
        let creatures_snapshot: Vec<_> = self.creatures.iter().enumerate().collect();
        for (i, creature) in creatures_snapshot.iter() {
            if creature.is_predator {
                for (j, prey) in creatures_snapshot.iter() {
                    if !prey.is_predator && creature.distance_to(prey) < EATING_RANGE {
                        energy_transfers.push((*i, EATING_GAIN));
                        dead_creatures.push(*j);
                    }
                }
            }
        }

        // Apply changes
        dead_creatures.sort_unstable();
        dead_creatures.dedup();
        for &index in dead_creatures.iter().rev() {
            self.creatures.swap_remove(index);
        }

        // Add new creatures
        self.creatures.extend(new_creatures);

        // Maintain food supply
        self.replenish_food();

        // Population control
        self.balance_population();
    }

    fn replenish_food(&mut self) {
        let mut rng = rand::thread_rng();
        
        // Remove eaten food
        self.food.retain(|f| !f.eaten);
        
        // Add new food to maintain constant supply
        while self.food.len() < FOOD_COUNT {
            self.food.push(Food::new(
                rng.gen_range(0.0..self.width),
                rng.gen_range(0.0..self.height)
            ));
        }
    }

    fn balance_population(&mut self) {
        let predator_count = self.creatures.iter()
            .filter(|c| c.is_predator)
            .count();
        let prey_count = self.creatures.iter()
            .filter(|c| !c.is_predator)
            .count();

        // Maintain minimum populations
        if predator_count < INITIAL_PREDATORS / 2 {
            let mut rng = rand::thread_rng();
            self.creatures.push(Creature::new(
                rng.gen_range(0.0..self.width),
                rng.gen_range(0.0..self.height),
                true
            ));
        }

        if prey_count < INITIAL_PREY / 2 {
            let mut rng = rand::thread_rng();
            self.creatures.push(Creature::new(
                rng.gen_range(0.0..self.width),
                rng.gen_range(0.0..self.height),
                false
            ));
        }

        // Limit maximum population
        const MAX_CREATURES: usize = 300;
        if self.creatures.len() > MAX_CREATURES {
            self.creatures.truncate(MAX_CREATURES);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Game State Implementation
//////////////////////////////////////////////////////////////////////////////

/// Manages the game state and rendering
struct GameState {
    world: World,
}

impl GameState {
    /// Creates a new game state
    fn new() -> GameResult<GameState> {
        let world = World::new(1920.0, 1080.0);
        Ok(GameState { world })
    }
}

impl event::EventHandler<ggez::GameError> for GameState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        self.world.update();
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::from_rgb(10, 10, 20));

        // Draw food with glow effect
        for food in &self.world.food {
            // Glow effect
            let glow = Mesh::new_circle(
                ctx,
                DrawMode::fill(),
                [food.x as f32, food.y as f32],
                4.0,
                0.1,
                Color::new(1.0, 1.0, 0.0, 0.2),
            )?;
            canvas.draw(&glow, graphics::DrawParam::default());

            // Food particle
            let circle = Mesh::new_circle(
                ctx,
                DrawMode::fill(),
                [food.x as f32, food.y as f32],
                2.0,
                0.1,
                Color::YELLOW,
            )?;
            canvas.draw(&circle, graphics::DrawParam::default());
        }

        // Draw creatures with enhanced visuals
        for creature in &self.world.creatures {
            let base_color = if creature.is_predator {
                Color::new(0.8, 0.2, 0.2, 1.0) // Red for predators
            } else {
                Color::new(0.2, 0.8, 0.2, 1.0) // Green for prey
            };

            // Energy indicator ring
            let energy_ratio = creature.energy / 100.0;
            let energy_color = Color::new(
                base_color.r,
                base_color.g,
                base_color.b,
                0.3 * energy_ratio as f32
            );
            let energy_ring = Mesh::new_circle(
                ctx,
                DrawMode::stroke(1.0),
                [creature.x as f32, creature.y as f32],
                creature.size as f32 * 6.0,
                0.1,
                energy_color,
            )?;
            canvas.draw(&energy_ring, graphics::DrawParam::default());

            // Stamina indicator
            let stamina_ratio = creature.stamina / STAMINA_MAX;
            let stamina_color = Color::new(1.0, 1.0, 1.0, stamina_ratio as f32);
            let stamina_bar = Mesh::new_rectangle(
                ctx,
                DrawMode::fill(),
                Rect::new(
                    (creature.x - creature.size) as f32,
                    (creature.y - creature.size * 1.5) as f32,
                    (creature.size * 2.0 * stamina_ratio) as f32,
                    2.0,
                ),
                stamina_color,
            )?;
            canvas.draw(&stamina_bar, graphics::DrawParam::default());

            // Direction indicator
            let speed = (creature.dx * creature.dx + creature.dy * creature.dy).sqrt();
            let direction_length = creature.size * 2.0 * (speed / creature.speed);
            let direction_line = Mesh::new_line(
                ctx,
                &[
                    [creature.x as f32, creature.y as f32],
                    [(creature.x + creature.dx * direction_length) as f32, 
                     (creature.y + creature.dy * direction_length) as f32],
                ],
                1.0,
                base_color,
            )?;
            canvas.draw(&direction_line, graphics::DrawParam::default());

            // Main body
            let body = Mesh::new_circle(
                ctx,
                DrawMode::fill(),
                [creature.x as f32, creature.y as f32],
                creature.size as f32 * 5.0,
                0.1,
                base_color,
            )?;
            canvas.draw(&body, graphics::DrawParam::default());

            // State indicator
            let state_color = match creature.state {
                CreatureState::Hunting => Color::RED,
                CreatureState::Stalking => Color::new(0.8, 0.4, 0.0, 1.0),
                CreatureState::Fleeing => Color::new(1.0, 0.6, 0.0, 1.0),
                CreatureState::Scattering => Color::new(1.0, 0.8, 0.0, 1.0),
                CreatureState::Grazing => Color::new(0.0, 0.8, 0.0, 1.0),
                CreatureState::Resting => Color::new(0.0, 0.6, 1.0, 1.0),
                CreatureState::Grouping => Color::new(0.6, 0.0, 1.0, 1.0),
            };
            let state_indicator = Mesh::new_circle(
                ctx,
                DrawMode::fill(),
                [creature.x as f32, creature.y as f32],
                2.0,
                0.1,
                state_color,
            )?;
            canvas.draw(&state_indicator, graphics::DrawParam::default());
        }

        canvas.finish(ctx)?;
        Ok(())
    }
}

//////////////////////////////////////////////////////////////////////////////
// Main Function
//////////////////////////////////////////////////////////////////////////////

fn main() -> GameResult {
    let cb = ggez::ContextBuilder::new("evolution_sim", "you")
        .window_setup(ggez::conf::WindowSetup::default().title("Evolution Simulation"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(1920.0, 1080.0));

    let (ctx, event_loop) = cb.build()?;
    let state = GameState::new()?;
    event::run(ctx, event_loop, state)
}
