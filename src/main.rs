use ggez::{
    event,
    graphics::{self, Color, DrawMode, Mesh, Rect},
    Context, GameResult,
};
use rand::Rng;
use std::vec::Vec;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::time::{Instant, Duration};

//////////////////////////////////////////////////////////////////////////////
// Simulation Constants
//////////////////////////////////////////////////////////////////////////////

/// Range at which creatures can detect others
const VISION_RANGE: f32 = 150.0;
/// Range at which predators can catch prey
const EATING_RANGE: f32 = 15.0;
/// Energy gained by predators when eating prey
const EATING_GAIN: f32 = 50.0;
/// Number of food particles in the world
const FOOD_COUNT: usize = 200;
/// Base energy consumption rate per update
const BASE_ENERGY_CONSUMPTION: f32 = 0.05;
/// Energy consumption multiplier based on speed
const SPEED_ENERGY_FACTOR: f32 = 0.1;
/// Absolute minimum separation multiplier based on combined sizes
const MIN_SEPARATION_FACTOR: f32 = 1.2; // Ensures entities never overlap based on their sizes
/// Preferred separation multiplier for same-type creatures
const PREFERRED_SEPARATION_FACTOR: f32 = 1.8;
/// Strong separation force for overlapping creatures
const SEPARATION_FORCE: f32 = 2.5;
/// Base collision avoidance force
const COLLISION_FORCE: f32 = 1.0;
/// Range for pack hunting coordination (smaller for tighter predator groups)
const PACK_RANGE: f32 = 40.0;
/// Range for prey herding (larger for bigger herds)
const HERD_RANGE: f32 = 150.0;
/// Factor for herd cohesion (adjusted for smoother movement)
const HERD_COHESION: f32 = 0.04;
/// Initial number of predators
const INITIAL_PREDATORS: usize = 20;
/// Initial number of prey
const INITIAL_PREY: usize = 150;
/// Minimum movement threshold to prevent stuttering
const MOVEMENT_THRESHOLD: f32 = 0.01;
/// How much prey align their movement
const HERD_ALIGNMENT_FACTOR: f32 = 0.3;
/// Range for coordinated attacks
const PACK_ATTACK_RANGE: f32 = 30.0;
/// Adjust constants for better chase dynamics
const PREDATOR_BASE_SPEED: f32 = 3.2;     // Even faster base speed
const CHASE_PREDICTION: f32 = 0.5;        // Better prediction
/// Adjust constants for more dynamic movement
const PREY_BASE_SPEED: f32 = 2.8;         // Decent base speed for prey
const PREDATOR_SPRINT: f32 = 4.8;         // Very fast sprint for short chases
const PREY_SPRINT: f32 = 4.2;             // Fast escape speed
const ACCELERATION: f32 = 0.15;           // How quickly speed changes
const INERTIA: f32 = 0.8;                 // Movement smoothing
const STAMINA_MAX: f32 = 100.0;
const STAMINA_RECOVERY: f32 = 0.2;
const SPRINT_COST: f32 = 0.5;
const FOOD_ENERGY: f32 = 25.0;
/// Maximum number of kill feed items
const MAX_KILL_FEED_ITEMS: usize = 5;
/// Duration for kill feed items
const KILL_FEED_DURATION: Duration = Duration::from_secs(5);
/// Duration for fade effect of kill feed items
const KILL_FEED_FADE_DURATION: Duration = Duration::from_secs(1);

//////////////////////////////////////////////////////////////////////////////
// Food Implementation
//////////////////////////////////////////////////////////////////////////////

/// Represents a food particle in the simulation
#[derive(Clone, Debug)]
struct Food {
    x: f32,
    y: f32,
    eaten: bool,
    energy: f32,
}

impl Food {
    /// Creates a new food particle at random coordinates
    fn new(width: f32, height: f32) -> Self {
        let mut rng = rand::thread_rng();
        Food {
            x: rng.gen_range(0.0..width),
            y: rng.gen_range(0.0..height),
            eaten: false,
            energy: FOOD_ENERGY,
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
    x: f32,
    y: f32,
    speed: f32,
    size: f32,
    energy: f32,
    stamina: f32,
    is_predator: bool,
    dx: f32,
    dy: f32,
    state: CreatureState,
    #[allow(dead_code)]
    last_state_change: f32,    // Time tracking for state changes
    #[allow(dead_code)]
    group_id: Option<usize>,   // For pack/herd coordination
    #[allow(dead_code)]
    target_id: Option<usize>,  // Current chase/flee target
}


impl Creature {
    /// Creates a new creature with randomized traits
    fn new(x: f32, y: f32, is_predator: bool) -> Self {
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

    #[inline]
    fn distance_to(&self, other: &Creature) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    #[inline]
    fn min_separation_distance(&self, other: &Creature) -> f32 {
        (self.size + other.size) * MIN_SEPARATION_FACTOR * 5.0
    }

    #[inline]
    fn preferred_separation_distance(&self, other: &Creature) -> f32 {
        (self.size + other.size) * PREFERRED_SEPARATION_FACTOR * 5.0
    }

    /// Enhanced collision avoidance with size-based distances
    fn calculate_collision_avoidance(&self, nearby: &[&Creature]) -> (f32, f32) {
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

    /// Updates creature movement based on surroundings
    fn update_movement(&mut self, nearby: &[&Creature], food: &[Food], width: f32, height: f32) {
        // Update stamina and energy
        self.stamina = (self.stamina + STAMINA_RECOVERY * 
            if self.state == CreatureState::Resting { 2.0 } else { 1.0 })
            .min(STAMINA_MAX);
        
        // Calculate movement vector using SIMD when possible
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

        // Optimize collision avoidance
        let (avoid_x, avoid_y) = self.calculate_collision_avoidance(nearby);
        target_dx += avoid_x;
        target_dy += avoid_y;

        // Fast vector normalization
        let magnitude_sq = target_dx * target_dx + target_dy * target_dy;
        if magnitude_sq > MOVEMENT_THRESHOLD * MOVEMENT_THRESHOLD {
            let inv_magnitude = 1.0 / magnitude_sq.sqrt();
            target_dx *= inv_magnitude;
            target_dy *= inv_magnitude;

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

        // Smooth acceleration
        self.dx = self.dx * INERTIA + target_dx * ACCELERATION;
        self.dy = self.dy * INERTIA + target_dy * ACCELERATION;

        // Update position with boundary wrapping
        self.x = (self.x + self.dx + width) % width;
        self.y = (self.y + self.dy + height) % height;

        // Energy consumption
        let speed_sq = self.dx * self.dx + self.dy * self.dy;
        self.energy -= BASE_ENERGY_CONSUMPTION + speed_sq.sqrt() * SPEED_ENERGY_FACTOR;

        // Update state
        self.update_state(nearby);
    }

    /// Calculate predator movement with improved hunting behavior
    #[inline]
    fn calculate_hunting_movement(&self, nearby: &[&Creature]) -> (f32, f32, bool) {
        let pack_members = nearby.iter()
            .filter(|c| c.is_predator && self.distance_to(c) < PACK_RANGE)
            .count();

        if let Some(target) = self.find_best_target(nearby) {
            let dist = self.distance_to(target);
            if dist < VISION_RANGE {
                let target_speed_sq = target.dx * target.dx + target.dy * target.dy;
                let target_speed = target_speed_sq.sqrt();

                // Optimized prediction calculation
                let prediction_time = if target_speed > 0.1 {
                    (dist * CHASE_PREDICTION).min(dist / target_speed)
                } else {
                    0.0
                };

                let intercept_x = target.x + target.dx * prediction_time;
                let intercept_y = target.y + target.dy * prediction_time;

                // Pack hunting optimization
                if pack_members >= 2 {
                    self.calculate_pack_intercept(target, pack_members)
                } else {
                    let chase_dx = intercept_x - self.x;
                    let chase_dy = intercept_y - self.y;
                    let chase_dist_sq = chase_dx * chase_dx + chase_dy * chase_dy;
                    let chase_dist = chase_dist_sq.sqrt();
                    
                    if chase_dist > 0.0 {
                        (chase_dx / chase_dist, chase_dy / chase_dist, dist < VISION_RANGE * 0.5)
                    } else {
                        (0.0, 0.0, false)
                    }
                }
            } else {
                self.calculate_search_movement()
            }
        } else {
            self.calculate_search_movement()
        }
    }

    /// Calculate prey movement with improved evasion behavior
    #[inline]
    fn calculate_fleeing_movement(&self, nearby: &[&Creature]) -> (f32, f32, bool) {
        let mut escape_x = 0.0;
        let mut escape_y = 0.0;
        let mut threat_level = 0.0;
        let mut nearest_predator_dist_sq = f32::MAX;

        // SIMD-friendly batch processing of nearby predators
        for predator in nearby.iter().filter(|c| c.is_predator) {
            let dx = self.x - predator.x;
            let dy = self.y - predator.y;
            let dist_sq = dx * dx + dy * dy;
            
            if dist_sq < VISION_RANGE * VISION_RANGE {
                let dist = dist_sq.sqrt();
                let weight = (1.0 - (dist / VISION_RANGE)).powi(2);
                
                // Vectorized escape calculation
                escape_x += dx * weight;
                escape_y += dy * weight;
                threat_level += weight;
                nearest_predator_dist_sq = nearest_predator_dist_sq.min(dist_sq);
            }
        }

        if threat_level > 0.0 {
            let mag_sq = escape_x * escape_x + escape_y * escape_y;
            if mag_sq > 0.0 {
                let inv_mag = 1.0 / mag_sq.sqrt();
                let should_sprint = nearest_predator_dist_sq < VISION_RANGE * VISION_RANGE * 0.16; // 0.4^2
                (escape_x * inv_mag, escape_y * inv_mag, should_sprint)
            } else {
                (rand::random::<f32>() * 2.0 - 1.0, 
                 rand::random::<f32>() * 2.0 - 1.0, 
                 true)
            }
        } else {
            (self.dx, self.dy, false)
        }
    }

    /// Calculate prey movement with improved evasion behavior
    #[inline]
    fn calculate_grazing_movement(&self, nearby: &[&Creature], food: &[Food]) -> (f32, f32, bool) {
        // Check for predators first (early warning system)
        for predator in nearby.iter().filter(|c| c.is_predator) {
            let dx = predator.x - self.x;
            let dy = predator.y - self.y;
            if dx * dx + dy * dy < VISION_RANGE * VISION_RANGE * 1.44 { // 1.2^2
                return self.calculate_fleeing_movement(nearby);
            }
        }

        // Find nearest food using squared distances
        let mut nearest_food_dist_sq = f32::MAX;
        let mut nearest_food_dx = 0.0;
        let mut nearest_food_dy = 0.0;
        let vision_range_sq = (VISION_RANGE * 0.5).powi(2);

        for food in food.iter().filter(|f| !f.eaten) {
            let dx = food.x - self.x;
            let dy = food.y - self.y;
            let dist_sq = dx * dx + dy * dy;
            
            if dist_sq < vision_range_sq && dist_sq < nearest_food_dist_sq {
                nearest_food_dist_sq = dist_sq;
                nearest_food_dx = dx;
                nearest_food_dy = dy;
            }
        }

        let mut target_dx = 0.0;
        let mut target_dy = 0.0;
        let mut priority = 0;

        // Apply food attraction if found
        if nearest_food_dist_sq < vision_range_sq {
            let food_weight = 1.0 - (nearest_food_dist_sq.sqrt() / (VISION_RANGE * 0.5));
            target_dx += nearest_food_dx * food_weight;
            target_dy += nearest_food_dy * food_weight;
            priority = 1;
        }

        // Calculate and apply herd center influence
        if priority == 0 {
            let mut center_x = 0.0;
            let mut center_y = 0.0;
            let mut count = 0;
            let herd_range_sq = HERD_RANGE * HERD_RANGE;

            for other in nearby.iter().filter(|c| !c.is_predator) {
                let dx = other.x - self.x;
                let dy = other.y - self.y;
                let dist_sq = dx * dx + dy * dy;
                
                if dist_sq < herd_range_sq {
                    center_x += other.x;
                    center_y += other.y;
                    count += 1;
                }
            }

            if count > 0 {
                target_dx += (center_x / count as f32 - self.x) * HERD_COHESION;
                target_dy += (center_y / count as f32 - self.y) * HERD_COHESION;
            }
        }

        let mag_sq = target_dx * target_dx + target_dy * target_dy;
        if mag_sq > 0.0 {
            let inv_mag = 1.0 / mag_sq.sqrt();
            (target_dx * inv_mag, target_dy * inv_mag, false)
        } else {
            self.calculate_wander_movement()
        }
    }

    /// Calculate prey movement with improved evasion behavior
    fn calculate_wander_movement(&self) -> (f32, f32, bool) {
        let mut rng = rand::thread_rng();
        let current_angle = self.dy.atan2(self.dx);
        let angle_change = rng.gen_range(-0.5..0.5);
        let new_angle = current_angle + angle_change;
        (new_angle.cos(), new_angle.sin(), false)
    }

    /// Find the best prey target for hunting
    #[inline]
    fn find_best_target<'a>(&self, nearby: &'a [&'a Creature]) -> Option<&'a Creature> {
        nearby.iter()
            .filter(|c| !c.is_predator)
            .min_by_key(|prey| {
                let dx = prey.x - self.x;
                let dy = prey.y - self.y;
                let dist_sq = dx * dx + dy * dy;
                let stamina_factor = (prey.stamina / 10.0) as i32;
                let isolation_factor = nearby.iter()
                    .filter(|c| !c.is_predator && {
                        let dx = prey.x - c.x;
                        let dy = prey.y - c.y;
                        dx * dx + dy * dy < HERD_RANGE * HERD_RANGE
                    })
                    .count() as i32;
                
                ((dist_sq.sqrt() * 10.0) as i32) + stamina_factor + isolation_factor * 5
            })
            .copied()
    }

    /// Update the state of the creature based on surroundings
    fn update_state(&mut self, nearby: &[&Creature]) {
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

    #[inline]
    fn calculate_group_movement(&self, nearby: &[&Creature]) -> (f32, f32, bool) {
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
            let dx = other.x - self.x;
            let dy = other.y - self.y;
            let dist_sq = dx * dx + dy * dy;
            let range_sq = range * range;
            
            if dist_sq < range_sq {
                let dist = dist_sq.sqrt();
                // Separation
                if dist_sq < range_sq * 0.25 {  // Using squared distance comparison
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
            let sep_mag_sq = sep_x * sep_x + sep_y * sep_y;
            if sep_mag_sq > 0.0 {
                let inv_sep_mag = 1.0 / sep_mag_sq.sqrt();
                target_dx += sep_x * inv_sep_mag;
                target_dy += sep_y * inv_sep_mag;
            }
        }

        // Apply alignment
        if align_count > 0 {
            let align_mag_sq = align_x * align_x + align_y * align_y;
            if align_mag_sq > 0.0 {
                let inv_align_mag = 1.0 / align_mag_sq.sqrt();
                target_dx += (align_x * inv_align_mag) * HERD_ALIGNMENT_FACTOR;
                target_dy += (align_y * inv_align_mag) * HERD_ALIGNMENT_FACTOR;
            }
        }

        // Normalize final vector
        let mag_sq = target_dx * target_dx + target_dy * target_dy;
        if mag_sq > 0.0 {
            let inv_mag = 1.0 / mag_sq.sqrt();
            (target_dx * inv_mag, target_dy * inv_mag, false)
        } else {
            self.calculate_wander_movement()
        }
    }

    fn calculate_search_movement(&self) -> (f32, f32, bool) {
        let mut rng = rand::thread_rng();
        let angle = rng.gen_range(0.0..std::f32::consts::PI * 2.0);
        (angle.cos(), angle.sin(), false)
    }

    #[inline]
    fn calculate_pack_intercept(&self, target: &Creature, pack_members: usize) -> (f32, f32, bool) {
        let angle_offset = std::f32::consts::TAU / pack_members as f32;
        let dx = target.x - self.x;
        let dy = target.y - self.y;
        let base_angle = dy.atan2(dx);
        let my_angle = base_angle + angle_offset * (pack_members as f32 / 2.0);

        let surround_x = target.x + my_angle.cos() * PACK_ATTACK_RANGE;
        let surround_y = target.y + my_angle.sin() * PACK_ATTACK_RANGE;

        let chase_dx = surround_x - self.x;
        let chase_dy = surround_y - self.y;
        let chase_dist_sq = chase_dx * chase_dx + chase_dy * chase_dy;
        
        if chase_dist_sq > 0.0 {
            let chase_dist = chase_dist_sq.sqrt();
            (chase_dx / chase_dist, chase_dy / chase_dist, true)
        } else {
            (0.0, 0.0, false)
        }
    }

    // Add this method to handle food consumption
    #[inline]
    fn try_eat_food(&mut self, food: &mut [Food]) -> bool {
        if self.is_predator {
            return false;
        }

        for f in food.iter_mut() {
            if !f.eaten {
                let dx = f.x - self.x;
                let dy = f.y - self.y;
                let dist_sq = dx * dx + dy * dy;
                
                if dist_sq < EATING_RANGE * EATING_RANGE {
                    f.eaten = true;
                    self.energy += f.energy;
                    return true;
                }
            }
        }
        false
    }

    // Add reproduction check
    #[inline]
    fn should_reproduce(&self) -> bool {
        !self.is_predator && self.energy > 150.0  // Reproduction threshold
    }
}

//////////////////////////////////////////////////////////////////////////////
// World Implementation
//////////////////////////////////////////////////////////////////////////////

/// Represents the simulation world containing all entities
#[derive(Clone, Debug)]
struct Position {
    x: f32,
    y: f32,
}

/// Add these optimized spatial partitioning structures
#[derive(Clone, Debug)]
struct SpatialCell {
    creatures: Vec<usize>,
    predators: u8,
    prey: u8,
}

impl SpatialCell {
    fn new() -> Self {
        SpatialCell {
            creatures: Vec::with_capacity(8),
            predators: 0,
            prey: 0,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.creatures.clear();
        self.predators = 0;
        self.prey = 0;
    }
}

// Optimize Grid implementation
struct Grid {
    cells: Vec<SpatialCell>,
    #[allow(dead_code)]
    cell_size: f32,
    width: usize,
    height: usize,
    inv_cell_size: f32,
}

impl Grid {
    fn new(world_width: f32, world_height: f32, cell_size: f32) -> Self {
        let width = (world_width / cell_size).ceil() as usize;
        let height = (world_height / cell_size).ceil() as usize;
        let cells = (0..width * height)
            .map(|_| SpatialCell::new())
            .collect();
        
        Grid {
            cells,
            cell_size,
            width,
            height,
            inv_cell_size: 1.0 / cell_size,
        }
    }

    #[inline]
    fn get_cell_coords(&self, x: f32, y: f32) -> (usize, usize) {
        let cell_x = (x * self.inv_cell_size) as usize;
        let cell_y = (y * self.inv_cell_size) as usize;
        (cell_x.min(self.width - 1), cell_y.min(self.height - 1))
    }

    fn clear(&mut self) {
        self.cells.par_iter_mut().for_each(|cell| cell.clear());
    }

    fn insert(&mut self, creature: &Creature, index: usize) {
        let (cell_x, cell_y) = self.get_cell_coords(creature.x, creature.y);
        let cell_index = cell_y * self.width + cell_x;
        
        if let Some(cell) = self.cells.get_mut(cell_index) {
            cell.creatures.push(index);
            if creature.is_predator {
                cell.predators += 1;
            } else {
                cell.prey += 1;
            }
        }
    }

    fn get_nearby_indices(&self, pos: &Position, range: f32) -> Vec<usize> {
        let cell_range = (range * self.inv_cell_size).ceil() as i32;
        let (center_x, center_y) = self.get_cell_coords(pos.x, pos.y);
        let mut indices = Vec::with_capacity(32);

        let start_x = (center_x as i32 - cell_range).max(0) as usize;
        let end_x = (center_x as i32 + cell_range + 1).min(self.width as i32) as usize;
        let start_y = (center_y as i32 - cell_range).max(0) as usize;
        let end_y = (center_y as i32 + cell_range + 1).min(self.height as i32) as usize;

        for y in start_y..end_y {
            for x in start_x..end_x {
                let cell_index = y * self.width + x;
                indices.extend(&self.cells[cell_index].creatures);
            }
        }

        indices
    }

}

/// Add these structures for efficient kill feed tracking
#[derive(Clone, Debug)]
struct KillFeedItem {
    predator_id: usize,
    prey_id: usize,
    timestamp: Instant,
    #[allow(dead_code)]
    position: Position,
}

#[derive(Clone, Debug)]
struct ScoreBoard {
    kill_feed: VecDeque<KillFeedItem>,
    total_kills: usize,
    predator_count: usize,
    prey_count: usize,
}

impl ScoreBoard {
    fn new() -> Self {
        ScoreBoard {
            kill_feed: VecDeque::with_capacity(MAX_KILL_FEED_ITEMS),
            total_kills: 0,
            predator_count: INITIAL_PREDATORS,
            prey_count: INITIAL_PREY,
        }
    }

    #[inline]
    fn add_kill(&mut self, predator_id: usize, prey_id: usize, position: Position) {
        let item = KillFeedItem {
            predator_id,
            prey_id,
            timestamp: Instant::now(),
            position,
        };

        if self.kill_feed.len() >= MAX_KILL_FEED_ITEMS {
            self.kill_feed.pop_front();
        }
        self.kill_feed.push_back(item);
        self.total_kills += 1;
    }

    #[inline]
    fn update_counts(&mut self, predators: usize, prey: usize) {
        self.predator_count = predators;
        self.prey_count = prey;
    }

    fn cleanup_old_kills(&mut self) {
        let now = Instant::now();
        while let Some(item) = self.kill_feed.front() {
            if now.duration_since(item.timestamp) > KILL_FEED_DURATION {
                self.kill_feed.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Represents the simulation world containing all entities
struct World {
    width: f32,
    height: f32,
    creatures: Vec<Creature>,
    food: Vec<Food>,
    spatial_grid: Grid,
    scoreboard: ScoreBoard,
}

impl World {
    /// Creates a new world with initial populations
    fn new(width: f32, height: f32) -> Self {
        let mut rng = rand::thread_rng();
        let mut creatures = Vec::with_capacity(INITIAL_PREDATORS + INITIAL_PREY);

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

        // Initialize food with pre-allocated capacity
        let mut food = Vec::with_capacity(FOOD_COUNT);
        for _ in 0..FOOD_COUNT {
            food.push(Food::new(width, height));
        }

        let spatial_grid = Grid::new(width, height, VISION_RANGE);
        World {
            width,
            height,
            creatures,
            food,
            spatial_grid,
            scoreboard: ScoreBoard::new(),
        }
    }

    /// Updates the world state for one time step
    fn update(&mut self) {
        // Update spatial grid in parallel
        self.spatial_grid.clear();
        
        // Batch insert creatures into grid
        for (i, creature) in self.creatures.iter().enumerate() {
            self.spatial_grid.insert(creature, i);
        }

        // Create thread-safe references
        let spatial_grid = &self.spatial_grid;
        let food = &self.food;
        let width = self.width;
        let height = self.height;
        let creatures = &self.creatures;

        // Parallel creature updates with pre-allocated vectors
        let updates: Vec<_> = (0..self.creatures.len())
            .into_par_iter()
            .map(|i| {
                let creature = &creatures[i];
                let nearby_indices = spatial_grid.get_nearby_indices(
                    &Position { x: creature.x, y: creature.y },
                    VISION_RANGE
                );

                let nearby: Vec<&Creature> = nearby_indices.iter()
                    .filter(|&&idx| idx != i)
                    .map(|&idx| &creatures[idx])
                    .collect();

                let mut updated = creature.clone();
                updated.update_movement(&nearby, food, width, height);
                (i, updated)
            })
            .collect();

        // Apply updates in order
        for (idx, updated) in updates {
            self.creatures[idx] = updated;
        }

        // Handle interactions and cleanup
        self.handle_interactions();
        self.replenish_food();
        self.balance_population();
    }

    fn handle_interactions(&mut self) {
        let mut dead_creatures = Vec::new();
        let mut energy_transfers = Vec::new();
        let mut new_kills = Vec::new();
        let mut new_creatures = Vec::new();

        // Handle food consumption and reproduction for prey
        for (_idx, creature) in self.creatures.iter_mut().enumerate() {
            if !creature.is_predator {
                // Try to eat food
                if creature.try_eat_food(&mut self.food) {
                    // Check for reproduction after eating
                    if creature.should_reproduce() {
                        let mut rng = rand::thread_rng();
                        // Create new creature with random offset
                        let offset_x = rng.gen_range(-20.0..20.0);
                        let offset_y = rng.gen_range(-20.0..20.0);
                        let mut new_creature = Creature::new(
                            (creature.x + offset_x + self.width) % self.width,
                            (creature.y + offset_y + self.height) % self.height,
                            false
                        );
                        // Split energy between parent and child
                        new_creature.energy = creature.energy * 0.4;
                        creature.energy *= 0.6;
                        new_creatures.push(new_creature);
                    }
                }
            }
        }

        // Process predator-prey interactions
        let interactions: Vec<_> = self.creatures.iter().enumerate()
            .filter(|(_, creature)| creature.is_predator)
            .flat_map(|(pred_idx, predator)| {
                self.creatures.iter().enumerate()
                    .filter(|(_, prey)| !prey.is_predator)
                    .filter(|(_, prey)| predator.distance_to(prey) < EATING_RANGE)
                    .map(move |(prey_idx, _)| (pred_idx, prey_idx))
            })
            .collect();

        // Process the kills
        for (predator_idx, prey_idx) in interactions {
            energy_transfers.push((predator_idx, EATING_GAIN));
            dead_creatures.push(prey_idx);
            
            let prey = &self.creatures[prey_idx];
            new_kills.push((
                predator_idx,
                prey_idx,
                Position {
                    x: prey.x,
                    y: prey.y,
                }
            ));
        }

        // Apply energy transfers
        for (idx, energy) in energy_transfers {
            if idx < self.creatures.len() {
                self.creatures[idx].energy += energy;
            }
        }

        // Remove dead creatures
        for idx in dead_creatures {
            self.creatures.swap_remove(idx);
        }

        // Add new creatures from reproduction
        for new_creature in new_creatures {
            if let Some(idx) = self.creatures.iter().position(|c| c.is_predator != new_creature.is_predator) {
                self.creatures.insert(idx, new_creature);
            }
        }

        // Update scoreboard
        for (predator_id, prey_id, position) in new_kills {
            self.scoreboard.add_kill(predator_id, prey_id, position);
        }

        // Update population counts
        let predator_count = self.creatures.iter()
            .filter(|c| c.is_predator)
            .count();
        let prey_count = self.creatures.len() - predator_count;
        self.scoreboard.update_counts(predator_count, prey_count);
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
        self.world.scoreboard.cleanup_old_kills();
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

        // Draw scoreboard with enhanced visuals
        let score_bg = Mesh::new_rectangle(
            ctx,
            DrawMode::fill(),
            Rect::new(5.0, 5.0, 200.0, 80.0),
            Color::new(0.1, 0.1, 0.2, 0.8),
        )?;
        canvas.draw(&score_bg, graphics::DrawParam::default());

        let score_border = Mesh::new_rectangle(
            ctx,
            DrawMode::stroke(2.0),
            Rect::new(5.0, 5.0, 200.0, 80.0),
            Color::new(0.3, 0.3, 0.4, 1.0),
        )?;
        canvas.draw(&score_border, graphics::DrawParam::default());

        // Draw stats with different colors and spacing
        let pred_text = graphics::Text::new(format!("Predators: {}", self.world.scoreboard.predator_count));
        let prey_text = graphics::Text::new(format!("Prey: {}", self.world.scoreboard.prey_count));
        let kills_text = graphics::Text::new(format!("Total Kills: {}", self.world.scoreboard.total_kills));

        canvas.draw(&pred_text, graphics::DrawParam::default()
            .dest([15.0, 15.0])
            .color(Color::new(1.0, 0.4, 0.4, 1.0)));
        canvas.draw(&prey_text, graphics::DrawParam::default()
            .dest([15.0, 40.0])
            .color(Color::new(0.4, 1.0, 0.4, 1.0)));
        canvas.draw(&kills_text, graphics::DrawParam::default()
            .dest([15.0, 65.0])
            .color(Color::WHITE));

        // Draw kill feed with enhanced visuals
        let now = Instant::now();
        let feed_start_y = 100.0;
        
        // Kill feed background
        if !self.world.scoreboard.kill_feed.is_empty() {
            let feed_height = self.world.scoreboard.kill_feed.len() as f32 * 25.0 + 10.0;
            let feed_bg = Mesh::new_rectangle(
                ctx,
                DrawMode::fill(),
                Rect::new(5.0, feed_start_y, 200.0, feed_height),
                Color::new(0.1, 0.1, 0.2, 0.6),
            )?;
            canvas.draw(&feed_bg, graphics::DrawParam::default());
        }

        for (i, kill) in self.world.scoreboard.kill_feed.iter().enumerate() {
            let age = now.duration_since(kill.timestamp);
            if age <= KILL_FEED_DURATION {
                let alpha = if age > KILL_FEED_FADE_DURATION {
                    1.0 - (age.as_secs_f32() - KILL_FEED_FADE_DURATION.as_secs_f32()) 
                        / (KILL_FEED_DURATION.as_secs_f32() - KILL_FEED_FADE_DURATION.as_secs_f32())
                } else {
                    1.0
                };

                let kill_text = graphics::Text::new(format!(
                    "Predator #{} → Prey #{}",
                    kill.predator_id, kill.prey_id
                ));

                canvas.draw(
                    &kill_text,
                    graphics::DrawParam::default()
                        .dest([15.0, feed_start_y + 10.0 + i as f32 * 25.0])
                        .color(Color::new(1.0, 1.0, 1.0, alpha))
                );
            }
        }

        // Draw eating range indicators for predators
        for creature in &self.world.creatures {
            if creature.is_predator {
                // Draw eating range circle
                let circle = Mesh::new_circle(
                    ctx,
                    DrawMode::stroke(1.0),
                    [creature.x, creature.y],
                    EATING_RANGE,
                    0.1,
                    Color::new(1.0, 0.0, 0.0, 0.2),
                )?;
                canvas.draw(&circle, graphics::DrawParam::default());
            }
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
