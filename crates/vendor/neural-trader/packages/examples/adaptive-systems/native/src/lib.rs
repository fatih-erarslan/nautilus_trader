// NAPI-RS implementation placeholder for adaptive systems
// Future optimizations for boids, ACO, and cellular automata

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct Vector2D {
  pub x: f64,
  pub y: f64,
}

#[napi]
impl Vector2D {
  #[napi(constructor)]
  pub fn new(x: f64, y: f64) -> Self {
    Vector2D { x, y }
  }

  #[napi]
  pub fn magnitude(&self) -> f64 {
    (self.x * self.x + self.y * self.y).sqrt()
  }

  #[napi]
  pub fn normalize(&self) -> Vector2D {
    let mag = self.magnitude();
    if mag == 0.0 {
      Vector2D { x: 0.0, y: 0.0 }
    } else {
      Vector2D {
        x: self.x / mag,
        y: self.y / mag,
      }
    }
  }

  #[napi]
  pub fn distance(&self, other: &Vector2D) -> f64 {
    let dx = other.x - self.x;
    let dy = other.y - self.y;
    (dx * dx + dy * dy).sqrt()
  }
}

#[napi]
pub fn calculate_separation(
  position: &Vector2D,
  neighbors: Vec<&Vector2D>,
  separation_radius: f64,
) -> Vector2D {
  let mut steer = Vector2D { x: 0.0, y: 0.0 };
  let mut count = 0;

  for neighbor in neighbors {
    let distance = position.distance(neighbor);

    if distance > 0.0 && distance < separation_radius {
      let dx = position.x - neighbor.x;
      let dy = position.y - neighbor.y;
      let weight = 1.0 / distance;

      steer.x += dx * weight;
      steer.y += dy * weight;
      count += 1;
    }
  }

  if count > 0 {
    steer.x /= count as f64;
    steer.y /= count as f64;
  }

  steer
}

#[napi]
pub fn calculate_alignment(
  velocities: Vec<&Vector2D>,
  max_speed: f64,
  max_force: f64,
) -> Vector2D {
  if velocities.is_empty() {
    return Vector2D { x: 0.0, y: 0.0 };
  }

  let mut avg_velocity = Vector2D { x: 0.0, y: 0.0 };

  for velocity in &velocities {
    avg_velocity.x += velocity.x;
    avg_velocity.y += velocity.y;
  }

  avg_velocity.x /= velocities.len() as f64;
  avg_velocity.y /= velocities.len() as f64;

  let normalized = avg_velocity.normalize();
  let mut desired = Vector2D {
    x: normalized.x * max_speed,
    y: normalized.y * max_speed,
  };

  let mag = desired.magnitude();
  if mag > max_force {
    let normalized = desired.normalize();
    desired.x = normalized.x * max_force;
    desired.y = normalized.y * max_force;
  }

  desired
}

#[napi]
pub fn calculate_cohesion(
  position: &Vector2D,
  neighbors: Vec<&Vector2D>,
  cohesion_radius: f64,
  max_speed: f64,
  max_force: f64,
) -> Vector2D {
  let mut avg_position = Vector2D { x: 0.0, y: 0.0 };
  let mut count = 0;

  for neighbor in neighbors {
    let distance = position.distance(neighbor);

    if distance > 0.0 && distance < cohesion_radius {
      avg_position.x += neighbor.x;
      avg_position.y += neighbor.y;
      count += 1;
    }
  }

  if count == 0 {
    return Vector2D { x: 0.0, y: 0.0 };
  }

  avg_position.x /= count as f64;
  avg_position.y /= count as f64;

  let mut desired = Vector2D {
    x: avg_position.x - position.x,
    y: avg_position.y - position.y,
  };

  let normalized = desired.normalize();
  desired.x = normalized.x * max_speed;
  desired.y = normalized.y * max_speed;

  let mag = desired.magnitude();
  if mag > max_force {
    let normalized = desired.normalize();
    desired.x = normalized.x * max_force;
    desired.y = normalized.y * max_force;
  }

  desired
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_vector_magnitude() {
    let v = Vector2D { x: 3.0, y: 4.0 };
    assert_eq!(v.magnitude(), 5.0);
  }

  #[test]
  fn test_vector_normalize() {
    let v = Vector2D { x: 3.0, y: 4.0 };
    let normalized = v.normalize();
    assert!((normalized.magnitude() - 1.0).abs() < 0.0001);
  }

  #[test]
  fn test_vector_distance() {
    let v1 = Vector2D { x: 0.0, y: 0.0 };
    let v2 = Vector2D { x: 3.0, y: 4.0 };
    assert_eq!(v1.distance(&v2), 5.0);
  }
}
