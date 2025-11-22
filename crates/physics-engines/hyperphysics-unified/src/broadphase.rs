//! SIMD-accelerated broadphase collision detection

use crate::AABB;
use std::collections::HashMap;

/// Broadphase collision detection trait
pub trait BroadPhase: Send + Sync {
    /// Handle type for tracked objects
    type Handle: Copy + Eq + std::hash::Hash;

    /// Insert AABB with handle
    fn insert(&mut self, handle: Self::Handle, aabb: AABB);

    /// Update existing AABB
    fn update(&mut self, handle: Self::Handle, aabb: AABB);

    /// Remove tracked object
    fn remove(&mut self, handle: Self::Handle);

    /// Find all potentially overlapping pairs
    fn find_pairs(&self) -> Vec<(Self::Handle, Self::Handle)>;

    /// Query all objects intersecting AABB
    fn query(&self, aabb: &AABB) -> Vec<Self::Handle>;

    /// Clear all tracked objects
    fn clear(&mut self);

    /// Number of tracked objects
    fn len(&self) -> usize;

    /// Is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple sweep-and-prune broadphase
pub struct SweepAndPrune<H: Copy + Eq + std::hash::Hash> {
    aabbs: HashMap<H, AABB>,
    sorted_x: Vec<(H, f32, bool)>, // handle, value, is_min
}

impl<H: Copy + Eq + std::hash::Hash> SweepAndPrune<H> {
    /// Create new SAP broadphase
    pub fn new() -> Self {
        Self {
            aabbs: HashMap::new(),
            sorted_x: Vec::new(),
        }
    }

    fn rebuild_sorted(&mut self) {
        self.sorted_x.clear();
        for (&handle, aabb) in &self.aabbs {
            self.sorted_x.push((handle, aabb.min.x, true));
            self.sorted_x.push((handle, aabb.max.x, false));
        }
        self.sorted_x.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    }
}

impl<H: Copy + Eq + std::hash::Hash> Default for SweepAndPrune<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Copy + Eq + std::hash::Hash + Send + Sync + 'static> BroadPhase for SweepAndPrune<H> {
    type Handle = H;

    fn insert(&mut self, handle: H, aabb: AABB) {
        self.aabbs.insert(handle, aabb);
    }

    fn update(&mut self, handle: H, aabb: AABB) {
        self.aabbs.insert(handle, aabb);
    }

    fn remove(&mut self, handle: H) {
        self.aabbs.remove(&handle);
    }

    fn find_pairs(&self) -> Vec<(H, H)> {
        let mut pairs = Vec::new();
        let handles: Vec<_> = self.aabbs.keys().copied().collect();

        // O(n²) but simple - use BVH for large counts
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                let a = &self.aabbs[&handles[i]];
                let b = &self.aabbs[&handles[j]];
                if a.intersects(b) {
                    pairs.push((handles[i], handles[j]));
                }
            }
        }
        pairs
    }

    fn query(&self, aabb: &AABB) -> Vec<H> {
        self.aabbs
            .iter()
            .filter(|(_, a)| a.intersects(aabb))
            .map(|(&h, _)| h)
            .collect()
    }

    fn clear(&mut self) {
        self.aabbs.clear();
        self.sorted_x.clear();
    }

    fn len(&self) -> usize {
        self.aabbs.len()
    }
}

/// Grid-based spatial hash broadphase
pub struct SpatialHash<H: Copy + Eq + std::hash::Hash> {
    cell_size: f32,
    inv_cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<H>>,
    object_cells: HashMap<H, Vec<(i32, i32, i32)>>,
    aabbs: HashMap<H, AABB>,
}

impl<H: Copy + Eq + std::hash::Hash> SpatialHash<H> {
    /// Create with cell size
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
            object_cells: HashMap::new(),
            aabbs: HashMap::new(),
        }
    }

    fn hash_point(&self, x: f32, y: f32, z: f32) -> (i32, i32, i32) {
        (
            (x * self.inv_cell_size).floor() as i32,
            (y * self.inv_cell_size).floor() as i32,
            (z * self.inv_cell_size).floor() as i32,
        )
    }

    fn cells_for_aabb(&self, aabb: &AABB) -> Vec<(i32, i32, i32)> {
        let min = self.hash_point(aabb.min.x, aabb.min.y, aabb.min.z);
        let max = self.hash_point(aabb.max.x, aabb.max.y, aabb.max.z);

        let mut cells = Vec::new();
        for x in min.0..=max.0 {
            for y in min.1..=max.1 {
                for z in min.2..=max.2 {
                    cells.push((x, y, z));
                }
            }
        }
        cells
    }
}

impl<H: Copy + Eq + std::hash::Hash + Send + Sync + 'static> BroadPhase for SpatialHash<H> {
    type Handle = H;

    fn insert(&mut self, handle: H, aabb: AABB) {
        let cells = self.cells_for_aabb(&aabb);
        for &cell in &cells {
            self.cells.entry(cell).or_default().push(handle);
        }
        self.object_cells.insert(handle, cells);
        self.aabbs.insert(handle, aabb);
    }

    fn update(&mut self, handle: H, aabb: AABB) {
        self.remove(handle);
        self.insert(handle, aabb);
    }

    fn remove(&mut self, handle: H) {
        if let Some(cells) = self.object_cells.remove(&handle) {
            for cell in cells {
                if let Some(list) = self.cells.get_mut(&cell) {
                    list.retain(|&h| h != handle);
                }
            }
        }
        self.aabbs.remove(&handle);
    }

    fn find_pairs(&self) -> Vec<(H, H)> {
        use std::collections::HashSet;
        let mut pairs = HashSet::new();

        for (_, handles) in &self.cells {
            for i in 0..handles.len() {
                for j in (i + 1)..handles.len() {
                    let a = handles[i];
                    let b = handles[j];
                    if let (Some(aabb_a), Some(aabb_b)) = (self.aabbs.get(&a), self.aabbs.get(&b)) {
                        if aabb_a.intersects(aabb_b) {
                            let pair = if std::mem::size_of_val(&a) < std::mem::size_of_val(&b) {
                                (a, b)
                            } else {
                                (b, a)
                            };
                            pairs.insert(pair);
                        }
                    }
                }
            }
        }
        pairs.into_iter().collect()
    }

    fn query(&self, aabb: &AABB) -> Vec<H> {
        use std::collections::HashSet;
        let mut result = HashSet::new();

        for cell in self.cells_for_aabb(aabb) {
            if let Some(handles) = self.cells.get(&cell) {
                for &h in handles {
                    if let Some(obj_aabb) = self.aabbs.get(&h) {
                        if aabb.intersects(obj_aabb) {
                            result.insert(h);
                        }
                    }
                }
            }
        }
        result.into_iter().collect()
    }

    fn clear(&mut self) {
        self.cells.clear();
        self.object_cells.clear();
        self.aabbs.clear();
    }

    fn len(&self) -> usize {
        self.aabbs.len()
    }
}
