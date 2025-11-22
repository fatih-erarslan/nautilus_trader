//! Ecology of Mind based on Gregory Bateson's theory
//! Mind as pattern of organization, learning levels, and double binds

use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};

use crate::Result;

/// Learning levels according to Bateson
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LearningLevel {
    /// Zero Learning - Simple response
    ZeroLearning {
        stimulus: String,
        response: String,
    },
    /// Learning I - Change in response
    LearningI {
        context: String,
        adaptation: String,
        success_rate: f64,
    },
    /// Learning II - Learning to learn (deutero-learning)
    DeuteroLearning {
        primary_patterns: Vec<Pattern>,
        meta_patterns: Vec<MetaPattern>,
    },
    /// Learning III - Change in Learning II
    LearningIII {
        worldview_shift: String,
        previous_framework: Box<LearningLevel>,
        new_framework: Box<LearningLevel>,
    },
    /// Learning IV - Evolutionary learning
    LearningIV {
        species_level_change: String,
        evolutionary_time: f64,
    },
}

/// A pattern in the ecology of mind
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pattern {
    pub name: String,
    pub frequency: f64,
    pub stability: f64,
    pub connections: Vec<String>,
}

/// A pattern of patterns (meta-pattern)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetaPattern {
    pub name: String,
    pub constituent_patterns: Vec<String>,
    pub emergence_threshold: f64,
    pub coherence: f64,
}

/// A double bind situation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DoubleBind {
    pub primary_injunction: String,
    pub secondary_injunction: String,
    pub tertiary_injunction: String, // "You cannot escape"
    pub context: String,
}

/// A paradox in the system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Paradox {
    pub description: String,
    pub logical_levels: Vec<LogicalLevel>,
    pub double_binds: Vec<DoubleBind>,
}

/// Logical levels in communication
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogicalLevel {
    pub level: u32,
    pub content: String,
    pub meta_communication: Option<String>,
}

/// Resolution of a paradox
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resolution {
    pub resolution_type: ResolutionType,
    pub new_perspective: String,
    pub learning_outcome: LearningLevel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResolutionType {
    Reframing,
    LevelJumping,
    Integration,
    Transcendence,
    Acceptance,
}

/// Trait for systems exhibiting ecology of mind
pub trait EcologyOfMind: Send + Sync {
    type Information: Send + Sync;
    type Context: Send + Sync;
    
    /// Deutero-learning: learning to learn
    fn deutero_learning(&mut self) -> LearningLevel;
    
    /// Resolve double bind paradoxes
    fn double_bind_resolution(&mut self, paradox: Paradox) -> Resolution;
    
    /// Get context markers for current state
    fn context_markers(&self) -> Vec<Self::Context>;
    
    /// Process distinctions (differences that make a difference)
    fn process_distinctions(&mut self, info: Self::Information) -> Vec<Distinction>;
    
    /// Navigate logical levels
    fn navigate_logical_levels(&self) -> LogicalLevelMap;
    
    /// Detect and handle recursive patterns
    fn recursive_pattern_detection(&self) -> Vec<RecursivePattern>;
}

/// A distinction (difference that makes a difference)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Distinction {
    pub difference: String,
    pub significance: f64,
    pub context_dependency: f64,
    pub information_content: f64,
}

/// Map of logical levels
#[derive(Clone, Debug)]
pub struct LogicalLevelMap {
    pub levels: HashMap<u32, Vec<String>>,
    pub level_transitions: Vec<(u32, u32, f64)>, // from, to, probability
}

/// A recursive pattern in the mind ecology
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursivePattern {
    pub pattern_id: String,
    pub recursion_depth: u32,
    pub self_similarity: f64,
    pub emergence_properties: Vec<String>,
}

/// Implementation of Bateson's mind ecology
pub struct BatesonianMind {
    /// Current patterns
    patterns: HashMap<String, Pattern>,
    /// Meta-patterns
    meta_patterns: HashMap<String, MetaPattern>,
    /// Learning history
    learning_history: VecDeque<LearningLevel>,
    /// Current logical level
    current_level: u32,
    /// Context stack
    context_stack: Vec<String>,
    /// Double binds encountered
    double_binds: Vec<DoubleBind>,
    /// Distinction threshold
    distinction_threshold: f64,
}

impl BatesonianMind {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            meta_patterns: HashMap::new(),
            learning_history: VecDeque::with_capacity(100),
            current_level: 1,
            context_stack: Vec::new(),
            double_binds: Vec::new(),
            distinction_threshold: 0.1,
        }
    }
    
    /// Add a new pattern
    pub fn add_pattern(&mut self, pattern: Pattern) {
        self.patterns.insert(pattern.name.clone(), pattern);
        self.check_for_meta_patterns();
    }
    
    /// Check if patterns form meta-patterns
    fn check_for_meta_patterns(&mut self) {
        // Look for patterns of patterns
        let pattern_names: Vec<String> = self.patterns.keys().cloned().collect();
        
        // Simple clustering based on connections
        for (name, pattern) in &self.patterns {
            let connected_patterns: Vec<String> = pattern.connections.iter()
                .filter(|conn| self.patterns.contains_key(*conn))
                .cloned()
                .collect();
            
            if connected_patterns.len() >= 3 {
                let meta_name = format!("meta_{}", name);
                let coherence = self.calculate_coherence(&connected_patterns);
                
                self.meta_patterns.insert(
                    meta_name.clone(),
                    MetaPattern {
                        name: meta_name,
                        constituent_patterns: connected_patterns,
                        emergence_threshold: 0.5,
                        coherence,
                    }
                );
            }
        }
    }
    
    /// Calculate coherence of a pattern set
    fn calculate_coherence(&self, patterns: &[String]) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }
        
        let mut total_connections = 0;
        let mut internal_connections = 0;
        
        for pattern_name in patterns {
            if let Some(pattern) = self.patterns.get(pattern_name) {
                total_connections += pattern.connections.len();
                internal_connections += pattern.connections.iter()
                    .filter(|conn| patterns.contains(conn))
                    .count();
            }
        }
        
        if total_connections == 0 {
            0.0
        } else {
            internal_connections as f64 / total_connections as f64
        }
    }
    
    /// Create a double bind situation
    pub fn create_double_bind(&mut self, primary: &str, secondary: &str, context: &str) {
        self.double_binds.push(DoubleBind {
            primary_injunction: primary.to_string(),
            secondary_injunction: secondary.to_string(),
            tertiary_injunction: "You cannot escape this situation".to_string(),
            context: context.to_string(),
        });
    }
}

impl EcologyOfMind for BatesonianMind {
    type Information = String;
    type Context = String;
    
    fn deutero_learning(&mut self) -> LearningLevel {
        // Analyze learning history to learn about learning
        let mut pattern_freq = HashMap::new();
        
        for learning in &self.learning_history {
            if let LearningLevel::LearningI { context, .. } = learning {
                *pattern_freq.entry(context.clone()).or_insert(0) += 1;
            }
        }
        
        // Extract patterns from learning history
        let primary_patterns: Vec<Pattern> = pattern_freq.into_iter()
            .map(|(context, freq)| Pattern {
                name: context.clone(),
                frequency: freq as f64,
                stability: 0.8,
                connections: vec![],
            })
            .collect();
        
        // Look for meta-patterns
        let meta_patterns = self.meta_patterns.values().cloned().collect();
        
        let deutero = LearningLevel::DeuteroLearning {
            primary_patterns,
            meta_patterns,
        };
        
        self.learning_history.push_back(deutero.clone());
        if self.learning_history.len() > 100 {
            self.learning_history.pop_front();
        }
        
        deutero
    }
    
    fn double_bind_resolution(&mut self, paradox: Paradox) -> Resolution {
        // Analyze the paradox structure
        let num_levels = paradox.logical_levels.len();
        let num_binds = paradox.double_binds.len();
        
        // Choose resolution strategy based on paradox complexity
        let resolution_type = if num_levels > 2 {
            ResolutionType::LevelJumping
        } else if num_binds > 1 {
            ResolutionType::Reframing
        } else {
            ResolutionType::Integration
        };
        
        // Create new perspective
        let new_perspective = match resolution_type {
            ResolutionType::LevelJumping => {
                format!("Jump to meta-level {} to resolve", self.current_level + 1)
            },
            ResolutionType::Reframing => {
                "Reframe the context to dissolve the paradox".to_string()
            },
            ResolutionType::Integration => {
                "Integrate both sides of the paradox".to_string()
            },
            _ => "Accept the paradox as fundamental".to_string(),
        };
        
        // Learning outcome from resolution
        let learning_outcome = LearningLevel::LearningIII {
            worldview_shift: "Paradox resolution led to perspective shift".to_string(),
            previous_framework: Box::new(LearningLevel::ZeroLearning {
                stimulus: paradox.description.clone(),
                response: "Confused".to_string(),
            }),
            new_framework: Box::new(LearningLevel::LearningI {
                context: new_perspective.clone(),
                adaptation: "Resolved".to_string(),
                success_rate: 0.9,
            }),
        };
        
        Resolution {
            resolution_type,
            new_perspective,
            learning_outcome,
        }
    }
    
    fn context_markers(&self) -> Vec<Self::Context> {
        self.context_stack.clone()
    }
    
    fn process_distinctions(&mut self, info: Self::Information) -> Vec<Distinction> {
        let mut distinctions = Vec::new();
        
        // Compare with existing patterns
        for (name, pattern) in &self.patterns {
            let difference = self.calculate_difference(&info, name);
            
            if difference > self.distinction_threshold {
                distinctions.push(Distinction {
                    difference: format!("{} vs {}", info, name),
                    significance: difference,
                    context_dependency: pattern.stability,
                    information_content: -difference.log2(), // Shannon information
                });
            }
        }
        
        // Update patterns based on new information
        if distinctions.len() > 0 {
            self.add_pattern(Pattern {
                name: info.clone(),
                frequency: 1.0,
                stability: 0.5,
                connections: distinctions.iter()
                    .map(|d| d.difference.split(" vs ").nth(1).unwrap_or("").to_string())
                    .collect(),
            });
        }
        
        distinctions
    }
    
    fn navigate_logical_levels(&self) -> LogicalLevelMap {
        let mut levels = HashMap::new();
        let mut transitions = Vec::new();
        
        // Level 1: Direct patterns
        levels.insert(1, self.patterns.keys().cloned().collect());
        
        // Level 2: Meta-patterns
        levels.insert(2, self.meta_patterns.keys().cloned().collect());
        
        // Level 3: Patterns of meta-patterns
        let meta_meta_patterns = vec!["emergence".to_string(), "coherence".to_string()];
        levels.insert(3, meta_meta_patterns);
        
        // Calculate transition probabilities
        for pattern in self.patterns.values() {
            if pattern.stability > 0.7 {
                transitions.push((1, 2, pattern.stability));
            }
        }
        
        LogicalLevelMap { levels, level_transitions: transitions }
    }
    
    fn recursive_pattern_detection(&self) -> Vec<RecursivePattern> {
        let mut recursive_patterns = Vec::new();
        
        // Look for self-referential patterns
        for (name, pattern) in &self.patterns {
            if pattern.connections.contains(name) {
                // Self-referential pattern found
                let recursion_depth = self.trace_recursion_depth(name, name, 0, 10);
                
                recursive_patterns.push(RecursivePattern {
                    pattern_id: name.clone(),
                    recursion_depth,
                    self_similarity: pattern.stability,
                    emergence_properties: vec!["self-reference".to_string()],
                });
            }
        }
        
        // Look for circular references
        for (name, pattern) in &self.patterns {
            for connection in &pattern.connections {
                if let Some(connected_pattern) = self.patterns.get(connection) {
                    if connected_pattern.connections.contains(name) {
                        recursive_patterns.push(RecursivePattern {
                            pattern_id: format!("{}<->{}", name, connection),
                            recursion_depth: 2,
                            self_similarity: (pattern.stability + connected_pattern.stability) / 2.0,
                            emergence_properties: vec!["circular-reference".to_string()],
                        });
                    }
                }
            }
        }
        
        recursive_patterns
    }
}

impl BatesonianMind {
    fn calculate_difference(&self, info1: &str, info2: &str) -> f64 {
        // Simple string difference metric
        let len1 = info1.len() as f64;
        let len2 = info2.len() as f64;
        
        let common_chars = info1.chars()
            .filter(|c| info2.contains(*c))
            .count() as f64;
        
        1.0 - (2.0 * common_chars) / (len1 + len2)
    }
    
    fn trace_recursion_depth(&self, start: &str, current: &str, depth: u32, max_depth: u32) -> u32 {
        if depth >= max_depth {
            return depth;
        }
        
        if let Some(pattern) = self.patterns.get(current) {
            for connection in &pattern.connections {
                if connection == start && depth > 0 {
                    return depth + 1;
                }
                
                let sub_depth = self.trace_recursion_depth(start, connection, depth + 1, max_depth);
                if sub_depth > depth {
                    return sub_depth;
                }
            }
        }
        
        depth
    }
}

/// Cognitive system inspired by Bateson
pub struct CognitiveSystem {
    mind: BatesonianMind,
    information_buffer: VecDeque<String>,
    context_sensitivity: f64,
}

impl CognitiveSystem {
    pub fn new() -> Self {
        Self {
            mind: BatesonianMind::new(),
            information_buffer: VecDeque::with_capacity(50),
            context_sensitivity: 0.7,
        }
    }
    
    pub fn process_information(&mut self, info: String) {
        self.information_buffer.push_back(info.clone());
        if self.information_buffer.len() > 50 {
            self.information_buffer.pop_front();
        }
        
        // Process through the mind ecology
        let distinctions = self.mind.process_distinctions(info);
        
        // Update context based on distinctions
        if distinctions.len() > 0 {
            let avg_significance = distinctions.iter()
                .map(|d| d.significance)
                .sum::<f64>() / distinctions.len() as f64;
            
            if avg_significance > self.context_sensitivity {
                self.mind.context_stack.push("High distinction context".to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_deutero_learning() {
        let mut mind = BatesonianMind::new();
        
        // Add some learning history
        mind.learning_history.push_back(LearningLevel::LearningI {
            context: "test_context".to_string(),
            adaptation: "adapted".to_string(),
            success_rate: 0.8,
        });
        
        let deutero = mind.deutero_learning();
        
        match deutero {
            LearningLevel::DeuteroLearning { primary_patterns, .. } => {
                assert!(primary_patterns.len() > 0);
            }
            _ => assert!(false, "Expected deutero learning, got different learning type"),
        }
    }
    
    #[test]
    fn test_double_bind_resolution() {
        let mut mind = BatesonianMind::new();
        
        let paradox = Paradox {
            description: "Test paradox".to_string(),
            logical_levels: vec![
                LogicalLevel {
                    level: 1,
                    content: "Do this".to_string(),
                    meta_communication: None,
                },
                LogicalLevel {
                    level: 2,
                    content: "Don't do this".to_string(),
                    meta_communication: Some("But you must".to_string()),
                },
            ],
            double_binds: vec![],
        };
        
        let resolution = mind.double_bind_resolution(paradox);
        assert!(matches!(resolution.resolution_type, ResolutionType::LevelJumping));
    }
}