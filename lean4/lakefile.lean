import Lake
open Lake DSL

package «hyperphysics» where
  -- Project configuration for HyperPhysics formal verification
  -- Note: 'version' field removed for Lean 4.x compatibility

@[default_target]
lean_lib «HyperPhysics» where
  -- Library for pBit lattice physics and Gillespie algorithm formalization
  globs := #[.andSubmodules `HyperPhysics]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
