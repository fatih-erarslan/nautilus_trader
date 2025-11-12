import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Algebra.Order.Field.Basic

namespace HyperPhysics

/-- A pBit state is a boolean value representing spin up (true) or down (false) -/
def PBitState := Bool

/-- A lattice is a finite collection of pBits indexed by natural numbers -/
def Lattice (n : Nat) := Fin n → PBitState

/-- Temperature is a positive real number in Kelvin -/
structure Temperature where
  val : ℝ
  pos : 0 < val

/-- Boltzmann constant in J/K -/
def k_B : ℝ := 1.380649e-23

/-- Effective field (magnetic field + interaction terms) at site i -/
def EffectiveField (n : Nat) := Fin n → ℝ

/-- Energy function for a single pBit in external field -/
def pbit_energy (state : PBitState) (h : ℝ) : ℝ :=
  match state with
  | true => -h    -- Energy when spin is up
  | false => h    -- Energy when spin is down

/-- Total energy of the lattice system -/
def lattice_energy (n : Nat) (lattice : Lattice n) (h_eff : EffectiveField n) : ℝ :=
  Finset.sum Finset.univ (fun i => pbit_energy (lattice i) (h_eff i))

/-- Time is a non-negative real number -/
structure Time where
  val : ℝ
  nonneg : 0 ≤ val

/-- Helper: Convert Bool to ℝ (true→1, false→0) -/
def bool_to_real (b : Bool) : ℝ :=
  if b then 1 else 0

/-- Magnetization: average spin of the lattice -/
noncomputable def magnetization (n : Nat) (lattice : Lattice n) : ℝ :=
  (Finset.sum Finset.univ (fun i => bool_to_real (lattice i))) / n

end HyperPhysics
