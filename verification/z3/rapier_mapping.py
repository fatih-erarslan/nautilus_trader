import z3
from typing import NamedTuple

def prove(claim, name):
    s = z3.Solver()
    s.add(z3.Not(claim))
    r = s.check()
    if r == z3.unsat:
        print(f"✅ [proven] {name}")
    else:
        print(f"❌ [failed] {name}")
        print("Counterexample:")
        print(s.model())

def verify_market_mapper():
    print("Verifying MarketMapper Properties...")

    # --- Constants from code ---
    # price_scale = 0.01
    # volume_scale = 0.001
    # bid_x_base = -1.0
    # ask_x_base = 1.0
    # bid_spacing = 0.1
    # ask_spacing = 0.1
    
    price_scale = z3.Real('price_scale')
    volume_scale = z3.Real('volume_scale')
    
    # Invariants for the mapper configuration
    config_invariants = z3.And(
        price_scale > 0,
        volume_scale > 0
    )

    # --- 1. Separation Theorem ---
    # Verify that bids and asks are always separated in X
    
    i = z3.Int('i') # Index in bid book
    j = z3.Int('j') # Index in ask book
    
    # Indices must be non-negative
    indices_valid = z3.And(i >= 0, j >= 0)
    
    # Position formulas from code:
    # bid_x = -1.0 - (i * 0.1)
    # ask_x = 1.0 + (j * 0.1)
    
    bid_x = -1.0 - (z3.ToReal(i) * 0.1)
    ask_x = 1.0 + (z3.ToReal(j) * 0.1)
    
    separation_claim = z3.Implies(
        z3.And(config_invariants, indices_valid),
        bid_x < ask_x
    )
    
    prove(separation_claim, "Bid/Ask Spatial Separation")

    # --- 2. Price Monotonicity ---
    # Verify that higher price always maps to higher Y
    
    price_a = z3.Real('price_a')
    price_b = z3.Real('price_b')
    mid_price = z3.Real('mid_price')
    
    # y = (price - mid_price) * price_scale
    y_a = (price_a - mid_price) * price_scale
    y_b = (price_b - mid_price) * price_scale
    
    monotonicity_claim = z3.Implies(
        z3.And(config_invariants, price_a > price_b),
        y_a > y_b
    )
    
    prove(monotonicity_claim, "Price Monotonicity (Y-axis)")

    # --- 3. Mass Positivity ---
    # Verify that positive volume results in positive mass
    
    volume = z3.Real('volume')
    
    # mass = volume * volume_scale
    mass = volume * volume_scale
    
    mass_positivity_claim = z3.Implies(
        z3.And(config_invariants, volume > 0),
        mass > 0
    )
    
    prove(mass_positivity_claim, "Mass Positivity")

    # --- 4. Participant Ordering ---
    # Verify vertical ordering of participants: Whale > Institutional > HFT > Retail
    # Based on:
    # Whale: base_y + 2.0
    # Inst:  base_y + 1.0
    # HFT:   base_y
    # Retail: base_y - 1.0
    
    pos_size = z3.Real('pos_size')
    base_y = pos_size * price_scale
    
    y_whale = base_y + 2.0
    y_inst = base_y + 1.0
    y_hft = base_y
    y_retail = base_y - 1.0
    
    ordering_claim = z3.Implies(
        config_invariants,
        z3.And(
            y_whale > y_inst,
            y_inst > y_hft,
            y_hft > y_retail
        )
    )
    
    prove(ordering_claim, "Participant Vertical Ordering")

if __name__ == "__main__":
    verify_market_mapper()
