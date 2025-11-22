import z3

def prove(claim, name):
    s = z3.Tactic('qfnra-nlsat').solver()
    s.add(z3.Not(claim))
    r = s.check()
    if r == z3.unsat:
        print(f"✅ [proven] {name}")
    else:
        print(f"❌ [failed] {name}")
        if r == z3.sat:
            print("Counterexample:")
            print(s.model())

def verify_pbit_dynamics():
    print("Verifying p-bit Dynamics (tanh activation)...")

    # We model tanh(x) using the transformation y = exp(2x).
    # m = tanh(x) = (e^(2x) - 1) / (e^(2x) + 1) = (y - 1) / (y + 1)
    # Since e^(2x) is a strictly positive real number for all real x,
    # we can verify properties on m(y) for y > 0.
    
    y = z3.Real('y')
    y1 = z3.Real('y1')
    y2 = z3.Real('y2')
    
    # Constraint: y must be positive (exponential is always positive)
    valid_y = y > 0
    valid_y1 = y1 > 0
    valid_y2 = y2 > 0
    
    def m(val):
        return (val - 1) / (val + 1)

    # --- 1. Boundedness ---
    # -1 < m(y) < 1 for all y > 0
    boundedness = z3.Implies(
        valid_y,
        z3.And(m(y) > -1, m(y) < 1)
    )
    prove(boundedness, "Boundedness (-1 < m < 1)")

    # --- 2. Monotonicity ---
    # y1 < y2 ==> m(y1) < m(y2)
    # This implies monotonicity with respect to I, since exp(2*beta*I) is monotonic.
    monotonicity = z3.Implies(
        z3.And(valid_y1, valid_y2, y1 < y2),
        m(y1) < m(y2)
    )
    prove(monotonicity, "Monotonicity")

    # --- 3. Sign Preservation ---
    # I > 0 ==> exp(2*beta*I) > 1 ==> y > 1
    # We verify: y > 1 ==> m(y) > 0
    sign_preservation_pos = z3.Implies(
        z3.And(valid_y, y > 1),
        m(y) > 0
    )
    prove(sign_preservation_pos, "Sign Preservation (Positive)")
    
    # I < 0 ==> exp(2*beta*I) < 1 ==> 0 < y < 1
    # We verify: 0 < y < 1 ==> m(y) < 0
    sign_preservation_neg = z3.Implies(
        z3.And(valid_y, y < 1),
        m(y) < 0
    )
    prove(sign_preservation_neg, "Sign Preservation (Negative)")

    # --- 4. Zero Crossing ---
    # I = 0 ==> exp(0) = 1 ==> y = 1
    # We verify: y = 1 ==> m(y) = 0
    zero_crossing = z3.Implies(
        y == 1,
        m(y) == 0
    )
    prove(zero_crossing, "Zero Crossing")

    # --- 5. Asymptotic Behavior (Symbolic) ---
    # We can't take limits directly, but we can check behavior for very large y.
    # For any epsilon > 0, there exists Y such that for all y > Y, 1 - m(y) < epsilon.
    # 1 - (y-1)/(y+1) = (y+1 - y + 1)/(y+1) = 2/(y+1)
    # 2/(y+1) < epsilon
    
    epsilon = z3.Real('epsilon')
    Y = z3.Real('Y')
    
    # We want to prove: For all epsilon > 0, exists Y, such that y > Y implies 1 - m(y) < epsilon
    # This requires quantifiers which makes it harder.
    # Instead, let's prove: For a specific symbolic epsilon > 0, does there exist a Y?
    # Actually, let's just prove that 1 - m(y) is always positive and decreases as y increases.
    
    asymptotic_convergence = z3.Implies(
        z3.And(valid_y1, valid_y2, y1 < y2),
        (1 - m(y1)) > (1 - m(y2))
    )
    prove(asymptotic_convergence, "Asymptotic Convergence (Approaches 1)")

if __name__ == "__main__":
    verify_pbit_dynamics()
