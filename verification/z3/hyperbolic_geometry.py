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

def verify_hyperbolic_geometry():
    print("Verifying HyperbolicSpace Properties...")

    # Define Points in 2D
    # We use Real arithmetic.
    
    def create_point(prefix):
        x = z3.Real(f'{prefix}_x')
        y = z3.Real(f'{prefix}_y')
        return x, y

    p1_x, p1_y = create_point('p1')
    p2_x, p2_y = create_point('p2')
    p3_x, p3_y = create_point('p3')

    # Helper for dot product / squared norm
    def dist_sq(ax, ay, bx, by):
        return (ax - bx)**2 + (ay - by)**2

    def norm_sq(ax, ay):
        return ax**2 + ay**2

    # Constraints: Points must be inside the unit disk (Poincaré model)
    # r^2 < 1
    def is_valid(x, y):
        return norm_sq(x, y) < 1.0

    valid_p1 = is_valid(p1_x, p1_y)
    valid_p2 = is_valid(p2_x, p2_y)
    valid_p3 = is_valid(p3_x, p3_y)
    
    all_valid = z3.And(valid_p1, valid_p2, valid_p3)

    # Implementation of the distance function from Rust code:
    # numerator = 2.0 * euclidean_dist_sq;
    # denominator = (1.0 - r1_sq) * (1.0 - r2_sq);
    # ratio = numerator / denominator;
    # distance = ln(1.0 + ratio);
    
    def get_ratio(ax, ay, bx, by):
        num = 2.0 * dist_sq(ax, ay, bx, by)
        den = (1.0 - norm_sq(ax, ay)) * (1.0 - norm_sq(bx, by))
        return num / den

    # --- 1. Non-negativity ---
    # distance >= 0  <==>  ln(1+ratio) >= 0  <==>  1+ratio >= 1  <==>  ratio >= 0
    
    ratio_12 = get_ratio(p1_x, p1_y, p2_x, p2_y)
    
    non_negativity = z3.Implies(
        z3.And(valid_p1, valid_p2),
        ratio_12 >= 0
    )
    prove(non_negativity, "Non-negativity (ratio >= 0)")

    # --- 2. Identity of Indiscernibles ---
    # distance = 0 <==> ratio = 0 <==> euclidean_dist_sq = 0 <==> p1 = p2
    
    identity = z3.Implies(
        z3.And(valid_p1, valid_p2),
        (ratio_12 == 0) == (z3.And(p1_x == p2_x, p1_y == p2_y))
    )
    prove(identity, "Identity of Indiscernibles")

    # --- 3. Symmetry ---
    # distance(p1, p2) == distance(p2, p1)
    
    ratio_21 = get_ratio(p2_x, p2_y, p1_x, p1_y)
    
    symmetry = z3.Implies(
        z3.And(valid_p1, valid_p2),
        ratio_12 == ratio_21
    )
    prove(symmetry, "Symmetry")

    # --- 4. Triangle Inequality ---
    # d(p1, p3) <= d(p1, p2) + d(p2, p3)
    # acosh(1 + r13) <= acosh(1 + r12) + acosh(1 + r23)
    # Since cosh is monotonic for x >= 0:
    # 1 + r13 <= cosh( acosh(1+r12) + acosh(1+r23) )
    # Using cosh(a+b) = cosh(a)cosh(b) + sinh(a)sinh(b)
    # Let u = 1+r12, v = 1+r23.
    # cosh(a) = u, cosh(b) = v
    # sinh(a) = sqrt(u^2 - 1), sinh(b) = sqrt(v^2 - 1)
    # We need to prove: 1 + r13 <= u*v + sqrt(u^2-1)*sqrt(v^2-1)
    
    ratio_13 = get_ratio(p1_x, p1_y, p3_x, p3_y)
    ratio_23 = get_ratio(p2_x, p2_y, p3_x, p3_y)
    
    u = 1.0 + ratio_12
    v = 1.0 + ratio_23
    w = 1.0 + ratio_13
    
    # We check the squared version to avoid sqrt in Z3 if possible, or just use algebraic constraints.
    # Target: w <= u*v + sqrt(u^2-1)*sqrt(v^2-1)
    # w - u*v <= sqrt(...)
    # If (w - u*v) < 0, it's trivially true (since sqrt >= 0).
    # If (w - u*v) >= 0, we check (w - u*v)^2 <= (u^2-1)(v^2-1).
    
    term_sqrt = (u**2 - 1) * (v**2 - 1)
    lhs = w - u*v
    
    triangle_ineq = z3.Implies(
        all_valid,
        z3.Or(
            lhs < 0,
            lhs**2 <= term_sqrt
        )
    )
    
    prove(triangle_ineq, "Triangle Inequality (Poincaré Metric)")

if __name__ == "__main__":
    verify_hyperbolic_geometry()
