//! Integration test demonstrating the CodependentRiskModel based on Buddhist philosophy
//!
//! This test showcases a realistic financial network where risk propagates through
//! dependencies following the principle of Pratītyasamutpāda (Dependent Origination).

use hyperphysics_risk::{
    CodependentRiskModel, AssetNode, DependencyEdge, DependencyType,
};

#[test]
fn test_realistic_financial_crisis_scenario() {
    // Create a model representing a financial system during a crisis
    let mut model = CodependentRiskModel::new(20, 0.75, 6);

    // 1. Major Investment Banks (high interconnectedness)
    let jpmorgan = AssetNode {
        id: 1,
        symbol: "JPM".to_string(),
        standalone_risk: 0.20,
        sector: "Investment Banking".to_string(),
    };

    let goldman = AssetNode {
        id: 2,
        symbol: "GS".to_string(),
        standalone_risk: 0.22,
        sector: "Investment Banking".to_string(),
    };

    let morgan_stanley = AssetNode {
        id: 3,
        symbol: "MS".to_string(),
        standalone_risk: 0.24,
        sector: "Investment Banking".to_string(),
    };

    // 2. Commercial Banks (exposed to investment banks)
    let bank_of_america = AssetNode {
        id: 4,
        symbol: "BAC".to_string(),
        standalone_risk: 0.25,
        sector: "Commercial Banking".to_string(),
    };

    let citigroup = AssetNode {
        id: 5,
        symbol: "C".to_string(),
        standalone_risk: 0.28,
        sector: "Commercial Banking".to_string(),
    };

    // 3. Insurance Companies (exposed through credit default swaps)
    let aig = AssetNode {
        id: 6,
        symbol: "AIG".to_string(),
        standalone_risk: 0.30, // Higher standalone risk during crisis
        sector: "Insurance".to_string(),
    };

    // 4. Technology companies (supply chain dependencies)
    let apple = AssetNode {
        id: 7,
        symbol: "AAPL".to_string(),
        standalone_risk: 0.15,
        sector: "Technology".to_string(),
    };

    let microsoft = AssetNode {
        id: 8,
        symbol: "MSFT".to_string(),
        standalone_risk: 0.12,
        sector: "Technology".to_string(),
    };

    // 5. Manufacturing (supply chain, credit exposure)
    let general_electric = AssetNode {
        id: 9,
        symbol: "GE".to_string(),
        standalone_risk: 0.30,
        sector: "Industrial".to_string(),
    };

    let ford = AssetNode {
        id: 10,
        symbol: "F".to_string(),
        standalone_risk: 0.32,
        sector: "Automotive".to_string(),
    };

    // Add all assets to model
    let assets = vec![
        jpmorgan,
        goldman,
        morgan_stanley,
        bank_of_america,
        citigroup,
        aig,
        apple,
        microsoft,
        general_electric,
        ford,
    ];

    for asset in assets {
        model.add_asset(asset);
    }

    // Build dependency network (reflecting 2008-style financial crisis)

    // Investment bank interconnectedness
    model.add_dependency(1, 2, DependencyEdge {
        weight: 0.85,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    model.add_dependency(2, 1, DependencyEdge {
        weight: 0.80,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    model.add_dependency(2, 3, DependencyEdge {
        weight: 0.75,
        dependency_type: DependencyType::Correlation,
    }).unwrap();

    model.add_dependency(3, 1, DependencyEdge {
        weight: 0.70,
        dependency_type: DependencyType::Correlation,
    }).unwrap();

    // Commercial banks exposed to investment banks
    model.add_dependency(1, 4, DependencyEdge {
        weight: 0.65,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    model.add_dependency(2, 5, DependencyEdge {
        weight: 0.70,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    model.add_dependency(3, 4, DependencyEdge {
        weight: 0.60,
        dependency_type: DependencyType::Correlation,
    }).unwrap();

    // AIG exposed through CDS (like 2008)
    model.add_dependency(1, 6, DependencyEdge {
        weight: 0.90, // Very high correlation
        dependency_type: DependencyType::Credit,
    }).unwrap();

    model.add_dependency(2, 6, DependencyEdge {
        weight: 0.85,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    model.add_dependency(3, 6, DependencyEdge {
        weight: 0.80,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    // Banks to corporates (credit exposure)
    model.add_dependency(4, 9, DependencyEdge {
        weight: 0.55,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    model.add_dependency(4, 10, DependencyEdge {
        weight: 0.60,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    model.add_dependency(5, 9, DependencyEdge {
        weight: 0.50,
        dependency_type: DependencyType::Credit,
    }).unwrap();

    // Tech companies with lower systemic risk
    model.add_dependency(7, 8, DependencyEdge {
        weight: 0.40,
        dependency_type: DependencyType::Correlation,
    }).unwrap();

    // Supply chain dependencies
    model.add_dependency(9, 7, DependencyEdge {
        weight: 0.35,
        dependency_type: DependencyType::Supply,
    }).unwrap();

    model.add_dependency(10, 7, DependencyEdge {
        weight: 0.30,
        dependency_type: DependencyType::Supply,
    }).unwrap();

    // Test 1: Calculate effective risk for a major investment bank
    let jpmorgan_risk = model.calculate_risk(1).unwrap();
    println!("\n=== JP Morgan Chase Risk Analysis ===");
    println!("Standalone risk: {:.4}", jpmorgan_risk.standalone);
    println!("Codependent risk: {:.4}", jpmorgan_risk.codependent);
    println!("Effective risk: {:.4}", jpmorgan_risk.effective);
    println!("Top contributors:");
    for (asset_id, contribution) in &jpmorgan_risk.top_contributors {
        println!("  Asset {}: {:.4}", asset_id, contribution);
    }

    // The effective risk should be higher than standalone due to network effects
    assert!(jpmorgan_risk.effective > jpmorgan_risk.standalone);
    assert!(jpmorgan_risk.codependent > 0.0);

    // Test 2: Calculate risk for AIG (should be very high due to CDS exposure)
    let aig_risk = model.calculate_risk(6).unwrap();
    println!("\n=== AIG Risk Analysis (CDS Exposure) ===");
    println!("Standalone risk: {:.4}", aig_risk.standalone);
    println!("Codependent risk: {:.4}", aig_risk.codependent);
    println!("Effective risk: {:.4}", aig_risk.effective);

    // AIG should have higher effective risk than standalone
    assert!(aig_risk.effective > aig_risk.standalone);
    assert!(aig_risk.codependent > 0.0);

    // Test 3: Calculate systemic risk
    let systemic = model.systemic_risk().unwrap();
    println!("\n=== Systemic Risk Analysis ===");
    println!("Total system risk: {:.4}", systemic.total);
    println!("Risk concentration: {:.4}", systemic.concentration);
    println!("Number of critical paths: {}", systemic.critical_paths.len());

    // System should have significant risk
    assert!(systemic.total > 0.15);
    assert!(systemic.concentration > 1.0); // Risk is concentrated

    // Test 4: Find contagion paths from JPM to Ford
    let contagion_paths = model.find_contagion_paths(1, 10).unwrap();
    println!("\n=== Contagion Paths: JPM -> Ford ===");
    for (i, path) in contagion_paths.iter().enumerate() {
        println!("Path {}: {:?}", i + 1, path);
    }

    assert!(!contagion_paths.is_empty(), "Should find at least one contagion path");

    // Test 5: Simulate AIG bailout (reduce risk)
    println!("\n=== Simulating AIG Bailout ===");
    model.update_standalone_risk(6, 0.10).unwrap(); // Government intervention reduces risk

    let aig_risk_after = model.calculate_risk(6).unwrap();
    println!("AIG risk after bailout: {:.4}", aig_risk_after.effective);
    println!("AIG standalone after bailout: {:.4}", aig_risk_after.standalone);
    assert!(aig_risk_after.standalone < aig_risk.standalone);

    // Check how this affects JPM - should have lower codependent component
    let jpmorgan_risk_after = model.calculate_risk(1).unwrap();
    println!("JPM risk after AIG bailout: {:.4}", jpmorgan_risk_after.effective);
    // Note: JPM risk may not always decrease if other dependencies dominate

    // Test 6: Technology companies should have lower effective risk
    let apple_risk = model.calculate_risk(7).unwrap();
    println!("\n=== Apple Risk Analysis ===");
    println!("Effective risk: {:.4}", apple_risk.effective);

    // Apple should have lower risk than financial institutions
    assert!(apple_risk.effective < jpmorgan_risk.effective);
    assert!(apple_risk.effective < aig_risk.effective);

    println!("\n=== Test Complete: Buddhist Codependent Risk Model ===");
    println!("Demonstrated Pratītyasamutpāda (Dependent Origination):");
    println!("  - No asset has independent risk");
    println!("  - Risk arises through network conditions");
    println!("  - Interventions propagate through dependencies");
}

#[test]
fn test_supply_chain_contagion() {
    // Test supply chain risk propagation
    let mut model = CodependentRiskModel::new(5, 0.6, 5);

    // Create supply chain: Raw Materials -> Manufacturer -> Retailer
    model.add_asset(AssetNode {
        id: 1,
        symbol: "RAW_MAT".to_string(),
        standalone_risk: 0.40, // High risk (e.g., commodity volatility)
        sector: "Raw Materials".to_string(),
    });

    model.add_asset(AssetNode {
        id: 2,
        symbol: "MANUFACTURER".to_string(),
        standalone_risk: 0.20,
        sector: "Manufacturing".to_string(),
    });

    model.add_asset(AssetNode {
        id: 3,
        symbol: "RETAILER".to_string(),
        standalone_risk: 0.15,
        sector: "Retail".to_string(),
    });

    // Build supply chain
    model.add_dependency(1, 2, DependencyEdge {
        weight: 0.80,
        dependency_type: DependencyType::Supply,
    }).unwrap();

    model.add_dependency(2, 3, DependencyEdge {
        weight: 0.70,
        dependency_type: DependencyType::Supply,
    }).unwrap();

    // Calculate risks
    let raw_risk = model.calculate_risk(1).unwrap();
    let mfg_risk = model.calculate_risk(2).unwrap();
    let retail_risk = model.calculate_risk(3).unwrap();

    println!("\n=== Supply Chain Risk Propagation ===");
    println!("Raw Materials: {:.4}", raw_risk.effective);
    println!("Manufacturer: {:.4}", mfg_risk.effective);
    println!("Retailer: {:.4}", retail_risk.effective);

    // Risk should propagate downstream
    assert!(mfg_risk.effective > mfg_risk.standalone);
    assert!(retail_risk.effective > retail_risk.standalone);

    // Find path from raw materials to retailer
    let paths = model.find_contagion_paths(1, 3).unwrap();
    assert!(!paths.is_empty());
    assert!(paths.iter().any(|p| p.len() == 3)); // Direct path exists
}

#[test]
fn test_market_correlation_network() {
    // Test market correlation-based risk
    let mut model = CodependentRiskModel::new(4, 0.7, 5);

    // Tech sector correlation
    let tech_stocks = vec![
        AssetNode {
            id: 1,
            symbol: "AAPL".to_string(),
            standalone_risk: 0.18,
            sector: "Technology".to_string(),
        },
        AssetNode {
            id: 2,
            symbol: "MSFT".to_string(),
            standalone_risk: 0.16,
            sector: "Technology".to_string(),
        },
        AssetNode {
            id: 3,
            symbol: "GOOGL".to_string(),
            standalone_risk: 0.20,
            sector: "Technology".to_string(),
        },
        AssetNode {
            id: 4,
            symbol: "META".to_string(),
            standalone_risk: 0.25,
            sector: "Technology".to_string(),
        },
    ];

    for asset in tech_stocks {
        model.add_asset(asset);
    }

    // High correlation within sector
    for i in 1..=4 {
        for j in 1..=4 {
            if i != j {
                model.add_dependency(i, j, DependencyEdge {
                    weight: 0.65,
                    dependency_type: DependencyType::Correlation,
                }).unwrap();
            }
        }
    }

    // Check systemic risk
    let systemic = model.systemic_risk().unwrap();
    println!("\n=== Tech Sector Systemic Risk ===");
    println!("Total risk: {:.4}", systemic.total);
    println!("Concentration: {:.4}", systemic.concentration);

    // High correlation should lead to high systemic risk
    assert!(systemic.total > 0.15);
}
