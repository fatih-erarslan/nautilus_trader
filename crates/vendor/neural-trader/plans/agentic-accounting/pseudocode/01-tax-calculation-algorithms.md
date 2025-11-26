# Tax Calculation Algorithms - Pseudocode

## 1. FIFO (First-In, First-Out) Method

```
FUNCTION calculate_fifo_disposal(sale_transaction, available_lots):
  INPUT:
    sale_transaction: Transaction to dispose
    available_lots: Array of TaxLot sorted by acquired_date ASC

  OUTPUT:
    disposals: Array of Disposal records
    remaining_lots: Updated available_lots

  ALGORITHM:
    disposals = []
    quantity_to_dispose = sale_transaction.quantity

    FOR EACH lot IN available_lots:
      IF quantity_to_dispose <= 0:
        BREAK

      # Determine how much to take from this lot
      disposal_quantity = MIN(lot.quantity, quantity_to_dispose)

      # Calculate cost basis for this disposal
      unit_cost_basis = lot.costBasis / lot.quantity
      disposal_cost_basis = disposal_quantity * unit_cost_basis

      # Calculate proceeds (proportional to quantity)
      total_proceeds = sale_transaction.price * sale_transaction.quantity
      disposal_proceeds = (disposal_quantity / sale_transaction.quantity) * total_proceeds

      # Calculate gain/loss
      gain = disposal_proceeds - disposal_cost_basis

      # Determine term (short vs long)
      holding_period = sale_transaction.timestamp - lot.acquiredDate
      term = holding_period >= 1_YEAR ? LONG : SHORT

      # Create disposal record
      disposal = Disposal {
        lotId: lot.id,
        transactionId: sale_transaction.id,
        disposalDate: sale_transaction.timestamp,
        quantity: disposal_quantity,
        proceeds: disposal_proceeds,
        costBasis: disposal_cost_basis,
        gain: gain,
        term: term,
        taxYear: get_tax_year(sale_transaction.timestamp),
        method: FIFO
      }

      disposals.append(disposal)

      # Update lot
      lot.quantity -= disposal_quantity
      IF lot.quantity == 0:
        lot.status = CLOSED
      ELSE:
        lot.status = PARTIAL

      lot.disposals.append(disposal)

      # Update remaining quantity
      quantity_to_dispose -= disposal_quantity

    IF quantity_to_dispose > 0:
      THROW InsufficientQuantityError("Not enough lots to cover disposal")

    RETURN disposals, available_lots
```

---

## 2. LIFO (Last-In, First-Out) Method

```
FUNCTION calculate_lifo_disposal(sale_transaction, available_lots):
  INPUT:
    sale_transaction: Transaction to dispose
    available_lots: Array of TaxLot sorted by acquired_date DESC

  OUTPUT:
    disposals: Array of Disposal records
    remaining_lots: Updated available_lots

  ALGORITHM:
    # Same as FIFO but with lots sorted DESC by acquisition date
    # This prioritizes most recent acquisitions

    disposals = []
    quantity_to_dispose = sale_transaction.quantity

    FOR EACH lot IN available_lots (sorted DESC by acquiredDate):
      # Same disposal logic as FIFO
      # ...

    RETURN disposals, available_lots
```

---

## 3. HIFO (Highest-In, First-Out) Method

```
FUNCTION calculate_hifo_disposal(sale_transaction, available_lots):
  INPUT:
    sale_transaction: Transaction to dispose
    available_lots: Array of TaxLot

  OUTPUT:
    disposals: Array of Disposal records
    remaining_lots: Updated available_lots

  ALGORITHM:
    disposals = []
    quantity_to_dispose = sale_transaction.quantity

    # Sort lots by unit cost basis (highest first)
    sorted_lots = SORT(available_lots, BY unit_cost_basis DESC)

    FOR EACH lot IN sorted_lots:
      IF quantity_to_dispose <= 0:
        BREAK

      # Calculate unit cost basis
      unit_cost_basis = lot.costBasis / lot.quantity

      # Same disposal logic as FIFO
      # ...

    RETURN disposals, available_lots
```

---

## 4. Specific Identification Method

```
FUNCTION calculate_specific_id_disposal(sale_transaction, selected_lot_ids, all_lots):
  INPUT:
    sale_transaction: Transaction to dispose
    selected_lot_ids: Array of lot IDs selected by user
    all_lots: All available TaxLot records

  OUTPUT:
    disposals: Array of Disposal records
    remaining_lots: Updated all_lots

  ALGORITHM:
    disposals = []
    quantity_to_dispose = sale_transaction.quantity

    # Filter to selected lots only
    selected_lots = FILTER(all_lots, WHERE id IN selected_lot_ids)

    FOR EACH lot IN selected_lots:
      IF quantity_to_dispose <= 0:
        BREAK

      # Validate lot availability
      IF lot.quantity <= 0:
        THROW InvalidLotError("Selected lot has no available quantity")

      # Same disposal logic as FIFO
      # ...

    IF quantity_to_dispose > 0:
      THROW InsufficientQuantityError("Selected lots don't cover disposal")

    RETURN disposals, all_lots
```

---

## 5. Average Cost Basis Method (for crypto)

```
FUNCTION calculate_average_cost_disposal(sale_transaction, available_lots):
  INPUT:
    sale_transaction: Transaction to dispose
    available_lots: Array of TaxLot for same asset

  OUTPUT:
    disposals: Array of Disposal records
    remaining_lots: Updated available_lots

  ALGORITHM:
    # Calculate weighted average cost basis
    total_quantity = SUM(lot.quantity FOR lot IN available_lots)
    total_cost_basis = SUM(lot.costBasis FOR lot IN available_lots)
    average_unit_cost = total_cost_basis / total_quantity

    # Create single disposal using average cost
    disposal_quantity = sale_transaction.quantity
    disposal_cost_basis = disposal_quantity * average_unit_cost
    disposal_proceeds = sale_transaction.price * sale_transaction.quantity
    gain = disposal_proceeds - disposal_cost_basis

    # Use oldest lot for holding period determination
    oldest_lot = MIN(available_lots, BY acquiredDate)
    holding_period = sale_transaction.timestamp - oldest_lot.acquiredDate
    term = holding_period >= 1_YEAR ? LONG : SHORT

    disposal = Disposal {
      lotId: "AVERAGE_" + sale_transaction.asset,
      transactionId: sale_transaction.id,
      disposalDate: sale_transaction.timestamp,
      quantity: disposal_quantity,
      proceeds: disposal_proceeds,
      costBasis: disposal_cost_basis,
      gain: gain,
      term: term,
      taxYear: get_tax_year(sale_transaction.timestamp),
      method: AVERAGE_COST
    }

    # Proportionally reduce all lots
    reduction_ratio = disposal_quantity / total_quantity
    FOR EACH lot IN available_lots:
      lot.quantity *= (1 - reduction_ratio)
      IF lot.quantity < EPSILON:
        lot.status = CLOSED

    RETURN [disposal], available_lots
```

---

## 6. Wash Sale Detection & Adjustment

```
FUNCTION detect_wash_sale(disposal, all_transactions, wash_period = 30_DAYS):
  INPUT:
    disposal: Disposal record to check
    all_transactions: All transactions for this asset
    wash_period: Days before/after to check (default 30)

  OUTPUT:
    is_wash_sale: Boolean
    disallowed_loss: Decimal
    adjustment_transaction: Transaction if wash sale detected

  ALGORITHM:
    # Wash sale only applies to losses
    IF disposal.gain >= 0:
      RETURN false, 0, null

    wash_start = disposal.disposalDate - wash_period
    wash_end = disposal.disposalDate + wash_period

    # Find acquisitions in wash sale window
    wash_acquisitions = FILTER(all_transactions, WHERE
      type == BUY AND
      asset == disposal.asset AND
      timestamp >= wash_start AND
      timestamp <= wash_end AND
      timestamp != disposal.disposalDate
    )

    IF wash_acquisitions.length > 0:
      # Wash sale detected - loss is disallowed
      disallowed_loss = ABS(disposal.gain)

      # Find the replacement lot(s) to adjust cost basis
      replacement = FIRST(wash_acquisitions, SORTED BY timestamp ASC)

      RETURN true, disallowed_loss, replacement

    RETURN false, 0, null

FUNCTION apply_wash_sale_adjustment(disposal, replacement_lot, disallowed_loss):
  INPUT:
    disposal: Original disposal with loss
    replacement_lot: Lot acquired in wash sale period
    disallowed_loss: Amount to adjust

  OUTPUT:
    adjusted_disposal: Updated disposal record
    adjusted_lot: Updated replacement lot

  ALGORITHM:
    # Disallow the loss on the disposal
    adjusted_disposal = disposal
    adjusted_disposal.gain = 0  # Loss disallowed
    adjusted_disposal.washSaleAdjustment = disallowed_loss

    # Add disallowed loss to replacement lot's cost basis
    adjusted_lot = replacement_lot
    adjusted_lot.costBasis += disallowed_loss
    adjusted_lot.unitCostBasis = adjusted_lot.costBasis / adjusted_lot.quantity

    RETURN adjusted_disposal, adjusted_lot
```

---

## 7. Tax-Loss Harvesting Opportunity Identification

```
FUNCTION identify_harvest_opportunities(positions, current_prices, wash_sale_window):
  INPUT:
    positions: Array of current Position records
    current_prices: Map of asset -> current market price
    wash_sale_window: Recent transactions to check for wash sales

  OUTPUT:
    opportunities: Array of HarvestOpportunity records, ranked by benefit

  ALGORITHM:
    opportunities = []

    FOR EACH position IN positions:
      current_price = current_prices[position.asset]
      unrealized_loss = position.totalCostBasis - (current_price * position.totalQuantity)

      # Only consider positions with losses
      IF unrealized_loss <= 0:
        CONTINUE

      # Check for wash sale risk
      wash_risk = check_wash_sale_risk(position, wash_sale_window)

      # Calculate benefit score
      benefit_score = calculate_harvest_benefit(
        loss_amount: unrealized_loss,
        holding_period: position.oldestLotDate,
        wash_risk: wash_risk,
        liquidity: get_market_liquidity(position.asset)
      )

      opportunity = HarvestOpportunity {
        position: position,
        unrealizedLoss: unrealized_loss,
        estimatedTaxBenefit: unrealized_loss * CAPITAL_GAINS_TAX_RATE,
        washSaleRisk: wash_risk,
        benefitScore: benefit_score,
        recommendedAction: generate_recommendation(position, wash_risk),
        correlatedAlternatives: find_correlated_assets(position.asset)
      }

      opportunities.append(opportunity)

    # Rank by benefit score (highest first)
    SORT(opportunities, BY benefitScore DESC)

    RETURN opportunities

FUNCTION calculate_harvest_benefit(loss_amount, holding_period, wash_risk, liquidity):
  INPUT:
    loss_amount: Decimal
    holding_period: Duration
    wash_risk: Boolean
    liquidity: Decimal (market depth)

  OUTPUT:
    benefit_score: Number (0-100)

  ALGORITHM:
    score = 0

    # Loss magnitude (40% weight)
    score += (loss_amount / 10000) * 40  # Normalize and scale

    # Holding period bonus (20% weight)
    IF holding_period > 1_YEAR:
      score += 20  # Long-term losses more valuable

    # Wash sale penalty (30% weight)
    IF wash_risk:
      score -= 30

    # Liquidity bonus (10% weight)
    IF liquidity > HIGH_LIQUIDITY_THRESHOLD:
      score += 10

    RETURN CLAMP(score, 0, 100)
```

---

## 8. Multi-Jurisdiction Tax Calculation

```
FUNCTION calculate_tax_by_jurisdiction(disposal, user_jurisdiction):
  INPUT:
    disposal: Disposal record
    user_jurisdiction: String (e.g., "US", "UK", "DE")

  OUTPUT:
    tax_liability: Decimal
    tax_details: Object with jurisdiction-specific info

  ALGORITHM:
    jurisdiction_config = load_jurisdiction_config(user_jurisdiction)

    SWITCH user_jurisdiction:
      CASE "US":
        RETURN calculate_us_tax(disposal, jurisdiction_config)

      CASE "UK":
        RETURN calculate_uk_tax(disposal, jurisdiction_config)

      CASE "DE":
        RETURN calculate_german_tax(disposal, jurisdiction_config)

      DEFAULT:
        THROW UnsupportedJurisdictionError(user_jurisdiction)

FUNCTION calculate_us_tax(disposal, config):
  tax_rate = disposal.term == LONG ? config.longTermRate : config.shortTermRate
  tax_liability = disposal.gain * tax_rate

  # Apply tax brackets if progressive
  IF config.progressive:
    tax_liability = apply_tax_brackets(disposal.gain, config.brackets, disposal.term)

  RETURN tax_liability, {
    term: disposal.term,
    rate: tax_rate,
    form: disposal.term == LONG ? "Schedule D" : "Form 8949"
  }
```

---

## Performance Optimizations

1. **Batch Processing**: Process multiple disposals in single pass
2. **Index Lots**: Maintain B-tree index on acquired_date for fast FIFO/LIFO
3. **Cache Average Cost**: Update incrementally, not recalculate
4. **Parallel Calculation**: Independent disposals can compute concurrently
5. **Rust Implementation**: Use Rust for hot path calculations (see architecture phase)
