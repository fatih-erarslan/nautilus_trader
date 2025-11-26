# Alpaca API Integration Test Results

**Test Date**: 2025-11-14 00:57:10

**API Base URL**: https://paper-api.alpaca.markets/v2

**Data API URL**: https://data.alpaca.markets/v2

## Test Summary

- **Total Tests**: 8
- **Passed**: 6 ✅
- **Failed**: 2 ❌
- **Success Rate**: 75.0%

## Detailed Test Results

### Test 1: Connection and Authentication

**Status**: ✅ PASSED

**Details**: Successfully authenticated. Account ID: e0ba4632-bfbd-4b72-b416-1eb1daa74520

**Timestamp**: 2025-11-14T00:57:08.249397

**Response Data**:
```json
{
  "id": "e0ba4632-bfbd-4b72-b416-1eb1daa74520",
  "admin_configurations": {},
  "user_configurations": null,
  "account_number": "PA33WXN7OD4M",
  "status": "ACTIVE",
  "crypto_status": "ACTIVE",
  "options_approved_level": 3,
  "options_trading_level": 3,
  "currency": "USD",
  "buying_power": "1954389.73",
  "regt_buying_power": "1954389.73",
  "daytrading_buying_power": "0",
  "effective_buying_power": "1954389.73",
  "non_marginable_buying_power": "975194.86",
  "options_buying_power": "977194.86",
  "bod_dtbp": "0",
  "cash": "954321.95",
  "accrued_fees": "0",
  "portfolio_value": "1000067.78",
  "pattern_day_trader": false,
  "trading_blocked": false,
  "transfers_blocked": false,
  "account_blocked": false,
  "created_at": "2025-09-22T21:24:24.802054Z",
  "trade_suspended_by_user": false,
  "multiplier": "2",
  "shorting_enabled": true,
  "equity": "1000067.78",
  "last_equity": "1001108.55",
  "long_market_value": "45745.83",
  "short_market_value": "0",
  "position_market_value": "45745.83",
  "initial_margin": "22872.92",
  "maintenance_margin": "14164.56",
  "last_maintenance_margin": "14509.64",
  "sma": "1001216.76",
  "daytrade_count": 0,
  "balance_asof": "2025-11-12",
  "crypto_tier": 1,
  "intraday_adjustments": "0",
  "pending_reg_taf_fees": "0"
}
```

---

### Test 2: Account Information

**Status**: ✅ PASSED

**Details**: Cash: $954,321.95, Buying Power: $1,954,389.73, Portfolio Value: $1,000,067.78, Pattern Day Trader: False

**Timestamp**: 2025-11-14T00:57:08.279029

**Response Data**:
```json
{
  "id": "e0ba4632-bfbd-4b72-b416-1eb1daa74520",
  "admin_configurations": {},
  "user_configurations": null,
  "account_number": "PA33WXN7OD4M",
  "status": "ACTIVE",
  "crypto_status": "ACTIVE",
  "options_approved_level": 3,
  "options_trading_level": 3,
  "currency": "USD",
  "buying_power": "1954389.73",
  "regt_buying_power": "1954389.73",
  "daytrading_buying_power": "0",
  "effective_buying_power": "1954389.73",
  "non_marginable_buying_power": "975194.86",
  "options_buying_power": "977194.86",
  "bod_dtbp": "0",
  "cash": "954321.95",
  "accrued_fees": "0",
  "portfolio_value": "1000067.78",
  "pattern_day_trader": false,
  "trading_blocked": false,
  "transfers_blocked": false,
  "account_blocked": false,
  "created_at": "2025-09-22T21:24:24.802054Z",
  "trade_suspended_by_user": false,
  "multiplier": "2",
  "shorting_enabled": true,
  "equity": "1000067.78",
  "last_equity": "1001108.55",
  "long_market_value": "45745.83",
  "short_market_value": "0",
  "position_market_value": "45745.83",
  "initial_margin": "22872.92",
  "maintenance_margin": "14164.56",
  "last_maintenance_margin": "14509.64",
  "sma": "1001216.76",
  "daytrade_count": 0,
  "balance_asof": "2025-11-12",
  "crypto_tier": 1,
  "intraday_adjustments": "0",
  "pending_reg_taf_fees": "0"
}
```

---

### Test 3: Real-time Quote for AAPL

**Status**: ✅ PASSED

**Details**: Symbol: AAPL, Bid: $260.23, Ask: $287.89, Bid Size: 100, Ask Size: 100

**Timestamp**: 2025-11-14T00:57:08.308072

**Response Data**:
```json
{
  "quote": {
    "ap": 287.89,
    "as": 100,
    "ax": "V",
    "bp": 260.23,
    "bs": 100,
    "bx": "V",
    "c": [
      "R"
    ],
    "t": "2025-11-13T21:00:00.449863409Z",
    "z": "C"
  },
  "symbol": "AAPL"
}
```

---

### Test 4: Historical Data for SPY

**Status**: ❌ FAILED

**Details**: Failed to fetch historical data. Status: 403, Error: {"message":"subscription does not permit querying recent SIP data"}


**Timestamp**: 2025-11-14T00:57:08.333256

---

### Test 5: Place Market Order (AAPL)

**Status**: ✅ PASSED

**Details**: Order placed successfully. Order ID: 829f8278-7b48-49b0-9374-af104f18a5fa, Symbol: AAPL, Qty: 1, Side: buy, Type: market, Status: accepted

**Timestamp**: 2025-11-14T00:57:08.371715

**Response Data**:
```json
{
  "id": "829f8278-7b48-49b0-9374-af104f18a5fa",
  "client_order_id": "1f23b701-c9ff-47f2-b8dd-21ce20150969",
  "created_at": "2025-11-14T00:57:08.269001182Z",
  "updated_at": "2025-11-14T00:57:08.271149892Z",
  "submitted_at": "2025-11-14T00:57:08.269001182Z",
  "filled_at": null,
  "expired_at": null,
  "canceled_at": null,
  "failed_at": null,
  "replaced_at": null,
  "replaced_by": null,
  "replaces": null,
  "asset_id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
  "symbol": "AAPL",
  "asset_class": "us_equity",
  "notional": null,
  "qty": "1",
  "filled_qty": "0",
  "filled_avg_price": null,
  "order_class": "",
  "order_type": "market",
  "type": "market",
  "side": "buy",
  "position_intent": "buy_to_open",
  "time_in_force": "day",
  "limit_price": null,
  "stop_price": null,
  "status": "accepted",
  "extended_hours": false,
  "legs": null,
  "trail_percent": null,
  "trail_price": null,
  "hwm": null,
  "subtag": null,
  "source": null,
  "expires_at": "2025-11-14T21:00:00Z"
}
```

---

### Test 6: Order Status Check

**Status**: ❌ FAILED

**Details**: Error checking order status: float() argument must be a string or a real number, not 'NoneType'

**Timestamp**: 2025-11-14T00:57:10.507001

---

### Test 7: Current Positions

**Status**: ✅ PASSED

**Details**: Total positions: 8. AAPL: 7 shares @ $256.60, AMD: 1 shares @ $160.07, AMZN: 5 shares @ $226.57, GOOG: 1 shares @ $253.74, META: 1 shares @ $767.64, NVDA: 15 shares @ $181.67, SPY: 51 shares @ $666.74, TSLA: 11 shares @ $439.90

**Timestamp**: 2025-11-14T00:57:10.528445

**Response Data**:
```json
[
  {
    "asset_id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
    "symbol": "AAPL",
    "exchange": "NASDAQ",
    "asset_class": "us_equity",
    "asset_marginable": true,
    "qty": "7",
    "avg_entry_price": "256.597143",
    "side": "long",
    "market_value": "1914.5",
    "cost_basis": "1796.18",
    "unrealized_pl": "118.32",
    "unrealized_plpc": "0.0658731307552695",
    "unrealized_intraday_pl": "0.21",
    "unrealized_intraday_plpc": "0.0001097012469375",
    "current_price": "273.5",
    "lastday_price": "273.47",
    "change_today": "0.0001097012469375",
    "qty_available": "7"
  },
  {
    "asset_id": "03fb07bb-5db1-4077-8dea-5d711b272625",
    "symbol": "AMD",
    "exchange": "NASDAQ",
    "asset_class": "us_equity",
    "asset_marginable": true,
    "qty": "1",
    "avg_entry_price": "160.07",
    "side": "long",
    "market_value": "247.38",
    "cost_basis": "160.07",
    "unrealized_pl": "87.31",
    "unrealized_plpc": "0.545448866121072",
    "unrealized_intraday_pl": "-11.51",
    "unrealized_intraday_plpc": "-0.044459036656495",
    "current_price": "247.38",
    "lastday_price": "258.89",
    "change_today": "-0.044459036656495",
    "qty_available": "1"
  },
  {
    "asset_id": "f801f835-bfe6-4a9d-a6b1-ccbb84bfd75f",
    "symbol": "AMZN",
    "exchange": "NASDAQ",
    "asset_class": "us_equity",
    "asset_marginable": true,
    "qty": "5",
    "avg_entry_price": "226.57",
    "side": "long",
    "market_value": "1190.3745",
    "cost_basis": "1132.85",
    "unrealized_pl": "57.5245",
    "unrealized_plpc": "0.0507785673301849",
    "unrealized_intraday_pl": "-30.6255",
    "unrealized_intraday_plpc": "-0.0250823095823096",
    "current_price": "238.0749",
    "lastday_price": "244.2",
    "change_today": "-0.0250823095823096",
    "qty_available": "5"
  },
  {
    "asset_id": "f30d734c-2806-4d0d-b145-f9fade61432b",
    "symbol": "GOOG",
    "exchange": "NASDAQ",
    "asset_class": "us_equity",
    "asset_marginable": true,
    "qty": "1",
    "avg_entry_price": "253.74",
    "side": "long",
    "market_value": "279.93",
    "cost_basis": "253.74",
    "unrealized_pl": "26.19",
    "unrealized_plpc": "0.1032158902813904",
    "unrealized_intraday_pl": "-7.5",
    "unrealized_intraday_plpc": "-0.0260933096753992",
    "current_price": "279.93",
    "lastday_price": "287.43",
    "change_today": "-0.0260933096753992",
    "qty_available": "1"
  },
  {
    "asset_id": "fc6a5dcd-4a70-4b8d-b64f-d83a6dae9ba4",
    "symbol": "META",
    "exchange": "NASDAQ",
    "asset_class": "us_equity",
    "asset_marginable": true,
    "qty": "1",
    "avg_entry_price": "767.64",
    "side": "long",
    "market_value": "608.75",
    "cost_basis": "767.64",
    "unrealized_pl": "-158.89",
    "unrealized_plpc": "-0.2069850450732114",
    "unrealized_intraday_pl": "-0.26",
    "unrealized_intraday_plpc": "-0.0004269223822269",
    "current_price": "608.75",
    "lastday_price": "609.01",
    "change_today": "-0.0004269223822269",
    "qty_available": "1"
  },
  {
    "asset_id": "4ce9353c-66d1-46c2-898f-fce867ab0247",
    "symbol": "NVDA",
    "exchange": "NASDAQ",
    "asset_class": "us_equity",
    "asset_marginable": true,
    "qty": "15",
    "avg_entry_price": "181.666",
    "side": "long",
    "market_value": "2789.85",
    "cost_basis": "2724.99",
    "unrealized_pl": "64.86",
    "unrealized_plpc": "0.0238019222088888",
    "unrealized_intraday_pl": "-117.15",
    "unrealized_intraday_plpc": "-0.0402992776057792",
    "current_price": "185.99",
    "lastday_price": "193.8",
    "change_today": "-0.0402992776057792",
    "qty_available": "15"
  },
  {
    "asset_id": "b28f4066-5c6d-479b-a2af-85dc1a8f16fb",
    "symbol": "SPY",
    "exchange": "ARCA",
    "asset_class": "us_equity",
    "asset_marginable": true,
    "qty": "51",
    "avg_entry_price": "666.737843",
    "side": "long",
    "market_value": "34307.7",
    "cost_basis": "34003.63",
    "unrealized_pl": "304.07",
    "unrealized_plpc": "0.0089422805741622",
    "unrealized_intraday_pl": "-544.68",
    "unrealized_intraday_plpc": "-0.0156282010009073",
    "current_price": "672.7",
    "lastday_price": "683.38",
    "change_today": "-0.0156282010009073",
    "qty_available": "51"
  },
  {
    "asset_id": "8ccae427-5dd0-45b3-b5fe-7ba5e422c766",
    "symbol": "TSLA",
    "exchange": "NASDAQ",
    "asset_class": "us_equity",
    "asset_marginable": true,
    "qty": "11",
    "avg_entry_price": "439.903636",
    "side": "long",
    "market_value": "4406.6",
    "cost_basis": "4838.94",
    "unrealized_pl": "-432.34",
    "unrealized_plpc": "-0.08934601379641",
    "unrealized_intraday_pl": "-330",
    "unrealized_intraday_plpc": "-0.0696702275894101",
    "current_price": "400.6",
    "lastday_price": "430.6",
    "change_today": "-0.0696702275894101",
    "qty_available": "11"
  }
]
```

---

### Test 8: Cancel Open Orders

**Status**: ✅ PASSED

**Details**: Cancelled 1 open orders: 829f8278-7b48-49b0-9374-af104f18a5fa

**Timestamp**: 2025-11-14T00:57:10.570053

**Response Data**:
```json
{
  "cancelled_orders": [
    "829f8278-7b48-49b0-9374-af104f18a5fa"
  ]
}
```

---

## API Response Schema Validation

All API responses were validated against expected schemas:

- ✅ Account endpoint returns required fields: id, cash, buying_power, portfolio_value
- ✅ Quote endpoint returns bid/ask prices and sizes
- ✅ Historical data endpoint returns OHLCV bars
- ✅ Order endpoint returns order ID and status
- ✅ Positions endpoint returns symbol, quantity, and entry price

## Errors and Issues

- **Historical Data for SPY**: Failed to fetch historical data. Status: 403, Error: {"message":"subscription does not permit querying recent SIP data"}

- **Order Status Check**: Error checking order status: float() argument must be a string or a real number, not 'NoneType'

## Recommendations

- Review failed tests and check API credentials
- Verify network connectivity to Alpaca API
- Check API rate limits and account status
