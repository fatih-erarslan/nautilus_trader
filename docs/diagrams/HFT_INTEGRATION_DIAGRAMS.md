# HyperPhysics-Nautilus Integration Diagrams

This document contains Mermaid diagrams documenting the integration architecture between HyperPhysics and Nautilus Trader.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Nautilus["Nautilus Trader"]
        DE[DataEngine]
        MB[MessageBus]
        EE[ExecEngine]
        RE[RiskEngine]
        VN[Venue/Exchange]
    end

    subgraph Bridge["hyperphysics-nautilus Bridge"]
        DA[NautilusDataAdapter]
        EB[NautilusExecBridge]
        HPS[HyperPhysicsStrategy]
    end

    subgraph HyperPhysics["HyperPhysics Core"]
        UP[UnifiedPipeline]
        PS[Physics Simulation]
        BO[Biomimetic Optimizer]
        BC[Byzantine Consensus]
    end

    DE -->|QuoteTick/TradeTick| MB
    MB -->|Events| HPS
    HPS --> DA
    DA -->|MarketFeed| UP
    UP --> PS
    PS --> BO
    BO --> BC
    BC -->|PipelineResult| EB
    EB -->|OrderCommand| HPS
    HPS -->|Orders| EE
    EE --> RE
    RE --> VN

    style Bridge fill:#f9f,stroke:#333,stroke-width:2px
    style HyperPhysics fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 2. Data Flow Sequence

```mermaid
sequenceDiagram
    participant DE as DataEngine
    participant MB as MessageBus
    participant HPS as HyperPhysicsStrategy
    participant DA as DataAdapter
    participant UP as UnifiedPipeline
    participant EB as ExecBridge
    participant EE as ExecEngine

    DE->>MB: QuoteTick
    MB->>HPS: on_quote()

    activate HPS
    HPS->>DA: Convert to MarketFeed
    DA-->>HPS: MarketFeed

    HPS->>UP: execute(feed)
    activate UP
    Note over UP: Physics Simulation
    Note over UP: Biomimetic Optimization
    Note over UP: Byzantine Consensus
    UP-->>HPS: PipelineResult
    deactivate UP

    HPS->>EB: process_result()

    alt confidence >= threshold
        EB-->>HPS: OrderCommand
        HPS->>EE: submit_order()
    else confidence < threshold
        EB-->>HPS: None
    end
    deactivate HPS
```

---

## 3. Type Conversion Layer

```mermaid
flowchart LR
    subgraph Nautilus["Nautilus Types (Fixed-Point)"]
        QT[QuoteTick<br/>bid_price: i64<br/>ask_price: i64]
        TT[TradeTick<br/>price: i64<br/>size: u64]
        BAR[Bar<br/>open/high/low/close: i64]
        OBD[OrderBookDelta<br/>price: i64<br/>size: u64]
    end

    subgraph Convert["Conversion Layer"]
        F2F[fixed_to_f64<br/>precision_scale]
    end

    subgraph HP["HyperPhysics Types (f64)"]
        MS[MarketSnapshot<br/>bid/ask/mid: f64]
        MT[MarketTick<br/>price/volume: f64]
        BD[BarData<br/>ohlc: f64]
        OBL[OrderBookLevel<br/>price/size: f64]
    end

    QT --> F2F --> MS
    TT --> F2F --> MT
    BAR --> F2F --> BD
    OBD --> F2F --> OBL

    style Convert fill:#ff9,stroke:#333,stroke-width:2px
```

---

## 4. Strategy State Machine

```mermaid
stateDiagram-v2
    [*] --> Initialized: new()

    Initialized --> Running: start()
    Running --> Stopped: stop()
    Stopped --> Initialized: reset()
    Running --> Initialized: reset()

    Running --> Faulted: error
    Faulted --> Initialized: reset()

    note right of Running
        Processing events:
        - on_quote()
        - on_trade()
        - on_bar()
    end note

    note right of Initialized
        Ready to start
        No event processing
    end note
```

---

## 5. HyperPhysics Pipeline Internals

```mermaid
flowchart TB
    subgraph Pipeline["UnifiedPipeline"]
        MF[MarketFeed Input]

        subgraph Physics["Physics Module"]
            WP[Warp Physics<br/>Spacetime modeling]
            HC[Hyperbolic<br/>Curvature analysis]
            EF[Energy Flow<br/>Market momentum]
        end

        subgraph Optimization["Biomimetic Optimizer"]
            ACO[Ant Colony<br/>Path optimization]
            PSO[Particle Swarm<br/>Global search]
            FF[Firefly<br/>Local refinement]
        end

        subgraph Consensus["Byzantine Consensus"]
            PV[Proposal<br/>Validation]
            VT[Voting<br/>Round]
            CM[Commit<br/>Decision]
        end

        MF --> WP
        WP --> HC
        HC --> EF

        EF --> ACO
        ACO --> PSO
        PSO --> FF

        FF --> PV
        PV --> VT
        VT --> CM

        CM --> TD[TradingDecision]
    end

    TD --> PR[PipelineResult<br/>• decision<br/>• confidence<br/>• latency_us<br/>• consensus_term]

    style Physics fill:#bbf
    style Optimization fill:#bfb
    style Consensus fill:#fbb
```

---

## 6. Backtest Architecture

```mermaid
flowchart TB
    subgraph Config["Configuration"]
        BC[BacktestConfig<br/>• initial_capital<br/>• commission_rate<br/>• slippage_model]
        SC[StrategyConfig<br/>• confidence_threshold<br/>• enable_consensus]
    end

    subgraph DataInput["Data Loading"]
        CSV[CSV Files]
        SYN[Synthetic Generator]
        API[External API]
    end

    subgraph Runner["BacktestRunner"]
        EL[Event Loop]
        POS[Position Tracker]
        EQ[Equity Curve]
        TR[Trade Records]
    end

    subgraph Results["BacktestResults"]
        RET[Total Return]
        SR[Sharpe Ratio]
        DD[Max Drawdown]
        WR[Win Rate]
        PF[Profit Factor]
    end

    BC --> Runner
    SC --> Runner

    CSV --> EL
    SYN --> EL
    API --> EL

    EL --> POS
    POS --> EQ
    EQ --> TR
    TR --> Results

    style Runner fill:#f9f,stroke:#333
```

---

## 7. Order Command Flow

```mermaid
flowchart LR
    subgraph HP["HyperPhysics Decision"]
        TD[TradingDecision<br/>• signal_type<br/>• confidence<br/>• position_size]
    end

    subgraph Bridge["ExecBridge Processing"]
        CV[Confidence<br/>Validation]
        QC[Quantity<br/>Calculation]
        TIF[Time-in-Force<br/>Selection]
        IDG[Order ID<br/>Generation]
    end

    subgraph Order["HyperPhysicsOrderCommand"]
        OC[OrderCommand<br/>• client_order_id<br/>• instrument_id<br/>• side, type, qty<br/>• hp_confidence<br/>• hp_algorithm<br/>• hp_latency_us]
    end

    TD --> CV
    CV -->|Pass| QC
    CV -->|Fail| REJ[Rejected]
    QC --> TIF
    TIF --> IDG
    IDG --> OC

    style Bridge fill:#ffd,stroke:#333
```

---

## 8. Module Dependencies

```mermaid
flowchart BT
    subgraph External["External Dependencies"]
        HPCore[hyperphysics-core]
        HPHFT[hyperphysics-hft-ecosystem]
        HPOpt[hyperphysics-optimization]
        HPMkt[hyperphysics-market]
        HPGeo[hyperphysics-geometry]
    end

    subgraph Crate["hyperphysics-nautilus"]
        LIB[lib.rs]
        ERR[error.rs]
        CFG[config.rs]

        subgraph Types["types/"]
            TMOD[mod.rs]
            CONV[conversions.rs]
            NAUT[nautilus_compat.rs]
        end

        subgraph Adapter["adapter/"]
            AMOD[mod.rs]
            DATA[data_adapter.rs]
            EXEC[exec_bridge.rs]
        end

        subgraph Strategy["strategy/"]
            SMOD[mod.rs]
            STRAT[hyperphysics_strategy.rs]
        end

        subgraph Backtest["backtest/"]
            BMOD[mod.rs]
            RUN[runner.rs]
            LOAD[data_loader.rs]
        end
    end

    HPCore --> CONV
    HPHFT --> STRAT
    HPOpt --> STRAT
    HPMkt --> DATA
    HPGeo --> CONV

    LIB --> Types
    LIB --> Adapter
    LIB --> Strategy
    LIB --> Backtest

    Types --> Adapter
    Types --> Strategy
    Adapter --> Strategy
    Strategy --> Backtest
```

---

## 9. Performance Metrics Flow

```mermaid
flowchart TB
    subgraph Events["Event Sources"]
        Q[Quote Events]
        T[Trade Events]
        B[Bar Events]
    end

    subgraph Metrics["StrategyMetrics"]
        QP[quotes_processed]
        TP[trades_processed]
        BP[bars_processed]
        SG[signals_generated]
        OS[orders_submitted]
        AL[avg_latency_us]
        ML[max_latency_us]
        RT[runtime_seconds]
    end

    subgraph Pipeline["Pipeline Stats"]
        PS[physics_latency_us]
        OL[optimization_latency_us]
        CL[consensus_latency_us]
    end

    subgraph Exec["ExecBridge Stats"]
        SP[signals_processed]
        OG[orders_generated]
        RJ[rejected_count]
    end

    Q --> QP
    T --> TP
    B --> BP

    QP --> SG
    TP --> SG
    BP --> SG

    SG --> AL
    SG --> ML
    SG --> OS

    Pipeline --> AL
    Exec --> OS
    Exec --> RJ
```

---

## 10. Error Handling Flow

```mermaid
flowchart TB
    subgraph Sources["Error Sources"]
        PE[Pipeline Error]
        CE[Conversion Error]
        IE[I/O Error]
        SE[Serialization Error]
    end

    subgraph Errors["IntegrationError"]
        PIE[Pipeline<br/>String]
        CNR[ConsensusNotReached<br/>confidence, threshold]
        ICV[InvalidConversion<br/>field, reason]
        SNR[StrategyNotRunning]
        SER[Serialization<br/>String]
        IOE[Io<br/>std::io::Error]
    end

    subgraph Handling["Error Handling"]
        LOG[Log Error]
        MET[Update Metrics]
        RET[Return Result]
    end

    PE --> PIE
    PE --> CNR
    CE --> ICV
    SE --> SER
    IE --> IOE

    PIE --> LOG
    CNR --> LOG
    ICV --> LOG
    SNR --> LOG
    SER --> LOG
    IOE --> LOG

    LOG --> MET
    MET --> RET
```

---

## 11. Integration Deployment Options

```mermaid
flowchart TB
    subgraph Mode1["Mode 1: Standalone Backtest"]
        BT1[BacktestRunner]
        SYN1[Synthetic Data]
        CSV1[CSV Data]

        SYN1 --> BT1
        CSV1 --> BT1
    end

    subgraph Mode2["Mode 2: Nautilus Integration"]
        NT[Nautilus Trader]
        HPS[HyperPhysicsStrategy]
        EX[Exchange]

        NT --> HPS
        HPS --> NT
        NT --> EX
    end

    subgraph Mode3["Mode 3: Hybrid"]
        NT3[Nautilus Data]
        HP3[HyperPhysics Signals]
        EX3[Execution]

        NT3 --> HP3
        HP3 --> EX3
    end

    style Mode1 fill:#bfb
    style Mode2 fill:#bbf
    style Mode3 fill:#fbf
```

---

## 12. Full Integration Overview

```mermaid
flowchart TB
    subgraph External["External Systems"]
        EX1[Binance]
        EX2[Coinbase]
        EX3[Interactive Brokers]
    end

    subgraph Nautilus["Nautilus Trader Layer"]
        VEN[Venue Adapters]
        DAT[DataEngine]
        MSG[MessageBus]
        RIS[RiskEngine]
        EXE[ExecEngine]
    end

    subgraph Integration["hyperphysics-nautilus"]
        ADP[Adapters]
        STR[Strategy]
        BCK[Backtest]
    end

    subgraph HyperPhysics["HyperPhysics Engine"]
        PIP[UnifiedPipeline]
        PHY[Physics Models]
        OPT[Optimization]
        CON[Consensus]
    end

    subgraph Storage["Data Storage"]
        TS[TimescaleDB]
        RD[Redis Cache]
        FI[File System]
    end

    External --> VEN
    VEN --> DAT
    DAT --> MSG
    MSG --> ADP
    ADP --> PIP
    PIP --> PHY
    PHY --> OPT
    OPT --> CON
    CON --> STR
    STR --> RIS
    RIS --> EXE
    EXE --> VEN

    DAT --> TS
    PIP --> RD
    BCK --> FI

    style Integration fill:#f9f,stroke:#333,stroke-width:2px
    style HyperPhysics fill:#bbf,stroke:#333,stroke-width:2px
```

---

## Usage Notes

These diagrams can be rendered using:
- GitHub Markdown (native Mermaid support)
- VS Code with Mermaid extension
- Mermaid Live Editor: https://mermaid.live
- Any documentation tool supporting Mermaid

To embed in other documentation:
```markdown
```mermaid
[diagram code here]
```
```

---

## Related Documentation

- [Nautilus Assessment](../integration/NAUTILUS_TRADER_ASSESSMENT.md)
- [Integration Architecture](../integration/HYPERPHYSICS_NAUTILUS_ARCHITECTURE.md)
- [HyperPhysics Architecture](../architecture/hyperphysics_unified_architecture_diagrams.md)
