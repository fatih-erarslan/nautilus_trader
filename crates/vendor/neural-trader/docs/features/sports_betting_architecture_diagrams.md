# Sports Betting Platform - Architecture Diagrams

## System Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        WEB[Web Application]
        MOBILE[Mobile Apps]
        API_SDK[API SDKs]
    end
    
    subgraph "API Gateway"
        KONG[Kong Gateway]
        AUTH[Auth Service]
        RATE[Rate Limiter]
    end
    
    subgraph "Core Services"
        subgraph "Trading Services"
            TS[Trading Engine]
            NS[News Analyzer]
            PS[Portfolio Service]
        end
        
        subgraph "Betting Services"
            BE[Betting Engine]
            OA[Odds Aggregator]
            SS[Settlement Service]
            SM[Syndicate Manager]
        end
        
        subgraph "Shared Services"
            RM[Risk Manager]
            ML[ML Pipeline]
            AN[Analytics Engine]
        end
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL)]
        TS_DB[(TimescaleDB)]
        REDIS[(Redis)]
        KAFKA[Kafka Streams]
    end
    
    subgraph "External APIs"
        BOOK[Bookmaker APIs]
        SPORT[Sports Data APIs]
        MARKET[Market Data APIs]
        NEWS[News APIs]
    end
    
    WEB --> KONG
    MOBILE --> KONG
    API_SDK --> KONG
    
    KONG --> AUTH
    KONG --> RATE
    
    KONG --> TS
    KONG --> BE
    KONG --> RM
    
    BE --> OA
    BE --> SS
    BE --> SM
    
    RM --> TS
    RM --> BE
    
    ML --> AN
    AN --> TS
    AN --> BE
    
    TS --> PG
    BE --> PG
    OA --> TS_DB
    
    BE --> REDIS
    OA --> REDIS
    
    OA --> KAFKA
    SS --> KAFKA
    
    OA --> BOOK
    BE --> SPORT
    TS --> MARKET
    NS --> NEWS
```

## Microservices Architecture

```mermaid
graph LR
    subgraph "Betting Engine Service"
        BE_API[REST API]
        BE_LOGIC[Business Logic]
        BE_VAL[Validation]
        BE_EXEC[Execution]
    end
    
    subgraph "Odds Aggregator Service"
        OA_COL[Collector]
        OA_NORM[Normalizer]
        OA_CALC[Calculator]
        OA_CACHE[Cache Manager]
    end
    
    subgraph "Risk Management Service"
        RM_VAR[VaR Engine]
        RM_MON[Monitor]
        RM_ALERT[Alerting]
        RM_HEDGE[Hedging]
    end
    
    subgraph "ML Prediction Service"
        ML_FEAT[Feature Store]
        ML_INF[Inference]
        ML_TRAIN[Training]
        ML_EVAL[Evaluation]
    end
    
    BE_API --> BE_LOGIC
    BE_LOGIC --> BE_VAL
    BE_VAL --> BE_EXEC
    
    OA_COL --> OA_NORM
    OA_NORM --> OA_CALC
    OA_CALC --> OA_CACHE
    
    BE_EXEC --> RM_MON
    RM_MON --> RM_VAR
    RM_VAR --> RM_ALERT
    RM_ALERT --> RM_HEDGE
    
    ML_FEAT --> ML_INF
    ML_TRAIN --> ML_EVAL
    ML_EVAL --> ML_INF
    
    BE_LOGIC --> ML_INF
    OA_CALC --> ML_INF
    ML_INF --> RM_VAR
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant API Gateway
    participant Betting Engine
    participant Odds Aggregator
    participant ML Service
    participant Risk Manager
    participant Bookmaker API
    participant Database
    
    User->>API Gateway: Place Bet Request
    API Gateway->>Betting Engine: Validate & Route
    
    Betting Engine->>ML Service: Get Prediction
    ML Service-->>Betting Engine: Prediction & Confidence
    
    Betting Engine->>Odds Aggregator: Get Best Odds
    Odds Aggregator->>Bookmaker API: Query Multiple Sources
    Bookmaker API-->>Odds Aggregator: Current Odds
    Odds Aggregator-->>Betting Engine: Best Odds & Provider
    
    Betting Engine->>Risk Manager: Check Exposure
    Risk Manager-->>Betting Engine: Risk Approval
    
    Betting Engine->>Bookmaker API: Place Bet
    Bookmaker API-->>Betting Engine: Confirmation
    
    Betting Engine->>Database: Store Bet Details
    Betting Engine-->>User: Bet Confirmation
```

## ML Pipeline Architecture

```mermaid
graph TD
    subgraph "Data Collection"
        HIST[Historical Data]
        LIVE[Live Data]
        NEWS[News Feeds]
        SOCIAL[Social Media]
    end
    
    subgraph "Feature Engineering"
        FE[Feature Extractor]
        FS[Feature Store]
        FV[Feature Validation]
    end
    
    subgraph "Model Training"
        MT[Model Trainer]
        ME[Model Evaluator]
        MR[Model Registry]
    end
    
    subgraph "Model Serving"
        MS[Model Server]
        MC[Model Cache]
        MM[Model Monitor]
    end
    
    subgraph "Inference"
        INF[Inference Engine]
        POST[Post-Processing]
        API[Prediction API]
    end
    
    HIST --> FE
    LIVE --> FE
    NEWS --> FE
    SOCIAL --> FE
    
    FE --> FS
    FS --> FV
    FV --> MT
    
    MT --> ME
    ME --> MR
    MR --> MS
    
    MS --> MC
    MC --> MM
    MM --> INF
    
    INF --> POST
    POST --> API
```

## Database Schema Overview

```mermaid
erDiagram
    USERS ||--o{ BETS : places
    USERS ||--o{ SYNDICATE_MEMBERS : joins
    SYNDICATES ||--o{ SYNDICATE_MEMBERS : has
    SYNDICATES ||--o{ BETS : places
    BETS ||--|| BET_SETTLEMENTS : settles
    EVENTS ||--o{ MARKETS : has
    MARKETS ||--o{ ODDS_HISTORY : tracks
    MARKETS ||--o{ BETS : placed_on
    
    USERS {
        uuid id PK
        string email
        string username
        boolean betting_enabled
        decimal unified_risk_limit
    }
    
    BETS {
        uuid id PK
        uuid user_id FK
        uuid syndicate_id FK
        string sport_id
        string event_id
        decimal stake
        decimal odds
        string status
        timestamp placed_at
    }
    
    SYNDICATES {
        uuid id PK
        string name
        uuid owner_id FK
        decimal total_capital
        json settings
    }
    
    ODDS_HISTORY {
        uuid id PK
        string event_id
        string market_type
        string bookmaker_id
        decimal odds
        timestamp timestamp
    }
```

## Risk Management Flow

```mermaid
flowchart TD
    A[New Bet Request] --> B{Risk Check}
    B -->|Pass| C[Calculate Exposure]
    B -->|Fail| D[Reject Bet]
    
    C --> E{Within Limits?}
    E -->|Yes| F[Check Correlations]
    E -->|No| G[Suggest Reduction]
    
    F --> H{Correlation Risk?}
    H -->|Low| I[Approve Bet]
    H -->|High| J[Warning + Hedging]
    
    I --> K[Place Bet]
    J --> L{User Decision}
    L -->|Accept Risk| K
    L -->|Hedge| M[Create Hedge]
    M --> K
    
    G --> N{User Decision}
    N -->|Reduce| O[Adjust Stake]
    N -->|Cancel| D
    O --> B
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        subgraph "Region 1 - Primary"
            LB1[Load Balancer]
            subgraph "Kubernetes Cluster 1"
                API1[API Pods]
                SVC1[Service Pods]
                ML1[ML Pods]
            end
            DB1[(Primary DB)]
            CACHE1[(Redis Master)]
        end
        
        subgraph "Region 2 - Secondary"
            LB2[Load Balancer]
            subgraph "Kubernetes Cluster 2"
                API2[API Pods]
                SVC2[Service Pods]
                ML2[ML Pods]
            end
            DB2[(Replica DB)]
            CACHE2[(Redis Replica)]
        end
        
        subgraph "Shared Services"
            CDN[CDN]
            KAFKA[Kafka Cluster]
            MONITORING[Monitoring Stack]
        end
    end
    
    subgraph "External Services"
        BOOKMAKERS[Bookmaker APIs]
        SPORTS[Sports Data]
        PAYMENT[Payment Providers]
    end
    
    CDN --> LB1
    CDN --> LB2
    
    LB1 --> API1
    LB2 --> API2
    
    API1 --> SVC1
    API2 --> SVC2
    
    SVC1 --> ML1
    SVC2 --> ML2
    
    SVC1 --> DB1
    SVC2 --> DB2
    DB1 -.->|Replication| DB2
    
    SVC1 --> CACHE1
    SVC2 --> CACHE2
    CACHE1 -.->|Replication| CACHE2
    
    SVC1 --> KAFKA
    SVC2 --> KAFKA
    
    SVC1 --> BOOKMAKERS
    SVC2 --> SPORTS
    
    MONITORING --> API1
    MONITORING --> API2
    MONITORING --> SVC1
    MONITORING --> SVC2
```

## Integration Points

```mermaid
graph LR
    subgraph "Trading Platform"
        TP_USER[User Service]
        TP_RISK[Risk Service]
        TP_ML[ML Service]
        TP_PORT[Portfolio Service]
    end
    
    subgraph "Shared Layer"
        AUTH[Authentication]
        RISK[Unified Risk]
        ML[ML Pipeline]
        DATA[Data Lake]
    end
    
    subgraph "Betting Platform"
        BP_BET[Betting Service]
        BP_ODDS[Odds Service]
        BP_SYND[Syndicate Service]
        BP_SETTLE[Settlement Service]
    end
    
    TP_USER --> AUTH
    BP_BET --> AUTH
    
    TP_RISK --> RISK
    BP_BET --> RISK
    
    TP_ML --> ML
    BP_ODDS --> ML
    
    TP_PORT --> DATA
    BP_SETTLE --> DATA
    
    style AUTH fill:#f9f,stroke:#333,stroke-width:4px
    style RISK fill:#f9f,stroke:#333,stroke-width:4px
    style ML fill:#f9f,stroke:#333,stroke-width:4px
    style DATA fill:#f9f,stroke:#333,stroke-width:4px
```

## Performance Monitoring Dashboard

```mermaid
graph TD
    subgraph "Metrics Collection"
        PROM[Prometheus]
        GRAF[Grafana]
        ELK[ELK Stack]
        JAEG[Jaeger]
    end
    
    subgraph "Key Metrics"
        subgraph "API Metrics"
            LAT[Latency p50/p95/p99]
            THR[Throughput req/s]
            ERR[Error Rate %]
        end
        
        subgraph "Business Metrics"
            BPM[Bets per Minute]
            AOV[Avg Odds Value]
            WIN[Win Rate %]
        end
        
        subgraph "System Metrics"
            CPU[CPU Usage]
            MEM[Memory Usage]
            GPU[GPU Utilization]
        end
    end
    
    subgraph "Alerts"
        ALERT[Alert Manager]
        SLACK[Slack]
        PAGE[PagerDuty]
    end
    
    LAT --> PROM
    THR --> PROM
    ERR --> PROM
    
    BPM --> PROM
    AOV --> PROM
    WIN --> PROM
    
    CPU --> PROM
    MEM --> PROM
    GPU --> PROM
    
    PROM --> GRAF
    PROM --> ALERT
    
    ALERT --> SLACK
    ALERT --> PAGE
```

---

These diagrams provide a comprehensive visual representation of the sports betting platform architecture, showing:

1. **System Overview**: High-level component relationships
2. **Microservices Architecture**: Internal service structure
3. **Data Flow**: Request processing sequence
4. **ML Pipeline**: Machine learning workflow
5. **Database Schema**: Entity relationships
6. **Risk Management**: Decision flow
7. **Deployment Architecture**: Infrastructure layout
8. **Integration Points**: Shared components with trading platform
9. **Performance Monitoring**: Metrics and alerting structure

Each diagram can be rendered using any Mermaid-compatible viewer or documentation tool.