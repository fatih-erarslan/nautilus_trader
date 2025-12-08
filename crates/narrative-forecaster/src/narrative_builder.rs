use crate::{MarketContext, NarrativeError};

pub struct NarrativeBuilder;

impl NarrativeBuilder {
    pub fn new() -> Self {
        Self
    }
    
    pub fn build_future_retrospective_prompt(
        &self,
        symbol: &str,
        context: &MarketContext,
    ) -> Result<String, NarrativeError> {
        let formatted_support = format!("${:.2}", context.support_level);
        let formatted_resistance = format!("${:.2}", context.resistance_level);
        let formatted_volume = format!("{:.2}", context.volume);
        
        // Build additional context section
        let mut context_section = String::new();
        if !context.additional_context.is_empty() {
            context_section.push_str("Additional Market Context:\n");
            for (key, value) in &context.additional_context {
                context_section.push_str(&format!("- {}: {}\n", key, value));
            }
        }
        
        let prompt = format!(r#"You are an elite financial analyst with deep expertise in cryptocurrency markets and behavioral economics. Using Claude Sonnet 4's advanced reasoning capabilities, perform a comprehensive analysis by looking back from tomorrow to describe what happened to {} given these current conditions:

Current Market Conditions:
- Current Price: {}
- 24h Trading Volume: {}
- Support Level: {}
- Resistance Level: {}
{}

Apply multi-layered reasoning to frame your response as a retrospective analysis:

**Technical Analysis Layer:**
1. Precise price movements in the next 24 hours with specific levels
2. Volume dynamics and their correlation with price action
3. Support/resistance level interactions and breakout scenarios
4. Pattern completion or failure analysis

**Behavioral Economics Layer:**
1. Market psychology shifts and crowd behavior patterns
2. Fear & Greed Index implications and sentiment cascades
3. Institutional vs retail trader positioning changes
4. Information asymmetry impacts and market efficiency gaps

**Multi-Factor Integration:**
1. Cross-correlations between technical signals and sentiment
2. Macro-economic backdrop influence on micro-movements
3. Liquidity flow analysis and market depth considerations
4. Risk-adjusted probability scenarios with confidence intervals

**Advanced Sentiment Analysis:**
1. Granular market mood assessment (not just bullish/bearish/neutral)
2. Trader confidence levels with uncertainty quantification
3. Fear vs greed balance with emotional intensity metrics
4. Volatility expectations with regime change probabilities
5. Momentum persistence vs mean reversion likelihood

Write in past tense as if looking back from tomorrow. Use Sonnet 4's reasoning to provide:
- Specific price targets with statistical confidence
- Causal chains linking multiple factors
- Probabilistic scenarios with their likelihood

End with a structured summary:
PRICE PREDICTION: [exact price with reasoning]
CONFIDENCE: [score 0.0-1.0 with justification]
KEY FACTORS: [3-4 interconnected factors with causal relationships]
REASONING DEPTH: [brief meta-analysis of prediction logic]"#,
            symbol,
            formatted_support,
            formatted_volume,
            formatted_support,
            formatted_resistance,
            context_section
        );
        
        Ok(prompt)
    }
}