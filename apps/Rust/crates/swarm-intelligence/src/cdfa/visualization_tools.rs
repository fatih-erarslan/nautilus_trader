//! Comprehensive Visualization Tools for CDFA
//! 
//! This module provides advanced visualization capabilities for CDFA analysis,
//! including real-time plotting, performance dashboards, and interactive charts
//! for trading signal analysis and algorithm performance monitoring.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use plotters::prelude::*;
use plotters::coord::Shift;

use crate::errors::SwarmError;
use super::ml_integration::{SignalFeatures, ProcessedSignal, MLExperience};
use super::performance_tracker::{PerformanceMetrics, TimestampedMetrics};

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Output directory for generated plots
    pub output_dir: String,
    
    /// Default plot dimensions
    pub plot_width: u32,
    pub plot_height: u32,
    
    /// Color scheme
    pub color_scheme: ColorScheme,
    
    /// Font configuration
    pub font_family: String,
    pub font_size: u32,
    
    /// Animation settings
    pub enable_animation: bool,
    pub animation_fps: u32,
    
    /// Real-time update frequency
    pub update_frequency_ms: u64,
    
    /// Data retention for plots
    pub max_data_points: usize,
    
    /// Export formats
    pub export_formats: Vec<ExportFormat>,
    
    /// Interactive features
    pub enable_interactivity: bool,
    pub enable_zoom: bool,
    pub enable_pan: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Default,
    Dark,
    Professional,
    HighContrast,
    Custom(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    PNG,
    SVG,
    PDF,
    HTML,
    JSON,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            output_dir: "./plots".to_string(),
            plot_width: 1200,
            plot_height: 800,
            color_scheme: ColorScheme::Professional,
            font_family: "Arial".to_string(),
            font_size: 12,
            enable_animation: true,
            animation_fps: 30,
            update_frequency_ms: 100,
            max_data_points: 10000,
            export_formats: vec![ExportFormat::PNG, ExportFormat::SVG],
            enable_interactivity: true,
            enable_zoom: true,
            enable_pan: true,
        }
    }
}

/// Plot data structure for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub timestamps: Vec<DateTime<Utc>>,
    pub values: Vec<f64>,
    pub label: String,
    pub color: String,
    pub line_style: LineStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

/// Heatmap data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    pub data: Vec<Vec<f64>>,
    pub x_labels: Vec<String>,
    pub y_labels: Vec<String>,
    pub color_scale: ColorScale,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScale {
    Viridis,
    Plasma,
    Inferno,
    Magma,
    BlueRed,
    Custom(Vec<String>),
}

/// 3D plot data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plot3DData {
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
    pub z_data: Vec<f64>,
    pub colors: Option<Vec<String>>,
    pub point_sizes: Option<Vec<f64>>,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub title: String,
    pub layout: DashboardLayout,
    pub panels: Vec<DashboardPanel>,
    pub update_interval_ms: u64,
    pub auto_refresh: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardLayout {
    Grid { rows: u32, cols: u32 },
    Flexible,
    Tabbed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPanel {
    pub id: String,
    pub title: String,
    pub plot_type: PlotType,
    pub data_source: String,
    pub position: PanelPosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelPosition {
    pub row: u32,
    pub col: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotType {
    TimeSeries,
    Heatmap,
    Scatter,
    Bar,
    Histogram,
    Violin,
    Box,
    Surface3D,
    Contour,
    Candlestick,
    Volume,
    PerformanceMetrics,
    ConfusionMatrix,
    ROCCurve,
    LearningCurve,
}

/// Comprehensive visualization engine
pub struct CDFAVisualizationEngine {
    /// Configuration
    config: VisualizationConfig,
    
    /// Active dashboards
    dashboards: Arc<RwLock<HashMap<String, Dashboard>>>,
    
    /// Data buffers for real-time plotting
    data_buffers: Arc<RwLock<HashMap<String, TimeSeriesBuffer>>>,
    
    /// Plot cache for performance
    plot_cache: Arc<RwLock<HashMap<String, CachedPlot>>>,
    
    /// Performance metrics history
    metrics_history: Arc<RwLock<Vec<TimestampedMetrics>>>,
    
    /// Signal processing history
    signal_history: Arc<RwLock<HashMap<String, Vec<ProcessedSignal>>>>,
}

/// Time series data buffer with efficient updates
pub struct TimeSeriesBuffer {
    data: Vec<TimeSeriesData>,
    max_points: usize,
    last_update: DateTime<Utc>,
}

impl TimeSeriesBuffer {
    pub fn new(max_points: usize) -> Self {
        Self {
            data: Vec::new(),
            max_points,
            last_update: Utc::now(),
        }
    }
    
    pub fn add_series(&mut self, series: TimeSeriesData) {
        self.data.push(series);
        self.trim_data();
        self.last_update = Utc::now();
    }
    
    pub fn update_series(&mut self, label: &str, timestamp: DateTime<Utc>, value: f64) {
        for series in &mut self.data {
            if series.label == label {
                series.timestamps.push(timestamp);
                series.values.push(value);
                
                // Maintain buffer size
                if series.timestamps.len() > self.max_points {
                    series.timestamps.remove(0);
                    series.values.remove(0);
                }
                break;
            }
        }
        self.last_update = Utc::now();
    }
    
    fn trim_data(&mut self) {
        for series in &mut self.data {
            while series.timestamps.len() > self.max_points {
                series.timestamps.remove(0);
                series.values.remove(0);
            }
        }
    }
}

/// Cached plot for performance optimization
pub struct CachedPlot {
    plot_data: Vec<u8>,
    last_update: DateTime<Utc>,
    data_hash: u64,
}

/// Interactive dashboard
pub struct Dashboard {
    config: DashboardConfig,
    panels: HashMap<String, DashboardPanel>,
    last_update: DateTime<Utc>,
}

impl CDFAVisualizationEngine {
    /// Create new visualization engine
    pub async fn new(config: VisualizationConfig) -> Result<Self, SwarmError> {
        // Create output directory
        std::fs::create_dir_all(&config.output_dir)
            .map_err(|e| SwarmError::IOError(format!("Failed to create output directory: {}", e)))?;
        
        Ok(Self {
            config,
            dashboards: Arc::new(RwLock::new(HashMap::new())),
            data_buffers: Arc::new(RwLock::new(HashMap::new())),
            plot_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            signal_history: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Create time series plot
    pub async fn plot_time_series(
        &self,
        data: &[TimeSeriesData],
        title: &str,
        x_label: &str,
        y_label: &str,
    ) -> Result<String, SwarmError> {
        let output_path = format!("{}/time_series_{}.png", self.config.output_dir, 
                                 title.replace(" ", "_").to_lowercase());
        
        let root = BitMapBackend::new(&output_path, (self.config.plot_width, self.config.plot_height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to fill background: {}", e)))?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("Arial", (self.config.font_size + 8) as i32))
            .margin(5)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(
                self.get_time_range(data)?,
                self.get_value_range(data)?
            )
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to build chart: {}", e)))?;
        
        chart.configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .axis_desc_style(("Arial", self.config.font_size as i32))
            .draw()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to configure mesh: {}", e)))?;
        
        // Plot each time series
        for (i, series) in data.iter().enumerate() {
            let color = self.get_color_for_series(i);
            let line_data: Vec<(DateTime<Utc>, f64)> = series.timestamps.iter()
                .zip(series.values.iter())
                .map(|(&t, &v)| (t, v))
                .collect();
            
            chart.draw_series(LineSeries::new(
                line_data.iter().map(|(t, v)| (*t, *v)),
                &color
            ))
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to draw series: {}", e)))?
            .label(&series.label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &color));
        }
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to draw legend: {}", e)))?;
        
        root.present()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to present plot: {}", e)))?;
        
        Ok(output_path)
    }
    
    /// Create heatmap visualization
    pub async fn plot_heatmap(
        &self,
        data: &HeatmapData,
        title: &str,
    ) -> Result<String, SwarmError> {
        let output_path = format!("{}/heatmap_{}.png", self.config.output_dir, 
                                 title.replace(" ", "_").to_lowercase());
        
        let root = BitMapBackend::new(&output_path, (self.config.plot_width, self.config.plot_height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to fill background: {}", e)))?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("Arial", (self.config.font_size + 8) as i32))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(80)
            .build_cartesian_2d(
                0f32..(data.x_labels.len() as f32),
                0f32..(data.y_labels.len() as f32)
            )
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to build heatmap chart: {}", e)))?;
        
        // Draw heatmap cells
        for (y, row) in data.data.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let color = self.value_to_color(value, &data.color_scale);
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x as f32, y as f32), ((x + 1) as f32, (y + 1) as f32)],
                    color.filled()
                )))
                .map_err(|e| SwarmError::VisualizationError(format!("Failed to draw heatmap cell: {}", e)))?;
            }
        }
        
        root.present()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to present heatmap: {}", e)))?;
        
        Ok(output_path)
    }
    
    /// Create performance dashboard
    pub async fn create_performance_dashboard(
        &self,
        metrics: &[PerformanceMetrics],
        title: &str,
    ) -> Result<String, SwarmError> {
        let output_path = format!("{}/dashboard_{}.png", self.config.output_dir, 
                                 title.replace(" ", "_").to_lowercase());
        
        let root = BitMapBackend::new(&output_path, (self.config.plot_width * 2, self.config.plot_height * 2))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to fill dashboard background: {}", e)))?;
        
        // Split into 4 quadrants
        let (upper, lower) = root.split_evenly((true, false));
        let (q1, q2) = upper.split_evenly((false, true));
        let (q3, q4) = lower.split_evenly((false, true));
        
        // Q1: Algorithm performance over time
        self.plot_algorithm_performance(&q1, metrics).await?;
        
        // Q2: Diversity metrics
        self.plot_diversity_metrics(&q2, metrics).await?;
        
        // Q3: Resource utilization
        self.plot_resource_utilization(&q3, metrics).await?;
        
        // Q4: Convergence analysis
        self.plot_convergence_analysis(&q4, metrics).await?;
        
        root.present()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to present dashboard: {}", e)))?;
        
        Ok(output_path)
    }
    
    /// Plot algorithm performance over time
    async fn plot_algorithm_performance<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        metrics: &[PerformanceMetrics],
    ) -> Result<(), SwarmError>
    where
        DB::ErrorType: 'static,
    {
        let mut chart = ChartBuilder::on(area)
            .caption("Algorithm Performance", ("Arial", self.config.font_size as i32))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0f32..(metrics.len() as f32),
                0f64..1f64
            )
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to build performance chart: {:?}", e)))?;
        
        chart.configure_mesh()
            .x_desc("Time")
            .y_desc("Performance")
            .draw()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to configure performance mesh: {:?}", e)))?;
        
        // Plot accuracy over time
        let accuracy_data: Vec<(f32, f64)> = metrics.iter()
            .enumerate()
            .map(|(i, m)| (i as f32, m.accuracy))
            .collect();
        
        chart.draw_series(LineSeries::new(accuracy_data, &BLUE))
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to draw accuracy series: {:?}", e)))?
            .label("Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));
        
        Ok(())
    }
    
    /// Plot diversity metrics
    async fn plot_diversity_metrics<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        metrics: &[PerformanceMetrics],
    ) -> Result<(), SwarmError>
    where
        DB::ErrorType: 'static,
    {
        let mut chart = ChartBuilder::on(area)
            .caption("Diversity Metrics", ("Arial", self.config.font_size as i32))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0f32..(metrics.len() as f32),
                0f64..1f64
            )
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to build diversity chart: {:?}", e)))?;
        
        chart.configure_mesh()
            .x_desc("Time")
            .y_desc("Diversity")
            .draw()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to configure diversity mesh: {:?}", e)))?;
        
        // Plot diversity metrics (using standard deviation as proxy)
        let diversity_data: Vec<(f32, f64)> = metrics.iter()
            .enumerate()
            .map(|(i, m)| (i as f32, m.standard_deviation))
            .collect();
        
        chart.draw_series(LineSeries::new(diversity_data, &GREEN))
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to draw diversity series: {:?}", e)))?
            .label("Diversity")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &GREEN));
        
        Ok(())
    }
    
    /// Plot resource utilization
    async fn plot_resource_utilization<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        metrics: &[PerformanceMetrics],
    ) -> Result<(), SwarmError>
    where
        DB::ErrorType: 'static,
    {
        let mut chart = ChartBuilder::on(area)
            .caption("Resource Utilization", ("Arial", self.config.font_size as i32))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0f32..(metrics.len() as f32),
                0f64..100f64
            )
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to build resource chart: {:?}", e)))?;
        
        chart.configure_mesh()
            .x_desc("Time")
            .y_desc("Utilization %")
            .draw()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to configure resource mesh: {:?}", e)))?;
        
        // Plot memory utilization (using max_value as proxy)
        let memory_data: Vec<(f32, f64)> = metrics.iter()
            .enumerate()
            .map(|(i, m)| (i as f32, m.max_value))
            .collect();
        
        chart.draw_series(LineSeries::new(memory_data, &RED))
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to draw memory series: {:?}", e)))?
            .label("Memory")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));
        
        Ok(())
    }
    
    /// Plot convergence analysis
    async fn plot_convergence_analysis<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        metrics: &[PerformanceMetrics],
    ) -> Result<(), SwarmError>
    where
        DB::ErrorType: 'static,
    {
        let mut chart = ChartBuilder::on(area)
            .caption("Convergence Analysis", ("Arial", self.config.font_size as i32))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0f32..(metrics.len() as f32),
                0f64..1f64
            )
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to build convergence chart: {:?}", e)))?;
        
        chart.configure_mesh()
            .x_desc("Time")
            .y_desc("Convergence")
            .draw()
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to configure convergence mesh: {:?}", e)))?;
        
        // Plot convergence rate (using min_value as proxy)
        let convergence_data: Vec<(f32, f64)> = metrics.iter()
            .enumerate()
            .map(|(i, m)| (i as f32, m.min_value))
            .collect();
        
        chart.draw_series(LineSeries::new(convergence_data, &MAGENTA))
            .map_err(|e| SwarmError::VisualizationError(format!("Failed to draw convergence series: {:?}", e)))?
            .label("Convergence")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &MAGENTA));
        
        Ok(())
    }
    
    /// Get time range for time series data
    fn get_time_range(&self, data: &[TimeSeriesData]) -> Result<std::ops::Range<DateTime<Utc>>, SwarmError> {
        let mut min_time = Utc::now();
        let mut max_time = Utc::now();
        
        for series in data {
            if let (Some(&first), Some(&last)) = (series.timestamps.first(), series.timestamps.last()) {
                if first < min_time { min_time = first; }
                if last > max_time { max_time = last; }
            }
        }
        
        Ok(min_time..max_time)
    }
    
    /// Get value range for time series data
    fn get_value_range(&self, data: &[TimeSeriesData]) -> Result<std::ops::Range<f64>, SwarmError> {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        
        for series in data {
            for &value in &series.values {
                if value < min_val { min_val = value; }
                if value > max_val { max_val = value; }
            }
        }
        
        // Add some padding
        let padding = (max_val - min_val) * 0.1;
        Ok((min_val - padding)..(max_val + padding))
    }
    
    /// Get color for series based on index
    fn get_color_for_series(&self, index: usize) -> RGBColor {
        let colors = [
            RGBColor(31, 119, 180),   // Blue
            RGBColor(255, 127, 14),   // Orange
            RGBColor(44, 160, 44),    // Green
            RGBColor(214, 39, 40),    // Red
            RGBColor(148, 103, 189),  // Purple
            RGBColor(140, 86, 75),    // Brown
            RGBColor(227, 119, 194),  // Pink
            RGBColor(127, 127, 127),  // Gray
            RGBColor(188, 189, 34),   // Olive
            RGBColor(23, 190, 207),   // Cyan
        ];
        
        colors[index % colors.len()]
    }
    
    /// Convert value to color for heatmap
    fn value_to_color(&self, value: f64, color_scale: &ColorScale) -> RGBColor {
        // Normalize value to 0-1 range (assuming input is already normalized)
        let normalized = value.max(0.0).min(1.0);
        
        match color_scale {
            ColorScale::Viridis => {
                // Simplified viridis color mapping
                let r = (normalized * 255.0) as u8;
                let g = ((1.0 - normalized) * 255.0) as u8;
                let b = (normalized * 0.5 * 255.0) as u8;
                RGBColor(r, g, b)
            },
            ColorScale::BlueRed => {
                if normalized < 0.5 {
                    let intensity = (normalized * 2.0 * 255.0) as u8;
                    RGBColor(0, 0, intensity)
                } else {
                    let intensity = ((normalized - 0.5) * 2.0 * 255.0) as u8;
                    RGBColor(intensity, 0, 0)
                }
            },
            _ => {
                // Default grayscale
                let intensity = (normalized * 255.0) as u8;
                RGBColor(intensity, intensity, intensity)
            }
        }
    }
    
    /// Add real-time data update
    pub async fn update_real_time_data(
        &self,
        buffer_id: &str,
        label: &str,
        value: f64,
    ) -> Result<(), SwarmError> {
        let mut buffers = self.data_buffers.write().await;
        
        if let Some(buffer) = buffers.get_mut(buffer_id) {
            buffer.update_series(label, Utc::now(), value);
        } else {
            // Create new buffer
            let mut new_buffer = TimeSeriesBuffer::new(self.config.max_data_points);
            let series = TimeSeriesData {
                timestamps: vec![Utc::now()],
                values: vec![value],
                label: label.to_string(),
                color: "#1f77b4".to_string(),
                line_style: LineStyle::Solid,
            };
            new_buffer.add_series(series);
            buffers.insert(buffer_id.to_string(), new_buffer);
        }
        
        Ok(())
    }
    
    /// Export dashboard as HTML
    pub async fn export_dashboard_html(
        &self,
        dashboard_id: &str,
        output_path: &str,
    ) -> Result<(), SwarmError> {
        // Generate HTML dashboard with embedded plots
        let html_content = self.generate_html_dashboard(dashboard_id).await?;
        
        std::fs::write(output_path, html_content)
            .map_err(|e| SwarmError::IOError(format!("Failed to write HTML dashboard: {}", e)))?;
        
        Ok(())
    }
    
    /// Generate HTML dashboard content
    async fn generate_html_dashboard(&self, dashboard_id: &str) -> Result<String, SwarmError> {
        let html = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>CDFA Dashboard - {}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .panel {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>CDFA Real-Time Dashboard</h1>
    <div class="dashboard">
        <div class="panel">
            <div class="title">Performance Metrics</div>
            <div id="performance-plot"></div>
        </div>
        <div class="panel">
            <div class="title">Diversity Analysis</div>
            <div id="diversity-plot"></div>
        </div>
        <div class="panel">
            <div class="title">Signal Processing</div>
            <div id="signals-plot"></div>
        </div>
        <div class="panel">
            <div class="title">Resource Utilization</div>
            <div id="resources-plot"></div>
        </div>
    </div>
    
    <script>
        // Initialize real-time plots
        function initializePlots() {{
            // Performance metrics plot
            Plotly.newPlot('performance-plot', [{{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Accuracy'
            }}], {{
                title: 'Algorithm Performance',
                xaxis: {{ title: 'Time' }},
                yaxis: {{ title: 'Performance' }}
            }});
            
            // Diversity analysis plot
            Plotly.newPlot('diversity-plot', [{{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Diversity'
            }}], {{
                title: 'Diversity Metrics',
                xaxis: {{ title: 'Time' }},
                yaxis: {{ title: 'Diversity Score' }}
            }});
            
            // Signal processing plot
            Plotly.newPlot('signals-plot', [{{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Signal Strength'
            }}], {{
                title: 'Signal Analysis',
                xaxis: {{ title: 'Time' }},
                yaxis: {{ title: 'Signal Value' }}
            }});
            
            // Resource utilization plot
            Plotly.newPlot('resources-plot', [{{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Memory Usage'
            }}], {{
                title: 'System Resources',
                xaxis: {{ title: 'Time' }},
                yaxis: {{ title: 'Usage %' }}
            }});
        }}
        
        // Update plots with real-time data
        function updatePlots() {{
            // This would connect to a WebSocket or REST API for real-time updates
            // For now, we'll simulate with random data
            const now = new Date();
            
            Plotly.extendTraces('performance-plot', {{
                x: [[now]],
                y: [[Math.random() * 0.3 + 0.7]]
            }}, [0]);
            
            Plotly.extendTraces('diversity-plot', {{
                x: [[now]],
                y: [[Math.random() * 0.5 + 0.3]]
            }}, [0]);
            
            Plotly.extendTraces('signals-plot', {{
                x: [[now]],
                y: [[Math.sin(Date.now() / 1000) * 0.5 + 0.5]]
            }}, [0]);
            
            Plotly.extendTraces('resources-plot', {{
                x: [[now]],
                y: [[Math.random() * 30 + 40]]
            }}, [0]);
        }}
        
        // Initialize and start updates
        initializePlots();
        setInterval(updatePlots, 1000);
    </script>
</body>
</html>
        "#, dashboard_id);
        
        Ok(html)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert!(!config.output_dir.is_empty());
        assert!(config.plot_width > 0);
        assert!(config.plot_height > 0);
        assert!(!config.export_formats.is_empty());
    }
    
    #[test]
    fn test_time_series_data_creation() {
        let data = TimeSeriesData {
            timestamps: vec![Utc::now()],
            values: vec![1.0],
            label: "test".to_string(),
            color: "#ff0000".to_string(),
            line_style: LineStyle::Solid,
        };
        
        assert_eq!(data.label, "test");
        assert_eq!(data.values.len(), 1);
        assert_eq!(data.timestamps.len(), 1);
    }
    
    #[test]
    fn test_time_series_buffer() {
        let mut buffer = TimeSeriesBuffer::new(5);
        
        let series = TimeSeriesData {
            timestamps: vec![Utc::now()],
            values: vec![1.0],
            label: "test".to_string(),
            color: "#ff0000".to_string(),
            line_style: LineStyle::Solid,
        };
        
        buffer.add_series(series);
        assert_eq!(buffer.data.len(), 1);
        
        // Test buffer overflow handling
        for i in 0..10 {
            buffer.update_series("test", Utc::now(), i as f64);
        }
        
        assert!(buffer.data[0].values.len() <= 5);
    }
    
    #[tokio::test]
    async fn test_visualization_engine_creation() {
        let config = VisualizationConfig::default();
        let engine = CDFAVisualizationEngine::new(config).await.unwrap();
        
        let buffers = engine.data_buffers.read().await;
        assert!(buffers.is_empty());
    }
}