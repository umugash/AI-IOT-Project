from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Premium Dark Color Palette ───
COLORS = {"Day": "#f59e0b", "Night": "#818cf8"}
ACCENT = "#6366f1"
SURFACE = "#1e293b"
BG = "#0f172a"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "rgba(148, 163, 184, 0.08)"
HOVER_BG = "rgba(30, 41, 59, 0.95)"

# Plotly layout template for consistent premium look without colliding kwargs
LAYOUT = {
    "template": "plotly_dark",
    "font": {
        "family": "Inter, Segoe UI, system-ui, sans-serif",
        "color": TEXT,
        "size": 13,
    },
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(15, 23, 42, 0.4)",
    "margin": {"l": 16, "r": 16, "t": 60, "b": 16},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "x": 0,
        "font": {"size": 12, "color": MUTED},
        "bgcolor": "rgba(0,0,0,0)",
    },
    "hoverlabel": {
        "bgcolor": HOVER_BG,
        "font_size": 13,
        "font_family": "Inter, Segoe UI, sans-serif",
        "bordercolor": "rgba(99, 102, 241, 0.3)",
    },
}

# ─── Extended color sequences for multi-series ───
STATION_COLORS = [
    "#818cf8", "#f59e0b", "#34d399", "#f472b6", "#38bdf8",
    "#a78bfa", "#fb923c", "#4ade80", "#f43f5e", "#22d3ee",
]

CITY_COLORS = [
    "#818cf8", "#f59e0b", "#34d399", "#f472b6", "#38bdf8",
    "#a78bfa", "#fb923c", "#22d3ee",
]


def _apply_axes_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    return fig


def monthly_trend(data: pd.DataFrame, title: str = "Monthly source values") -> go.Figure:
    long = data.melt(
        id_vars=["observation_period", "Station", "Name"],
        value_vars=["Day", "Night"],
        var_name="metric",
        value_name="source_value",
    ).dropna(subset=["source_value"])
    grouped = (
        long.groupby(["observation_period", "metric"], as_index=False)["source_value"]
        .mean()
        .sort_values("observation_period")
    )
    fig = px.line(
        grouped,
        x="observation_period",
        y="source_value",
        color="metric",
        markers=True,
        color_discrete_map=COLORS,
        title=title,
        labels={"observation_period": "Historical month", "source_value": "Source value"},
    )
    fig.update_traces(line={"width": 2.5}, marker={"size": 5})
    return _apply_axes_style(fig)


def station_time_series(data: pd.DataFrame, metric: str) -> go.Figure:
    ordered = data.sort_values("observation_period")
    fig = px.line(
        ordered,
        x="observation_period",
        y=metric,
        color="Name",
        markers=True,
        title=f"{metric} source values by station",
        labels={"observation_period": "Historical month", metric: "Source value"},
        color_discrete_sequence=STATION_COLORS,
    )
    fig.update_traces(line={"width": 2}, marker={"size": 4})
    return _apply_axes_style(fig)


def day_night_station_comparison(data: pd.DataFrame) -> go.Figure:
    summary = data.groupby(["Station", "Name"], as_index=False)[["Day", "Night"]].mean()
    long = summary.melt(
        id_vars=["Station", "Name"], var_name="metric", value_name="mean_source_value"
    )
    fig = px.bar(
        long,
        x="mean_source_value",
        y="Name",
        color="metric",
        barmode="group",
        orientation="h",
        color_discrete_map=COLORS,
        title="Day / Night station mean comparison",
        labels={"mean_source_value": "Mean source value", "Name": "Station label"},
    )
    fig.update_traces(marker_line_width=0, opacity=0.92)
    return _apply_axes_style(fig)


def distribution_plot(data: pd.DataFrame) -> go.Figure:
    long = data.melt(
        id_vars=["Name"], value_vars=["Day", "Night"], var_name="metric", value_name="value"
    ).dropna(subset=["value"])
    fig = px.histogram(
        long,
        x="value",
        color="metric",
        marginal="box",
        barmode="overlay",
        opacity=0.72,
        color_discrete_map=COLORS,
        title="Distribution of available source values",
        labels={"value": "Source value"},
    )
    return _apply_axes_style(fig)


def monthly_variation(data: pd.DataFrame) -> go.Figure:
    long = data.melt(
        id_vars=["Month"], value_vars=["Day", "Night"], var_name="metric", value_name="value"
    ).dropna(subset=["value"])
    fig = px.box(
        long,
        x="Month",
        y="value",
        color="metric",
        points="outliers",
        color_discrete_map=COLORS,
        title="Month-of-year variation (descriptive)",
        labels={"Month": "Calendar month", "value": "Source value"},
    )
    return _apply_axes_style(fig)


def annual_comparison(data: pd.DataFrame) -> go.Figure:
    annual = data.groupby("Year", as_index=False)[["Day", "Night"]].mean()
    long = annual.melt(id_vars="Year", var_name="metric", value_name="mean_source_value")
    fig = px.line(
        long,
        x="Year",
        y="mean_source_value",
        color="metric",
        markers=True,
        color_discrete_map=COLORS,
        title="Annual mean source values",
        labels={"mean_source_value": "Mean source value"},
    )
    fig.update_traces(line={"width": 3}, marker={"size": 8})
    return _apply_axes_style(fig)


def hotspot_timeline(data: pd.DataFrame) -> go.Figure:
    grouped = (
        data.groupby(["observation_period", "metric"], as_index=False)["relative_high"]
        .mean()
        .sort_values("observation_period")
    )
    fig = px.line(
        grouped,
        x="observation_period",
        y="relative_high",
        color="metric",
        markers=True,
        color_discrete_map=COLORS,
        title="Share of selected stations above pooled historical threshold",
        labels={"relative_high": "Relative-high share", "observation_period": "Historical month"},
    )
    fig.update_traces(line={"width": 2.5}, marker={"size": 5})
    fig.update_yaxes(range=[0, 1])
    return _apply_axes_style(fig)


def hotspot_matrix(data: pd.DataFrame) -> go.Figure:
    matrix = data.pivot_table(
        index="Name", columns="observation_period", values="relative_high", aggfunc="mean"
    )
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale=[
            [0, "rgba(15, 23, 42, 0.8)"],
            [0.3, "rgba(99, 102, 241, 0.3)"],
            [0.7, "#f59e0b"],
            [1, "#ef4444"],
        ],
        zmin=0,
        zmax=1,
        title="Station × month relative-high matrix (not geographic)",
        labels={"x": "Historical month", "y": "Station label", "color": "Relative high"},
    )
    return _apply_axes_style(fig)


def city_comparison(data: pd.DataFrame, metric: str) -> go.Figure:
    summary = data.groupby("City", as_index=False)[metric].mean().sort_values(metric)
    fig = px.bar(
        summary,
        x=metric,
        y="City",
        orientation="h",
        color=metric,
        color_continuous_scale=["#312e81", "#6366f1", "#818cf8", "#a5b4fc"],
        title=f"City-label mean — {metric}",
        labels={metric: "Mean source value", "City": "Source city label"},
    )
    fig.update_traces(marker_line_width=0)
    _apply_axes_style(fig)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def city_year_trends(data: pd.DataFrame, cities: list[str]) -> go.Figure:
    selected = data.loc[data["City"].isin(cities)]
    long = selected.melt(
        id_vars=["City", "Year"],
        value_vars=["Day", "Night"],
        var_name="metric",
        value_name="source_value",
    )
    grouped = (
        long.groupby(["City", "Year", "metric"], as_index=False)["source_value"]
        .mean()
    )
    fig = px.line(
        grouped,
        x="Year",
        y="source_value",
        color="City",
        line_dash="metric",
        markers=True,
        title="Yearly city-label trends",
        labels={"source_value": "Mean source value"},
        color_discrete_sequence=CITY_COLORS,
    )
    fig.update_traces(line={"width": 2}, marker={"size": 5})
    return _apply_axes_style(fig)


def day_night_difference_plot(data: pd.DataFrame) -> go.Figure:
    summary = data.groupby(["Station", "Name"], as_index=False)[["Day", "Night"]].mean()
    summary["delta"] = summary["Day"] - summary["Night"]
    summary = summary.sort_values("delta", ascending=True)
    fig = px.bar(
        summary,
        x="delta",
        y="Name",
        orientation="h",
        color="delta",
        color_continuous_scale=["#3b82f6", "#6366f1", "#f59e0b", "#ef4444"],
        title="Diurnal Noise Variation (Day Mean − Night Mean)",
        labels={"delta": "Day − Night Difference", "Name": "Monitoring Station"},
    )
    fig.update_traces(marker_line_width=0)
    _apply_axes_style(fig)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def forecast_actual_plot(
    data: pd.DataFrame,
    metric: str,
    methods: list[str],
    station: str | None = None,
    station_name: str | None = None,
) -> go.Figure:
    subset = data.loc[data["metric"].eq(metric) & data["method"].isin(methods)].copy()
    if station:
        subset = subset.loc[subset["Station"].eq(station)]
        title = f"{station_name or station} — Actual vs. Backtest Predictions ({metric})"
        actual_label = f"Actual ({station_name or station})"
    else:
        title = f"{metric} Historical Walk-Forward Backtest (Station Mean)"
        actual_label = "Actual Station Mean"

    actual = (
        subset.drop_duplicates(["Station", "observation_period"])
        .groupby("observation_period", as_index=False)["actual"]
        .mean()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual["observation_period"],
            y=actual["actual"],
            name=actual_label,
            mode="lines+markers",
            line={"color": TEXT, "width": 3},
            marker={"size": 6},
        )
    )
    method_colors = ["#818cf8", "#f59e0b", "#34d399", "#f472b6", "#38bdf8", "#a78bfa"]
    for i, method in enumerate(methods):
        forecast = (
            subset.loc[subset["method"].eq(method)]
            .groupby("observation_period", as_index=False)["predicted"]
            .mean()
        )
        if not forecast.empty:
            clean_name = method.replace("_", " ").title()
            fig.add_trace(
                go.Scatter(
                    x=forecast["observation_period"],
                    y=forecast["predicted"],
                    name=clean_name,
                    mode="lines",
                    line={"width": 2, "color": method_colors[i % len(method_colors)]},
                )
            )
    fig.update_layout(
        **LAYOUT,
        title=title,
        xaxis_title="Historical Month",
        yaxis_title="Source Value",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    return fig


def forecast_station_plot(
    data: pd.DataFrame,
    metric: str,
    highlight_station: str | None = None,
) -> go.Figure:
    subset = data.loc[data["metric"].eq(metric)].sort_values("predicted_value", ascending=True)
    fig = go.Figure()
    colors = [
        "#f43f5e" if highlight_station and s == highlight_station else ACCENT
        for s in subset["Station"]
    ]
    sizes = [
        16 if highlight_station and s == highlight_station else 10
        for s in subset["Station"]
    ]
    fig.add_trace(
        go.Scatter(
            x=subset["predicted_value"],
            y=subset["Name"],
            mode="markers",
            name="Persistence Estimate",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": subset["prediction_upper_heuristic"] - subset["predicted_value"],
                "arrayminus": subset["predicted_value"] - subset["prediction_lower_heuristic"],
                "color": "rgba(129, 140, 248, 0.4)",
                "thickness": 2,
            },
            marker={"color": colors, "size": sizes, "line": {"width": 2, "color": "#a5b4fc"}},
        )
    )
    if "historical_relative_hotspot_threshold" in subset.columns and not subset.empty:
        thresh = float(subset["historical_relative_hotspot_threshold"].iloc[0])
        fig.add_vline(
            x=thresh,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Hotspot Threshold ({thresh:.1f})",
            annotation_position="top right",
        )
    fig.update_layout(
        **LAYOUT,
        title=f"Station Spatiotemporal Forecast Comparison — {metric} (Next Period)",
        xaxis_title="Predicted Source Value (with heuristic residual range)",
        yaxis_title="Monitoring Station",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
    return fig


def feature_importance_plot(data: pd.DataFrame, metric: str, model: str) -> go.Figure:
    subset = data.loc[data["metric"].eq(metric) & data["model"].eq(model)].copy()
    top10 = subset.sort_values("importance", ascending=True).tail(10)
    fig = px.bar(
        top10,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#312e81", "#6366f1", "#818cf8", "#a5b4fc"],
        title=f"Top feature importances — {model} ({metric})",
        labels={"importance": "Relative Importance", "feature": "Feature"},
    )
    fig.update_traces(marker_line_width=0)
    _apply_axes_style(fig)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def residual_distribution_plot(data: pd.DataFrame, metric: str, methods: list[str]) -> go.Figure:
    subset = data.loc[data["metric"].eq(metric) & data["method"].isin(methods)].copy()
    fig = px.box(
        subset,
        x="method",
        y="error",
        color="method",
        title=f"Residual distribution by model — {metric} (Error = Actual − Predicted)",
        labels={"error": "Residual error", "method": "Forecasting method"},
        color_discrete_sequence=STATION_COLORS,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, opacity=0.8)
    fig.update_traces(boxpoints="outliers")
    return _apply_axes_style(fig)


def actual_vs_predicted_scatter(data: pd.DataFrame, metric: str, methods: list[str]) -> go.Figure:
    subset = data.loc[data["metric"].eq(metric) & data["method"].isin(methods)].copy()
    fig = px.scatter(
        subset,
        x="actual",
        y="predicted",
        color="method",
        title=f"Actual vs. Predicted values — {metric}",
        labels={"actual": "Observed source value", "predicted": "Backtest prediction"},
        color_discrete_sequence=STATION_COLORS,
        opacity=0.75,
    )
    if not subset.empty:
        min_val = min(subset["actual"].min(), subset["predicted"].min())
        max_val = max(subset["actual"].max(), subset["predicted"].max())
        fig.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line=dict(color=MUTED, width=1.5, dash="dash"),
        )
    return _apply_axes_style(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT INTELLIGENCE CHARTS FOR CPCB 10-MINUTE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def cpcb_continuous_timeline(
    data: pd.DataFrame,
    selected_stations: list[int | str],
    metric_col: str = "noise_value",
) -> go.Figure:
    """Plot multi-station continuous 10-minute noise levels over time."""
    subset = data.loc[data["station_id"].isin(selected_stations)].copy()
    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title="No observations for selected stations", **LAYOUT)
        return fig
        
    fig = px.line(
        subset,
        x="timestamp",
        y=metric_col,
        color="station_name",
        title="Continuous Ambient Noise Levels (10-Minute Resolution)",
        labels={"timestamp": "Observation Time (IST)", metric_col: "Sound Pressure Level (dBA)", "station_name": "Station"},
        color_discrete_sequence=STATION_COLORS,
    )
    fig.update_traces(line=dict(width=1.5))
    return _apply_axes_style(fig)


def cpcb_diurnal_profile(
    data: pd.DataFrame,
    selected_stations: list[int | str],
) -> go.Figure:
    """Plot 24-hour diurnal profile (00:00 to 23:00) by station showing rush-hour peaks."""
    subset = data.loc[data["station_id"].isin(selected_stations)].copy()
    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title="No observations for diurnal curve", **LAYOUT)
        return fig
        
    hourly = (
        subset.groupby(["hour", "station_name"], as_index=False)["noise_value"]
        .mean()
        .sort_values("hour")
    )
    fig = px.line(
        hourly,
        x="hour",
        y="noise_value",
        color="station_name",
        markers=True,
        title="Diurnal Noise Profile Across 24 Hours (Diurnal Hotspot Cycles)",
        labels={"hour": "Hour of Day (00:00 - 23:00 IST)", "noise_value": "Average Sound Level (dBA)", "station_name": "Station"},
        color_discrete_sequence=STATION_COLORS,
    )
    # Highlight statutory day window (06:00 to 22:00)
    fig.add_vrect(
        x0=6, x1=22,
        fillcolor="rgba(245, 158, 11, 0.05)",
        layer="below", line_width=0,
        annotation_text="Day Window (06:00–22:00)",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color=MUTED
    )
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=2)
    return _apply_axes_style(fig)


def cpcb_weekday_weekend_profile(
    data: pd.DataFrame,
    station_id: int | str,
) -> go.Figure:
    """Plot weekday vs weekend hourly comparison for a selected station."""
    subset = data.loc[data["station_id"] == station_id].copy()
    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title="No observations for weekday/weekend comparison", **LAYOUT)
        return fig
        
    st_name = subset["station_name"].iloc[0]
    subset["day_type"] = subset["day_period"].map(lambda x: "Weekday" if x == 0 else "Weekend")
    subset["is_weekend_str"] = subset["timestamp"].dt.day_of_week.map(lambda d: "Weekend (Sat-Sun)" if d >= 5 else "Weekday (Mon-Fri)")
    
    grouped = (
        subset.groupby(["hour", "is_weekend_str"], as_index=False)["noise_value"]
        .mean()
        .sort_values("hour")
    )
    fig = px.line(
        grouped,
        x="hour",
        y="noise_value",
        color="is_weekend_str",
        markers=True,
        title=f"Weekday vs. Weekend Acoustic Variation — {st_name}",
        labels={"hour": "Hour of Day (IST)", "noise_value": "Sound Level (dBA)", "is_weekend_str": "Day Type"},
        color_discrete_map={"Weekday (Mon-Fri)": "#6366f1", "Weekend (Sat-Sun)": "#f59e0b"},
    )
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=2)
    return _apply_axes_style(fig)


def cpcb_standards_compliance_bar(
    daily_data: pd.DataFrame,
    stations_metadata: dict[int, dict[str, Any]],
) -> go.Figure:
    """Compare observed station Day/Night Leq against statutory limits under Noise Rules 2000."""
    records = []
    for sid, meta in stations_metadata.items():
        st_sub = daily_data.loc[daily_data["station_id"] == sid]
        if st_sub.empty:
            continue
            
        day_sub = st_sub.loc[st_sub["day_period"] == "Day"]
        night_sub = st_sub.loc[st_sub["day_period"] == "Night"]
        
        # Energetic Leq mean
        def log_mean(s):
            v = s.dropna().values
            return round(float(10 * np.log10(np.mean(10 ** (v / 10)))), 1) if len(v) > 0 else np.nan
            
        day_leq = log_mean(day_sub["leq"])
        night_leq = log_mean(night_sub["leq"])
        
        records.append({
            "station_name": meta["display_name"],
            "zone": meta["zone_type"],
            "period": "Day (06:00-22:00)",
            "observed_leq": day_leq,
            "statutory_limit": meta["day_limit_dba"],
            "exceedance": round(day_leq - meta["day_limit_dba"], 1) if pd.notna(day_leq) else np.nan,
        })
        records.append({
            "station_name": meta["display_name"],
            "zone": meta["zone_type"],
            "period": "Night (22:00-06:00)",
            "observed_leq": night_leq,
            "statutory_limit": meta["night_limit_dba"],
            "exceedance": round(night_leq - meta["night_limit_dba"], 1) if pd.notna(night_leq) else np.nan,
        })
        
    df_comp = pd.DataFrame(records)
    df_comp["status"] = df_comp["exceedance"].map(lambda x: "Breaches Standard" if x > 0 else "Compliant")
    
    fig = px.bar(
        df_comp,
        x="station_name",
        y="observed_leq",
        color="period",
        barmode="group",
        title="Statutory Ambient Noise Compliance (Day & Night Energetic Leq vs Standards)",
        labels={"station_name": "Monitoring Station", "observed_leq": "Observed Leq dB(A)", "period": "Regulated Period"},
        color_discrete_map={"Day (06:00-22:00)": COLORS["Day"], "Night (22:00-06:00)": COLORS["Night"]},
    )
    # Add horizontal threshold indicators
    fig.add_hline(y=75, line_dash="dot", line_color="#ef4444", annotation_text="Industrial Day (75 dBA)")
    fig.add_hline(y=65, line_dash="dot", line_color="#f59e0b", annotation_text="Commercial Day (65 dBA)")
    fig.add_hline(y=55, line_dash="dot", line_color="#3b82f6", annotation_text="Residential Day (55 dBA)")
    fig.add_hline(y=50, line_dash="dot", line_color="#10b981", annotation_text="Silence Day (50 dBA)")
    return _apply_axes_style(fig)


def cpcb_forecast_comparison_chart(
    test_times: Any,
    y_true: Any,
    y_pred: Any,
    lower_bound: float | Any,
    upper_bound: float | Any,
    station_name: str,
    horizon_label: str,
    best_model_name: str,
    coverage_pct: float | None = None,
    mpiw_dba: float | None = None,
) -> go.Figure:
    """Plot actual observations vs forecasted levels with calibrated empirical prediction interval."""
    fig = go.Figure()
    
    # Prepare conformal band y-coordinates
    times_list = list(test_times)
    n_pts = len(times_list)
    if isinstance(upper_bound, (list, np.ndarray, pd.Series)):
        y_upper = list(upper_bound)
    else:
        y_upper = [float(upper_bound)] * n_pts
        
    if isinstance(lower_bound, (list, np.ndarray, pd.Series)):
        y_lower = list(lower_bound)
    else:
        y_lower = [float(lower_bound)] * n_pts
        
    band_name = (
        f"Calibrated 95% Interval (Achieved: {coverage_pct:.1f}%, Width: {mpiw_dba:.1f} dBA)"
        if coverage_pct is not None and mpiw_dba is not None
        else "Calibrated 95% Prediction Interval"
    )
    
    # Calibrated empirical conformal uncertainty band
    fig.add_trace(go.Scatter(
        x=times_list + times_list[::-1],
        y=y_upper + y_lower[::-1],
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.18)",
        line=dict(color="rgba(99, 102, 241, 0.05)"),
        hoverinfo="skip",
        showlegend=True,
        name=band_name,
    ))
    
    # Ground truth actuals
    fig.add_trace(go.Scatter(
        x=test_times,
        y=y_true,
        mode="lines",
        name="Actual Noise (LAF_avg Leq)",
        line=dict(color="#38bdf8", width=2),
    ))
    
    # Model forecast
    fig.add_trace(go.Scatter(
        x=test_times,
        y=y_pred,
        mode="lines",
        name=f"Forecast: {best_model_name}",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    ))
    
    fig.update_layout(
        title=f"Spatiotemporal Forecast Validation — {station_name} ({horizon_label})",
        xaxis_title="Observation Timestamp (IST)",
        yaxis_title="Sound Pressure Level (dBA)",
        **LAYOUT
    )
    _apply_axes_style(fig)
    return fig

