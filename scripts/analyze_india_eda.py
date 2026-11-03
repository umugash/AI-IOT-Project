from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "india"
MONTHLY_FILE = DATA_DIR / "cpcb_noise_2011_2018" / "station_month.csv"
STATIONS_FILE = DATA_DIR / "cpcb_noise_2011_2018" / "stations.csv"
OUTPUT_DIR = ROOT / "outputs" / "india" / "eda"
METRICS = ("Day", "Night")
COLORS = {"Day": "#d97706", "Night": "#2563eb"}


def _month_label(index: int) -> str:
    year, month_zero_based = divmod(int(index), 12)
    return f"{year}-{month_zero_based + 1:02d}"


def _load_chennai_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(MONTHLY_FILE)
    stations = pd.read_csv(STATIONS_FILE)
    required_monthly = {
        "Station",
        "Year",
        "Month",
        "Day",
        "Night",
        "DayLimit",
        "NightLimit",
    }
    required_stations = {"Station", "Name", "City", "State", "Type"}
    if not required_monthly.issubset(records.columns):
        raise ValueError(f"{MONTHLY_FILE} is missing required columns.")
    if not required_stations.issubset(stations.columns):
        raise ValueError(f"{STATIONS_FILE} is missing required columns.")
    if stations["Station"].duplicated().any():
        raise ValueError(f"{STATIONS_FILE} contains duplicate station IDs.")

    chennai_stations = stations.loc[stations["City"].eq("Chennai")].copy()
    if chennai_stations.empty:
        raise ValueError("No Chennai stations were found in station metadata.")
    data = records.merge(
        chennai_stations,
        on="Station",
        how="inner",
        validate="many_to_one",
    )
    if data.empty:
        raise ValueError("No monthly observations matched Chennai stations.")

    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data["Month"] = pd.to_numeric(data["Month"], errors="raise").astype(int)
    if not data["Month"].between(1, 12).all():
        raise ValueError("Monthly data contain invalid month values.")
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")

    if data.duplicated(["Station", "Year", "Month"]).any():
        raise ValueError("Chennai data contain duplicate station-month keys.")

    data["month_index"] = data["Year"] * 12 + data["Month"] - 1
    data["month_label"] = data["month_index"].map(_month_label)
    return data.sort_values(["Station", "month_index"]), chennai_stations


def _write_coverage(
    data: pd.DataFrame, stations: pd.DataFrame, output_file: Path
) -> pd.DataFrame:
    rows = []
    for station in stations.sort_values("Station").itertuples(index=False):
        subset = data.loc[data["Station"].eq(station.Station)].sort_values(
            "month_index"
        )
        if subset.empty:
            raise ValueError(
                f"Station {station.Station} has no matching monthly observations."
            )
        first_index = int(subset["month_index"].min())
        last_index = int(subset["month_index"].max())
        expected = last_index - first_index + 1
        present = set(subset["month_index"].astype(int))
        missing_months = [
            _month_label(index)
            for index in range(first_index, last_index + 1)
            if index not in present
        ]
        rows.append(
            {
                "station_id": station.Station,
                "station_name": station.Name,
                "source_type": station.Type,
                "first_month": _month_label(first_index),
                "last_month": _month_label(last_index),
                "monthly_rows": len(subset),
                "expected_months_within_station_span": expected,
                "missing_months_within_station_span": len(missing_months),
                "missing_month_labels": ";".join(missing_months),
                "missing_day_values": int(subset["Day"].isna().sum()),
                "missing_night_values": int(subset["Night"].isna().sum()),
                "duplicate_station_month_keys": int(
                    subset.duplicated(["Station", "Year", "Month"]).sum()
                ),
            }
        )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(output_file, index=False)
    return coverage


def _write_station_summary(data: pd.DataFrame, output_file: Path) -> pd.DataFrame:
    rows = []
    for (station_id, station_name), subset in data.groupby(
        ["Station", "Name"], sort=True
    ):
        row: dict[str, object] = {
            "station_id": station_id,
            "station_name": station_name,
            "source_type": subset["Type"].iloc[0],
            "monthly_rows": len(subset),
        }
        for metric in METRICS:
            values = subset[metric].dropna()
            row[f"{metric.lower()}_mean"] = float(values.mean())
            row[f"{metric.lower()}_median"] = float(values.median())
            row[f"{metric.lower()}_std_dev"] = float(values.std(ddof=1))
            row[f"{metric.lower()}_minimum"] = float(values.min())
            row[f"{metric.lower()}_maximum"] = float(values.max())
        row["mean_day_minus_night"] = float(
            (subset["Day"] - subset["Night"]).mean()
        )
        elapsed_years = (
            subset["month_index"].to_numpy(dtype=float)
            - float(subset["month_index"].min())
        ) / 12
        row["day_descriptive_linear_slope_per_year"] = float(
            np.polyfit(elapsed_years, subset["Day"].to_numpy(dtype=float), 1)[0]
        )
        row["night_descriptive_linear_slope_per_year"] = float(
            np.polyfit(elapsed_years, subset["Night"].to_numpy(dtype=float), 1)[0]
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_file, index=False, float_format="%.4f")
    return summary


def _plot_monthly_trends(data: pd.DataFrame, output_file: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, metric in zip(axes, METRICS):
        for station_id, subset in data.groupby("Station", sort=True):
            ax.plot(
                subset["month_index"],
                subset[metric],
                linewidth=1.05,
                alpha=0.78,
                label=f"{station_id} {subset['Name'].iloc[0]}",
            )
        ax.axvline(2015 * 12, color="#555555", linestyle="--", linewidth=1)
        ax.set_ylabel("Source value\n(unit/weighting unspecified)")
        ax.set_title(f"Monthly {metric.lower()} reading by station")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=8, loc="best")
    ticks = sorted(data["month_index"].unique())
    axes[-1].set_xticks(ticks[::12])
    axes[-1].set_xticklabels(
        [_month_label(tick) for tick in ticks[::12]], rotation=45, ha="right"
    )
    axes[-1].set_xlabel("Source year-month (not an exact observation timestamp)")
    fig.suptitle("Chennai CPCB-attributed monthly station readings")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_station_comparison(data: pd.DataFrame, output_file: Path) -> None:
    means = data.groupby(["Station", "Name"], sort=True)[list(METRICS)].mean()
    labels = [f"{station}\n{name}" for station, name in means.index]
    x = np.arange(len(means))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, means["Day"], width, label="Day", color=COLORS["Day"])
    ax.bar(x + width / 2, means["Night"], width, label="Night", color=COLORS["Night"])
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.set_ylabel("Mean source value (unit/weighting unspecified)")
    ax.set_title("Station means across each station's available months")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_day_night_comparison(data: pd.DataFrame, output_file: Path) -> None:
    means = data.groupby(["Station", "Name"], sort=True)[list(METRICS)].mean()
    lower = float(min(means["Day"].min(), means["Night"].min()))
    upper = float(max(means["Day"].max(), means["Night"].max()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(means["Day"], means["Night"], color="#2a6f97")
    axes[0].plot([lower, upper], [lower, upper], "--", color="#777777")
    for (station, name), row in means.iterrows():
        axes[0].annotate(station, (row["Day"], row["Night"]), xytext=(3, 3), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Mean Day value")
    axes[0].set_ylabel("Mean Night value")
    axes[0].set_title("Paired station means")
    axes[0].grid(alpha=0.25)

    difference = means["Day"] - means["Night"]
    colors = ["#d97706" if value >= 0 else "#2563eb" for value in difference]
    axes[1].bar(means.index.get_level_values("Station"), difference, color=colors)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xlabel("Station ID")
    axes[1].set_ylabel("Mean Day minus Night")
    axes[1].set_title("Difference between source-reported periods")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Day/night comparison (source metrics retained as supplied)")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_distributions(data: pd.DataFrame, output_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.arange(
        min(data["Day"].min(), data["Night"].min()) - 0.5,
        max(data["Day"].max(), data["Night"].max()) + 1.5,
        1,
    )
    for metric in METRICS:
        axes[0].hist(
            data[metric].dropna(),
            bins=bins,
            alpha=0.55,
            label=metric,
            color=COLORS[metric],
        )
    axes[0].set_xlabel("Source value (unit/weighting unspecified)")
    axes[0].set_ylabel("Monthly station records")
    axes[0].set_title("Pooled distributions")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].boxplot(
        [data["Day"].dropna(), data["Night"].dropna()],
        showmeans=True,
    )
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Day", "Night"])
    axes[1].set_ylabel("Source value (unit/weighting unspecified)")
    axes[1].set_title("Pooled spread; points are not independent replicates")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Distribution of Chennai monthly readings")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_missingness(
    data: pd.DataFrame, stations: pd.DataFrame, output_file: Path
) -> None:
    first_index = int(data["month_index"].min())
    last_index = int(data["month_index"].max())
    month_indices = np.arange(first_index, last_index + 1)
    station_ids = stations.sort_values("Station")["Station"].tolist()
    present = np.zeros((len(station_ids), len(month_indices)), dtype=int)
    for row_index, station_id in enumerate(station_ids):
        months = set(
            data.loc[data["Station"].eq(station_id), "month_index"].astype(int)
        )
        present[row_index] = [
            int(month in months) for month in month_indices
        ]

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].imshow(present, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    axes[0].set_yticks(np.arange(len(station_ids)), station_ids)
    axes[0].set_title("Station-month presence: green = row present, white = no row")
    ticks = np.arange(0, len(month_indices), 12)
    axes[0].set_xticks(ticks, [_month_label(month_indices[i]) for i in ticks], rotation=45, ha="right")

    y = np.arange(len(station_ids))
    bar_height = 0.36
    missing_day = [
        int(data.loc[data["Station"].eq(station_id), "Day"].isna().sum())
        for station_id in station_ids
    ]
    missing_night = [
        int(data.loc[data["Station"].eq(station_id), "Night"].isna().sum())
        for station_id in station_ids
    ]
    axes[1].barh(y - bar_height / 2, missing_day, bar_height, label="Day missing")
    axes[1].barh(y + bar_height / 2, missing_night, bar_height, label="Night missing")
    axes[1].set_yticks(y, station_ids)
    axes[1].set_xlim(0, max(1, max(missing_day + missing_night, default=0) + 1))
    for row_index, (day_count, night_count) in enumerate(
        zip(missing_day, missing_night)
    ):
        axes[1].text(
            max(day_count, night_count) + 0.03,
            row_index,
            f"{day_count} / {night_count}",
            va="center",
            fontsize=8,
        )
    axes[1].set_title("Missing Day / Night values within existing station-month rows")
    axes[1].set_xlabel("Missing values (labels show Day / Night)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Coverage and missing-value audit")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_station_timeseries(data: pd.DataFrame, output_file: Path) -> None:
    station_ids = sorted(data["Station"].unique())
    fig, axes = plt.subplots(5, 2, figsize=(15, 17), sharex=True)
    for ax, station_id in zip(axes.flat, station_ids):
        subset = data.loc[data["Station"].eq(station_id)].sort_values("month_index")
        ax.plot(subset["month_index"], subset["Day"], color=COLORS["Day"], linewidth=1, label="Day")
        ax.plot(subset["month_index"], subset["Night"], color=COLORS["Night"], linewidth=1, label="Night")
        ax.set_title(f"{station_id} — {subset['Name'].iloc[0]} ({len(subset)} months)")
        ax.set_ylabel("Source value")
        ax.grid(alpha=0.22)
    ticks = np.arange(data["month_index"].min(), data["month_index"].max() + 1, 12)
    for ax in axes[-1, :]:
        ax.set_xticks(ticks, [_month_label(tick) for tick in ticks], rotation=45, ha="right")
        ax.set_xlabel("Source year-month (not an exact timestamp)")
    axes[0, 0].legend()
    fig.suptitle("Station-level monthly time series; gaps are not imputed")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_seasonality(data: pd.DataFrame, output_file: Path) -> None:
    within_station = data.copy()
    for metric in METRICS:
        within_station[f"{metric}_station_anomaly"] = within_station[metric] - within_station.groupby("Station")[metric].transform("mean")
    station_month = data.groupby(["Station", "Month"])[list(METRICS)].mean()
    raw = station_month.groupby("Month").mean()
    station_month_anomalies = within_station.groupby(
        ["Station", "Month"]
    )[[f"{metric}_station_anomaly" for metric in METRICS]].mean()
    anomalies = station_month_anomalies.groupby("Month").mean()
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(1, 13)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric in METRICS:
        axes[0].plot(x, raw[metric], marker="o", label=metric, color=COLORS[metric])
        axes[1].plot(
            x,
            anomalies[f"{metric}_station_anomaly"],
            marker="o",
            label=metric,
            color=COLORS[metric],
        )
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_title("Station-balanced month-of-year averages")
    axes[0].set_ylabel("Source value (unit/weighting unspecified)")
    axes[1].set_title("Within-station centered month profile")
    axes[1].set_ylabel("Deviation from each station's mean")
    fig.suptitle("Descriptive seasonality only; no significance test or forecast")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _summary(
    data: pd.DataFrame,
    coverage: pd.DataFrame,
    station_summary: pd.DataFrame,
) -> dict[str, object]:
    result: dict[str, object] = {
        "source": "CPCB-attributed Kaggle monthly noise dataset",
        "source_period": "2011-02 through 2018-12",
        "city_filter": "Chennai",
        "monthly_rows": int(len(data)),
        "unique_stations": int(data["Station"].nunique()),
        "unique_years": sorted(int(year) for year in data["Year"].unique()),
        "unique_month_numbers": sorted(int(month) for month in data["Month"].unique()),
        "duplicate_station_month_keys": int(
            data.duplicated(["Station", "Year", "Month"]).sum()
        ),
        "missing_values": {
            column: int(data[column].isna().sum())
            for column in ("Station", "Year", "Month", "Day", "Night", "DayLimit", "NightLimit")
        },
        "station_coverage": coverage.to_dict(orient="records"),
        "station_statistics": station_summary.to_dict(orient="records"),
        "distributions": {},
        "day_minus_night_mean": float((data["Day"] - data["Night"]).mean()),
        "annual_station_month_means": (
            data.groupby("Year")[list(METRICS)].mean().round(4).to_dict(orient="index")
        ),
        "interpretation_note": (
            "Year-Month is a monthly period only. No observation timestamps, coordinates, "
            "or missing-value imputations are created."
        ),
    }
    for metric in METRICS:
        values = data[metric].dropna().to_numpy(dtype=float)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        lower = q1 - 1.5 * (q3 - q1)
        upper = q3 + 1.5 * (q3 - q1)
        result["distributions"][metric] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "median": float(median),
            "standard_deviation": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "q1": float(q1),
            "q3": float(q3),
            "maximum": float(values.max()),
            "iqr_fence_lower": float(lower),
            "iqr_fence_upper": float(upper),
            "iqr_flagged_count": int(((values < lower) | (values > upper)).sum()),
        }
        flags = data.loc[
            data[metric].lt(lower) | data[metric].gt(upper),
            ["Station", "Name", "Year", "Month", metric],
        ]
        result["distributions"][metric]["iqr_flagged_observations"] = [
            {
                "station_id": row.Station,
                "station_name": row.Name,
                "year_month": _month_label(int(row.Year) * 12 + int(row.Month) - 1),
                "value": float(getattr(row, metric)),
            }
            for row in flags.itertuples(index=False)
        ]
    station_centered = data.copy()
    for metric in METRICS:
        station_centered[f"{metric}_station_anomaly"] = (
            station_centered[metric]
            - station_centered.groupby("Station")[metric].transform("mean")
        )
    station_month_anomalies = station_centered.groupby(
        ["Station", "Month"]
    )[
        [f"{metric}_station_anomaly" for metric in METRICS]
    ].mean()
    result["within_station_centered_monthly_anomalies"] = (
        station_month_anomalies.groupby("Month")
        .mean()
        .round(4)
        .to_dict(orient="index")
    )
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, stations = _load_chennai_data()
    coverage = _write_coverage(
        data, stations, OUTPUT_DIR / "chennai_station_coverage.csv"
    )
    station_summary = _write_station_summary(
        data, OUTPUT_DIR / "chennai_station_descriptive_summary.csv"
    )
    summary = _summary(data, coverage, station_summary)
    (OUTPUT_DIR / "chennai_eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_monthly_trends(data, OUTPUT_DIR / "monthly_noise_trends.png")
    _plot_station_comparison(data, OUTPUT_DIR / "station_wise_comparison.png")
    _plot_day_night_comparison(data, OUTPUT_DIR / "day_night_comparison.png")
    _plot_distributions(data, OUTPUT_DIR / "noise_distributions.png")
    _plot_missingness(
        data, stations, OUTPUT_DIR / "missing_data_and_coverage.png"
    )
    _plot_station_timeseries(data, OUTPUT_DIR / "station_time_series.png")
    _plot_seasonality(data, OUTPUT_DIR / "descriptive_monthly_seasonality.png")
    print(
        f"Created Chennai EDA outputs for {len(data)} monthly station records "
        f"across {data['Station'].nunique()} stations in {OUTPUT_DIR}."
    )


if __name__ == "__main__":
    main()
