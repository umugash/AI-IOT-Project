"""Project-wide configuration: paths, Chennai CPCB stations and noise limits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CPCB_RAW_DIR = DATA_DIR / "cpcb" / "raw"
CPCB_SNAPSHOT_DIR = DATA_DIR / "cpcb" / "live_snapshots"
REFERENCE_DIR = DATA_DIR / "reference"
DB_PATH = DATA_DIR / "noisescope.db"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
EXTERNAL_DIR = DATA_DIR / "external"  # large downloaded datasets (git-ignored)

# Timestamps from the CPCB NANMN portal are Indian Standard Time wall-clock.
TIMEZONE = "Asia/Kolkata"

# Ambient noise limits, Noise Pollution (Regulation and Control) Rules, 2000, Schedule.
# Day = 06:00-22:00, Night = 22:00-06:00. Values are LAeq in dB(A).
ZONE_LIMITS: dict[str, tuple[int, int]] = {
    "Industrial": (75, 70),
    "Commercial": (65, 55),
    "Residential": (55, 45),
    "Silence": (50, 40),
}
DAY_START_HOUR = 6
NIGHT_START_HOUR = 22


@dataclass(frozen=True)
class Station:
    station_id: int
    name: str           # portal name
    label: str          # short display label
    zone: str           # Industrial / Commercial / Residential / Silence (as listed by the CPCB portal)
    lat: float
    lon: float
    location_note: str  # how the coordinate was obtained

    @property
    def day_limit(self) -> int:
        return ZONE_LIMITS[self.zone][0]

    @property
    def night_limit(self) -> int:
        return ZONE_LIMITS[self.zone][1]


# The ten Chennai stations of the CPCB National Ambient Noise Monitoring Network (NANMN).
# Coordinates are approximate locality / landmark points (OpenStreetMap), not surveyed
# sensor positions. Zones are as reported by the CPCB portal's station list.
STATIONS: dict[int, Station] = {s.station_id: s for s in [
    Station(264, "Chennai_Eye Hospital Pudupet", "Egmore Eye Hospital", "Silence", 13.0703627, 80.2600948, "landmark (OSM)"),
    Station(265, "Chennai_Thiyagaraya Nagar", "T. Nagar", "Commercial", 13.0385946, 80.2335793, "locality (OSM)"),
    Station(266, "Chennai_Perambur Paper Mill Rd.", "Perambur", "Commercial", 13.1121242, 80.2450220, "locality (OSM)"),
    Station(267, "Chennai_Guidy Mount Salai", "Guindy", "Industrial", 13.0087095, 80.2203646, "locality (OSM)"),
    Station(268, "Chennai_Ice House Triplicane", "Triplicane", "Residential", 13.0604285, 80.2750704, "locality (OSM)"),
    Station(269, "Chennai_Pallikaranai", "Pallikaranai", "Commercial", 12.9304748, 80.2077692, "locality (OSM)"),
    Station(270, "Chennai_Velacherry", "Velachery", "Residential", 12.9772611, 80.2209183, "locality (OSM)"),
    Station(271, "Chennai_Washermanpet", "Washermanpet", "Commercial", 13.1143, 80.2872, "locality centroid (approx.)"),
    Station(272, "Chennai_Anna Nagar", "Anna Nagar", "Silence", 13.0850, 80.2101, "locality centroid (approx.)"),
    Station(273, "Chennai_Sowcarpet", "Sowcarpet", "Residential", 13.0884967, 80.2783092, "locality (OSM)"),
]}


def limit_for(zone: str, hour: int) -> int:
    day, night = ZONE_LIMITS[zone]
    return day if DAY_START_HOUR <= hour < NIGHT_START_HOUR else night


def period_for(hour: int) -> str:
    return "Day" if DAY_START_HOUR <= hour < NIGHT_START_HOUR else "Night"
