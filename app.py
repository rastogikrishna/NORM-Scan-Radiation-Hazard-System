# app.py — Radiation Hazard Assessment System (Redesigned)

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────

from pdf_generator import generate_pdf
import dash
from dash import Dash, html, Input, Output, State, dcc, callback_context
import dash_bootstrap_components as dbc
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

import requests
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import dash_leaflet as dl

from formula_engine import (
    calculate_dose_rate,
    calculate_raeq,
    calculate_hex,
    calculate_hin,
    calculate_aed,
    calculate_elcr
)

from isotope_lookup import detect_isotope, get_all_isotopes, ISOTOPE_STEPS


# ─────────────────────────────────────────────────────────────
# LOAD ML MODEL & DATASETS
# ─────────────────────────────────────────────────────────────

anomaly_model = joblib.load("model/anomaly_model.pkl")

# Load samples dataset and metadata
radiation_df = pd.read_csv("dataset/Radiation_dataset.csv")
metadata_df = pd.read_csv("dataset/sample_metadata.csv")
samples_merged_df = pd.merge(radiation_df, metadata_df, on="Sample")

# Load cross-validation metrics
import json
from formula_engine import calculate_raeq

try:
    with open("model/evaluation_metrics.json", "r") as f:
        cv_metrics = json.load(f)
except Exception as e:
    cv_metrics = {
        "task": "anomaly_detection",
        "target": "None (Unsupervised)",
        "model": "Isolation Forest",
        "training sample count": 850,
        "contamination": 0.05,
        "n_estimators": 150
    }

ml_training_samples = cv_metrics.get("training sample count", 850)
contamination = cv_metrics.get("contamination", 0.05)
n_estimators = cv_metrics.get("n_estimators", 150)
best_model_name = "Isolation Forest"

# Load model metadata for feature importances
try:
    with open("model/model_metadata.json", "r") as f:
        model_metadata = json.load(f)
    global_feature_importances = model_metadata["feature_importances"]
except Exception as e:
    global_feature_importances = [("Ra-226", 0.4), ("Th-232", 0.3), ("K-40", 0.2), ("U-235", 0.1)]

# Create a dictionary for quick lookup by Sample name
sample_lookup = {}
for idx, row in samples_merged_df.iterrows():
    s_id = row["Sample"]
    num = s_id.replace("S", "")
    display_name = f"Sample {num}"
    sample_lookup[display_name] = row.to_dict()


# ─────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────

C = {
    "bg":        "#0A0E1A",
    "surface":   "#111827",
    "border":    "#1E2D40",
    "accent":    "#00E5FF",
    "accent2":   "#FF6B35",
    "accent3":   "#7C3AED",
    "text":      "#E2E8F0",
    "muted":     "#64748B",
    "low":       "#22C55E",
    "mod":       "#F59E0B",
    "high":      "#EF4444",
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'JetBrains Mono', monospace", color=C["text"], size=12),
    margin=dict(l=20, r=20, t=50, b=20),
)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def glow_card(children, glow_color=C["accent"], **kwargs):
    return html.Div(
        children,
        style={
            "background":    "linear-gradient(135deg, #111827 0%, #0D1520 100%)",
            "border":        f"1px solid {glow_color}33",
            "borderRadius":  "16px",
            "padding":       "24px",
            "boxShadow":     f"0 0 30px {glow_color}22, inset 0 1px 0 {glow_color}22",
            "backdropFilter": "blur(10px)",
            **kwargs.get("style", {})
        },
        **{k: v for k, v in kwargs.items() if k != "style"}
    )


def metric_card(label, value, unit, color, icon=""):
    return html.Div([
        html.Div(icon, style={"fontSize": "28px", "marginBottom": "8px"}),
        html.Div(label, style={
            "fontSize": "10px",
            "fontFamily": "'JetBrains Mono', monospace",
            "letterSpacing": "2px",
            "color": C["muted"],
            "textTransform": "uppercase",
            "marginBottom": "6px"
        }),
        html.Div(str(value), style={
            "fontSize": "28px",
            "fontWeight": "700",
            "fontFamily": "'Orbitron', sans-serif",
            "color": color,
            "lineHeight": "1",
            "marginBottom": "4px"
        }),
        html.Div(unit, style={
            "fontSize": "10px",
            "color": C["muted"],
            "fontFamily": "'JetBrains Mono', monospace",
        }),
    ], style={
        "background":   f"linear-gradient(135deg, {color}11 0%, transparent 100%)",
        "border":       f"1px solid {color}44",
        "borderRadius": "12px",
        "padding":      "20px",
        "textAlign":    "center",
        "transition":   "transform 0.2s, box-shadow 0.2s",
    })


def isotope_sub_card(radionuclide, energy_peak, isotope, decay_chain):
    is_detected = isotope != "Not Detected" and isotope != "Unknown"
    text_color = C["mod"] if is_detected else C["muted"]
    return html.Div([
        html.Div(radionuclide, style={
            "fontFamily": "'Orbitron', sans-serif",
            "fontSize": "11px",
            "fontWeight": "bold",
            "letterSpacing": "1.5px",
            "color": C["accent"],
            "marginBottom": "6px",
            "borderBottom": f"1px solid {C['border']}66",
            "paddingBottom": "3px"
        }),
        html.Div([
            html.Span("Peak: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "10px"}),
            html.Span(f"{energy_peak} keV" if energy_peak else "—", style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "10.5px"}),
        ], style={"marginBottom": "3px"}),
        html.Div([
            html.Span("Isotope: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "10px"}),
            html.Span(isotope, style={"color": text_color, "fontFamily": "JetBrains Mono", "fontSize": "10.5px", "fontWeight": "600"}),
        ], style={"marginBottom": "3px"}),
        html.Div([
            html.Span("Chain: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "10px"}),
            html.Span(decay_chain, style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "10px"}),
        ]),
    ], style={
        "background": "rgba(255,255,255,0.02)",
        "border": f"1px solid {C['border']}",
        "borderRadius": "8px",
        "padding": "12px",
        "flex": "1",
        "minWidth": "160px",
    })



def input_field(label, id_, placeholder, icon="", min_val=None, max_val=None):
    return html.Div([
        html.Label(
            [html.Span(icon + "  ", style={"marginRight": "6px"}), label],
            style={
                "color": C["muted"],
                "fontSize": "11px",
                "letterSpacing": "1.5px",
                "textTransform": "uppercase",
                "fontFamily": "'JetBrains Mono', monospace",
                "marginBottom": "8px",
                "display": "block"
            }
        ),
        dcc.Input(
            id=id_,
            type="number",
            placeholder=placeholder,
            min=min_val,
            max=max_val,
            style={
                "width":           "100%",
                "background":      "#0A0E1A",
                "border":          f"1px solid {C['border']}",
                "borderRadius":    "8px",
                "color":           C["text"],
                "fontFamily":      "'JetBrains Mono', monospace",
                "fontSize":        "14px",
                "padding":         "10px 14px",
                "outline":         "none",
                "transition":      "border-color 0.2s",
                "boxSizing":       "border-box",
            },
            debounce=False,
        )
    ], style={"marginBottom": "20px"})


def dropdown_field(label, id_, options, icon=""):
    return html.Div([
        html.Label(
            [html.Span(icon + "  ", style={"marginRight": "6px"}), label],
            style={
                "color": C["muted"],
                "fontSize": "11px",
                "letterSpacing": "1.5px",
                "textTransform": "uppercase",
                "fontFamily": "'JetBrains Mono', monospace",
                "marginBottom": "8px",
                "display": "block"
            }
        ),
        dcc.Dropdown( 
            id=id_,
            options=[{"label": opt, "value": opt} for opt in options],
            value=options[0] if options else None,
            clearable=False,
            style={
                "width":           "100%",
                "background":      "#0A0E1A",
                "color":           C["text"],
                "fontFamily":      "'JetBrains Mono', monospace",
                "fontSize":        "14px",
                "boxSizing":       "border-box",
            }
        )
    ], style={"marginBottom": "20px"})


def reverse_geocode(lat, lon):
    if lat is None or lon is None:
        return {"location_name": "N/A", "city": "N/A", "state": "N/A", "country": "N/A"}
    
    # Chad Yapala / Mayo-Dallah check
    if 7.8 <= lat <= 9.8 and 14.0 <= lon <= 16.0:
        return {
            "location_name": "Yapala",
            "city": "Mayo-Dallah",
            "state": "Southern Chad",
            "country": "Chad"
        }
        
    import threading
    result = {}
    
    def worker():
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        headers = {
            "User-Agent": "RAD-SCAN-Radiation-Prediction-App/2.0 (contact: user@example.com)"
        }
        try:
            response = requests.get(url, headers=headers, timeout=(0.5, 1.0))
            if response.status_code == 200:
                data = response.json()
                address = data.get("address", {})
                city = address.get("city") or address.get("town") or address.get("village") or address.get("suburb") or ""
                state = address.get("state") or address.get("region") or ""
                country = address.get("country") or ""
                
                if not city:
                    city = address.get("county") or address.get("state_district") or address.get("municipality") or ""
                
                location_name = city if city else (state if state else country)
                if location_name:
                    result["data"] = {
                        "location_name": location_name,
                        "city": city if city else "N/A",
                        "state": state if state else "N/A",
                        "country": country if country else "N/A"
                    }
        except Exception:
            pass

    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    t.join(timeout=0.3)
    
    if "data" in result:
        return result["data"]

    # Local fallback (replacing India references with Chad)
    fallbacks = [
        {"lat": 8.5000, "lon": 15.3000, "location_name": "Yapala", "city": "Mayo-Dallah", "state": "Southern Chad", "country": "Chad"},
        {"lat": 40.7128, "lon": -74.0060, "location_name": "New York", "city": "New York", "state": "New York", "country": "United States"},
        {"lat": 51.5074, "lon": -0.1278, "location_name": "London", "city": "London", "state": "England", "country": "United Kingdom"},
        {"lat": 48.8566, "lon": 2.3522, "location_name": "Paris", "city": "Paris", "state": "Île-de-France", "country": "France"},
        {"lat": 35.6762, "lon": 139.6503, "location_name": "Tokyo", "city": "Tokyo", "state": "Tokyo", "country": "Japan"},
        {"lat": -33.8688, "lon": 151.2093, "location_name": "Sydney", "city": "Sydney", "state": "New South Wales", "country": "Australia"},
        {"lat": 22.3193, "lon": 114.1694, "location_name": "Hong Kong", "city": "Hong Kong", "state": "Kowloon", "country": "China"},
    ]
    
    closest = None
    min_dist = float('inf')
    for f in fallbacks:
        dist = (lat - f["lat"])**2 + (lon - f["lon"])**2
        if dist < min_dist:
            min_dist = dist
            closest = f
            
    if min_dist < 25.0:
        return {
            "location_name": closest["location_name"],
            "city": closest["city"],
            "state": closest["state"],
            "country": closest["country"]
        }
    else:
        region_lat = "Northern" if lat >= 0 else "Southern"
        region_lon = "Eastern" if lon >= 0 else "Western"
        return {
            "location_name": f"Sample Zone ({region_lat}-{region_lon})",
            "city": "Remote Station",
            "state": f"Zone {region_lat}",
            "country": f"Global {region_lon} Region"
        }


def get_soil_interpretation(texture, ph):
    if ph is None:
        ph_text = "Soil pH data is missing or incomplete. Cannot evaluate pH influence."
    elif ph < 5.1:
        ph_text = "Strongly acidic conditions may increase radionuclide mobility."
    elif ph < 5.9:
        ph_text = "Moderately acidic ferruginous soil conditions typical of the Yapala study area."
    elif ph <= 6.5:
        ph_text = "Slightly acidic conditions with moderate radionuclide retention."
    else:
        ph_text = "Near-neutral soil conditions exceeding the typical study-area average."
        
    texture_interpretations = {
        "Sandy Tropical Ferruginous Soil": "Highly permeable ferruginous soil with low retention capacity, typical of tropical weathered profiles.",
        "Granitic Sandy Soil":            "Coarse-textured sandy soil derived from granitic bedrock; high water drainage and minimal radionuclide retention.",
        "Weathered Granitic Soil":         "Partially weathered granitic material with moderate permeability and intermediate adsorption."
    }
    
    texture_text = texture_interpretations.get(texture, "Unknown soil texture characteristics.")
    return texture_text, ph_text


# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  (injected via index_string)
# ─────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

body {
    background-color: #0A0E1A !important;
    background-image:
        radial-gradient(ellipse at 20% 10%, #00E5FF08 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, #7C3AED08 0%, transparent 50%),
        linear-gradient(180deg, #0A0E1A 0%, #060810 100%);
    min-height: 100vh;
    color: #E2E8F0 !important;
}

.rad-input:focus { border-color: #00E5FF !important; box-shadow: 0 0 0 2px #00E5FF22 !important; }

.scan-line {
    position: relative;
    overflow: hidden;
}
.scan-line::after {
    content: '';
    position: absolute;
    top: -100%;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00E5FF88, transparent);
    animation: scan 3s linear infinite;
}
@keyframes scan {
    0%   { top: -2px; }
    100% { top: 100%; }
}

@keyframes pulse-ring {
    0%   { transform: scale(1);   opacity: 0.8; }
    100% { transform: scale(1.6); opacity: 0; }
}
.pulse-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #22C55E;
    position: relative;
    margin-right: 8px;
}
.pulse-dot::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #22C55E;
    animation: pulse-ring 1.4s ease-out infinite;
}

.tab-content { animation: fadeIn 0.4s ease; }
@keyframes fadeIn { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform: none; } }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2D40; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00E5FF44; }

.Select-control, .dash-dropdown .Select-control {
    background-color: #0A0E1A !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}
.Select-menu-outer, .dash-dropdown .Select-menu-outer {
    background-color: #0A0E1A !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 8px !important;
    z-index: 1000 !important;
}
.Select-option, .dash-dropdown .Select-option {
    background-color: #0A0E1A !important;
    color: #E2E8F0 !important;
}
.Select-option.is-focused, .dash-dropdown .Select-option.is-focused {
    background-color: #00E5FF22 !important;
    color: #00E5FF !important;
}
.Select-value-label, .dash-dropdown .Select-value-label,
.Select-value, .dash-dropdown .Select-value,
.Select-placeholder, .dash-dropdown .Select-placeholder {
    color: #E2E8F0 !important;
}
.Select-input > input {
    color: #E2E8F0 !important;
}
.Select-arrow {
    border-color: #64748B transparent transparent !important;
}
"""


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@300;400;500&display=swap",
    ],
    suppress_callback_exceptions=True
)

app.title = "RAD-SCAN // Radiation Hazard Assessment"

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{CUSTOM_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────

SIDEBAR = html.Div([

    # Logo / Title
    html.Div([
        html.Div("☢", style={"fontSize": "48px", "lineHeight": "1"}),
        html.Div("RAD-SCAN", style={
            "fontFamily": "'Orbitron', sans-serif",
            "fontSize": "22px",
            "fontWeight": "900",
            "background": f"linear-gradient(135deg, {C['accent']}, {C['accent3']})",
            "WebkitBackgroundClip": "text",
            "WebkitTextFillColor": "transparent",
            "letterSpacing": "3px",
        }),
        html.Div("HAZARD ASSESSMENT v2.0", style={
            "fontSize": "9px",
            "letterSpacing": "2px",
            "color": C["muted"],
            "fontFamily": "'JetBrains Mono', monospace",
            "marginTop": "2px",
        }),
    ], style={"marginBottom": "40px", "paddingTop": "10px"}),

    # Status indicator
    html.Div([
        html.Span(className="pulse-dot"),
        html.Span("SYSTEM ONLINE", style={
            "fontSize": "10px",
            "letterSpacing": "2px",
            "color": C["low"],
            "fontFamily": "'JetBrains Mono', monospace",
        })
    ], style={"marginBottom": "36px", "display": "flex", "alignItems": "center"}),

    # Preloaded Sample Database
    html.Div([
        html.Div("◈ PRELOADED SAMPLE DATABASE", style={
            "fontSize": "9px",
            "letterSpacing": "3px",
            "color": C["accent"],
            "fontFamily": "'JetBrains Mono', monospace",
            "marginBottom": "20px",
            "borderBottom": f"1px solid {C['border']}",
            "paddingBottom": "10px",
        }),
        dcc.Dropdown(
            id="sample-select",
            options=[{"label": f"Sample {i}", "value": f"Sample {i}"} for i in range(1, 21)],
            placeholder="Select Preloaded Sample...",
            clearable=True,
            style={
                "width":           "100%",
                "background":      "#0A0E1A",
                "color":           C["text"],
                "fontFamily":      "'JetBrains Mono', monospace",
                "fontSize":        "14px",
                "boxSizing":       "border-box",
                "marginBottom":    "24px",
            }
        ),
    ]),

    # Inputs
    html.Div([
        html.Div("◈ RADIONUCLIDE ANALYSIS", style={
            "fontSize": "9px",
            "letterSpacing": "3px",
            "color": C["accent"],
            "fontFamily": "'JetBrains Mono', monospace",
            "marginBottom": "20px",
            "borderBottom": f"1px solid {C['border']}",
            "paddingBottom": "10px",
        }),

        input_field("Ra-226 Activity (Bq/kg)", "ra-input",   "e.g. 35.4",  "⚛", min_val=0),
        input_field("Th-232 Activity (Bq/kg)", "th-input",   "e.g. 17.4",  "⚛", min_val=0),
        input_field("K-40 Activity (Bq/kg)", "k-input",    "e.g. 215.7", "⚛", min_val=0),
        input_field("U-235 Activity (Bq/kg)", "u235-input", "e.g. 3.5",   "⚛", min_val=0),

        html.Div("◈ ISOTOPE SPECTROSCOPY SELECTION", style={
            "fontSize": "9px",
            "letterSpacing": "3px",
            "color": C["accent"],
            "fontFamily": "'JetBrains Mono', monospace",
            "marginBottom": "20px",
            "borderBottom": f"1px solid {C['border']}",
            "paddingBottom": "10px",
        }),

        dropdown_field("Isotope Family", "isotope-family-input", [
            "Ra-226 Series", "Th-232 Series", "K-40", "U-235 Series", "Cs-137"
        ], "⚛"),
        dropdown_field("Energy Peak", "isotope-energy-input", [], "⚡"),

        html.Div("◈ SOIL CHARACTERISTICS", style={
            "fontSize": "9px",
            "letterSpacing": "3px",
            "color": C["accent"],
            "fontFamily": "'JetBrains Mono', monospace",
            "marginBottom": "20px",
            "marginTop": "10px",
            "borderBottom": f"1px solid {C['border']}",
            "paddingBottom": "10px",
        }),

        dropdown_field("Soil Texture", "soil-texture-input", [
            "Sandy Tropical Ferruginous Soil",
            "Granitic Sandy Soil",
            "Weathered Granitic Soil"
        ], "🌱"),

        input_field("Soil pH", "soil-ph-input", "e.g. 7.2", "🧪", min_val=0, max_val=14),

        html.Div("◈ LOCATION", style={
            "fontSize": "9px",
            "letterSpacing": "3px",
            "color": C["accent"],
            "fontFamily": "'JetBrains Mono', monospace",
            "marginBottom": "20px",
            "marginTop": "10px",
            "borderBottom": f"1px solid {C['border']}",
            "paddingBottom": "10px",
        }),

        input_field("Latitude",  "lat-input",  "e.g. 28.6139", "📍", min_val=-90, max_val=90),
        input_field("Longitude", "long-input", "e.g. 77.2090", "📍", min_val=-180, max_val=180),

        # Calculate Button
        html.Button(
            [html.Span("⚡  "), "ANALYSE SAMPLE"],
            id="calculate-button",
            n_clicks=0,
            style={
                "width": "100%",
                "padding": "14px",
                "background": f"linear-gradient(135deg, {C['accent']}22, {C['accent3']}22)",
                "border": f"1px solid {C['accent']}88",
                "borderRadius": "10px",
                "color": C["accent"],
                "fontFamily": "'Orbitron', sans-serif",
                "fontSize": "13px",
                "letterSpacing": "2px",
                "cursor": "pointer",
                "marginTop": "8px",
                "transition": "all 0.25s",
            }
        ),

        # Generate Report Button
        html.Button(
            [html.Span("📄  "), "GENERATE REPORT"],
            id="report-button",
            n_clicks=0,
            disabled=True,
            style={
                "width": "100%",
                "padding": "14px",
                "background": "linear-gradient(135deg, #28a74511, #28a74522)",
                "border": "1px solid #28a74544",
                "borderRadius": "10px",
                "color": "#28a74588",
                "fontFamily": "'Orbitron', sans-serif",
                "fontSize": "13px",
                "letterSpacing": "2px",
                "cursor": "not-allowed",
                "marginTop": "10px",
                "transition": "all 0.25s",
                "opacity": "0.5"
            }
        ),

        # Reset Button
        html.Button(
            "↺  RESET",
            id="reset-button",
            n_clicks=0,
            style={
                "width":         "100%",
                "padding":       "10px",
                "background":    "transparent",
                "border":        f"1px solid {C['border']}",
                "borderRadius":  "10px",
                "color":         C["muted"],
                "fontFamily":    "\'JetBrains Mono\', monospace",
                "fontSize":      "12px",
                "cursor":        "pointer",
                "marginTop":     "10px",
                "letterSpacing": "1px",
            }
        ),

    ]),

], style={
    "width":        "280px",
    "minWidth":     "280px",
    "background":   "linear-gradient(180deg, #0D1520 0%, #080C14 100%)",
    "borderRight":  f"1px solid {C['border']}",
    "padding":      "30px 24px",
    "height":       "100vh",
    "overflowY":    "auto",
    "position":     "sticky",
    "top":          "0",
})


MAIN_AREA = html.Div([

    # Top bar
    html.Div([
        html.Div([
            html.Span("◉ ", style={"color": C["accent"]}),
            html.Span("RADIATION MONITORING SYSTEM", style={
                "fontFamily":    "'Orbitron', sans-serif",
                "fontSize":      "14px",
                "letterSpacing": "3px",
                "color":         C["text"],
            }),
        ]),
        html.Div(id="timestamp-display", style={
            "fontFamily": "'JetBrains Mono', monospace",
            "fontSize":   "11px",
            "color":      C["muted"],
        }),
    ], style={
        "display":        "flex",
        "justifyContent": "space-between",
        "alignItems":     "center",
        "padding":        "16px 32px",
        "borderBottom":   f"1px solid {C['border']}",
        "background":     "#0A0E1A99",
        "backdropFilter": "blur(10px)",
        "position":       "sticky",
        "top":            "0",
        "zIndex":         "100",
    }),

    # Output area
    dcc.Loading(
        id="loading-output-results",
        type="default",
        color=C["accent"],
        children=html.Div(
            id="output-results",
            style={"padding": "32px"},
            children=[
                # Default welcome screen
                html.Div([
                    html.Div("☢", style={
                        "fontSize": "80px",
                        "opacity": "0.15",
                        "marginBottom": "20px",
                        "animation": "spin 20s linear infinite",
                    }),
                    html.Div("AWAITING SAMPLE DATA", style={
                        "fontFamily": "'Orbitron', sans-serif",
                        "fontSize":   "24px",
                        "letterSpacing": "6px",
                        "color": C["muted"],
                        "marginBottom": "12px",
                    }),
                    html.Div("Enter isotope concentrations in the panel and click ANALYSE SAMPLE", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "color":      C["muted"],
                        "fontSize":   "13px",
                    }),
                ], style={
                    "display":        "flex",
                    "flexDirection":  "column",
                    "alignItems":     "center",
                    "justifyContent": "center",
                    "height":         "60vh",
                    "textAlign":      "center",
                })
            ]
        )
    ),

    # Interval for clock
    dcc.Interval(id="clock-interval", interval=60000, n_intervals=0),

], style={"flex": "1", "overflowY": "auto", "height": "100vh"})


app.layout = html.Div(
    [SIDEBAR, MAIN_AREA, dcc.Download(id="download-report-pdf")],
    style={"display": "flex", "height": "100vh", "overflow": "hidden"}
)


# ─────────────────────────────────────────────────────────────
# CLOCK CALLBACK
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("timestamp-display", "children"),
    Input("clock-interval",     "n_intervals")
)
def update_clock(n):
    try:
        from datetime import datetime
        now = datetime.now()
        val = now.strftime("UTC+5:30  //  %Y-%m-%d  %H:%M:%S")
        return val
    except Exception as e:
        logging.error(f"Callback Error in update_clock: {e}")
        return "Time Sync Error"


# ─────────────────────────────────────────────────────────────
# CALCULATE CALLBACK
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("output-results", "children"),
    Input("calculate-button", "n_clicks"),
    Input("reset-button", "n_clicks"),
    Input("sample-select", "value"),
    State("ra-input",            "value"),
    State("th-input",            "value"),
    State("k-input",             "value"),
    State("u235-input",          "value"),
    State("lat-input",           "value"),
    State("long-input",          "value"),
    State("soil-texture-input",  "value"),
    State("soil-ph-input",       "value"),
    State("isotope-family-input", "value"),
    State("isotope-energy-input", "value"),
    prevent_initial_call=True
)
def calculate_results(calc_clicks, reset_clicks, sample_name, ra, th, k, u235, latitude, longitude, soil_texture, soil_ph, family, energy_peak_str):
    logging.debug(
        f"calculate_results callback triggered. calc_clicks={calc_clicks}, reset_clicks={reset_clicks}, "
        f"sample_name={sample_name}, ra={ra}, th={th}, k={k}, u235={u235}"
    )
    try:
        return _calculate_results_impl(calc_clicks, reset_clicks, sample_name, ra, th, k, u235, latitude, longitude, soil_texture, soil_ph, family, energy_peak_str)
    except Exception as e:
        logging.exception(f"Callback Error in calculate_results: {e}")
        return glow_card([
            html.Div("❌", style={"fontSize": "36px", "color": C["high"], "marginBottom": "12px"}),
            html.Div("CALCULATION ERROR", style={
                "fontFamily": "'Orbitron', sans-serif",
                "color": C["high"],
                "fontSize": "18px",
                "letterSpacing": "3px",
                "marginBottom": "8px",
            }),
            html.Div(f"An unexpected error occurred: {str(e)}", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "color": C["muted"],
                "fontSize": "13px",
            }),
        ], glow_color=C["high"], style={"textAlign": "center", "maxWidth": "500px", "margin": "80px auto"})

def _calculate_results_impl(calc_clicks, reset_clicks, sample_name, ra, th, k, u235, latitude, longitude, soil_texture, soil_ph, family, energy_peak_str):

    # ── Check if Reset or Clear Sample was triggered ──────────
    ctx = callback_context
    try:
        triggered = ctx.triggered
    except Exception:
        triggered = []

    trigger_id = ""
    if triggered:
        trigger_id = triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "reset-button" or (trigger_id == "sample-select" and not sample_name):
        return html.Div([
            html.Div("☢", style={
                "fontSize": "80px",
                "opacity": "0.15",
                "marginBottom": "20px",
                "animation": "spin 20s linear infinite",
            }),
            html.Div("AWAITING SAMPLE DATA", style={
                "fontFamily": "'Orbitron', sans-serif",
                "fontSize":   "24px",
                "letterSpacing": "6px",
                "color": C["muted"],
                "marginBottom": "12px",
            }),
            html.Div("Enter isotope concentrations in the panel and click ANALYSE SAMPLE", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "color":      C["muted"],
                "fontSize":   "13px",
            }),
        ], style={
            "display":        "flex",
            "flexDirection":  "column",
            "alignItems":     "center",
            "justifyContent": "center",
            "height":         "60vh",
            "textAlign":      "center",
        })

    # ── Determine Mode and Variables ──────────────────────────
    if trigger_id == "sample-select" and sample_name:
        row = sample_lookup.get(sample_name)
        if not row:
            return html.Div("Sample not found.", style={"color": C["high"]})
        ra_val = row["Ra226"]
        th_val = row["Th232"]
        k_val = row["K40"]
        u235_val = row["U235"]
        lat_val = row["Latitude"]
        lon_val = row["Longitude"]
        soil_texture_val = row["SoilTexture"]
        soil_ph_val = row["SoilPH"]
        family_val = row["IsotopeFamily"]
        energy_peak_val = row["EnergyPeak"]
        current_sample_id = row["Sample"]
    else:
        ra_val = ra
        th_val = th
        k_val = k
        u235_val = u235
        lat_val = latitude
        lon_val = longitude
        soil_texture_val = soil_texture
        soil_ph_val = soil_ph
        family_val = family
        energy_peak_val = energy_peak_str
        if sample_name:
            row = sample_lookup.get(sample_name)
            current_sample_id = row["Sample"] if row else None
        else:
            current_sample_id = None

    # Reassign for subsequent logic
    ra = ra_val
    th = th_val
    k = k_val
    u235 = u235_val
    latitude = lat_val
    longitude = lon_val
    soil_texture = soil_texture_val
    soil_ph = soil_ph_val
    family = family_val
    energy_peak_str = energy_peak_val

    # ── Validation ────────────────────────────────────────────
    if any(v is None for v in [ra, th, k, u235]):
        return glow_card([
            html.Div("⚠", style={"fontSize": "36px", "color": C["mod"], "marginBottom": "12px"}),
            html.Div("INCOMPLETE INPUT", style={
                "fontFamily": "'Orbitron', sans-serif",
                "color": C["mod"],
                "fontSize": "18px",
                "letterSpacing": "3px",
                "marginBottom": "8px",
            }),
            html.Div("Please enter Ra226, Th232, K40, and U235 values to proceed.", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "color": C["muted"],
                "fontSize": "13px",
            }),
        ], glow_color=C["mod"], style={"textAlign": "center", "maxWidth": "500px", "margin": "80px auto"})

    if any(v < 0 for v in [ra, th, k, u235]):
        return glow_card([
            html.Div("⚠", style={"fontSize": "36px", "color": C["high"], "marginBottom": "12px"}),
            html.Div("INVALID INPUT", style={
                "fontFamily": "'Orbitron', sans-serif",
                "color": C["high"],
                "fontSize": "18px",
                "letterSpacing": "3px",
                "marginBottom": "8px",
            }),
            html.Div("Radioactivity concentrations cannot be negative.", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "color": C["muted"],
                "fontSize": "13px",
            }),
        ], glow_color=C["high"], style={"textAlign": "center", "maxWidth": "500px", "margin": "80px auto"})

    if (latitude is not None and not (-90 <= latitude <= 90)) or (longitude is not None and not (-180 <= longitude <= 180)):
        return glow_card([
            html.Div("⚠", style={"fontSize": "36px", "color": C["high"], "marginBottom": "12px"}),
            html.Div("INVALID COORDINATES", style={
                "fontFamily": "'Orbitron', sans-serif",
                "color": C["high"],
                "fontSize": "18px",
                "letterSpacing": "3px",
                "marginBottom": "8px",
            }),
            html.Div("Latitude must be between -90 and 90. Longitude must be between -180 and 180.", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "color": C["muted"],
                "fontSize": "13px",
            }),
        ], glow_color=C["high"], style={"textAlign": "center", "maxWidth": "500px", "margin": "80px auto"})

    # ── Calculations ──────────────────────────────────────────
    dose      = calculate_dose_rate(ra, th, k)
    raeq      = calculate_raeq(ra, th, k)
    hex_val   = calculate_hex(ra, th, k)
    hin_val   = calculate_hin(ra, th, k)
    aed       = calculate_aed(dose)
    elcr      = calculate_elcr(aed)

    # ── ML Prediction (Anomaly Detection) ─────────────────────
    # Features expected by pipeline: Ra226, Th232, K40, U235, SoilPH, SoilTexture, MaterialType
    input_df = pd.DataFrame([{
        "Ra226": ra,
        "Th232": th,
        "K40": k,
        "U235": u235,
        "SoilPH": soil_ph if soil_ph is not None else np.nan,
        "SoilTexture": soil_texture if soil_texture is not None else np.nan,
        "MaterialType": np.nan
    }])
    
    # Predict outlier status: 1 = Normal, -1 = Anomalous
    anomaly_pred = anomaly_model.predict(input_df)[0]
    raw_decision_score = anomaly_model.decision_function(input_df)[0]
    anomaly_score = float(1.0 / (1.0 + np.exp(raw_decision_score * 5.0)))
    
    if anomaly_pred == -1:
        anomaly_status = "ANOMALOUS — VERIFY MEASUREMENT"
        anomaly_message = "The sample is statistically unusual compared with the learned development-data pattern. Verify the measurement and sample information before drawing environmental conclusions."
        anomaly_color = C["high"]
    else:
        anomaly_status = "NORMAL"
        anomaly_message = "The sample is within the learned development-data pattern."
        anomaly_color = C["low"]
        
    # Radiological screening risk level based on Raeq thresholds (NOT model target)
    if raeq < 100:
        predicted_label = "Low"
    elif raeq <= 370:
        predicted_label = "Moderate"
    else:
        predicted_label = "High"

    # Feature Importance (loaded from pre-calculated model_metadata)
    feat_importances = global_feature_importances
    
    # Explanation Bullet Points
    concentrations = {"Ra-226": ra, "Th-232": th, "K-40": k, "U-235": u235}
    rank1_name = feat_importances[0][0]
    rank2_name = feat_importances[1][0]
    rank3_name = feat_importances[2][0]
    rank4_name = feat_importances[3][0]
    
    explanation_items = []
    # Rank 1 explanation based on feature importance
    explanation_items.append(
        f"Based on Random Forest Feature Importance, {rank1_name} has the highest overall training importance, contributing most to prediction decisions across the dataset. In this sample, its concentration is {concentrations[rank1_name]:.1f} Bq/kg."
    )
    # Rank 2 explanation
    explanation_items.append(
        f"{rank2_name} is the secondary training feature. This sample has a concentration of {concentrations[rank2_name]:.1f} Bq/kg, indicating a moderate influence on predictions."
    )
    # Rank 3 explanation
    explanation_items.append(
        f"{rank3_name} concentration ({concentrations[rank3_name]:.1f} Bq/kg) represents the third most important feature in model training splits."
    )
    # Rank 4 explanation
    explanation_items.append(
        f"{rank4_name} concentration ({concentrations[rank4_name]:.1f} Bq/kg) carries minimal overall Gini importance, playing a minor role in final predictions."
    )

    # ── Figures ──
    # 1. Feature Importance chart
    feat_imp_y = [item[0] for item in reversed(feat_importances)]
    feat_imp_x = [item[1] * 100 for item in reversed(feat_importances)]
    feat_imp_fig = go.Figure(go.Bar(
        x=feat_imp_x,
        y=feat_imp_y,
        orientation="h",
        marker=dict(color=C["accent"], line=dict(color="rgba(255, 255, 255, 0.1)", width=1)),
        text=[f"{v:.1f}%" for v in feat_imp_x],
        textposition="inside",
        textfont=dict(family="JetBrains Mono", size=10, color="#000000"),
    ))
    feat_imp_fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Importance (%)"),
        yaxis=dict(showgrid=False),
        font={"family": "JetBrains Mono", "size": 10, "color": C["text"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=70, r=20, t=10, b=30),
        height=150,
    )

    # 2. Radionuclide Concentrations chart
    rad_categories = ["U-235", "Ra-226", "Th-232", "K-40"]
    rad_vals = [u235, ra, th, k]
    rad_colors = [C["accent2"], C["accent"], C["low"], C["muted"]]
    
    prob_fig = go.Figure(go.Bar(
        x=rad_vals,
        y=rad_categories,
        orientation="h",
        marker=dict(color=rad_colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.1f} Bq/kg" for v in rad_vals],
        textposition="inside",
        textfont=dict(family="JetBrains Mono", size=10, color="#ffffff"),
    ))
    prob_fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Activity Concentration (Bq/kg)"),
        yaxis=dict(showgrid=False),
        font={"family": "JetBrains Mono", "size": 10, "color": C["text"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=70, r=20, t=10, b=30),
        height=150,
    )

    # 3. Anomaly Score Distribution (Density Plot)
    train_scores_raw = cv_metrics.get("train_scores", [])
    train_anomaly_scores = [float(1.0 / (1.0 + np.exp(s * 5.0))) for s in train_scores_raw]
    
    if len(train_anomaly_scores) > 0:
        cm_fig = go.Figure()
        cm_fig.add_trace(go.Histogram(
            x=train_anomaly_scores,
            nbinsx=30,
            marker_color="rgba(0, 229, 255, 0.2)",
            marker_line=dict(color=C["border"], width=0.5),
            name="Training Data"
        ))
        cm_fig.add_shape(
            type="line",
            x0=anomaly_score, y0=0, x1=anomaly_score, y1=1,
            yref="paper",
            line=dict(color=anomaly_color, width=2, dash="dash")
        )
        cm_fig.add_trace(go.Scatter(
            x=[anomaly_score],
            y=[0],
            mode="markers",
            marker=dict(color=anomaly_color, size=8),
            name="Current Sample"
        ))
        cm_fig.update_layout(
            font={"family": "JetBrains Mono", "size": 10, "color": C["text"]},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=10, b=40),
            height=150,
            showlegend=False
        )
        cm_fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", title="Anomaly Score (0=Normal, 1=Outlier)", range=[0.0, 1.0])
        cm_fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", title="Count")
    else:
        cm_fig = go.Figure()

    # Validation status for 20 real samples
    validation_status = "Independent numerical dose-rate validation is not available because directly measured dose-rate ground truth is not present in the authorized dataset."

    # ML details dictionary for the PDF generator
    ml_details = {
        "model_name": best_model_name,
        "prediction_class": predicted_label,
        "anomaly_status": anomaly_status,
        "anomaly_score": f"{anomaly_score * 100:.1f}%",
        "anomaly_message": anomaly_message,
        "validation_status": validation_status,
        "feature_importances": feat_importances,
        "n_splits": 5,
        "performance_metrics": {
            "contamination": f"{contamination * 100:.1f}%",
            "n_estimators": str(n_estimators),
            "sample_count": str(ml_training_samples)
        },
        "explanation": explanation_items
    }

    # ── Isotope Detection ─────────────────────────────────────
    detected_isotope = "Not Detected"
    parent_series = "N/A"
    decay_chain = "N/A"
    energy_peak = "N/A"

    if family and energy_peak_str:
        details = ISOTOPE_STEPS.get(family, {}).get(energy_peak_str)
        if details:
            detected_isotope = details["isotope"]
            parent_series = details["parent"]
            decay_chain = details["chain"]
            energy_peak = f"{details['peak_val']} keV"

    # ── Location Geocoding ────────────────────────────────────
    if current_sample_id:
        row = sample_lookup.get(sample_name)
        loc_name = row["LocationName"]
        loc_city = row["District"]
        loc_state = row["Region"]
        loc_country = row["Country"]
    else:
        loc_data = reverse_geocode(latitude, longitude)
        loc_name = loc_data["location_name"]
        loc_city = loc_data["city"]
        loc_state = loc_data["state"]
        loc_country = loc_data["country"]

    # ── Soil Interpretation ──────────────────────────────────
    soil_texture_val = soil_texture if soil_texture else "Sandy Tropical Ferruginous Soil"
    texture_interp, ph_interp = get_soil_interpretation(soil_texture_val, soil_ph)

    generate_pdf(
        ra,
        th,
        k,
        u235,
        latitude,
        longitude,
        detected_isotope,
        parent_series,
        decay_chain,
        energy_peak,
        dose,
        raeq,
        hex_val,
        hin_val,
        aed,
        elcr,
        predicted_label,
        loc_name,
        loc_city,
        loc_state,
        loc_country,
        soil_texture_val,
        soil_ph,
        texture_interp,
        ph_interp,
        sample_id=current_sample_id,
        ml_details=ml_details
    )

    # ── Risk styling ──────────────────────────────────────────
    risk_color = C["low"] if predicted_label == "Low" else (C["mod"] if predicted_label == "Moderate" else C["high"])
    risk_icon  = "✅" if predicted_label == "Low" else ("⚠️" if predicted_label == "Moderate" else "🚨")

    # ── Percentages vs. safe limits ───────────────────────────
    raeq_pct = min(100, round(raeq / 370 * 100, 1))
    hex_pct  = min(100, round(hex_val * 100, 1))
    hin_pct  = min(100, round(hin_val * 100, 1))

    # ─────────────────────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────────────────────

    # 1. Bar chart
    bar_fig = go.Figure(go.Bar(
        x=["Dose Rate", "Raeq", "Hex", "Hin", "AED", "ELCR"],
        y=[dose, raeq, hex_val, hin_val, aed, elcr],
        marker=dict(
            color=[C["accent"], C["mod"], C["low"], C["high"], "#42A5F5", C["accent3"]],
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[str(v) for v in [dose, raeq, hex_val, hin_val, aed, elcr]],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono"),
    ))
    bar_fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="◈ Parameter Analysis", font=dict(size=14, family="Orbitron")),
        yaxis=dict(gridcolor="#1E2D40", showline=False),
        xaxis=dict(showgrid=False),
        showlegend=False,
        height=320,
    )

    # 2. Gauge
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dose,
        delta={"reference": 60, "increasing": {"color": C["high"]}, "decreasing": {"color": C["low"]}},
        number={"suffix": " nGy/h", "font": {"size": 22, "family": "Orbitron"}},
        title={"text": "◈ DOSE RATE", "font": {"size": 13, "family": "Orbitron"}},
        gauge={
            "axis":       {"range": [0, 300], "tickwidth": 1, "tickcolor": C["muted"]},
            "bar":        {"color": risk_color, "thickness": 0.25},
            "bgcolor":    "rgba(0,0,0,0)",
            "bordercolor": C["border"],
            "steps": [
                {"range": [0, 60], "color": "rgba(34,197,94,0.2)"},
                {"range": [60, 150], "color": "rgba(245,158,11,0.2)"},
                {"range": [150, 300], "color": "rgba(239,68,68,0.2)"},
            ],
            "threshold": {
                "line":  {"color": C["accent"], "width": 3},
                "thickness": 0.75,
                "value": 60
            },
        }
    ))
    gauge_fig.update_layout(**PLOTLY_LAYOUT, height=320)

    # 3. Spider / Radar chart
    categories  = ["Dose Rate", "Raeq", "Hex×100", "Hin×100", "AED×10", "ELCR×1k"]
    norm_vals   = [
        min(100, dose / 3),
        min(100, raeq / 3.7),
        min(100, hex_val * 100),
        min(100, hin_val * 100),
        min(100, aed * 10),
        min(100, elcr * 1000),
    ]
    radar_fig = go.Figure(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(245,158,11,0.2)",
        line=dict(color=risk_color, width=2),
        marker=dict(color=risk_color, size=6),
    ))
    radar_fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="◈ Risk Profile Radar", font=dict(size=14, family="Orbitron")),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=C["border"], linecolor=C["border"]),
            angularaxis=dict(gridcolor=C["border"], linecolor=C["border"]),
        ),
        height=320,
    )

    # 4. Safety threshold bar
    def threshold_bar(label, value, max_val, color):
        pct = min(100, value / max_val * 100)
        status = "SAFE" if pct < 70 else ("CAUTION" if pct < 100 else "EXCEEDED")
        s_color = C["low"] if pct < 70 else (C["mod"] if pct < 100 else C["high"])
        return html.Div([
            html.Div([
                html.Span(label, style={"fontFamily": "'JetBrains Mono', monospace", "fontSize": "11px", "color": C["muted"]}),
                html.Span(f"{value} / {max_val}", style={"fontFamily": "'JetBrains Mono', monospace", "fontSize": "11px", "color": C["text"]}),
                html.Span(status, style={"fontFamily": "'JetBrains Mono', monospace", "fontSize": "10px", "color": s_color, "letterSpacing": "1px"}),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"}),
            html.Div(
                html.Div(style={
                    "width":         f"{pct}%",
                    "height":        "6px",
                    "borderRadius":  "3px",
                    "background":    f"linear-gradient(90deg, {s_color}88, {s_color})",
                    "transition":    "width 1s ease",
                }),
                style={
                    "width": "100%", "height": "6px",
                    "background": "#1E2D40", "borderRadius": "3px",
                    "marginBottom": "16px"
                }
            ),
        ])

    # ─────────────────────────────────────────────────────────
    # MAP
    # ─────────────────────────────────────────────────────────

    if latitude is not None and longitude is not None:
        map_component = dl.Map(
            center=[latitude, longitude],
            zoom=7,
            style={"width": "100%", "height": "400px", "borderRadius": "12px"},
            children=[
                dl.TileLayer(
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                    attribution="© CartoDB"
                ),
                dl.Marker(
                    position=[latitude, longitude],
                    children=[dl.Popup([
                        html.H4("☢ Radiation Sample", style={"color": risk_color, "fontFamily": "Orbitron", "fontSize": "14px"}),
                        html.Hr(style={"borderColor": "#333"}),
                        html.P(f"Sample Number: {current_sample_id if current_sample_id else 'Manual Mode'}", style={"fontFamily": "monospace", "fontSize": "12px", "fontWeight": "bold", "color": C["accent"]}),
                        html.P(f"Location Name: {loc_name}",        style={"fontFamily": "monospace", "fontSize": "11.5px"}),
                        html.P(f"District:      {loc_city}",        style={"fontFamily": "monospace", "fontSize": "11.5px"}),
                        html.P(f"Region:        {loc_state}",       style={"fontFamily": "monospace", "fontSize": "11.5px"}),
                        html.P(f"Country:       {loc_country}",     style={"fontFamily": "monospace", "fontSize": "11.5px"}),
                        html.P(f"Soil Texture:  {soil_texture_val}",style={"fontFamily": "monospace", "fontSize": "11.5px", "color": C["accent2"]}),
                        html.P(f"Soil pH:       {soil_ph if soil_ph is not None else 'N/A'}", style={"fontFamily": "monospace", "fontSize": "11.5px", "color": C["accent"]}),
                        html.P(f"pH Interpretation: {ph_interp}",   style={"fontFamily": "monospace", "fontSize": "10.5px", "color": C["muted"]}),
                        html.Hr(style={"borderColor": "#222", "margin": "6px 0"}),
                        html.P(f"Risk Level:    {predicted_label}", style={"fontFamily": "monospace", "fontSize": "12px"}),
                        html.P(f"Dose Rate:     {dose} nGy/h",      style={"fontFamily": "monospace", "fontSize": "12px"}),
                        html.P(f"Detected Isotope: {detected_isotope}",   style={"fontFamily": "monospace", "fontSize": "11px", "color": C["accent"]}),
                        html.P(f"Parent Series:    {parent_series}",      style={"fontFamily": "monospace", "fontSize": "11px", "color": C["mod"]}),
                        html.P(f"Decay Chain:       {decay_chain}",        style={"fontFamily": "monospace", "fontSize": "11px", "color": C["text"]}),
                        html.P(f"Energy Peak:       {energy_peak}",        style={"fontFamily": "monospace", "fontSize": "11px", "color": C["muted"]}),
                        html.P(f"Coords:        {latitude:.4f}, {longitude:.4f}", style={"fontFamily": "monospace", "fontSize": "11px", "color": C["muted"]}),
                    ])]
                ),
            ]
        )
    else:
        map_component = html.Div("📍 Enter latitude & longitude to view map.", style={
            "textAlign": "center", "padding": "80px", "color": C["muted"],
            "fontFamily": "'JetBrains Mono', monospace", "fontSize": "13px",
        })

    # ─────────────────────────────────────────────────────────
    # ISOTOPE REFERENCE TABLE
    # ─────────────────────────────────────────────────────────

    all_isotopes = get_all_isotopes()
    isotope_rows = []
    
    selected_peak_val = None
    if family and energy_peak_str:
        details = ISOTOPE_STEPS.get(family, {}).get(energy_peak_str)
        if details:
            selected_peak_val = details["peak_val"]

    for peak, (name, chain) in all_isotopes.items():
        matches = []
        if selected_peak_val is not None and abs(selected_peak_val - peak) <= 5:
            matches.append(detected_isotope)
            
        is_match = len(matches) > 0
        match_str = f"◀ MATCH ({', '.join(matches)})" if is_match else ""
        row_style = {
            "background":  f"{C['accent']}11" if is_match else "transparent",
            "borderLeft":  f"3px solid {C['accent']}" if is_match else "3px solid transparent",
        }
        isotope_rows.append(html.Tr([
            html.Td(f"{peak} keV", style={"padding": "8px 12px", "fontFamily": "JetBrains Mono", "fontSize": "12px", "color": C["accent"] if is_match else C["text"]}),
            html.Td(name,          style={"padding": "8px 12px", "fontFamily": "JetBrains Mono", "fontSize": "12px", "color": C["mod"]    if is_match else C["text"]}),
            html.Td(chain,         style={"padding": "8px 12px", "fontFamily": "JetBrains Mono", "fontSize": "12px", "color": C["muted"]}),
            html.Td(match_str,
                    style={"padding": "8px 12px", "fontFamily": "JetBrains Mono", "fontSize": "10px", "color": C["accent"], "letterSpacing": "1px"}),
        ], style=row_style))

    # ─────────────────────────────────────────────────────────
    # COMPARATIVE SCORE
    # ─────────────────────────────────────────────────────────

    safety_score = max(0, round(100 - raeq_pct * 0.6 - hex_pct * 0.2 - hin_pct * 0.2, 1))
    score_color  = C["low"] if safety_score >= 70 else (C["mod"] if safety_score >= 40 else C["high"])

    # ─────────────────────────────────────────────────────────
    # RETURN LAYOUT
    # ─────────────────────────────────────────────────────────

    return html.Div([

        # ── RISK BANNER ──────────────────────────────────────
        html.Div([
            html.Div([
                html.Span(risk_icon + " ", style={"fontSize": "28px"}),
                html.Span(f"RISK LEVEL: {predicted_label.upper()}", style={
                    "fontFamily":    "'Orbitron', sans-serif",
                    "fontSize":      "24px",
                    "letterSpacing": "4px",
                    "color":         risk_color,
                }),
            ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),

            html.Div([
                html.Div(f"{safety_score}", style={
                    "fontFamily": "'Orbitron', sans-serif",
                    "fontSize":   "42px",
                    "fontWeight": "900",
                    "color":      score_color,
                    "lineHeight": "1",
                }),
                html.Div("SAFETY SCORE / 100", style={
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontSize":   "10px",
                    "color":      C["muted"],
                    "letterSpacing": "2px",
                }),
            ], style={"textAlign": "right"}),
        ], style={
            "display":        "flex",
            "justifyContent": "space-between",
            "alignItems":     "center",
            "background":     f"linear-gradient(135deg, {risk_color}11, transparent)",
            "border":         f"1px solid {risk_color}44",
            "borderRadius":   "14px",
            "padding":        "20px 28px",
            "marginBottom":   "28px",
        }),

        # ── METRIC CARDS ─────────────────────────────────────
        html.Div([
            metric_card("Dose Rate", dose,    "nGy/h",  C["accent"],  "☢"),
            metric_card("Raeq",      raeq,    "Bq/kg",  C["mod"],     "⚛"),
            metric_card("Hex",       hex_val, "index",  C["low"],     "🔵"),
            metric_card("Hin",       hin_val, "index",  C["high"],    "🔴"),
            metric_card("AED",       aed,     "mSv/y",  "#42A5F5",   "💊"),
            metric_card("ELCR",      elcr,    "×10⁻³",  C["accent3"], "🧬"),
        ], style={
            "display":      "grid",
            "gridTemplate": "repeat(2, 1fr) / repeat(3, 1fr)",
            "gap":          "14px",
            "marginBottom": "28px",
        }),

        # ── ENVIRONMENTAL ASSESSMENTS (NEW SECTIONS) ─────────
        html.Div([
            # Location details card
            glow_card([
                html.Div("◈ LOCATION DETAILS", style={
                    "fontFamily": "'Orbitron', sans-serif",
                    "fontSize":   "12px",
                    "letterSpacing": "3px",
                    "color":      C["accent"],
                    "marginBottom": "16px",
                }),
                html.Div([
                    html.Div([
                        html.Span("Sample Number: ", style={"color": C["accent"], "fontFamily": "JetBrains Mono", "fontSize": "12px", "fontWeight": "bold"}),
                        html.Span(current_sample_id if current_sample_id else "Manual Mode", style={"color": C["accent"], "fontFamily": "JetBrains Mono", "fontSize": "13px", "fontWeight": "bold"}),
                    ], style={"marginBottom": "12px", "borderBottom": f"1px solid {C['border']}66", "paddingBottom": "6px"}),
                    html.Div([
                        html.Span("Location Name: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(loc_name, style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "13px", "fontWeight": "600"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("District:      ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(loc_city, style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "13px"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Region:        ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(loc_state, style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "13px"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Country:       ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(loc_country, style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "13px"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Latitude: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(f"{latitude:.4f}" if latitude is not None else "N/A", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Longitude:", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(f"{longitude:.4f}" if longitude is not None else "N/A", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                    ]),
                ], style={"display": "flex", "flexDirection": "column"}),
            ], style={"flex": "1"}),

            # Soil analysis card
            glow_card([
                html.Div("◈ SOIL ANALYSIS", style={
                    "fontFamily": "'Orbitron', sans-serif",
                    "fontSize":   "12px",
                    "letterSpacing": "3px",
                    "color":      C["accent"],
                    "marginBottom": "16px",
                }),
                html.Div([
                    html.Div([
                        html.Span("Soil Texture: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(soil_texture_val, style={"color": C["accent2"], "fontFamily": "JetBrains Mono", "fontSize": "13px", "fontWeight": "600"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Soil pH:      ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(f"{soil_ph:.1f}" if soil_ph is not None else "N/A", style={"color": C["accent"], "fontFamily": "JetBrains Mono", "fontSize": "13px", "fontWeight": "600"}),
                    ], style={"marginBottom": "12px"}),
                    
                    html.Hr(style={"borderColor": C["border"], "margin": "8px 0"}),
                    
                    html.Div("INTERPRETATION", style={
                        "fontSize": "9px", "letterSpacing": "2px",
                        "color": C["muted"], "fontFamily": "JetBrains Mono", "marginBottom": "8px"
                    }),
                    html.Div(f"Texture: {texture_interp}", style={
                        "fontFamily": "'JetBrains Mono', monospace", "fontSize": "11px",
                        "color": C["text"], "marginBottom": "6px"
                    }),
                    html.Div(f"pH: {ph_interp}", style={
                        "fontFamily": "'JetBrains Mono', monospace", "fontSize": "11px",
                        "color": C["text"]
                    }),
                ], style={"display": "flex", "flexDirection": "column"}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

        # ── SAFETY THRESHOLDS ────────────────────────────────
        glow_card([
            html.Div("◈ SAFETY THRESHOLD ANALYSIS", style={
                "fontFamily": "'Orbitron', sans-serif",
                "fontSize":   "12px",
                "letterSpacing": "3px",
                "color":      C["accent"],
                "marginBottom": "20px",
            }),
            threshold_bar("Raeq vs limit (370 Bq/kg)",     raeq,    370,  risk_color),
            threshold_bar("Hex vs limit (1.0)",             hex_val, 1.0,  risk_color),
            threshold_bar("Hin vs limit (1.0)",             hin_val, 1.0,  risk_color),
            threshold_bar("Dose Rate vs normal (60 nGy/h)", dose,    60,   risk_color),
        ], style={"marginBottom": "28px"}),

        # ── CHARTS ROW 1 ─────────────────────────────────────
        html.Div([
            glow_card([dcc.Graph(figure=bar_fig,   config={"displayModeBar": False})], style={"flex": "1"}),
            glow_card([dcc.Graph(figure=gauge_fig, config={"displayModeBar": False})], style={"width": "380px"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

        # ── CHARTS ROW 2 ─────────────────────────────────────
        html.Div([
            glow_card([dcc.Graph(figure=radar_fig, config={"displayModeBar": False})], style={"flex": "1"}),

            # Isotope Detection Panel
            glow_card([
                html.Div("◈ RADIONUCLIDE ISOTOPE DETECTION", style={
                    "fontFamily": "'Orbitron', sans-serif",
                    "fontSize":   "12px",
                    "letterSpacing": "3px",
                    "color":      C["accent"],
                    "marginBottom": "16px",
                }),
                html.Div([
                    html.Div([
                        html.Span("Detected Isotope: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(detected_isotope, style={"color": C["accent2"], "fontFamily": "JetBrains Mono", "fontSize": "13px", "fontWeight": "600"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Parent Series:    ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(parent_series, style={"color": C["accent"], "fontFamily": "JetBrains Mono", "fontSize": "13px", "fontWeight": "600"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Decay Chain:      ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(decay_chain, style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "13px"}),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Energy Peak:      ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                        html.Span(energy_peak, style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "13px"}),
                    ]),
                ], style={
                    "background": "rgba(255,255,255,0.02)",
                    "border": f"1px solid {C['border']}",
                    "borderRadius": "8px",
                    "padding": "16px",
                    "display": "flex",
                    "flexDirection": "column",
                    "marginBottom": "12px",
                    "width": "100%"
                }),
                
                html.Hr(style={"borderColor": C["border"], "margin": "16px 0"}),
                
                html.Div("CLASSIFICATION SUMMARY", style={
                    "fontSize": "9px", "letterSpacing": "3px",
                    "color": C["muted"], "fontFamily": "JetBrains Mono", "marginBottom": "12px"
                }),
                html.Div([
                    html.Div([
                        html.Span("Risk Level: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                        html.Span(predicted_label, style={"color": risk_color, "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "600"}),
                    ]),
                    html.Div([
                        html.Span("Exposure Risk: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                        html.Span("Safe" if hex_val < 1 else "Unsafe", style={"color": C["low"] if hex_val < 1 else C["high"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "600"}),
                    ]),
                    html.Div([
                        html.Span("Contamination: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                        html.Span("Normal" if dose < 60 else ("Elevated" if dose < 150 else "Critical"), style={"color": C["low"] if dose < 60 else (C["mod"] if dose < 150 else C["high"]), "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "600"}),
                    ]),
                ], style={"display": "flex", "flexDirection": "column", "gap": "6px"}),
            ], style={"width": "420px"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

        # ── MACHINE LEARNING DIAGNOSTIC PANEL ─────────────────
        html.Div([
            html.Div([
                html.Span("◈ ", style={"color": C["accent"]}),
                html.Span("MACHINE LEARNING DIAGNOSTIC REPORT", style={
                    "fontFamily": "'Orbitron', sans-serif",
                    "fontSize": "14px",
                    "letterSpacing": "4px",
                    "color": C["text"],
                }),
            ], style={"borderBottom": f"1px solid {C['border']}aa", "paddingBottom": "10px", "marginBottom": "24px", "marginTop": "14px"}),

            # ROW 1: Anomaly Screening & Model Specifications
            html.Div([
                # ML Anomaly Screening Summary Card
                glow_card([
                    html.Div("◈ ANOMALY SCREENING", style={
                        "fontFamily": "'Orbitron', sans-serif",
                        "fontSize": "11px",
                        "letterSpacing": "2px",
                        "color": C["accent"],
                        "marginBottom": "16px"
                    }),
                    html.Div([
                        html.Div([
                            html.Span("Screening Status:  ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                            html.Span(anomaly_status, style={"color": anomaly_color, "fontFamily": "JetBrains Mono", "fontSize": "12px", "fontWeight": "bold"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span("Anomaly Score:      ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                            html.Span(f"{anomaly_score * 100:.2f}%", style={"color": C["accent"], "fontFamily": "JetBrains Mono", "fontSize": "12px", "fontWeight": "bold"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span("Analysis Message:   ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                            html.Div(anomaly_message, style={
                                "color": C["text"], 
                                "fontFamily": "JetBrains Mono", 
                                "fontSize": "11px", 
                                "marginTop": "6px",
                                "lineHeight": "1.4",
                                "padding": "10px",
                                "background": "rgba(255, 255, 255, 0.03)",
                                "borderRadius": "4px",
                                "borderLeft": f"3px solid {anomaly_color}"
                            }),
                        ]),
                    ], style={"display": "flex", "flexDirection": "column"}),
                ], style={"flex": "1"}),

                # Model Specifications Card
                glow_card([
                    html.Div("◈ MODEL SPECIFICATIONS", style={
                        "fontFamily": "'Orbitron', sans-serif",
                        "fontSize": "11px",
                        "letterSpacing": "2px",
                        "color": C["accent"],
                        "marginBottom": "16px"
                    }),
                    html.Div([
                        html.Div([
                            html.Span("Algorithm:        ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                            html.Span("Isolation Forest (sklearn)", style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "600"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span("Training Samples: ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                            html.Span(f"{ml_training_samples} (measured + synthetic)", style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span("Contamination:    ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                            html.Span(f"{contamination * 100:.1f}%", style={"color": C["low"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "bold"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span("Number of Trees:  ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                            html.Span(str(n_estimators), style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span("Input Features:   ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                            html.Span("Ra-226, Th-232, K-40, U-235, Soil pH, Texture", style={"color": C["accent2"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                        ]),
                    ], style={"display": "flex", "flexDirection": "column"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

            # ROW 2: Radionuclide Concentrations & Prediction Explanation
            html.Div([
                # Radionuclide Concentrations Card
                glow_card([
                    html.Div([
                        html.Span("◈ RADIONUCLIDE CONCENTRATIONS  //  ", style={"fontFamily": "'Orbitron', sans-serif", "fontSize": "11px", "letterSpacing": "2px", "color": C["accent"]}),
                        html.Span("Bq/kg", style={"fontFamily": "JetBrains Mono", "fontSize": "11px", "color": C["muted"]}),
                    ], style={"marginBottom": "12px"}),
                    dcc.Graph(figure=prob_fig, config={"displayModeBar": False}),
                ], style={"flex": "1"}),

                # Prediction Explanation Card
                glow_card([
                    html.Div("◈ DETECTION EXPLANATION", style={
                        "fontFamily": "'Orbitron', sans-serif",
                        "fontSize": "11px",
                        "letterSpacing": "2px",
                        "color": C["accent"],
                        "marginBottom": "16px"
                    }),
                    html.Ul([
                        html.Li(explanation_items[0], style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "marginBottom": "8px"}),
                        html.Li(explanation_items[1], style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "marginBottom": "8px"}),
                        html.Li(explanation_items[2], style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "marginBottom": "8px"}),
                        html.Li(explanation_items[3], style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                    ], style={"paddingLeft": "16px", "margin": "0"}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

            # ROW 3: ML Performance & Anomaly Score Density
            html.Div([
                # ML Performance Card
                glow_card([
                    html.Div("◈ ANOMALY SCREENING SUMMARY", style={
                        "fontFamily": "'Orbitron', sans-serif",
                        "fontSize": "11px",
                        "letterSpacing": "2px",
                        "color": C["accent"],
                        "marginBottom": "16px"
                    }),
                    html.Table([
                        html.Tbody([
                            html.Tr([
                                html.Td("Training Set Size", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "padding": "8px 0", "borderBottom": f"1px solid {C['border']}22"}),
                                html.Td(f"{ml_training_samples} samples", style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "bold", "textAlign": "right", "borderBottom": f"1px solid {C['border']}22"}),
                            ]),
                            html.Tr([
                                html.Td("Contamination Factor", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "padding": "8px 0", "borderBottom": f"1px solid {C['border']}22"}),
                                html.Td(f"{contamination * 100:.1f}%", style={"color": C["accent"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "bold", "textAlign": "right", "borderBottom": f"1px solid {C['border']}22"}),
                            ]),
                            html.Tr([
                                html.Td("Independent Validation", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "padding": "8px 0"}),
                                html.Td("Not Available", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "bold", "textAlign": "right"}),
                            ]),
                        ])
                    ], style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "12px"}),
                    html.Div(
                        validation_status,
                        style={
                            "fontFamily": "JetBrains Mono",
                            "fontSize": "10px",
                            "color": C["muted"],
                            "borderTop": f"1px solid {C['border']}44",
                            "paddingTop": "8px",
                            "lineHeight": "1.3"
                        }
                    )
                ], style={"flex": "1"}),

                # Confusion Matrix Card (Parity Plot)
                glow_card([
                    html.Div("◈ ANOMALY SCORE DISTRIBUTION (DENSITY PLOT)", style={
                        "fontFamily": "'Orbitron', sans-serif",
                        "fontSize": "11px",
                        "letterSpacing": "2px",
                        "color": C["accent"],
                        "marginBottom": "12px"
                    }),
                    dcc.Graph(figure=cm_fig, config={"displayModeBar": False}),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

            # ROW 4: Risk Screening & Feature Importance
            html.Div([
                # Risk Screening Card
                glow_card([
                    html.Div("◈ RADIOLOGICAL RISK SCREENING", style={
                        "fontFamily": "'Orbitron', sans-serif",
                        "fontSize": "11px",
                        "letterSpacing": "2px",
                        "color": C["accent"],
                        "marginBottom": "16px"
                    }),
                    html.Div([
                        html.Div([
                            html.Span("Equivalent Radium (Raeq):", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                            html.Span(f" {raeq:.2f} Bq/kg", style={"color": C["text"], "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "bold"}),
                        ], style={"marginBottom": "8px"}),
                        html.Div([
                            html.Span("Screening Risk Level:     ", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "11px"}),
                            html.Span(f"{predicted_label} Risk", style={"color": risk_color, "fontFamily": "JetBrains Mono", "fontSize": "11px", "fontWeight": "bold"}),
                        ], style={"marginBottom": "12px"}),
                        html.Div([
                            html.Div("Radiological Screening Thresholds:", style={"color": C["muted"], "fontFamily": "JetBrains Mono", "fontSize": "10px", "marginBottom": "4px"}),
                            html.Div("• Low Risk:      Raeq < 100 Bq/kg", style={"color": C["low"], "fontFamily": "JetBrains Mono", "fontSize": "10px"}),
                            html.Div("• Moderate Risk: Raeq 100 - 370 Bq/kg", style={"color": C["mod"], "fontFamily": "JetBrains Mono", "fontSize": "10px"}),
                            html.Div("• High Risk:     Raeq > 370 Bq/kg", style={"color": C["high"], "fontFamily": "JetBrains Mono", "fontSize": "10px"}),
                        ], style={"padding": "8px", "background": "rgba(255,255,255,0.02)", "borderRadius": "4px"}),
                    ], style={"display": "flex", "flexDirection": "column"}),
                ], style={"flex": "1"}),

                # Feature Importance Card
                glow_card([
                    html.Div("◈ RANDOM FOREST FEATURE IMPORTANCE", style={
                        "fontFamily": "'Orbitron', sans-serif",
                        "fontSize": "11px",
                        "letterSpacing": "2px",
                        "color": C["accent"],
                        "marginBottom": "12px"
                    }),
                    dcc.Graph(figure=feat_imp_fig, config={"displayModeBar": False}),
                    html.Div(
                        f"Feature importance shows the relative contribution of input variables to the model's predictions. The Random Forest model identified {feat_importances[0][0]} as the most influential feature.",
                        style={
                            "fontFamily": "JetBrains Mono", 
                            "fontSize": "11px", 
                            "color": C["muted"], 
                            "marginTop": "8px",
                            "borderTop": f"1px solid {C['border']}44",
                            "paddingTop": "8px"
                        }
                    )
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),
        ]),

        # ── ISOTOPE REFERENCE TABLE ────────────────────
        glow_card([
            html.Div("◈ ISOTOPE SPECTROSCOPY REFERENCE DATABASE", style={
                "fontFamily": "'Orbitron', sans-serif",
                "fontSize":   "12px",
                "letterSpacing": "3px",
                "color":      C["accent"],
                "marginBottom": "16px",
            }),
            html.Div("Matching rows are highlighted based on your entered energy peaks ± 5 keV tolerance.", style={
                "fontFamily": "'JetBrains Mono', monospace",
                "fontSize":   "11px",
                "color":      C["muted"],
                "marginBottom": "16px",
            }),
            html.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={
                        "padding": "10px 12px",
                        "textAlign": "left",
                        "fontFamily": "JetBrains Mono",
                        "fontSize": "10px",
                        "letterSpacing": "2px",
                        "color": C["muted"],
                        "borderBottom": f"1px solid {C['border']}",
                        "background": "#080C14",
                    })
                    for h in ["ENERGY PEAK", "ISOTOPE", "DECAY CHAIN", "STATUS"]
                ])),
                html.Tbody(isotope_rows),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ], style={"marginBottom": "28px"}),

        # ── MAP ──────────────────────────────────────────────
        glow_card([
            html.Div("◈ RADIATION MONITORING MAP", style={
                "fontFamily": "'Orbitron', sans-serif",
                "fontSize":   "12px",
                "letterSpacing": "3px",
                "color":      C["accent"],
                "marginBottom": "16px",
            }),
            map_component,
        ], style={"marginBottom": "28px"}),

    ])



REPORT_BUTTON_DISABLED_STYLE = {
    "width": "100%",
    "padding": "14px",
    "background": "linear-gradient(135deg, #28a74511, #28a74522)",
    "border": "1px solid #28a74544",
    "borderRadius": "10px",
    "color": "#28a74588",
    "fontFamily": "'Orbitron', sans-serif",
    "fontSize": "13px",
    "letterSpacing": "2px",
    "cursor": "not-allowed",
    "marginTop": "10px",
    "transition": "all 0.25s",
    "opacity": "0.5"
}

REPORT_BUTTON_ENABLED_STYLE = {
    "width": "100%",
    "padding": "14px",
    "background": "linear-gradient(135deg, #28a74522, #28a74544)",
    "border": "1px solid #28a745",
    "borderRadius": "10px",
    "color": "#28a745",
    "fontFamily": "'Orbitron', sans-serif",
    "fontSize": "13px",
    "letterSpacing": "2px",
    "cursor": "pointer",
    "marginTop": "10px",
    "transition": "all 0.25s",
    "opacity": "1"
}


# ─────────────────────────────────────────────────────────────
# REPORT BUTTON STATE CALLBACK
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("report-button", "disabled"),
    Output("report-button", "style"),
    Input("calculate-button", "n_clicks"),
    Input("reset-button", "n_clicks"),
    Input("sample-select", "value"),
    State("ra-input", "value"),
    State("th-input", "value"),
    State("k-input", "value"),
    State("u235-input", "value"),
    prevent_initial_call=True
)
def update_report_button(calc_clicks, reset_clicks, sample_name, ra, th, k, u235):
    logging.debug(f"update_report_button callback triggered. triggers={callback_context.triggered}")
    try:
        ctx = callback_context
        if not ctx.triggered:
            return True, REPORT_BUTTON_DISABLED_STYLE
        
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "reset-button":
            return True, REPORT_BUTTON_DISABLED_STYLE
            
        if trigger_id == "sample-select":
            if sample_name and sample_name != "Select Preloaded Sample...":
                return False, REPORT_BUTTON_ENABLED_STYLE
            else:
                return True, REPORT_BUTTON_DISABLED_STYLE
        
        if any(v is None for v in [ra, th, k, u235]) or any(v < 0 for v in [ra, th, k, u235]):
            return True, REPORT_BUTTON_DISABLED_STYLE
            
        return False, REPORT_BUTTON_ENABLED_STYLE
    except Exception as e:
        logging.error(f"Callback Error in update_report_button: {e}")
        return True, REPORT_BUTTON_DISABLED_STYLE


# ─────────────────────────────────────────────────────────────
# REPORT DOWNLOAD CALLBACK
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("download-report-pdf", "data"),
    Input("report-button", "n_clicks"),
    prevent_initial_call=True
)
def download_report(n_clicks):
    logging.debug(f"download_report callback triggered with n_clicks={n_clicks}")
    try:
        import os
        filename = "reports/Radiation_Report.pdf"
        if os.path.exists(filename):
            logging.debug(f"download_report sending file: {filename}")
            return dcc.send_file(filename)
        else:
            logging.warning(f"download_report: Report file not found at {filename}")
            raise dash.exceptions.PreventUpdate
    except Exception as e:
        if isinstance(e, dash.exceptions.PreventUpdate):
            raise
        logging.error(f"Callback Error in download_report: {e}")
        raise dash.exceptions.PreventUpdate


# ─────────────────────────────────────────────────────────────
# INPUTS SYNC, RESET & POPULATE CALLBACK
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("ra-input",             "value"),
    Output("th-input",             "value"),
    Output("k-input",              "value"),
    Output("u235-input",           "value"),
    Output("lat-input",            "value"),
    Output("long-input",           "value"),
    Output("soil-texture-input",   "value"),
    Output("soil-ph-input",        "value"),
    Output("isotope-family-input", "value"),
    Output("isotope-energy-input", "options"),
    Output("isotope-energy-input", "value"),
    Output("sample-select",        "value"),
    Input("sample-select",         "value"),
    Input("reset-button",          "n_clicks"),
    Input("isotope-family-input",  "value"),
    State("ra-input",              "value"),
    State("th-input",              "value"),
    State("k-input",               "value"),
    State("u235-input",            "value"),
    State("lat-input",             "value"),
    State("long-input",            "value"),
    State("soil-texture-input",    "value"),
    State("soil-ph-input",         "value"),
    State("isotope-energy-input",  "value"),
    prevent_initial_call=True
)
def sync_sidebar_inputs(sample_name, reset_clicks, family_val,
                        curr_ra, curr_th, curr_k, curr_u235,
                        curr_lat, curr_lon, curr_texture, curr_ph,
                        curr_energy):
    logging.debug(
        f"sync_sidebar_inputs callback triggered. sample_name={sample_name}, reset_clicks={reset_clicks}, "
        f"family_val={family_val}"
    )
    try:
        val = _sync_sidebar_inputs_impl(
            sample_name, reset_clicks, family_val,
            curr_ra, curr_th, curr_k, curr_u235,
            curr_lat, curr_lon, curr_texture, curr_ph,
            curr_energy
        )
        logging.debug("sync_sidebar_inputs returned successfully")
        return val
    except Exception as e:
        if isinstance(e, dash.exceptions.PreventUpdate):
            logging.debug("sync_sidebar_inputs raising PreventUpdate")
            raise
        logging.error(f"Callback Error in sync_sidebar_inputs: {e}")
        raise dash.exceptions.PreventUpdate

def _sync_sidebar_inputs_impl(sample_name, reset_clicks, family_val,
                            curr_ra, curr_th, curr_k, curr_u235,
                            curr_lat, curr_lon, curr_texture, curr_ph,
                            curr_energy):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Helper function to get energy peak options list
    def get_opts(fam):
        if not fam or fam not in ISOTOPE_STEPS:
            return []
        return [{"label": opt, "value": opt} for opt in ISOTOPE_STEPS[fam].keys()]

    # 1. Reset Button
    if triggered_id == "reset-button":
        default_fam = "Ra-226 Series"
        default_energy = "295 keV (Pb-214)"
        return (
            None, None, None, None, None, None,
            "Sandy Tropical Ferruginous Soil", None,
            default_fam, get_opts(default_fam), default_energy,
            None
        )

    # 2. Sample Select Dropdown
    if triggered_id == "sample-select":
        if not sample_name or sample_name == "Select Preloaded Sample...":
            default_fam = "Ra-226 Series"
            default_energy = "295 keV (Pb-214)"
            return (
                None, None, None, None, None, None,
                "Sandy Tropical Ferruginous Soil", None,
                default_fam, get_opts(default_fam), default_energy,
                None
            )

        row = sample_lookup.get(sample_name)
        if row:
            fam = row["IsotopeFamily"]
            energy = row["EnergyPeak"]
            return (
                row["Ra226"],
                row["Th232"],
                row["K40"],
                row["U235"],
                row["Latitude"],
                row["Longitude"],
                row["SoilTexture"],
                row["SoilPH"],
                fam,
                get_opts(fam),
                energy,
                sample_name
            )

    # 3. Isotope Family Dropdown
    if triggered_id == "isotope-family-input":
        # Update the peak options and select value based on family change
        opts = get_opts(family_val)
        if curr_energy and family_val in ISOTOPE_STEPS and curr_energy in ISOTOPE_STEPS[family_val]:
            energy_val = curr_energy
        else:
            energy_val = list(ISOTOPE_STEPS[family_val].keys())[0] if family_val in ISOTOPE_STEPS else None

        return (
            curr_ra, curr_th, curr_k, curr_u235,
            curr_lat, curr_lon, curr_texture, curr_ph,
            family_val, opts, energy_val,
            sample_name
        )

    raise dash.exceptions.PreventUpdate


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)

