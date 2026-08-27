# isotope_lookup.py

# ─────────────────────────────────────────────────────────────
# EXTENDED ISOTOPE DATABASE
# ─────────────────────────────────────────────────────────────
# Format:  peak_keV : (isotope_name, decay_chain)

ISOTOPES = {

    # ── U-238 Series ──────────────────────────────────────────
    295:  ("214Pb",   "U238 Series"),
    352:  ("214Pb",   "U238 Series"),
    609:  ("214Bi",   "U238 Series"),
    1120: ("214Bi",   "U238 Series"),
    1765: ("214Bi",   "U238 Series"),
    242:  ("214Pb",   "U238 Series"),
    1238: ("214Bi",   "U238 Series"),
    2204: ("214Bi",   "U238 Series"),
    186:  ("226Ra",   "U238 Series"),
    63:   ("234Th",   "U238 Series"),
    92:   ("234Th",   "U238 Series"),

    # ── Th-232 Series ─────────────────────────────────────────
    338:  ("228Ac",   "Th232 Series"),
    583:  ("208Tl",   "Th232 Series"),
    911:  ("228Ac",   "Th232 Series"),
    969:  ("228Ac",   "Th232 Series"),
    2614: ("208Tl",   "Th232 Series"),
    239:  ("212Pb",   "Th232 Series"),
    300:  ("212Pb",   "Th232 Series"),
    727:  ("212Bi",   "Th232 Series"),
    1588: ("228Ac",   "Th232 Series"),

    # ── Potassium Series ──────────────────────────────────────
    1461: ("40K",     "Potassium Series"),

    # ── U-235 Series ──────────────────────────────────────────
    143:  ("235U",    "U235 Series"),
    163:  ("235U",    "U235 Series"),
    185:  ("235U",    "U235 Series"),
    205:  ("235U",    "U235 Series"),

    # ── Artificial / Anthropogenic ────────────────────────────
    662:  ("137Cs",   "Artificial Isotope"),
    1173: ("60Co",    "Artificial Isotope"),
    1332: ("60Co",    "Artificial Isotope"),
    1274: ("22Na",    "Artificial Isotope"),
    511:  ("22Na",    "Artificial Isotope"),   # annihilation
    834:  ("54Mn",    "Artificial Isotope"),
    1115: ("65Zn",    "Artificial Isotope"),
    411:  ("198Au",   "Artificial Isotope"),
    80:   ("133Ba",   "Artificial Isotope"),
    356:  ("133Ba",   "Artificial Isotope"),

}


# ─────────────────────────────────────────────────────────────
# DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────

def detect_isotope(user_peak, tolerance: int = 5):
    """
    Match a measured gamma-ray peak (keV) to a known isotope.

    Parameters
    ----------
    user_peak : float | int   — measured energy peak in keV
    tolerance : int           — ± window in keV (default 5)

    Returns
    -------
    (isotope_name, decay_chain) or ("Unknown", "No Matching Decay Chain")
    """
    best_match = None
    best_delta = tolerance + 1

    for peak, details in ISOTOPES.items():
        delta = abs(user_peak - peak)
        if delta <= tolerance and delta < best_delta:
            best_delta = delta
            best_match = details

    if best_match:
        return best_match[0], best_match[1]

    return "Unknown", "No Matching Decay Chain"


# ─────────────────────────────────────────────────────────────
# HELPER — return the full database (used by dashboard table)
# ─────────────────────────────────────────────────────────────

def get_all_isotopes() -> dict:
    """Return the complete ISOTOPES dictionary sorted by energy peak."""
    return dict(sorted(ISOTOPES.items()))


# ─────────────────────────────────────────────────────────────
# HELPER — list all isotopes in a given decay series
# ─────────────────────────────────────────────────────────────

def get_isotopes_by_series(series_name: str) -> dict:
    """
    Filter isotopes by decay chain / series name (case-insensitive substring).

    Example
    -------
    get_isotopes_by_series("Th232")  →  {338: ("228Ac", "Th232 Series"), ...}
    """
    return {
        peak: details
        for peak, details in ISOTOPES.items()
        if series_name.lower() in details[1].lower()
    }


# ─────────────────────────────────────────────────────────────
# TWO-LEVEL ISOTOPE SPECIFIC STEPS
# ─────────────────────────────────────────────────────────────

ISOTOPE_STEPS = {
    "Ra-226 Series": {
        "295 keV (Pb-214)": {
            "isotope": "Pb-214",
            "parent": "Ra-226",
            "chain": "Radium Series",
            "peak_val": 295
        },
        "352 keV (Pb-214)": {
            "isotope": "Pb-214",
            "parent": "Ra-226",
            "chain": "Radium Series",
            "peak_val": 352
        },
        "609 keV (Bi-214)": {
            "isotope": "Bi-214",
            "parent": "Ra-226",
            "chain": "Radium Series",
            "peak_val": 609
        },
        "1120 keV (Bi-214)": {
            "isotope": "Bi-214",
            "parent": "Ra-226",
            "chain": "Radium Series",
            "peak_val": 1120
        },
        "1764 keV (Bi-214)": {
            "isotope": "Bi-214",
            "parent": "Ra-226",
            "chain": "Radium Series",
            "peak_val": 1764
        }
    },
    "Th-232 Series": {
        "238 keV (Pb-212)": {
            "isotope": "Pb-212",
            "parent": "Th-232",
            "chain": "Thorium Series",
            "peak_val": 238
        },
        "338 keV (Ac-228)": {
            "isotope": "Ac-228",
            "parent": "Th-232",
            "chain": "Thorium Series",
            "peak_val": 338
        },
        "583 keV (Tl-208)": {
            "isotope": "Tl-208",
            "parent": "Th-232",
            "chain": "Thorium Series",
            "peak_val": 583
        },
        "911 keV (Ac-228)": {
            "isotope": "Ac-228",
            "parent": "Th-232",
            "chain": "Thorium Series",
            "peak_val": 911
        },
        "2614 keV (Tl-208)": {
            "isotope": "Tl-208",
            "parent": "Th-232",
            "chain": "Thorium Series",
            "peak_val": 2614
        }
    },
    "K-40": {
        "1461 keV": {
            "isotope": "K-40",
            "parent": "K-40",
            "chain": "Potassium Decay",
            "peak_val": 1461
        }
    },
    "U-235 Series": {
        "143 keV": {
            "isotope": "U-235",
            "parent": "U-235",
            "chain": "Actinium Series",
            "peak_val": 143
        },
        "163 keV": {
            "isotope": "U-235",
            "parent": "U-235",
            "chain": "Actinium Series",
            "peak_val": 163
        },
        "185 keV": {
            "isotope": "U-235",
            "parent": "U-235",
            "chain": "Actinium Series",
            "peak_val": 185
        },
        "205 keV": {
            "isotope": "U-235",
            "parent": "U-235",
            "chain": "Actinium Series",
            "peak_val": 205
        }
    },
    "Cs-137": {
        "662 keV": {
            "isotope": "Cs-137",
            "parent": "Cs-137",
            "chain": "Anthropogenic",
            "peak_val": 662
        }
    }
}