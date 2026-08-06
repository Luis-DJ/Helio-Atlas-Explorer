"""
HelioAtlas Explorer
Configuration module

All user-editable settings live here.
No executable code should appear in this module.
"""

# ==========================================================
# Application
# ==========================================================

APP_NAME = "HelioAtlas"
APP_VERSION = "1.0.0-alpha1"

# ==========================================================
# Timeline
# ==========================================================

START = "2025-07-01"
STOP = "2037-03-31"
STEP = "1d"

MARK = "2026-01-10"

DATE_FMT = "%d-%b-%Y"

# ==========================================================
# Planet visibility
# ==========================================================

SHOW_PLANET_JUPITER = True
SHOW_PLANET_MARS = True
SHOW_PLANET_SATURN = True

# ==========================================================
# Orbit radii (AU)
# ==========================================================

EARTH_ORBIT_AU = 1.000
MARS_ORBIT_AU = 1.524
JUPITER_ORBIT_AU = 5.204
SATURN_ORBIT_AU = 9.582

# ==========================================================
# Marker sizes
# ==========================================================

SUN_GLOW_SIZE = 220
SUN_CORE_SIZE = 70

EARTH_SIZE = 60
MARS_SIZE = 65
JUPITER_SIZE = 80
SATURN_SIZE = 85

EVENT_SIZE = 95

# ==========================================================
# Event markers
# ==========================================================

EVENT_OPP_COLOR = "white"
EVENT_CONJ_COLOR = "crimson"

EVENT_MARKER_LW = 1.8

SHOW_EVENT_LABELS = True

# ==========================================================
# Solar glare limits
# ==========================================================

SHOW_SOLAR_GLARE_MASK = True

ELONG_HIDE_DEG = 10.0
ELONG_DIFFICULT_DEG = 15.0
ELONG_USABLE_DEG = 25.0
ELONG_EXCELLENT_DEG = 60.0

GLARE_SHADE_BELOW_DEG = ELONG_DIFFICULT_DEG

GLARE_SHADE_COLOR = "crimson"
GLARE_SHADE_ALPHA = 0.10

# ==========================================================
# Connector appearance
# ==========================================================

CONNECTOR_LW_DIFFICULT = 3.0
CONNECTOR_LW_MARGINAL = 3.2
CONNECTOR_LW_USABLE = 3.2
CONNECTOR_LW_EXCELLENT = 2.8

HIDE_CONNECTOR_BELOW_DEG = ELONG_HIDE_DEG

# ==========================================================
# Colours
# ==========================================================

SUN_CORE_COLOR = "orangered"
SUN_GLOW_COLOR = "yellow"

EARTH_COLOR = "dodgerblue"
MARS_COLOR = "tomato"
JUPITER_COLOR = "orange"
SATURN_COLOR = "gold"

GUIDE_COLOR = "gray"

BACKGROUND_COLOR = "#0d1117"