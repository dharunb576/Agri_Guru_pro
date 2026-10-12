import streamlit as st
import json
import requests
import math
import numpy as np
import folium
from streamlit_folium import st_folium
from groq import Groq
from datetime import datetime

# Shapely & SciPy for organic land boundary partitioning
SHAPELY_AVAILABLE = True
try:
    from shapely.geometry import shape, Polygon, MultiPolygon, Point, MultiPoint, LineString, GeometryCollection, mapping
    from shapely.ops import voronoi_diagram, unary_union
    from scipy.cluster.vq import kmeans
except ImportError:
    SHAPELY_AVAILABLE = False


# ─────────────────────────────────────────────
# CONFIG — keys loaded from Streamlit Secrets
# ─────────────────────────────────────────────
GROQ_API_KEY = ""
AGRO_API_KEY = ""

try:
    if hasattr(st, "secrets"):
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
        AGRO_API_KEY = st.secrets.get("AGRO_API_KEY", "")
except Exception:
    pass

# Fallback model list for maximum uptime
GROQ_MODELS  = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"]

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


SYSTEM_PROMPT = """You are Agri-Guru, a world-class Agricultural Consultant AI specialising
in biological crop pairing, companion planting, soil management, and smart farming systems.
Answer ONLY agriculture-related questions. For anything else, say:
"I'm Agri-Guru — I can only assist with agriculture-related questions. 🌾"
Keep answers practical, structured, and concise. Use metric units. Suggest eco-friendly practices.
When farm land zoning data or weather data is provided in the prompt context, use it to provide
hyper-local, zone-specific crop advice and actionable farming insights.
"""

# ── Bio zone definitions with extended agro attributes ──
BIO_ZONES = [
    {
        "id": "Z1", "name": "Nitrogen Fixer Zone",
        "main_crop": "Groundnut", "companion": "Sorghum",
        "pair_type": "Nitrogen-Fixer + Cereal",
        "reason": "Groundnut fixes atmospheric nitrogen in root nodules. Sorghum absorbs excess nitrogen as a heavy feeder. Deep + shallow root systems avoid moisture competition.",
        "soil": "Red Sandy Loam", "irrigation": "Drip Irrigation",
        "color": "#4ecb61", "icon": "🌿",
        "ph": "6.0 - 6.8", "n_pk": "High N, Low P demand"
    },
    {
        "id": "Z2", "name": "Three Sisters Zone",
        "main_crop": "Maize + Beans", "companion": "Pumpkin",
        "pair_type": "Three Sisters Polyculture",
        "reason": "Ancient bio-symbiosis: Maize provides sturdy stalks for beans to climb. Pole beans fix soil nitrogen. Pumpkin vines cover soil to retain moisture and suppress weeds.",
        "soil": "Alluvial Loam", "irrigation": "Canal / Furrow Fed",
        "color": "#e0b84a", "icon": "🌽",
        "ph": "6.5 - 7.5", "n_pk": "Balanced NPK"
    },
    {
        "id": "Z3", "name": "Pest Repellent Border",
        "main_crop": "Tomato + Chilli", "companion": "Marigold (border)",
        "pair_type": "Pest-Repellent + Nightshade",
        "reason": "Marigold roots emit alpha-terthienyl which suppresses root-knot nematodes and repels whiteflies. Chilli acts as a secondary deterrence layer for pests.",
        "soil": "Black Cotton Soil", "irrigation": "Micro-Sprinkler",
        "color": "#e07a4a", "icon": "🍅",
        "ph": "6.0 - 7.0", "n_pk": "High Potassium (K)"
    },
    {
        "id": "Z4", "name": "Pollinator Corridor",
        "main_crop": "Sunflower", "companion": "Lavender + Mustard",
        "pair_type": "Pollinator Attractor",
        "reason": "Bright sunflower and scented lavender attract honeybees and beneficial predators. Boosts cross-pollination rates across adjacent zones by ~40%.",
        "soil": "Sandy Loam", "irrigation": "Rainfed / Drip",
        "color": "#f0d060", "icon": "🌻",
        "ph": "6.0 - 7.2", "n_pk": "Moderate Phosphorous"
    },
    {
        "id": "Z5", "name": "Orchard Understory",
        "main_crop": "Mango + Coconut", "companion": "Turmeric + Ginger",
        "pair_type": "Canopy + Shade-Tolerant Understory",
        "reason": "Turmeric & ginger thrive under partial shade of tree canopy. Their aromatic rhizome secretions naturalize soil pathogen populations.",
        "soil": "Laterite Loam", "irrigation": "Overhead Sprinkler",
        "color": "#c47fe0", "icon": "🥭",
        "ph": "5.5 - 6.5", "n_pk": "Organic Matter Rich"
    },
    {
        "id": "Z6", "name": "Soil Recovery Block",
        "main_crop": "Green Manure (Dhaincha)", "companion": "Sesbania",
        "pair_type": "Soil Bio-Restoration",
        "reason": "Dhaincha fixes 80-100 kg atmospheric Nitrogen per hectare. Ploughed back before flowering to restore degraded organic carbon in 1 season.",
        "soil": "Degraded Sandy Clay", "irrigation": "Rainfed",
        "color": "#e05a5a", "icon": "🌱",
        "ph": "5.0 - 8.5 (Tolerant)", "n_pk": "Bio-accumulator"
    },
    {
        "id": "Z7", "name": "High-Value Spices Zone",
        "main_crop": "Cardamom + Pepper", "companion": "Silver Oak",
        "pair_type": "Shade Tree + Climbing Spice",
        "reason": "Pepper vines trail up silver oak trunks while cardamom grows in humid understory microclimate, maximizing 3D vertical spatial yield.",
        "soil": "Forest Loam", "irrigation": "Drip & Mist",
        "color": "#38bdf8", "icon": "🌱",
        "ph": "5.5 - 6.5", "n_pk": "High Potash"
    },
    {
        "id": "Z8", "name": "Bio-Dynamic Pulse Ring",
        "main_crop": "Cowpea + Pigeon Pea", "companion": "Finger Millet (Ragi)",
        "pair_type": "Pulse-Cereal Intercrop",
        "reason": "Drought-hardy pulses build soil nitrogen reserves while ragi provides coarse root structure that prevents soil erosion during monsoons.",
        "soil": "Red Clay Loam", "irrigation": "Deficit Irrigation",
        "color": "#a855f7", "icon": "🌾",
        "ph": "6.0 - 7.0", "n_pk": "Low Nitrogen input"
    },
    {
        "id": "Z9", "name": "Root & Tuber Sanctuary",
        "main_crop": "Sweet Potato + Tapioca", "companion": "Elephant Foot Yam",
        "pair_type": "Tuber Polyculture",
        "reason": "Deep rooting tuber architectures utilize subsoil moisture layers efficiently without interfering with shallow vegetable root zones.",
        "soil": "Friable Sandy Loam", "irrigation": "Sub-surface Drip",
        "color": "#f97316", "icon": "🍠",
        "ph": "5.8 - 6.8", "n_pk": "High Potash & Humus"
    },
    {
        "id": "Z10", "name": "Medicinal & Aromatic Belt",
        "main_crop": "Tulsi + Vetiver", "companion": "Lemongrass",
        "pair_type": "Aromatic Soil Anchor",
        "reason": "Vetiver roots penetrate up to 3m deep, stabilizing slope contours against runoff while aromatic oils repel sap-sucking insects.",
        "symbol": "🌿", "soil": "Gravelly Loam", "irrigation": "Rainfed",
        "color": "#10b981", "icon": "🌿",
        "ph": "6.0 - 7.5", "n_pk": "Low Nutrient demand"
    }
]

# ─────────────────────────────────────────────
# GEOMETRY & ORGANIC LAND PARTITIONING
# ─────────────────────────────────────────────
def parse_geojson_to_shapely(data):
    """
    Parses any GeoJSON structure (FeatureCollection, Feature, Geometry)
    and returns a clean, valid Shapely geometry (Polygon or MultiPolygon).
    """
    if isinstance(data, str):
        data = json.loads(data)

    geom = None
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError("GeoJSON FeatureCollection contains no features.")
        geoms = []
        for f in features:
            if f.get("geometry"):
                g = shape(f["geometry"])
                if not g.is_empty:
                    geoms.append(g)
        if not geoms:
            raise ValueError("No valid geometries found in FeatureCollection.")
        geom = unary_union(geoms)
    elif data.get("type") == "Feature":
        geom = shape(data.get("geometry"))
    else:
        geom = shape(data)

    if geom is None or geom.is_empty:
        raise ValueError("Loaded GeoJSON geometry is empty.")

    # Convert non-polygons (LineString/Point) into valid buffered polygons
    if geom.geom_type in ("LineString", "MultiLineString"):
        coords = list(geom.coords) if geom.geom_type == "LineString" else [c for g in geom.geoms for c in g.coords]
        if len(coords) >= 3 and coords[0] == coords[-1]:
            geom = Polygon(coords)
        else:
            geom = geom.convex_hull.buffer(0.0001)
    elif geom.geom_type in ("Point", "MultiPoint"):
        geom = geom.buffer(0.001)
    elif geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        geom = unary_union(polys) if polys else geom.convex_hull.buffer(0.0001)

    if not geom.is_valid:
        geom = geom.buffer(0)

    return geom


def split_polygon_into_organic_zones(land_geom, n_zones=6):
    """
    Splits any land polygon into n_zones organic, realistic sub-polygons
    conforming strictly to the input land boundary shape (Centroidal Voronoi Tessellation).
    """
    if land_geom.is_empty:
        return []
    if not land_geom.is_valid:
        land_geom = land_geom.buffer(0)

    minx, miny, maxx, maxy = land_geom.bounds
    width, height = maxx - minx, maxy - miny

    if width < 1e-9 or height < 1e-9:
        return [land_geom]

    # Sample interior points for centroidal Voronoi seed computation
    points = []
    np.random.seed(42)
    attempts = 0
    target_samples = max(400, n_zones * 60)

    while len(points) < target_samples and attempts < 3500:
        attempts += 1
        rx = np.random.uniform(minx, maxx)
        ry = np.random.uniform(miny, maxy)
        pt = Point(rx, ry)
        if land_geom.contains(pt):
            points.append([rx, ry])

    # Fallback to grid sampling if Monte Carlo misses complex geometry
    if len(points) < n_zones:
        gx = np.linspace(minx, maxx, 25)
        gy = np.linspace(miny, maxy, 25)
        for x in gx:
            for y in gy:
                pt = Point(x, y)
                if land_geom.contains(pt) or land_geom.intersects(pt):
                    points.append([x, y])

    if len(points) < n_zones:
        return [land_geom]

    points = np.array(points)

    try:
        centroids, _ = kmeans(points, n_zones)
    except Exception:
        centroids = points[:n_zones]

    # Ensure exactly n_zones seed points
    while len(centroids) < n_zones:
        jitter = np.random.normal(0, max(width, height) * 0.01, size=(1, 2))
        centroids = np.vstack([centroids, centroids[0:1] + jitter])

    centroid_pts = [Point(c[0], c[1]) for c in centroids]

    # Generate Voronoi diagram clipped to land boundary polygon
    env = land_geom.envelope.buffer(max(width, height) * 3)
    vor = voronoi_diagram(MultiPoint(centroid_pts), envelope=env)

    zones = []
    for cell in vor.geoms:
        inter = cell.intersection(land_geom)
        if not inter.is_empty and inter.area > 1e-12:
            if isinstance(inter, Polygon):
                zones.append(inter)
            elif isinstance(inter, MultiPolygon):
                for p in inter.geoms:
                    if p.area > 1e-12:
                        zones.append(p)

    # Sort zones by centroid position for consistent zone numbering (North to South, West to East)
    zones.sort(key=lambda z: (-z.centroid.y, z.centroid.x))
    return zones[:n_zones] if len(zones) >= n_zones else zones


def calculate_area_metrics(poly, center_lat):
    """
    Calculates polygon area in hectares, acres, and square meters based on latitude.
    """
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * math.cos(lat_rad)
    clon, clat = poly.centroid.x, poly.centroid.y

    if poly.geom_type == "Polygon":
        exterior = list(poly.exterior.coords)
    else:
        # MultiPolygon fallback
        exterior = list(poly.convex_hull.exterior.coords)

    pts_m = [((x - clon) * m_per_deg_lon, (y - clat) * m_per_deg_lat) for x, y in exterior]
    n = len(pts_m)
    area_m2 = 0.0
    for i in range(n - 1):
        area_m2 += pts_m[i][0] * pts_m[i+1][1] - pts_m[i+1][0] * pts_m[i][1]
    area_m2 = abs(area_m2) / 2.0

    ha = area_m2 / 10000.0
    acres = area_m2 / 4046.85642
    return round(ha, 3), round(acres, 3), round(area_m2, 1)


# ─────────────────────────────────────────────
# WEATHER FETCH FUNCTION
# ─────────────────────────────────────────────
def fetch_agro_weather(lat, lon, api_key):
    """
    Fetches current weather + agro data from Agromonitoring API.
    """
    try:
        weather_url = "http://api.agromonitoring.com/agro/1.0/weather"
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
        r = requests.get(weather_url, params=params, timeout=8)
        r.raise_for_status()
        w = r.json()

        result = {
            "success":       True,
            "temperature":   round(w.get("main", {}).get("temp", 0), 1),
            "feels_like":    round(w.get("main", {}).get("feels_like", 0), 1),
            "temp_min":      round(w.get("main", {}).get("temp_min", 0), 1),
            "temp_max":      round(w.get("main", {}).get("temp_max", 0), 1),
            "humidity":      w.get("main", {}).get("humidity", "N/A"),
            "pressure":      w.get("main", {}).get("pressure", "N/A"),
            "wind_speed":    round(w.get("wind", {}).get("speed", 0), 1),
            "wind_deg":      w.get("wind", {}).get("deg", 0),
            "clouds":        w.get("clouds", {}).get("all", 0),
            "visibility":    w.get("visibility", "N/A"),
            "description":   w.get("weather", [{}])[0].get("description", "N/A").title(),
            "weather_icon":  w.get("weather", [{}])[0].get("icon", "01d"),
            "rain_1h":       w.get("rain", {}).get("1h", 0),
            "soil_moisture": w.get("soil_moisture", "N/A"),
            "soil_temp":     w.get("soil_temp", "N/A"),
            "location_name": w.get("name", f"{lat:.4f}, {lon:.4f}"),
            "sunrise":       datetime.fromtimestamp(w.get("sys", {}).get("sunrise", 0)).strftime("%H:%M") if w.get("sys", {}).get("sunrise") else "N/A",
            "sunset":        datetime.fromtimestamp(w.get("sys", {}).get("sunset", 0)).strftime("%H:%M") if w.get("sys", {}).get("sunset") else "N/A",
            "timestamp":     datetime.now().strftime("%d %b %Y, %H:%M"),
        }
        return result

    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Network connection issue."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Weather request timed out."}
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        if code == 401:
            return {"success": False, "error": "Invalid Agromonitoring API key."}
        elif code == 429:
            return {"success": False, "error": "API rate limit reached."}
        else:
            return {"success": False, "error": f"API HTTP Error {code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def wind_direction(deg):
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]

def temp_color(temp):
    if temp < 15:   return "#5ab4e0"
    elif temp < 25: return "#4ecb61"
    elif temp < 32: return "#e0b84a"
    else:           return "#e05a5a"

def humidity_color(h):
    if isinstance(h, (int, float)):
        if h < 30:   return "#e05a5a"
        elif h < 60: return "#4ecb61"
        else:        return "#5ab4e0"
    return "#4ecb61"


# ─────────────────────────────────────────────
# PAGE CONFIG & CSS DESIGN SYSTEM
# ─────────────────────────────────────────────
st.set_page_config(page_title="Agri-Guru Pro — Smart Land Zoning", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #050b06;
    --surface:  #09140b;
    --card:     #0f1f12;
    --card2:    #152a19;
    --accent:   #3ddc6e;
    --accent2:  #27a84e;
    --glow:     rgba(61,220,110,0.2);
    --gold:     #eab308;
    --blue:     #38bdf8;
    --text:     #e2f2e5;
    --muted:    #62956b;
    --border:   #1b3820;
    --user-msg: #0e2712;
    --bot-msg:  #0a150c;
}

/* ── Global Styles ── */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}
.block-container { padding-top: 0.8rem !important; max-width: 1600px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #050d07 0%, #09190c 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; letter-spacing: 0.3px; }
h1 { color: var(--accent) !important; }
h2, h3 { color: #86efac !important; }

/* ── Chat Styling ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    margin-bottom: 4px !important;
    padding: 2px 0 !important;
}

[data-testid="stChatMessageContent"] {
    background: var(--bot-msg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 16px 16px 16px !important;
    padding: 12px 16px !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.4), inset 0 1px 0 rgba(61,220,110,0.06) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #0f2b14 0%, #15381c 100%) !important;
    border-color: #27562e !important;
    border-radius: 16px 0 16px 16px !important;
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    color: var(--text) !important;
    font-size: 0.94rem !important;
    line-height: 1.7 !important;
}
[data-testid="stChatMessageContent"] strong { color: var(--accent) !important; }

[data-testid="stChatInput"] {
    background: var(--card2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 20px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px var(--glow) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #050b06 !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(61,220,110,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(61,220,110,0.45) !important;
}

/* ── Custom Cards & Badges ── */
.zone-card {
    background: var(--card);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    border-left: 4px solid var(--accent);
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    transition: transform 0.15s, border-color 0.15s;
}
.zone-card:hover { transform: translateX(3px); border-color: var(--accent); }

.weather-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.65rem 0.8rem;
    text-align: center;
}
.weather-tile .val { font-size: 1.2rem; font-weight: 700; }
.weather-tile .lbl { font-size: 0.68rem; color: var(--muted); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.4px; }

.chat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: linear-gradient(90deg, rgba(61,220,110,0.09) 0%, transparent 100%);
    border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0;
    margin-bottom: 10px;
}
.chat-header .title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--accent);
}
.chat-header .subtitle { font-size: 0.72rem; color: var(--muted); margin-top: 1px; }

@keyframes blink { 0%,80%,100%{opacity:0.2} 40%{opacity:1} }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SAMPLE GEOJSON GENERATOR (Tamil Nadu Farm Plot)
# ─────────────────────────────────────────────
SAMPLE_FARM_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Kuttur Smart Farm Plot, Tamil Nadu"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [79.3850, 11.0920],
                        [79.3875, 11.0922],
                        [79.3882, 11.0905],
                        [79.3868, 11.0890],
                        [79.3845, 11.0898],
                        [79.3850, 11.0920]
                    ]
                ]
            }
        }
    ]
}


# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:0.6rem 0 0.2rem 0'>
      <div style='font-size:2.2rem;filter:drop-shadow(0 0 12px rgba(61,220,110,0.5))'>🌾</div>
      <div style='font-family:Cormorant Garamond,serif;font-size:1.35rem;font-weight:700;color:#3ddc6e'>
        Agri-Guru Pro
      </div>
      <div style='font-size:0.72rem;color:#62956b;margin-top:1px'>AI Smart Agriculture & Bio-Zoning</div>
    </div>
    <div style='height:1px;background:linear-gradient(90deg,transparent,#27562e,transparent);margin:0.6rem 0 0.8rem 0'></div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Farm Boundary (GeoJSON)")
    uploaded_file = st.file_uploader("Upload Land Boundary (.geojson)", type=["geojson", "json"])

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⚡ Load Sample", help="Load sample Tamil Nadu farm boundary"):
            st.session_state.active_geojson = SAMPLE_FARM_GEOJSON
            st.toast("Loaded sample farm plot!", icon="🌾")
            st.rerun()

    with col_btn2:
        if st.button("🗑️ Reset", help="Clear uploaded land data"):
            st.session_state.active_geojson = None
            st.session_state.weather_data   = None
            st.session_state.last_coords    = (None, None)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🌦️ Agro-Weather API")
    agro_key_input = st.text_input(
        "Agromonitoring API Key",
        value="",
        type="password",
        placeholder="Uses secret by default — paste to override"
    )
    manual_lat = st.number_input("Override Latitude",  value=0.0, format="%.4f", help="Leave 0 to auto-detect from land GeoJSON")
    manual_lon = st.number_input("Override Longitude", value=0.0, format="%.4f", help="Leave 0 to auto-detect from land GeoJSON")

    st.markdown("---")
    st.markdown("### ⚙️ Map & Partition Settings")
    n_zones     = st.slider("Number of Sub-Zones", 2, 10, 6, help="Divides land into organic land shapes matching boundary contours")
    map_style   = st.selectbox("Base Map Layer", ["Esri World Imagery", "OpenStreetMap", "CartoDB dark_matter", "CartoDB positron"], index=0)
    zoom_level  = st.slider("Initial Zoom Level", 12, 19, 17)
    show_labels = st.toggle("Show Crop Markers on Map", value=True)

    st.markdown("---")
    if st.button("💬 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.76rem;color:#4ecb61;line-height:1.9'>
      ✅ <b>Organic Land Partitioning:</b>voronoi tessellation clipped to shape<br>
      🔒 <b>Role-Locked AI:</b> Groq Agriculture Consultant<br>
      🌦️ <b>Agro-Weather:</b> Live temperature, humidity & soil params<br>
      🌿 <b>Bio-Pairing Engine:</b> Companion crops & ecological advice
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "weather_data"   not in st.session_state: st.session_state.weather_data   = None
if "last_coords"    not in st.session_state: st.session_state.last_coords    = (None, None)
if "active_geojson" not in st.session_state: st.session_state.active_geojson = None

# If user uploaded a new file, parse it into active_geojson
if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        st.session_state.active_geojson = json.loads(file_bytes.decode("utf-8"))
    except Exception as e:
        st.sidebar.error(f"Error reading GeoJSON file: {e}")


# ─────────────────────────────────────────────
# HEADER SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding:0.4rem 0 0.8rem 0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem'>
  <div>
    <div style='display:flex;align-items:center;gap:12px'>
      <div style='font-size:2.4rem;filter:drop-shadow(0 0 14px rgba(61,220,110,0.5))'>🌾</div>
      <div>
        <h1 style='margin:0;font-size:2.1rem;font-family:Cormorant Garamond,serif;
                   background:linear-gradient(135deg,#3ddc6e,#27a84e);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
          Agri-Guru Pro — Smart Farming & Organic Land Zoning
        </h1>
        <div style='font-size:0.83rem;color:#62956b;margin-top:2px;letter-spacing:0.3px'>
          Intelligent GeoJSON Land Splitting &nbsp;·&nbsp; Biological Crop Pairing &nbsp;·&nbsp; Agro-Weather Analytics
        </div>
      </div>
    </div>
  </div>
  <div style='display:flex;gap:8px;flex-wrap:wrap'>
    <div style='background:#09140b;border:1px solid #1b3820;border-radius:20px;padding:4px 14px;font-size:0.74rem;color:#3ddc6e'>
      🌱 Organic Land Shapes
    </div>
    <div style='background:#09140b;border:1px solid #1b3820;border-radius:20px;padding:4px 14px;font-size:0.74rem;color:#38bdf8'>
      ⚡ Groq LLaMA AI
    </div>
    <div style='background:#09140b;border:1px solid #1b3820;border-radius:20px;padding:4px 14px;font-size:0.74rem;color:#eab308'>
      🌦️ Agromonitoring Live
    </div>
  </div>
</div>
<div style='height:1px;background:linear-gradient(90deg,#3ddc6e33,#3ddc6e88,#3ddc6e33);margin-bottom:0.8rem'></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# THREE COLUMNS LAYOUT: Chat | Map | Weather
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1.0, 1.25, 0.85], gap="medium")


# ══════════════════════════════════════════════
# COL 1 — AI AGRI-CONSULTANT CHATBOT
# ══════════════════════════════════════════════
with col1:
    st.markdown("""
    <div class='chat-header'>
      <div style='font-size:1.4rem;filter:drop-shadow(0 0 8px rgba(61,220,110,0.6))'>🤖</div>
      <div>
        <div class='title'>AI Farming Assistant</div>
        <div class='subtitle'>Role-locked to Agriculture · Land & Weather aware</div>
      </div>
      <div style='margin-left:auto;display:flex;align-items:center;gap:6px'>
        <div style='width:7px;height:7px;border-radius:50%;background:#3ddc6e;
                    box-shadow:0 0 8px #3ddc6e;animation:blink 2s infinite'></div>
        <span style='font-size:0.7rem;color:#3ddc6e'>ONLINE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    chat_box = st.container(height=430)
    with chat_box:
        if not st.session_state.messages:
            st.markdown("""
            <div style='text-align:center;padding:2.2rem 1rem;'>
              <div style='font-size:3.5rem;filter:drop-shadow(0 0 20px rgba(61,220,110,0.4));
                          animation:blink 3s infinite'>🌾</div>
              <div style='font-family:Cormorant Garamond,serif;font-size:1.35rem;
                          font-weight:700;color:#3ddc6e;margin-top:0.8rem;'>
                Agri-Guru is ready
              </div>
              <div style='font-size:0.85rem;margin-top:0.4rem;color:#86efac;line-height:1.6'>
                Upload your farm GeoJSON to get zone-specific companion crop plans,<br>
                soil management & weather-based irrigation advice.
              </div>
              <div style='margin-top:1.2rem;display:flex;flex-wrap:wrap;gap:6px;justify-content:center'>
                <div style='background:#09140b;border:1px solid #1b3820;border-radius:16px;
                            padding:4px 12px;font-size:0.72rem;color:#62956b'>
                  🌱 Companion Planting
                </div>
                <div style='background:#09140b;border:1px solid #1b3820;border-radius:16px;
                            padding:4px 12px;font-size:0.72rem;color:#62956b'>
                  🪸 Soil Management
                </div>
                <div style='background:#09140b;border:1px solid #1b3820;border-radius:16px;
                            padding:4px 12px;font-size:0.72rem;color:#62956b'>
                  💧 Weather Irrigation
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Quick prompt buttons
    st.markdown("<div style='margin-top:6px;margin-bottom:6px;'>", unsafe_allow_html=True)
    qp_col1, qp_col2 = st.columns(2)
    quick_prompt = None
    with qp_col1:
        if st.button("🌿 Explain Crop Rotation", use_container_width=True):
            quick_prompt = "Explain a 3-year crop rotation plan for the sub-zones in my farm."
    with qp_col2:
        if st.button("💧 Weather Irrigation Plan", use_container_width=True):
            quick_prompt = "Based on current weather, how should I schedule irrigation for each zone?"
    st.markdown("</div>", unsafe_allow_html=True)

    chat_input_val = st.chat_input("💬 Ask Agri-Guru about your land, crops, soil or weather…")
    prompt = quick_prompt if quick_prompt else chat_input_val

    if prompt:
        # Build enriched context with land & weather details
        context_parts = []
        if st.session_state.weather_data and st.session_state.weather_data.get("success"):
            wd = st.session_state.weather_data
            context_parts.append(
                f"[LIVE FARM WEATHER: Temp={wd['temperature']}°C (Feels {wd['feels_like']}°C), "
                f"Humidity={wd['humidity']}%, Rain={wd['rain_1h']}mm, Wind={wd['wind_speed']}m/s {wind_direction(wd['wind_deg'])}, "
                f"Conditions={wd['description']}, Soil Temp={wd['soil_temp']}, Soil Moisture={wd['soil_moisture']}]"
            )

        if st.session_state.active_geojson:
            context_parts.append(
                f"[LAND BOUNDARY LOADED: Divided into {n_zones} organic land shape sub-zones with assigned crop pairings: " +
                ", ".join([f"{z['id']} ({z['name']}: {z['main_crop']} + {z['companion']})" for z in BIO_ZONES[:n_zones]]) + "]"
            )

        weather_context = "\n\n" + "\n".join(context_parts) if context_parts else ""
        enriched_prompt = prompt + weather_context

        st.session_state.messages.append({"role": "user", "content": prompt})

        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in st.session_state.messages[-10:]:
            role    = m["role"]
            content = enriched_prompt if (role == "user" and m == st.session_state.messages[-1]) else m["content"]
            history.append({"role": role, "content": content})

        # Try API call with model fallbacks
        reply = None
        active_client = client
        if not active_client and GROQ_API_KEY:
            try:
                active_client = Groq(api_key=GROQ_API_KEY)
            except Exception:
                active_client = None

        if not active_client:
            reply = "⚠️ **Missing Groq API Key** — Please add your GROQ_API_KEY to `.streamlit/secrets.toml`."
        else:
            for m_name in GROQ_MODELS:
                try:
                    response = active_client.chat.completions.create(model=m_name, messages=history, max_tokens=1024, temperature=0.65)
                    reply    = response.choices[0].message.content
                    break
                except Exception as ex:
                    err_str = str(ex)
                    if "401" in err_str:
                        reply = "⚠️ **Invalid Groq Key** — please check your API key in secrets.toml."
                        break
                    elif "429" in err_str:
                        reply = "⚠️ **Rate limit hit** — please wait a moment and try again."
                        break
                    continue

        if not reply:

            reply = "⚠️ Could not connect to Groq AI service. Please check your API key and internet connection."

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()


# ══════════════════════════════════════════════
# COL 2 — ORGANIC LAND SHAPE MAP & BIO ZONES
# ══════════════════════════════════════════════
with col2:
    st.markdown("""
    <div class='chat-header' style='border-left-color:#38bdf8;background:linear-gradient(90deg,rgba(56,189,248,0.08),transparent)'>
      <div style='font-size:1.4rem'>🗺️</div>
      <div>
        <div class='title' style='color:#38bdf8'>Smart Bio-Zone Land Map</div>
        <div class='subtitle'>Organic land shapes matching boundary contours (Tamil Nadu district parcel style)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.active_geojson:
        try:
            # Parse land boundary with Shapely
            land_geom = parse_geojson_to_shapely(st.session_state.active_geojson)
            clon, clat = land_geom.centroid.x, land_geom.centroid.y

            # Calculate total farm area
            tot_ha, tot_acres, tot_m2 = calculate_area_metrics(land_geom, clat)

            # Auto-fetch weather when coordinates change
            fetch_lat = manual_lat if manual_lat != 0.0 else clat
            fetch_lon = manual_lon if manual_lon != 0.0 else clon

            if (fetch_lat, fetch_lon) != st.session_state.last_coords:
                st.session_state.last_coords = (fetch_lat, fetch_lon)
                _sidebar_key = agro_key_input.strip() if agro_key_input.strip() not in ["", "YOUR_AGROMONITORING_API_KEY_HERE"] else ""
                active_key   = _sidebar_key if _sidebar_key else (AGRO_API_KEY.strip() if AGRO_API_KEY.strip() else None)
                if active_key:
                    with st.spinner("🌦️ Fetching live agro-weather..."):
                        st.session_state.weather_data = fetch_agro_weather(fetch_lat, fetch_lon, active_key)
                else:
                    st.session_state.weather_data = None

            # Split polygon into organic sub-zones (Centroidal Voronoi tessellation clipped to boundary)
            sub_polys = split_polygon_into_organic_zones(land_geom, n_zones=n_zones)
            zones_to_draw = BIO_ZONES[:len(sub_polys)]

            # Map tiles definition
            tile_opts = {
                "Esri World Imagery": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Esri World Imagery"),
                "OpenStreetMap": ("OpenStreetMap", "OpenStreetMap"),
                "CartoDB dark_matter": ("CartoDB dark_matter", "CartoDB dark_matter"),
                "CartoDB positron": ("CartoDB positron", "CartoDB positron"),
            }
            t_url, t_attr = tile_opts.get(map_style, ("OpenStreetMap", "OpenStreetMap"))

            m = folium.Map(location=[clat, clon], zoom_start=zoom_level, tiles=t_url, attr=t_attr if t_url != map_style else None)

            # Draw outer land boundary with glowing white/gold vector stroke
            minx, miny, maxx, maxy = land_geom.bounds
            
            # Format boundary for GeoJson overlay
            boundary_geojson = mapping(land_geom)
            folium.GeoJson(
                {"type": "Feature", "geometry": boundary_geojson, "properties": {"name": "Land Boundary"}},
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "color": "#facc15",
                    "weight": 3.5,
                    "dashArray": "6 4",
                    "fillOpacity": 0
                },
                tooltip=folium.Tooltip("📍 Outer Land Boundary"),
            ).add_to(m)

            # Prepare Export GeoJSON collection
            export_features = []

            # Draw each organic sub-zone polygon
            for idx, (zone, z_poly) in enumerate(zip(zones_to_draw, sub_polys)):
                z_ha, z_acres, z_m2 = calculate_area_metrics(z_poly, clat)
                z_clat, z_clon = z_poly.centroid.y, z_poly.centroid.x

                # Weather hint for popup
                wd = st.session_state.weather_data
                weather_hint = ""
                if wd and wd.get("success"):
                    weather_hint = f"""
                    <div style="margin-top:6px;padding:5px 8px;background:#09190c;
                                border-radius:6px;font-size:0.74rem;color:#86efac">
                      🌡️ {wd['temperature']}°C &nbsp; 💧 {wd['humidity']}% &nbsp; 🌬️ {wd['wind_speed']}m/s &nbsp; ☁️ {wd['description']}
                    </div>"""

                area_disp = f"{z_ha} Ha ({z_acres} Acres)" if z_ha > 0.05 else f"{z_m2} m²"

                tooltip_html = f"""
                <div style="font-family:'Inter',sans-serif;min-width:240px;max-width:300px;
                            background:#0f2413;border:1.5px solid {zone['color']};
                            border-radius:10px;padding:10px 13px;color:#e2f2e5">
                  <div style="font-size:1.3rem;text-align:center">{zone['icon']}</div>
                  <b style="color:{zone['color']};font-size:0.98rem">{zone['name']} ({zone['id']})</b>
                  <div style="font-size:0.74rem;color:#38bdf8;margin-bottom:4px">📐 Area: <b>{area_disp}</b></div>
                  <div style="margin:3px 0;font-size:0.82rem">🌱 <b>Main Crop:</b> {zone['main_crop']}</div>
                  <div style="margin:3px 0;font-size:0.82rem">🤝 <b>Companion:</b> {zone['companion']}</div>
                  <div style="margin:3px 0;font-size:0.80rem">🪸 <b>Soil:</b> {zone['soil']}</div>
                  <div style="margin:3px 0;font-size:0.80rem">💧 <b>Irrigation:</b> {zone['irrigation']}</div>
                  {weather_hint}
                  <div style="margin-top:6px;padding:6px;background:#09140b;
                              border-radius:6px;font-size:0.75rem;color:#bbf7d0;line-height:1.5">
                    💡 {zone['reason']}
                  </div>
                </div>"""

                z_geojson_geom = mapping(z_poly)

                folium.GeoJson(
                    {"type": "Feature", "geometry": z_geojson_geom, "properties": {"zone_id": zone["id"], "name": zone["name"]}},
                    style_function=lambda x, col=zone["color"]: {
                        "fillColor": col,
                        "color": "#09140b",
                        "weight": 2.5,
                        "fillOpacity": 0.55
                    },
                    highlight_function=lambda x: {"fillOpacity": 0.85, "weight": 3.5},
                    tooltip=folium.Tooltip(tooltip_html, sticky=True)
                ).add_to(m)

                # Add to export features
                export_features.append({
                    "type": "Feature",
                    "geometry": z_geojson_geom,
                    "properties": {
                        "zone_id": zone["id"],
                        "zone_name": zone["name"],
                        "main_crop": zone["main_crop"],
                        "companion_crop": zone["companion"],
                        "pair_type": zone["pair_type"],
                        "soil_type": zone["soil"],
                        "irrigation": zone["irrigation"],
                        "area_ha": z_ha,
                        "area_acres": z_acres
                    }
                })

                # Marker labels on map
                if show_labels:
                    folium.Marker(
                        [z_clat, z_clon],
                        icon=folium.DivIcon(html=f"""
                        <div style="font-family:'Inter',sans-serif;text-align:center;pointer-events:none;white-space:nowrap;">
                          <div style="font-size:15px">{zone['icon']}</div>
                          <div style="font-size:9px;font-weight:700;color:#fff;text-shadow:0 1px 4px #000;
                                      background:rgba(9,20,11,0.85);border:1px solid {zone['color']};
                                      border-radius:4px;padding:1px 5px;">
                            {zone['id']}: {zone['main_crop'][:14]}</div>
                          <div style="font-size:8px;color:#86efac;text-shadow:0 1px 3px #000;
                                      background:rgba(9,20,11,0.65);border-radius:3px;padding:0 4px;">
                            +{zone['companion'][:12]}</div>
                        </div>""", icon_size=(130, 48), icon_anchor=(65, 24))
                    ).add_to(m)

            folium.FitBounds([[miny, minx], [maxy, maxx]]).add_to(m)
            st_folium(m, width="100%", height=440, returned_objects=[], use_container_width=True)

            # Export GeoJSON string
            export_geojson_dict = {
                "type": "FeatureCollection",
                "features": export_features
            }
            export_json_str = json.dumps(export_geojson_dict, indent=2)

            # Metrics Summary Bar
            tot_disp = f"{tot_ha} Ha ({tot_acres} Acres)" if tot_ha > 0.05 else f"{tot_m2} m²"
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("📍 Sub-Zones", len(sub_polys))
            mc2.metric("📐 Total Area", tot_disp)
            mc3.metric("🌐 Center Lat", f"{clat:.4f}°")
            mc4.metric("🌐 Center Lon", f"{clon:.4f}°")

            # GeoJSON Download Button
            st.download_button(
                label="📥 Export Zone Plan (GeoJSON)",
                data=export_json_str,
                file_name=f"smart_farm_zoning_{n_zones}zones.geojson",
                mime="application/json",
                help="Download divided organic land shapes with embedded crop metadata"
            )

            # Biological Zone Breakdown Grid
            st.markdown("---")
            st.markdown("**🌿 Organic Land Shape & Bio-Zone Breakdown**")
            card_cols = st.columns(2)
            for i, (zone, z_poly) in enumerate(zip(zones_to_draw, sub_polys)):
                z_ha, z_acres, z_m2 = calculate_area_metrics(z_poly, clat)
                area_str = f"{z_ha} Ha ({z_acres} Ac)" if z_ha > 0.05 else f"{z_m2} m²"
                with card_cols[i % 2]:
                    st.markdown(f"""
                    <div class='zone-card' style='border-left-color:{zone["color"]}'>
                      <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:3px'>
                        <div style='display:flex;align-items:center;gap:7px'>
                          <span style='font-size:1.1rem'>{zone['icon']}</span>
                          <b style='color:{zone["color"]};font-size:0.88rem'>{zone['id']}: {zone['name']}</b>
                        </div>
                        <span style='font-size:0.72rem;color:#38bdf8;background:#09140b;padding:1px 6px;border-radius:4px;border:1px solid #1b3820'>
                          {area_str}
                        </span>
                      </div>
                      <div style='font-size:0.82rem'>🌱 <b>{zone['main_crop']}</b> + {zone['companion']}</div>
                      <div style='font-size:0.76rem;color:#86efac;margin-top:2px'>⚗️ {zone['pair_type']} &nbsp;|&nbsp; 🪸 {zone['soil']}</div>
                      <div style='font-size:0.73rem;color:#62956b;font-style:italic;margin-top:4px;line-height:1.4'>
                        💡 {zone['reason'][:100]}…</div>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Geometry processing error: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    else:
        st.markdown("""
        <div style='background:#0f2413;border:2px dashed #27562e;border-radius:18px;
                    padding:3.5rem 2rem;text-align:center;'>
          <div style='font-size:3.5rem;filter:drop-shadow(0 0 16px rgba(61,220,110,0.4))'>🗺️</div>
          <div style='font-size:1.1rem;margin-top:0.8rem;color:#86efac;font-weight:600'>No Land Data Loaded</div>
          <div style='font-size:0.84rem;color:#62956b;margin-top:0.6rem;line-height:1.7'>
            1. Upload a <code style="background:#09140b;padding:2px 6px;border-radius:4px;color:#3ddc6e">.geojson</code> file in sidebar<br>
            <b>OR</b> click <code style="background:#09140b;padding:2px 6px;border-radius:4px;color:#3ddc6e">⚡ Load Sample</code> to view the Tamil Nadu plot!<br><br>
            The land will be automatically divided into <b>organic land shapes</b> conforming strictly to the plot boundary.
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# COL 3 — LIVE AGRO-WEATHER DASHBOARD
# ══════════════════════════════════════════════
with col3:
    st.markdown("""
    <div class='chat-header' style='border-left-color:#eab308;background:linear-gradient(90deg,rgba(234,179,8,0.08),transparent)'>
      <div style='font-size:1.4rem'>🌦️</div>
      <div>
        <div class='title' style='color:#eab308'>Agro-Weather</div>
        <div class='subtitle'>Live data via Agromonitoring API</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    wd = st.session_state.weather_data

    if wd is None:
        st.markdown("""
        <div style='background:#0f2413;border:2px dashed #27562e;border-radius:14px;
                    padding:2rem 1rem;text-align:center;'>
          <div style='font-size:2.5rem'>🌦️</div>
          <div style='font-size:0.95rem;color:#86efac;margin-top:0.7rem;font-weight:500'>
            No Weather Loaded
          </div>
          <div style='font-size:0.78rem;color:#62956b;margin-top:0.4rem;line-height:1.6'>
            1. Load or upload your farm GeoJSON<br>
            2. Weather auto-fetches from farm coordinates<br><br>
            <b style='color:#3ddc6e'>Free Key:</b> agromonitoring.com
          </div>
        </div>""", unsafe_allow_html=True)

    elif not wd.get("success"):
        st.error(f"⚠️ Weather fetch failed:\n{wd.get('error','Unknown error')}")
        st.markdown("""
        <div style='font-size:0.8rem;color:#62956b;padding:0.5rem;'>
          Get a free key at <a href='https://agromonitoring.com' style='color:#3ddc6e' target='_blank'>agromonitoring.com</a><br>
          → Dashboard → API Keys → Paste key in sidebar.
        </div>""", unsafe_allow_html=True)

    else:
        tc = temp_color(wd["temperature"])
        hc = humidity_color(wd["humidity"])

        # Location header
        st.markdown(f"""
        <div class='zone-card' style='text-align:center;border-left-color:#3ddc6e'>
          <div style='font-size:0.72rem;color:#62956b'>{wd['timestamp']}</div>
          <div style='font-size:1.05rem;font-weight:700;color:#3ddc6e;margin:2px 0'>
            📍 {wd['location_name']}
          </div>
          <div style='font-size:1.6rem;margin:3px 0'>{wd['description']}</div>
        </div>""", unsafe_allow_html=True)

        # Main temp + humidity
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:0.5rem'>
          <div class='weather-tile'>
            <div class='val' style='color:{tc}'>{wd['temperature']}°C</div>
            <div class='lbl'>🌡️ Temperature</div>
            <div style='font-size:0.68rem;color:#62956b'>Feels {wd['feels_like']}°C</div>
          </div>
          <div class='weather-tile'>
            <div class='val' style='color:{hc}'>{wd['humidity']}%</div>
            <div class='lbl'>💧 Humidity</div>
            <div style='font-size:0.68rem;color:#62956b'>{
              'Dry — Irrigate' if isinstance(wd['humidity'], (int, float)) and wd['humidity'] < 35
              else 'Optimal' if isinstance(wd['humidity'], (int, float)) and wd['humidity'] < 65
              else 'High Humidity'
            }</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Grid of agro params
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:0.5rem'>
          <div class='weather-tile'>
            <div class='val' style='color:#38bdf8'>{wd['wind_speed']} m/s</div>
            <div class='lbl'>🌬️ Wind ({wind_direction(wd['wind_deg'])})</div>
          </div>
          <div class='weather-tile'>
            <div class='val' style='color:#eab308'>{wd['clouds']}%</div>
            <div class='lbl'>☁️ Cloud Cover</div>
          </div>
          <div class='weather-tile'>
            <div class='val' style='color:#38bdf8'>{wd['rain_1h']} mm</div>
            <div class='lbl'>🌧️ Rain (1h)</div>
          </div>
          <div class='weather-tile'>
            <div class='val' style='color:#86efac'>{wd['pressure']}</div>
            <div class='lbl'>📊 Pressure hPa</div>
          </div>
          <div class='weather-tile'>
            <div class='val' style='color:#fde047'>{wd['sunrise']}</div>
            <div class='lbl'>🌅 Sunrise</div>
          </div>
          <div class='weather-tile'>
            <div class='val' style='color:#f97316'>{wd['sunset']}</div>
            <div class='lbl'>🌇 Sunset</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Agro-specific params
        st.markdown(f"""
        <div class='zone-card' style='border-left-color:#c47fe0'>
          <div style='font-size:0.8rem;font-weight:700;color:#c47fe0;margin-bottom:5px'>
            🌾 Soil & Agro Metrics
          </div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem'>
            <div class='weather-tile'>
              <div class='val' style='color:#c47fe0'>{wd['soil_moisture']}</div>
              <div class='lbl'>💧 Soil Moisture</div>
            </div>
            <div class='weather-tile'>
              <div class='val' style='color:#e07a4a'>{wd['soil_temp']}</div>
              <div class='lbl'>🌡️ Soil Temp</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Temp range bar
        t_min, t_max = wd['temp_min'], wd['temp_max']
        t_range = max(t_max - t_min, 0.1)
        t_pos   = max(0, min(100, ((wd['temperature'] - t_min) / t_range) * 100))
        st.markdown(f"""
        <div class='zone-card' style='border-left-color:#38bdf8'>
          <div style='font-size:0.76rem;color:#62956b;margin-bottom:4px'>
            🌡️ Today's Range: {t_min}°C — {t_max}°C
          </div>
          <div style='background:#09140b;border-radius:6px;height:8px;position:relative;overflow:hidden'>
            <div style='position:absolute;left:0;top:0;height:100%;
                        background:linear-gradient(90deg,#38bdf8,#4ecb61,#eab308,#ef4444);
                        width:100%;border-radius:6px'></div>
            <div style='position:absolute;top:-3px;left:{t_pos:.0f}%;
                        transform:translateX(-50%);width:14px;height:14px;
                        background:#fff;border-radius:50%;border:2px solid #09140b;'></div>
          </div>
          <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#62956b;margin-top:3px'>
            <span>{t_min}°C</span><span style='color:{tc};font-weight:700'>{wd['temperature']}°C</span><span>{t_max}°C</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # AI Agro Advice Chip
        st.markdown(f"""
        <div style='background:#0f2413;border:1px solid #3ddc6e;border-radius:10px;
                    padding:0.75rem 0.9rem;font-size:0.78rem;color:#e2f2e5;line-height:1.6'>
          <b style='color:#3ddc6e'>🤖 AI Agro Advice:</b><br>
          {
            f"Temperature {wd['temperature']}°C is optimal for warm-season cereals & pulses (Maize, Groundnut, Pulses)." if 24 <= wd['temperature'] <= 34
            else f"Cool temperature ({wd['temperature']}°C) favors mustard, wheat & cool-season tubers." if wd['temperature'] < 24
            else f"High heat ({wd['temperature']}°C) detected — ensure mulch layers & activate drip irrigation."
          }
          {' High atmospheric humidity — monitor for leaf blight & fungal spores.' if isinstance(wd['humidity'], (int, float)) and wd['humidity'] > 70 else ''}
          {' Dry air — retain soil moisture with organic cover crops.' if isinstance(wd['humidity'], (int, float)) and wd['humidity'] < 35 else ''}
        </div>""", unsafe_allow_html=True)
