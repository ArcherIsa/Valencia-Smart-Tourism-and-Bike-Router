import streamlit as st
import pandas as pd
import json
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import osmnx as ox
import networkx as nx
import numpy as np
import itertools
import ssl
import urllib3
import requests
import re
from streamlit_searchbox import st_searchbox
import base64

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

ox.settings.requests_kwargs = {'verify': False}


# Initialise state variables
if 'origin' not in st.session_state:
    st.session_state.origin = None
if 'destination' not in st.session_state:
    st.session_state.destination = None
if 'highlighted_route' not in st.session_state:
    st.session_state.highlighted_route = None 


# Page Configuration & CSS
st.set_page_config(
    page_title="Valencia Bike Tour", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

def apply_custom_layout(image_path):
    # Read the image and convert it to base64
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
        
    st.markdown(
        f"""
        <div class="viewport-footer"></div>
        
        <style>
        /* Header across entire screen */
        .viewport-footer {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 150px; 
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center 55%;
            z-index: 1000000; 
            pointer-events: none;
        }}
        
        /* Top padding main container */
        .block-container {{
            padding-top: 120px;
            padding-bottom: 2rem; 
        }}

        /* Loading symbol */
        header[data-testid="stHeader"] {{
            background: transparent; 
            box-shadow: none;
        }}
        header[data-testid="stHeader"]::before {{
            display: none;
        }}
        [data-testid="stToolbar"] {{
            display: none;
        }}
        
        /* Pull the main title up */
        h1 {{
            margin-top: 0rem;
        }}
        
        /* Push sidebar down to match main page */
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            padding-top: 200px;
            padding-bottom: 2rem; 
        }}
        [data-testid="stSidebarHeader"] {{
            display: none !important; /* Hides the native sidebar top gap */
        }}
        
        /* Hide Streamlit footer text */
        footer {{
            visibility: hidden;
        }}

        /* Hide header links */
        a > svg {{
            display: none;
        }}
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {{
            pointer-events: none;
            cursor: default;
        }}
        
        /* Disable sidebar collapse button */
        [data-testid="stSidebarCollapseButton"] {{
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with image file
apply_custom_layout("city of arts and sciences.jpeg")

st.markdown("""
<style>
span[data-baseweb="tag"] {
    background-color: darkgreen;
}
span[data-baseweb="tag"] span {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: darkgreen;'>Valencia Smart Tourism & Bike Router</h1>", unsafe_allow_html=True)
st.markdown("Select your **Origin** and **Destination** by clicking on the map or typing in the search bar. " \
"Click on the routes on the map or the name on the right to highlight them.")


# Data Loading & Caching
@st.cache_data
def load_monument_data():
    with open('monumentos.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    features = data.get('features', [])
    records = []
    for f in features:
        attr = f['attributes']
        geom = f.get('geometry', {})
        if 'x' in geom and 'y' in geom:
            records.append({
                'name': attr.get('nombre', 'Unknown Monument'),
                'geometry': Point(geom['x'], geom['y'])
            })
            
    # Create GeoDataFrame and convert from EPSG:25830 (UTM 30N) to EPSG:4326 (Lat/Lon WGS84)
    gdf = gpd.GeoDataFrame(records, crs="EPSG:25830").to_crs("EPSG:4326")

    # Extract lat/lon into standard columns
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    
    # Load tags from the CSV
    tags_df = pd.read_csv("monument_tags_images.csv", encoding="utf-8")

    # Merge the spatial data with the tags
    gdf = gdf.merge(tags_df, on='name', how='left')

    # Fallback just in case a name didn't match
    gdf['tags'] = gdf['tags'].fillna("historical") 

    return pd.DataFrame(gdf.drop(columns=['geometry']))

@st.cache_resource
def load_valencia_graph():
    return ox.graph_from_place("Valencia, Spain", network_type="bike", simplify=True)

@st.cache_data
def get_bike_network_geojson(_city_graph):
    nodes, edges = ox.graph_to_gdfs(_city_graph)
    return edges.to_json()

df_monuments = load_monument_data()
with st.spinner("Loading the bike lanes of Valencia..."):
    city_graph = load_valencia_graph()


# Sidebar (interests and instructions)
st.sidebar.markdown("<h1 style='color: darkgreen; margin-top: 45px; padding-top: 0px;'>Your interests</h1>", unsafe_allow_html=True)
available_interests = ['Historical', 'Gothic', 'Modern', 'Science', 'Nature', 'Park', 'Religious', 'Museum', 'Art', 'Food']
user_interests = st.sidebar.multiselect("What types of monuments would you like to see on your route?", available_interests)

st.sidebar.markdown("---")
st.sidebar.markdown("<h1 style='color: darkgreen;'>Map Controls</h1>", unsafe_allow_html=True)

mostrar_red = st.sidebar.checkbox("Show complete bike network", value=False)

if st.sidebar.button("Clear selection"):
    st.session_state.origin = None
    st.session_state.destination = None
    st.session_state.highlighted_route = None
    st.rerun()

# Recommender system (content-based filtering)
if user_interests:
    user_query = " ".join(user_interests).lower()
   
    # Vectorise tags to calculate similarities
    cv = CountVectorizer()
    monument_matrix = cv.fit_transform(df_monuments['tags'])
    user_matrix = cv.transform([user_query])

    # Calculate cosine similarity
    df_monuments['similarity'] = cosine_similarity(user_matrix, monument_matrix).flatten()

    # Filter and sort recommendations
    recommended_monuments = df_monuments[df_monuments['similarity'] > 0].copy()
else:
    recommended_monuments = pd.DataFrame()


# Function to handle the type-ahead suggestions
def search_valencia_locations(searchterm: str):
    if not searchterm:
        return []
    
    # Check if the user typed direct coordinates (e.g., "39.4699, -0.3763")
    coord_match = re.match(r"^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$", searchterm)
    if coord_match:
        lat, lon = float(coord_match.group(1)), float(coord_match.group(3))
        return [(f"Coordinates: {lat}, {lon}", (lat, lon))]

    url = "https://photon.komoot.io/api/"
    params = {
        "q": searchterm,
        "lat": 39.4699,
        "lon": -0.3763,
        "limit": 5
    }
    
    # A custom User-Agent to prevent the API from blocking the request
    headers = {
        "User-Agent": "ValenciaSmartBikeTour/1.0"
    }
    
    try:
        # Pass the headers into the request
        res = requests.get(url, params=params, headers=headers, timeout=5, verify=False)
        
        # Check if API returned an error
        if res.status_code != 200:
            print(f"API Error: {res.status_code} - {res.text}")
            return []

        data = res.json()
        features = data.get("features", [])
        suggestions = []
        
        for f in features:
            props = f["properties"]
            name = props.get("name", "")
            street = props.get("street", "")
            city = props.get("city", "")
            
            # Build a clean label for the dropdown
            parts = [p for p in [name, street, city] if p]
            label = ", ".join(parts)
            if not label:
                continue
                
            coords = (f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])
            suggestions.append((label, coords))
            
        return suggestions
    except Exception as e:
        # Print the exact error to terminal
        print(f"Python Error in Search: {str(e)}")
        return []
    

def get_address_from_coords(lat, lon):
    """Translates a map click (lat, lon) into a readable street name."""
    try:
        url = f"https://photon.komoot.io/reverse?lat={lat}&lon={lon}"
        headers = {"User-Agent": "ValenciaSmartBikeTour/1.0"}
        
        # verify=False to bypass the proxy issues
        res = requests.get(url, headers=headers, verify=False, timeout=5)
        
        if res.status_code == 200:
            features = res.json().get("features", [])
            if features:
                props = features[0]["properties"]
                name = props.get("name", "")
                street = props.get("street", "")
                
                # Combine name and street if they exist
                parts = [p for p in [name, street] if p]
                if parts:
                    return ", ".join(parts)
                    
        # Fallback if the API can't find a street name
        return f"Coordinates: {lat:.4f}, {lon:.4f}"
    except Exception:
        return f"Coordinates: {lat:.4f}, {lon:.4f}"
        
    
def get_route_directions(graph, path, poi_nodes_map=None, poi_sequence=None):
    """Extracts street-level directions and injects tourist stops."""
    directions = []
    current_street = None
    current_length = 0
    
    # Create a set of the POIs on this specific route for fast lookup
    poi_nodes_set = set(poi_sequence) if poi_sequence else set()
    visited_pois = set()

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        # Check if user has reached a monument
        if u in poi_nodes_set and u not in visited_pois:
            # If we were tracking a street, close it out before the stop
            if current_street is not None:
                directions.append(f"Head on **{current_street}** for {int(current_length)}m")
                current_street = None
                current_length = 0
            
            # Inject the tourist stop
            monument_name = poi_nodes_map[u]['name']
            directions.append(f"📸 **Stop at: {monument_name}**")
            visited_pois.add(u)
        
        # Normal street tracking
        edge_data = graph.get_edge_data(u, v)[0]
        street_name = edge_data.get('name', 'Unnamed street')
        if isinstance(street_name, list):
            street_name = street_name[0]
            
        length = edge_data.get('length', 0)
        
        if street_name == current_street:
            current_length += length
        else:
            if current_street is not None:
                directions.append(f"Head on **{current_street}** for {int(current_length)}m")
            current_street = street_name
            current_length = length
            
    # Add the very last street segment
    last_node = path[-1]
    if current_street is not None:
        directions.append(f"Head on **{current_street}** for {int(current_length)}m")
        
    # Check if the final destination itself is a monument
    if last_node in poi_nodes_set and last_node not in visited_pois:
        monument_name = poi_nodes_map[last_node]['name']
        directions.append(f"📸 **Stop at: {monument_name}**")
        
    directions.append("🏁 **Arrive at Destination**")
        
    return directions

# Main layout
col1, col2 = st.columns([2, 1])

routes_to_plot = []
dist_directa = 0
coords_direct = []

with col1:
    # Initialise folium map
    m = folium.Map(location=[39.4699, -0.3763], zoom_start=13, tiles="CartoDB positron")

    m.get_root().header.add_child(folium.Element("""
<style>
    /* Remove focus outline from the map container */
    .leaflet-container:focus {
        outline: none;
    }
    /* Remove focus outline from clicked lines/routes */
    path.leaflet-interactive:focus {
        outline: none;
    }
    /* Catch-all for any other clicked elements inside the map */
    *:focus {
        outline: none;
    }
</style>
"""))
    
    if mostrar_red:
        network_json = get_bike_network_geojson(city_graph)
        folium.GeoJson(
            network_json,
            name="Complete Bike Network (OSMnx)",
            style_function=lambda feature: {'color': '#2ca25f', 'weight': 2, 'opacity': 0.3}
        ).add_to(m)

    if st.session_state.origin:
        folium.CircleMarker(
            location=st.session_state.origin,
            radius=9, color="darkgreen", fill=True, fill_color="darkgreen", fill_opacity=1, popup="Origin"
        ).add_to(m)
    if st.session_state.destination:
        folium.CircleMarker(
            location=st.session_state.destination,
            radius=9, color="darkgreen", fill=True, fill_color="darkgreen", fill_opacity=1, popup="Destination"
        ).add_to(m)

    lines_to_draw = []

    if st.session_state.origin and st.session_state.destination:
        orig_node = ox.distance.nearest_nodes(city_graph, X=st.session_state.origin[1], Y=st.session_state.origin[0])
        dest_node = ox.distance.nearest_nodes(city_graph, X=st.session_state.destination[1], Y=st.session_state.destination[0])
        
        try:
            # DIRECT ROUTE (without POIs)
            route_direct = nx.shortest_path(city_graph, orig_node, dest_node, weight='length')
            dist_directa = nx.shortest_path_length(city_graph, orig_node, dest_node, weight='length')
            coords_direct = [(city_graph.nodes[n]['y'], city_graph.nodes[n]['x']) for n in route_direct]
            
            is_hl_directa = st.session_state.highlighted_route == "Direct Route"
            lines_to_draw.append({'coords': coords_direct, 'name': "Direct Route", 'is_hl': is_hl_directa, 'nodes': []})
            
            # TOURIST ROUTE
            if not recommended_monuments.empty:
                mid_lat = (st.session_state.origin[0] + st.session_state.destination[0]) / 2
                mid_lon = (st.session_state.origin[1] + st.session_state.destination[1]) / 2
                
                recommended_monuments['dist_to_mid'] = np.sqrt(
                    (recommended_monuments['lat'] - mid_lat)**2 + 
                    (recommended_monuments['lon'] - mid_lon)**2
                )
                
                top_monuments = recommended_monuments.sort_values(
                    by=['similarity', 'dist_to_mid'], ascending=[False, True]
                ).head(15)
                
                poi_nodes_map = {}
                for _, row in top_monuments.iterrows():
                    n = ox.distance.nearest_nodes(city_graph, X=row['lon'], Y=row['lat'])
                    poi_nodes_map[n] = row
                
                poi_nodes = list(poi_nodes_map.keys())
                nodes_to_calc = [orig_node, dest_node] + poi_nodes
                
                dist_matrix = {}
                for src in nodes_to_calc:
                    try:
                        lengths = nx.single_source_dijkstra_path_length(city_graph, src, weight='length')
                        for tgt in nodes_to_calc:
                            dist_matrix[(src, tgt)] = lengths.get(tgt, float('inf'))
                    except nx.NetworkXNoPath:
                        pass
                        
                def get_seq_dist(seq):
                    d = dist_matrix.get((orig_node, seq[0]), float('inf'))
                    for i in range(len(seq)-1):
                        d += dist_matrix.get((seq[i], seq[i+1]), float('inf'))
                    d += dist_matrix.get((seq[-1], dest_node), float('inf'))
                    return d

                valid_seqs = []
                for k in range(1, 5):
                    for seq in itertools.permutations(poi_nodes, k):
                        d = get_seq_dist(seq)
                        if d != float('inf'):
                            dev = d - dist_directa
                            if dev <= 2000:
                                valid_seqs.append({'seq': seq, 'dist': d, 'dev': dev, 'num_pois': k})
                
                def get_best_seq(bucket):
                    if not bucket: return None
                    sorted_b = sorted(bucket, key=lambda x: (-x['num_pois'], x['dist']))
                    return sorted_b[0]

                b1 = [s for s in valid_seqs if s['dev'] < 500]
                b2 = [s for s in valid_seqs if 500 <= s['dev'] <= 1000]
                b3 = [s for s in valid_seqs if s['dev'] < 2000]

                best_r1 = get_best_seq(b1)
                best_r2 = get_best_seq(b2)
                best_r3 = get_best_seq(b3)

                plotted_seqs = set()
                
                if best_r1:
                    nombres = [poi_nodes_map[n]['name'] for n in best_r1['seq']]
                    routes_to_plot.append(('Touristic (<500m deviation)', best_r1, nombres))
                    plotted_seqs.add(best_r1['seq'])
                if best_r2 and best_r2['seq'] not in plotted_seqs:
                    nombres = [poi_nodes_map[n]['name'] for n in best_r2['seq']]
                    routes_to_plot.append(('Touristic (500-1000m deviation)', best_r2, nombres))
                    plotted_seqs.add(best_r2['seq'])
                if best_r3 and best_r3['seq'] not in plotted_seqs:
                    nombres = [poi_nodes_map[n]['name'] for n in best_r3['seq']]
                    routes_to_plot.append(('Touristic Max (<2000m deviation>)', best_r3, nombres))

                for r_name, r_data, r_names in routes_to_plot:
                    seq = r_data['seq']
                    full_path = nx.shortest_path(city_graph, orig_node, seq[0], weight='length')[:-1]
                    for i in range(len(seq)-1):
                        full_path += nx.shortest_path(city_graph, seq[i], seq[i+1], weight='length')[:-1]
                    full_path += nx.shortest_path(city_graph, seq[-1], dest_node, weight='length')
                    
                    coords = [(city_graph.nodes[n]['y'], city_graph.nodes[n]['x']) for n in full_path]
                    is_hl = st.session_state.highlighted_route == r_name
                    
                    lines_to_draw.append({
                        'coords': coords, 'name': r_name, 'is_hl': is_hl, 'nodes': seq, 'names': r_names
                    })

            # Drawing on the map
            # First draw the not selected routes (light green and in the background)
            for line in lines_to_draw:
                if not line['is_hl']:
                    folium.PolyLine(line['coords'], color="#74c476", weight=4, opacity=0.6, tooltip=line['name']).add_to(m)
                    for n in line['nodes']:
                        monument = poi_nodes_map[n]
                        name = monument['name']
                        img_url = monument.get('image_url', None)

                        # Build popup HTML with image if it exists
                        if pd.notna(img_url) and str(img_url).strip() not in ["None", "", "nan"]:
                            popup_html = f"""
                            <div style='width: 200px;'>
                                <b style='font-size: 14px;'>{name}</b><br><br>
                                <img src='{img_url}' width='100%' style='border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);'>
                            </div>
                            """
                        else:
                            popup_html = f"<b style='font-size: 14px;'>{name}</b>"

                        folium.Marker(
                            [monument['lat'], monument['lon']], 
                            popup=folium.Popup(popup_html, max_width=250),
                            icon=folium.Icon(color="lightgreen", icon="camera", prefix="fa", icon_color="white")
                        ).add_to(m)

            # Second draw the selected route (dark green, thicker and on top)
            for line in lines_to_draw:
                if line['is_hl']:
                    folium.PolyLine(line['coords'], color="darkgreen", weight=7, opacity=1.0, tooltip=line['name']).add_to(m)
                    for n in line['nodes']:
                        monument = poi_nodes_map[n]
                        name = monument['name']
                        img_url = monument.get('image_url', None)

                        # Build popup HTML with image if it exists
                        if pd.notna(img_url) and str(img_url).strip() not in ["None", "", "nan"]:
                            popup_html = f"""
                            <div style='width: 200px;'>
                                <b style='font-size: 14px;'>{name}</b><br><br>
                                <img src='{img_url}' width='100%' style='border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);'>
                            </div>
                            """
                        else:
                            popup_html = f"<b style='font-size: 14px;'>{name}</b>"

                        folium.Marker(
                            [monument['lat'], monument['lon']], 
                            popup=folium.Popup(popup_html, max_width=250),
                            icon=folium.Icon(color="darkgreen", icon="camera", prefix="fa", icon_color="white")
                        ).add_to(m)

        except nx.NetworkXNoPath:
            st.error("Could not calculate a cycling route between these points.")

    # Ask st_folium to return tooltip and popup data alongside the click coordinates
    map_data = st_folium(
        m, 
        width=800, 
        height=500, 
        return_on_hover=False, 
        returned_objects=["last_clicked", "last_object_clicked_popup", "last_object_clicked_tooltip"]
    )
    
    if map_data:
        # Check if the user clicked on a map object with a tooltip or popup
        clicked_popup = map_data.get("last_object_clicked_popup")
        clicked_tooltip = map_data.get("last_object_clicked_tooltip")
        
        # Create a list of the route names currently on the map
        valid_route_names = [line['name'] for line in lines_to_draw] if 'lines_to_draw' in locals() else []
        
        clicked_route_name = None
        if clicked_popup in valid_route_names:
            clicked_route_name = clicked_popup
        elif clicked_tooltip in valid_route_names:
            clicked_route_name = clicked_tooltip
            
        # If a route was clicked, change the highlighted route and rerun
        if clicked_route_name:
            if st.session_state.highlighted_route != clicked_route_name:
                st.session_state.highlighted_route = clicked_route_name
                st.rerun()
                
        # If no route was clicked, treat it as a standard map click to set Origin/Destination
        elif map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            
            # Get the street name from the click
            street_name = get_address_from_coords(clicked_lat, clicked_lon)
            
            if st.session_state.origin is None:
                st.session_state.origin = (clicked_lat, clicked_lon)
                st.session_state.origin_name = street_name # Save the name
                st.session_state.highlighted_route = None 
                st.rerun()
            elif st.session_state.destination is None:
                st.session_state.destination = (clicked_lat, clicked_lon)
                st.session_state.dest_name = street_name # Save the name
                st.session_state.highlighted_route = None 
                st.rerun()

with col2:
    st.markdown("<h3 style='color: darkgreen;'>Search Locations</h3>", unsafe_allow_html=True)
        
    # Origin container
    with st.container():
        if st.session_state.origin:
            # Safely fetch the name, defaulting to "Map Selection" if it's missing
            orig_display = st.session_state.get('origin_name', 'Selected on map')
            st.success(f"**📍 Origin:** {orig_display}")
        else:
            st.info("**📍 Origin:** Not set yet")
            
        # Origin Search Box
        selected_origin = st_searchbox(
            search_valencia_locations,
            key="origin_search",
            label="Search to change Origin:",
            placeholder="Search origin..."
        )

        if selected_origin and st.session_state.origin != selected_origin:
            st.session_state.origin = selected_origin
            st.session_state.origin_name = "Selected from search bar" 
            st.session_state.highlighted_route = None
            st.rerun()
    st.write("") 

    # Destination container
    with st.container():
        if st.session_state.destination:
            dest_display = st.session_state.get('dest_name', 'Selected on map')
            st.success(f"**🏁 Destination:** {dest_display}")
        else:
            st.info("**🏁 Destination:** Not set yet")

        # Destination Search Box
        selected_dest = st_searchbox(
            search_valencia_locations,
            key="dest_search",
            label="Search to change Destination:",
            placeholder="Search destination..."
        )

        if selected_dest and st.session_state.destination != selected_dest:
            st.session_state.destination = selected_dest
            st.session_state.dest_name = "Selected from search bar"
            st.session_state.highlighted_route = None
            st.rerun()
            
    st.markdown("---")

    st.markdown("<h3 style='color: darkgreen;'>Route information</h3>", unsafe_allow_html=True)
    
    if st.session_state.origin and st.session_state.destination:
        if dist_directa > 0:
            dist_km_directa = dist_directa / 1000
            tiempo_directa = int(round((dist_km_directa / 15) * 60))
            
            # Interactive button for Direct Route
            is_hl_directa = st.session_state.highlighted_route == "Direct Route"
            lbl_directa = "🟢 Direct Route" if is_hl_directa else "⚪ Direct Route"
            
            if st.button(lbl_directa, use_container_width=True):
                st.session_state.highlighted_route = "Direct Route"
                st.rerun()
                
            st.write(f"Distance: **{dist_km_directa:.2f} km** | Time: **~{tiempo_directa} min**")
            st.caption("No tourist stops.")

            if is_hl_directa:
                with st.expander("📍 View Step-by-Step Directions"):
                    # We pass None for the POIs since this is the direct route
                    steps = get_route_directions(city_graph, route_direct, poi_nodes_map=None, poi_sequence=None)
                    for idx, step in enumerate(steps):
                        st.markdown(f"{idx + 1}. {step}")
            st.markdown("---")
            
            if not routes_to_plot:
                st.info("No monuments of your interests are close enough for these deviations.")
            else:
                for r_name, r_data, r_names in routes_to_plot:
                    dist_km = r_data['dist'] / 1000
                    tiempo_min = int(round((dist_km / 15) * 60))
                    
                    # Interactive button for tourist routes
                    is_hl = st.session_state.highlighted_route == r_name
                    lbl = f"🟢 {r_name}" if is_hl else f"⚪ {r_name}"
                    
                    if st.button(lbl, key=f"btn_{r_name}", use_container_width=True):
                        st.session_state.highlighted_route = r_name
                        st.rerun()
                    
                    st.write(f"Distance: **{dist_km:.2f} km** | Time: **~{tiempo_min} min**")
                    nombres_formateados = " ➔ ".join(r_names)
                    st.caption(f"**You will pass through:** {nombres_formateados}")

                    if is_hl:
                        with st.expander("📍 View Step-by-Step Directions"):
                            seq = r_data['seq']
                            
                            # Recalculate full path for directions
                            full_path = nx.shortest_path(city_graph, orig_node, seq[0], weight='length')[:-1]
                            for i in range(len(seq)-1):
                                full_path += nx.shortest_path(city_graph, seq[i], seq[i+1], weight='length')[:-1]
                            full_path += nx.shortest_path(city_graph, seq[-1], dest_node, weight='length')
                            
                            # Pass the monument data into the directions function!
                            steps = get_route_directions(city_graph, full_path, poi_nodes_map=poi_nodes_map, poi_sequence=seq)
                            for idx, step in enumerate(steps):
                                st.markdown(f"{idx + 1}. {step}")
                    st.write("") 
    else:
        st.warning("Click on the map or use the search bars to set your origin and destination.")