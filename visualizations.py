#This script generates the geographical scatter map, the top regions bar chart, and the interactive HTML map.

import pandas as pd
import matplotlib.pyplot as plt
import ast
import json
from collections import Counter

# --- 1. Data Loading & Cleaning ---
df = pd.read_csv('missile_attacks_daily.csv')

# Function to parse the 'affected region' column (strings of lists)
def parse_regions(x):
    if pd.isna(x):
        return []
    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return val
        return [str(val)]
    except:
        return [str(x)]

df['affected_regions_list'] = df['affected region'].apply(parse_regions)

# Flatten list to count frequency
all_regions = []
for regions in df['affected_regions_list']:
    all_regions.extend(regions)

region_counts = Counter(all_regions)

# Approximate coordinates for Ukrainian Oblasts (Longitude, Latitude)
# Note: Matplotlib uses (x, y) = (Lon, Lat)
region_coords = {
    'Kharkiv oblast': (36.2, 50.0), 'Sumy oblast': (34.8, 50.9),
    'Dnipropetrovsk oblast': (35.0, 48.4), 'Odesa oblast': (30.7, 46.5),
    'Donetsk oblast': (37.8, 48.0), 'Kyiv oblast': (30.5, 50.4),
    'Zaporizhzhia oblast': (35.1, 47.8), 'Poltava oblast': (34.5, 49.6),
    'Chernihiv oblast': (31.3, 51.5), 'Cherkasy oblast': (32.0, 49.4),
    'Mykolaiv oblast': (32.0, 47.0), 'Vinnytsia oblast': (28.5, 49.2),
    'Kherson oblast': (32.6, 46.6), 'Khmelnytskyi oblast': (27.0, 49.4),
    'Zhytomyr oblast': (28.7, 50.2), 'Lviv oblast': (24.0, 49.8),
    'Rivne oblast': (26.2, 50.6), 'Kirovohrad oblast': (32.3, 48.5),
    'Ivano-Frankivsk oblast': (24.7, 48.9), 'Ternopil oblast': (25.6, 49.5),
    'Volyn oblast': (25.3, 50.7), 'Zakarpattia oblast': (23.2, 48.6),
    'Chernivtsi oblast': (25.9, 48.3), 'Luhansk oblast': (39.3, 48.6),
    'Crimea': (34.0, 45.0)
}

# --- 2. Static Scatter Map & Bar Chart ---
x_coords = []
y_coords = []
sizes = []
labels = []

for region, count in region_counts.items():
    if region in region_coords:
        lon, lat = region_coords[region]
        x_coords.append(lon)
        y_coords.append(lat)
        sizes.append(count * 2) # Scale marker size
        labels.append(region)

# Plot 1: Scatter Map
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, s=sizes, c='red', alpha=0.6, edgecolors='k')
plt.title('Geographical Distribution of Attacks (Static Scatter)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.5)

for x, y, label, s in zip(x_coords, y_coords, labels, sizes):
    if s > 50: 
        plt.text(x, y, label, fontsize=9, ha='right')

plt.xlim(22, 41)
plt.ylim(44, 53)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('geo_scatter_map.png')
print("Generated geo_scatter_map.png")
plt.close()

# Plot 2: Top Regions Bar Chart
top_regions = pd.DataFrame(region_counts.most_common(15), columns=['Region', 'Count'])
plt.figure(figsize=(12, 6))
plt.bar(top_regions['Region'], top_regions['Count'], color='darkred')
plt.xticks(rotation=45, ha='right')
plt.title('Top 15 Affected Regions')
plt.tight_layout()
plt.savefig('top_regions_bar.png')
print("Generated top_regions_bar.png")
plt.close()

# --- 3. Interactive Leaflet Map (HTML) ---
markers_data = []
for region, count in region_counts.items():
    if region in region_coords:
        lon, lat = region_coords[region]
        # Leaflet needs (Lat, Lon)
        markers_data.append({
            'name': region,
            'lat': lat,
            'lon': lon,
            'count': count
        })

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Missile Attacks Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>#map {{ height: 600px; width: 100%; }}</style>
</head>
<body>
    <h2 style="text-align:center;">Geographical Frequency of Missile Attacks</h2>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([48.3794, 31.1656], 6);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);
        var data = {json.dumps(markers_data)};
        data.forEach(function(item) {{
            L.circleMarker([item.lat, item.lon], {{
                radius: Math.sqrt(item.count) * 2,
                fillColor: "#ff0000",
                color: "#000",
                weight: 1,
                opacity: 1,
                fillOpacity: 0.6
            }}).addTo(map)
            .bindPopup("<b>" + item.name + "</b><br>Attacks: " + item.count);
        }});
    </script>
</body>
</html>
"""

with open('missile_attacks_map.html', 'w') as f:
    f.write(html_content)
print("Generated missile_attacks_map.html")
