import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt

# 1. Europa-Karte laden & projizieren
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)
europe = world[world['CONTINENT'] == 'Europe'].to_crs(epsg=3035)

# 2. 100x100 km Raster erstellen
minx, miny, maxx, maxy = europe.total_bounds
pixel_size = 50000 
x_coords = np.arange(minx, maxx, pixel_size)
y_coords = np.arange(miny, maxy, pixel_size)
cells = [shapely.geometry.box(x, y, x + pixel_size, y + pixel_size) for x in x_coords for y in y_coords]
grid = gpd.GeoDataFrame(cells, columns=['geometry'], crs=europe.crs)

# 3. Räumlicher Filter auf Landmasse
europe_grid = gpd.sjoin(grid, europe[['geometry']], how='inner', predicate='intersects').drop(columns=['index_right'])

# 4. Temperatur-Extrema simulieren (Gradient + saisonaler Offset)
y_centers = europe_grid.geometry.centroid.y
y_norm = (y_centers - y_centers.min()) / (y_centers.max() - y_centers.min())

# Kältester Monat (Winter): Nord-Süd Gefälle von -15°C bis +10°C
europe_grid['Temp_Min'] = (y_norm * -25) + 10 + np.random.normal(0, 1, len(europe_grid))

# Wärmster Monat (Sommer): Nord-Süd Gefälle von +10°C bis +35°C
europe_grid['Temp_Max'] = (y_norm * -25) + 35 + np.random.normal(0, 1, len(europe_grid))

# 5. Visualisierung: Zwei Karten nebeneinander
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Gemeinsame Einstellungen
for ax, col, title, cmap in zip([ax1, ax2], 
                                 ['Temp_Min', 'Temp_Max'], 
                                 ['Kältester Monat (Winter Minimum)', 'Wärmster Monat (Sommer Maximum)'],
                                 ['Blues_r', 'YlOrRd']):
    europe.plot(ax=ax, color='#e0e0e0', edgecolor='none')
    europe_grid.plot(column=col, ax=ax, cmap=cmap, legend=True, 
                     legend_kwds={'label': "Temperatur (°C)", 'orientation': "horizontal", 'pad': 0.05, 'shrink': 0.8},
                     edgecolor='none')
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlim(minx - 100000, maxx + 100000)
    ax.set_ylim(miny - 100000, maxy + 100000)
    ax.set_axis_off()

plt.suptitle('Thermisches Design-Spektrum für europäische CO2-Pipelines', fontsize=22, y=1.05)
plt.tight_layout()
plt.show()