from __future__ import annotations
from collections import defaultdict
from typing import Dict, Any, Iterable, List, Tuple

from iberian_co2_network.scenarios import get_scenarios

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from shapely import wkt
from collections import defaultdict
import math
import ast

INTEREST_RATE = 0.08 # interest rate used for all the present value calculations


"""

PREDICTION ON SPAIN AND PORTUGAL EMISSIONS BETWEEN 2025 AND 2050

Data is a mix between the E-PRTR database for Portugal (2017 data) and the GeoPortal database for Spain (2020 data).

* Energy sector emissions are assumed to continue following the trend of the CO2 emissions between 2008 and 2022, according to the IEA: https://www.iea.org/countries/spain/emissions and https://www.iea.org/countries/portugal/emissions
* All other emissions are assumed to grow with the average GDP growth rate per year, assumed to be 1.4% for both countries, based on: https://www.pwc.com/gx/en/research-insights/economy/the-world-in-2050.html

"""

spain_emissions_from_power_plants_2000 = 97 # MtCO2 in 2000
spain_emissions_from_power_plants_2022 = 49 # MtCO2 in 2022
avg_spain_emission_annual_variation_perc = ((spain_emissions_from_power_plants_2022 - spain_emissions_from_power_plants_2000) / (2022 - 2000))/spain_emissions_from_power_plants_2000 * 100  # Average percentual annual variation in MtCO2

portugal_emissions_from_power_plants_2000 = 22 # MtCO2 in 2000
portugal_emissions_from_power_plants_2022 = 8 # MtCO2 in 2022
avg_portugal_emission_annual_variation_perc = ((portugal_emissions_from_power_plants_2022 - portugal_emissions_from_power_plants_2000) / (2022 - 2000))/portugal_emissions_from_power_plants_2000 * 100  # Average percentual annual variation in MtCO2

gdp_growth_rate = 1.4  # Average GDP growth rate per year (1.4%)

emissions_df = pd.read_excel('Spanish and Portuguese emissions and sinks database.xlsx', sheet_name='2020 Spain + 2017 Portugal em.')  # Read emissions data from Excel file

years = list(range(2026, 2051))

emissions_projection_df = emissions_df[['Facility name', 'Country', 'Main Sector']].copy() # Create a new DataFrame with the base columns

# Function to compute projected emissions for each facility
def project_emissions(row):
    # Determine base year based on country
    base_year = 2020 if row['Country'] == 'Spain' else 2017
    base_emission = row['CO2 emissions']
    projections = {}

    # Determine the annual variation rate
    if row['Main Sector'] == 'Energy sector':
        if row['Country'] == 'Spain':
            annual_rate = avg_spain_emission_annual_variation_perc / 100
        else:
            annual_rate = avg_portugal_emission_annual_variation_perc / 100
    else:
        annual_rate = gdp_growth_rate / 100

    # Project emissions year by year
    for year in years:
        years_since_base = year - base_year
        projected_value = base_emission * ((1 + annual_rate) ** years_since_base)
        projections[year] = projected_value

    return pd.Series(projections)

# Apply the projection function to each row in the DataFrame
projections_df = emissions_df.apply(project_emissions, axis=1)

# Concatenate the original identifying columns with the projection columns
emissions_projection_df = pd.concat([emissions_projection_df, projections_df], axis=1)

# print(emissions_projection_df.tail(50))

# Read cluster mapping
cluster_map_df = pd.read_excel('iberian_co2_network_data.xlsx', sheet_name='Emissions clustering map')

# Merge with emissions projections
emissions_projection_df = (
    emissions_projection_df
    .merge(cluster_map_df[['Facility name', 'Source cluster']],
           on='Facility name', how='left')
)

# Identify year columns (those that are digits)
year_cols = [c for c in emissions_projection_df.columns if isinstance(c, int) or (isinstance(c, str) and c.isdigit())]

# Group by Source cluster and sum emissions
node_emissions_df = (
    emissions_projection_df
        .groupby('Source cluster', as_index=False)[year_cols]
        .sum()
        .sort_values('Source cluster')
        .reset_index(drop=True)
)

# Convert Source cluster to integer (if it's float-like)
node_emissions_df['Source cluster'] = node_emissions_df['Source cluster'].astype(int)

# Convert emissions to Mt and round to 3 decimals
node_emissions_df[year_cols] = (node_emissions_df[year_cols] / 1e9).round(3)

# --- 5-year aggregation of emissions (sum of annual MtCO2) ---

# Mapping: final year of each 5-year block -> list of annual columns to sum
five_year_bins = {
    2030: list(range(2026, 2031)),   # 2026-2030 inclusive
    2035: list(range(2031, 2036)),   # 2031-2035 inclusive
    2040: list(range(2036, 2041)),   # 2036-2040 inclusive
    2045: list(range(2041, 2046)),   # 2041-2045 inclusive
    2050: list(range(2046, 2051))    # 2046-2050 inclusive
}

# Create reduced dataframe with identifier column
node_emissions_5yr_df = node_emissions_df[['Source cluster']].copy()

# Aggregate each 5-year window
for target_year, cols in five_year_bins.items():
    # Sum annual MtCO2 values and keep three-decimal precision
    node_emissions_5yr_df[target_year] = (
        node_emissions_df[cols].sum(axis=1).round(2)
    )

ordered_cols = ['Source cluster', 2030, 2035, 2040, 2045, 2050]
node_emissions_5yr_df = node_emissions_5yr_df[ordered_cols]

# node_emissions_5yr_df now contains cumulative emissions per node for each 5-year period
# print(node_emissions_5yr_df.head(20))


# ###############################
# ###############################

# # Code for visualization of the emissions projections

# ###############################
# ###############################

# years = list(range(2026, 2051))

# # Dictionary to store the results
# results = {
#     "Year": [],
#     "Spain - Energy [Mt CO2]": [],
#     "Portugal - Energy [Mt CO2]": [],
#     "Spain - Non-Energy [Mt CO2]": [],
#     "Portugal - Non-Energy [Mt CO2]": [],
#     "Total [Mt CO2]": []
# }

# # Loop over each year to calculate the sums
# for year in years:
#     # Filter by country and sector
#     spain_energy = emissions_projection_df[
#         (emissions_projection_df["Country"] == "Spain") &
#         (emissions_projection_df["Main Sector"] == "Energy sector")
#     ][year].sum()

#     portugal_energy = emissions_projection_df[
#         (emissions_projection_df["Country"] == "Portugal") &
#         (emissions_projection_df["Main Sector"] == "Energy sector")
#     ][year].sum()

#     spain_non_energy = emissions_projection_df[
#         (emissions_projection_df["Country"] == "Spain") &
#         (emissions_projection_df["Main Sector"] != "Energy sector")
#     ][year].sum()

#     portugal_non_energy = emissions_projection_df[
#         (emissions_projection_df["Country"] == "Portugal") &
#         (emissions_projection_df["Main Sector"] != "Energy sector")
#     ][year].sum()

#     total = spain_energy + portugal_energy + spain_non_energy + portugal_non_energy

#     # Store results
#     results["Year"].append(int(year))
#     results["Spain - Energy [Mt CO2]"].append(spain_energy)
#     results["Portugal - Energy [Mt CO2]"].append(portugal_energy)
#     results["Spain - Non-Energy [Mt CO2]"].append(spain_non_energy)
#     results["Portugal - Non-Energy [Mt CO2]"].append(portugal_non_energy)
#     results["Total [Mt CO2]"].append(total)

# # Create the summary DataFrame
# emission_proj_summary_df = pd.DataFrame(results)

# emission_proj_summary_df['Spain - Energy [Mt CO2]'] = (emission_proj_summary_df['Spain - Energy [Mt CO2]'] /10**9).round(2)
# emission_proj_summary_df['Portugal - Energy [Mt CO2]'] = (emission_proj_summary_df['Portugal - Energy [Mt CO2]'] /10**9).round(2)
# emission_proj_summary_df['Spain - Non-Energy [Mt CO2]'] = (emission_proj_summary_df['Spain - Non-Energy [Mt CO2]'] /10**9).round(2) 
# emission_proj_summary_df['Portugal - Non-Energy [Mt CO2]'] = (emission_proj_summary_df['Portugal - Non-Energy [Mt CO2]'] /10**9).round(2)
# emission_proj_summary_df['Total [Mt CO2]'] = (emission_proj_summary_df['Total [Mt CO2]'] /10**9).round(2)  # Convert to MtCO2 and round

# # print(emission_proj_summary_df)

# k = 10  # Scaling factor for offset
# min_offset = 0.2
# max_offset = 2.0

# label_map = {
#     "Spain - Energy [Mt CO2]": "Spain - Power sector",
#     "Spain - Non-Energy [Mt CO2]": "Spain - Other sectors",
#     "Portugal - Energy [Mt CO2]": "Portugal - Power sector",
#     "Portugal - Non-Energy [Mt CO2]": "Portugal - Other sectors",
#     "Total [Mt CO2]": "Total"
# }

# colors = {
#     "Spain - Power sector": "#1f77b4",         # Strong blue
#     "Spain - Other sectors": "#6495ED",        # Light blue
#     "Portugal - Power sector": "#d62728",      # Dark red
#     "Portugal - Other sectors": "#f08080",     # Light red
#     "Total": "#9467bd"                         # Purple/lilac
# }

# fig, ax = plt.subplots(figsize=(11, 7))  # CORRECTO: fig, ax

# # Plot each line + points
# ax.plot(emission_proj_summary_df["Year"], emission_proj_summary_df["Spain - Energy [Mt CO2]"],
#         color=colors["Spain - Power sector"], linewidth=2, label="Spain - Power sector")

# ax.plot(emission_proj_summary_df["Year"], emission_proj_summary_df["Spain - Non-Energy [Mt CO2]"],
#         color=colors["Spain - Other sectors"], linewidth=2, label="Spain - Other sectors")

# ax.plot(emission_proj_summary_df["Year"], emission_proj_summary_df["Portugal - Energy [Mt CO2]"],
#         color=colors["Portugal - Power sector"], linewidth=2, label="Portugal - Power sector")

# ax.plot(emission_proj_summary_df["Year"], emission_proj_summary_df["Portugal - Non-Energy [Mt CO2]"],
#         color=colors["Portugal - Other sectors"], linewidth=2, label="Portugal - Other sectors")

# ax.plot(emission_proj_summary_df["Year"], emission_proj_summary_df["Total [Mt CO2]"],
#         color=colors["Total"], linewidth=2, label="Total")

# # Annotate first and last year values (2026 and 2050) for each series
# for col in emission_proj_summary_df.columns:
#     if col == "Year":
#         continue
    
#     x_start = emission_proj_summary_df["Year"].iloc[0]
#     y_start = emission_proj_summary_df[col].iloc[0]
#     x_end = emission_proj_summary_df["Year"].iloc[-1]
#     y_end = emission_proj_summary_df[col].iloc[-1]

#     # Offsets
#     offset_start = min(max(k / y_start, min_offset), max_offset)
#     offset_end = min(max(k / y_end, min_offset), max_offset)
#     offset_start *= 1 if y_end >= y_start else -1
#     offset_end *= 1 if y_end >= y_start else -1

#     # Annotations (with bold font)
#     ax.text(
#         x_start,
#         y_start + offset_start + 1,
#         f"{y_start:.1f}",
#         fontweight="bold",
#         fontsize=10,
#         color=colors[label_map[col]],
#         ha="left",
#         va="bottom"
#     )
#     ax.text(
#         x_end,
#         y_end + offset_end + 1,
#         f"{y_end:.1f}",
#         fontweight="bold",
#         fontsize=10,
#         color=colors[label_map[col]],
#         ha="right",
#         va="bottom"
#     )

# # Axis settings
# ax.set_xlim(2025.5, 2050.5)
# ax.set_ylim(0, 145)
# ax.set_xlabel("Year", fontsize=12)
# ax.set_ylabel("CO₂ emissions [Mt CO₂]", fontsize=12)
# ax.set_title("Projected CO₂ annual emissions by sector and country (2026–2050)", fontsize=14)
# ax.grid(False)

# # Legend
# ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=11, frameon=False)

# # Layout and save
# # plt.tight_layout()
# # plt.show()
# # fig.savefig("emissions_projection_plot.png", dpi=300, bbox_inches='tight')



"""

PREDICTION ON EUROPEAN UNION CARBON ALLOWANCES BETWEEN 2026 AND 2050

The prediction is made by considering a logistic growth model, with a maximum value of 400 €/tCO2, whcich allows to capture initial growth, and then a plateau and a final acceleration towards the maximum value.

"""

def doble_logistic_growth(c, t, Pmax_1, Pmax_2, k_1, k_2, t0_1, t0_2):

    """
    Logistic growth function.
    
    Parameters:
        c(float): Constant to adjust the curve.
        t (float): Time variable (year).
        Pmax (float): Maximum value of the function.
        k (float): Growth rate.
        t0 (float): Inflection point (midpoint of the curve).
        
    Returns:
        float: Value of the logistic function at time t.
    """
    return c + Pmax_1 / (1 + np.exp(-k_1 * (t - t0_1))) + Pmax_2 / (1 + np.exp(-k_2 * (t - t0_2)))

# Parameters for the logistic growth model
c = 25.2  # Constant to adjust the curve (€/tCO2)
Pmax_1 = 107  # Maximum value of the function (€/tCO2)
k_1 = 0.127  # Growth rate (adjustable)
t0_1 = 2027.3  # Inflection point (midpoint of the curve)
Pmax_2 = 453  # Maximum value of the function (€/tCO2)
k_2 = 0.394  # Growth rate (adjustable)
t0_2 = 2047.4 # Inflection point (midpoint of the curve)

# Years for prediction
years = np.arange(2025, 2052)
# Calculate the predicted values using the logistic growth function
predicted_values = doble_logistic_growth(c, years, Pmax_1, Pmax_2, k_1, k_2, t0_1, t0_2)

# Create a DataFrame for the predicted values
predicted_values_df = pd.DataFrame({
    "Year": years,
    "Predicted Carbon Allowance Price [€/tCO2]": predicted_values
})

# Round the predicted values to 2 decimal places
predicted_values_df["Predicted Carbon Allowance Price [€/tCO2]"] = predicted_values_df["Predicted Carbon Allowance Price [€/tCO2]"].round(2)

# --- Select 5-year milestone allowance prices ---

# Years we want to keep
milestone_years = [2030, 2035, 2040, 2045, 2050]

# Filter the existing DataFrame
carbon_price_5yr_df = (
    predicted_values_df[predicted_values_df['Year'].isin(milestone_years)]
    .reset_index(drop=True)          # Clean sequential index
    .sort_values('Year')             # Ensure chronological order
)

# print(carbon_price_5yr_df)

df = carbon_price_5yr_df.copy()

# 1) Identify base year (earliest year in the column)
base_year = df['Year'].min()        # e.g. 2030

# 2) Compute discount factor per row
df['discount_factor'] = (1 + INTEREST_RATE) ** (df['Year'] - base_year)

# 3) Present Value of the nominal carbon price
df['Predicted Carbon Allowance PV [€/tCO2]'] = df['Predicted Carbon Allowance Price [€/tCO2]'] / df['discount_factor']

# 4) (Optional) tidy up the result
carbon_price_pv_df = df[['Year', 'Predicted Carbon Allowance PV [€/tCO2]']].round(2)
carbon_price_pv_df = carbon_price_pv_df.rename(
    columns=lambda c: c.replace('€/tCO2', 'M€/MtCO2') if '€/tCO2' in c else c
)

# print(carbon_price_pv_df)

###############################
###############################

# Draw the predicted values

###############################
###############################
    
# plt.figure(figsize=(10, 6))
# plt.plot(predicted_values_df["Year"], predicted_values_df["Predicted Carbon Allowance Price [€/tCO2]"], marker='o', markersize=5, color="#1f77b4", linewidth=1.5, label='Predicted price')
# plt.title('Predicted European Union Carbon Allowance Price (2026–2050)', fontsize=14)
# plt.xlabel('Year', fontsize=12)
# plt.ylabel('Price [€/tCO2]', fontsize=12)
# plt.xticks(predicted_values_df["Year"], rotation=45)
# plt.yticks(np.arange(0, 600, 50))
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.axhline(y=500, color= "#a71a1a", linewidth=1, linestyle='--', label='Maximum price')

# plt.xlim(2025.5, 2050.5)
# plt.ylim(50, 550)

# plt.legend(loc='lower right', fontsize=10)
# plt.tight_layout()

# plt.savefig("predicted_carbon_allowance_price.png", dpi=300, bbox_inches='tight')
# plt.show()




"""

PRELIMINARY INVESTMENT COST FOR ONSHORE PIPELINES

Approach taken from: https://catalogplus.tuwien.at/primo-explore/fulldisplay?docid=TN_cdi_scopus_primary_2_s2_0_85197242287&context=PC&vid=UTW&lang=en_US&search_scope=UTW&adaptor=primo_central_multiple_fe&tab=default_tab&query=any,contains,co2%20transport%20pressure%20drop&offset=0

Preliminary investment cost for onshore pipelines between 1 and 64 inches diameter, in €/m.

An approximate investment cost assumption of 40 € per meter of pipeline length per inch of diameter (€/inch/m) is made [21].
The cost is generalized for carbon steel pipelines and is used for the calculations in this research.
This cost value (2010) is then scaled to the current market (2024) for different pipeline diameters. HIPC data are taken from: https://ec.europa.eu/eurostat/web/main/data/database.

"""

###############################
###############################

# Read HICP data,  compute cumulative infation since 2010-2025, and predict inflation for 2026-2050

###############################
###############################

hicp_df = pd.read_excel('model_parameters.xlsx', sheet_name='HICP') # Read HICP data from Excel file
hicp_df.columns = ["DATE", "TIME_PERIOD", "HICP"]
hicp_df["DATE"] = pd.to_datetime(hicp_df["DATE"])

hicp_df["Year"] = hicp_df["DATE"].dt.year # Extract year from the date
hicp_df = hicp_df[hicp_df["Year"] >= 2010]

cumulative_infl_df = hicp_df.groupby("Year")["HICP"].mean().reset_index() # Compute average inflation per year
cumulative_infl_df.columns = ["Year", "Average Inflation"]

cumulative_infl_df["Factor"] = 1 + cumulative_infl_df["Average Inflation"] / 100 # Convert percentage to inflation factor

cumulative_infl_df["Cumulative Inflation"] = cumulative_infl_df["Factor"].cumprod() - 1 # Compute cumulative inflation since 2010
cumulative_infl_df["Cumulative Inflation"] = cumulative_infl_df["Cumulative Inflation"] * 100  # convert to %


# --- PREDICTIONS FOR 2026–2050 ---

mean_inflation = cumulative_infl_df["Average Inflation"].mean()
future_years_df = pd.DataFrame({"Year": range(2026, 2051)})

future_years_df["Average Inflation Prediction"] = mean_inflation
future_years_df["Factor Prediction"] = 1 + mean_inflation / 100

last_real_factor = cumulative_infl_df["Factor"].prod() # Get the cumulative factor from 2025

future_years_df["Cumulative Factor"] = last_real_factor * future_years_df["Factor Prediction"].cumprod() # Calculate cumulative prediction based on previous cumulative factor
future_years_df["Cumulative Inflation Prediction"] = (future_years_df["Cumulative Factor"] - 1) * 100
future_years_df.drop(columns=["Cumulative Factor"], inplace=True) # Drop the helper column

future_years_df["Average Inflation"] = pd.NA # add empty columns for existing metrics (will remain NaN for predictions)
future_years_df["Factor"] = pd.NA 
future_years_df["Cumulative Inflation"] = pd.NA

# Reorder columns to match original + prediction structure
cols = [
    "Year",
    "Average Inflation",
    "Factor",
    "Cumulative Inflation",
    "Average Inflation Prediction",
    "Factor Prediction",
    "Cumulative Inflation Prediction"
]

# Combine historical and prediction data
cumulative_infl_df = pd.concat([cumulative_infl_df, future_years_df], ignore_index=True)[cols]

# print(cumulative_infl_df)


###############################
###############################

# Calculate present value cost for pipelines

###############################
###############################

PIPE_LIFESPAN = 50
COST_PER_INCH_EUR = 40 # in €/m
cost_per_inch = COST_PER_INCH_EUR * 1e3 / 1e6 # in M€/km

# complete set of diameters - 39 possible options
# diameter_inch_str = [
#     '1', '1 1/4', '1 1/2', '2', '2 1/2', '3', '3 1/2', '4', '5', '6', '8','10',
#     '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32', '34', '36',
#     '38', '40', '42', '44', '46', '48', '50', '52', '54', '56', '58', '60', '62', '64'
# ]

# selected set of diameters - 10 possible options
diameter_inch_str = ['6', '10', '14', '18', '22', '26', '30', '34', '38', '42']

# Convert to numeric (float) inches
def parse_inch(inch_str):
    parts = inch_str.split()
    if len(parts) == 2:
        whole = float(parts[0])
        fraction = eval(parts[1])
        return whole + fraction
    else:
        return float(parts[0])

diameter_inch = [parse_inch(d) for d in diameter_inch_str]

onshore_pipe_inv_cost_df = pd.DataFrame({
    "Diameter [inch]": diameter_inch_str,
    "Diameter_numeric": diameter_inch,
})

onshore_pipe_inv_cost_df["Diameter [mm]"] = onshore_pipe_inv_cost_df["Diameter_numeric"] * 25.4
pred_years = cumulative_infl_df[cumulative_infl_df["Year"] >= 2026]

for _, row in pred_years.iterrows(): # Iterate over each prediction year
    year = int(row["Year"])
    cumulative_infl = row["Cumulative Inflation Prediction"]
    
    onshore_pipe_inv_cost_df[str(year)] = round(cost_per_inch * onshore_pipe_inv_cost_df["Diameter_numeric"] * (1 + cumulative_infl / 100), 6) # Calculate cost in M€/km for each year

onshore_pipe_inv_cost_df.drop(columns="Diameter_numeric", inplace=True)

# Mapping from each milestone to its 5-year window (inclusive)
five_year_bins = {yr: list(range(yr - 4, yr + 1)) for yr in milestone_years}
# → {2030: [2026,2027,2028,2029,2030], 2035: [2031,…,2035], …}

# Create reduced DataFrame with identifier columns
onshore_pipe_inv_cost_5yr_df = onshore_pipe_inv_cost_df[['Diameter [inch]', 'Diameter [mm]']].copy()

# Compute the 5-year average for every milestone
for target_year, cols in five_year_bins.items():
    # Ensure column names are strings (matches DataFrame dtypes)
    cols = [str(c) for c in cols]
    # Row-wise mean, rounded to 2 decimals
    onshore_pipe_inv_cost_5yr_df[str(target_year)] = (
        onshore_pipe_inv_cost_df[cols].mean(axis=1).round(6)
    )

# Optional: enforce final column order
ordered_cols = ['Diameter [inch]', 'Diameter [mm]'] + [str(y) for y in milestone_years]
onshore_pipe_inv_cost_5yr_df = onshore_pipe_inv_cost_5yr_df[ordered_cols]

# print(onshore_pipe_inv_cost_5yr_df)

# Calculation of the Present Value

df = onshore_pipe_inv_cost_5yr_df.copy()

# 1) Detect numeric year columns (2030, 2035, …)
year_cols = [c for c in df.columns if str(c).isdigit()]          # keep as strings
year_ints = list(map(int, year_cols))                            # same years as int
base_year = min(year_ints)                                       # e.g. 2030

# 2) Pre-compute the discount factors for every year column
discount_factors = {str(y): (1 + INTEREST_RATE) ** (y - base_year)
                    for y in year_ints}

# 3) Build the PV DataFrame
pv_inv_onsh_pipe_df = df.copy()
for col, factor in discount_factors.items():
    # Divide the nominal cost by its discount factor
    pv_inv_onsh_pipe_df[col] = df[col] / factor

# 4) Round for neat printing
pv_inv_onsh_pipe_df[year_cols] = pv_inv_onsh_pipe_df[year_cols].round(6)

# print(pv_inv_onsh_pipe_df)





"""

PRELIMINARY OPERATION AND MAINTENANCE COST FOR ONSHORE PIPELINES

Approach taken from: https://catalogplus.tuwien.at/primo-explore/fulldisplay?docid=TN_cdi_scopus_primary_2_s2_0_85197242287&context=PC&vid=UTW&lang=en_US&search_scope=UTW&adaptor=primo_central_multiple_fe&tab=default_tab&query=any,contains,co2%20transport%20pressure%20drop&offset=0

Preliminary operation and maintenance cost for onshore pipelines between 1 and 64 inches diameter, in €/m.

In this research, an annual fixed O&M factor of 2.6% has been assumed based on [25], where insurance and property taxes account
for 1% each, maintenance and repairs for 0.5%, and licensing and permitting for 0.1%.

"""

onshore_pipe_om_cost_5yr_df = onshore_pipe_inv_cost_5yr_df.copy()

year_columns = [col for col in onshore_pipe_om_cost_5yr_df.columns if col.isdigit()] # Identify the columns that are years (i.e., the cost columns)
onshore_pipe_om_cost_5yr_df[year_columns] = round(onshore_pipe_om_cost_5yr_df[year_columns] * 0.026, 6) # Multiply all those year columns by 0.026 to get O&M costs

# print(onshore_pipe_om_cost_5yr_df)

pv_om_onsh_pipe_df = pv_inv_onsh_pipe_df.copy()
pv_om_onsh_pipe_df[year_columns] = round(pv_om_onsh_pipe_df[year_columns] * 0.026, 6) # Multiply all those year columns by 0.026 to get O&M costs

# print(pv_om_onsh_pipe_df)



"""

PRELIMINARY INVESTMENT COST FOR OFFSHORE PIPELINES

Approach taken from: https://publications.jrc.ec.europa.eu/repository/handle/JRC62502

Preliminary investment cost for offshore pipelines is considered to be twice that of onshore piping.


"""

offshore_pipe_inv_cost_5yr_df = onshore_pipe_inv_cost_5yr_df.copy()

year_columns = [col for col in offshore_pipe_inv_cost_5yr_df.columns if col.isdigit()] # Identify the columns that are years (i.e., the cost columns)
offshore_pipe_inv_cost_5yr_df[year_columns] = round(offshore_pipe_inv_cost_5yr_df[year_columns] * 2, 6) # Multiply all those year columns by 2 to get offshore piping costs

# print(offshore_pipe_inv_cost_5yr_df)

pv_inv_offsh_pipe_df = pv_inv_onsh_pipe_df.copy()
pv_inv_offsh_pipe_df[year_columns] = round(pv_inv_offsh_pipe_df[year_columns] * 2, 6) # Multiply all those year columns by 2 to get offshore piping costs

# print(pv_inv_offsh_pipe_df)




"""

PRELIMINARY operation and maintenance COST FOR OFFSHORE PIPELINES

Preliminary operation and maintenance cost for offshore pipelines is calculated by applying the same 2.6% to offshore pipeline
investment costs.


"""

offshore_pipe_om_cost_5yr_df = offshore_pipe_inv_cost_5yr_df.copy()

year_columns = [col for col in offshore_pipe_om_cost_5yr_df.columns if col.isdigit()] # Identify the columns that are years (i.e., the cost columns)
offshore_pipe_om_cost_5yr_df[year_columns] = round(offshore_pipe_om_cost_5yr_df[year_columns] * 0.026, 6) # Multiply all those year columns by 0.026 to get O&M costs

# print(offshore_pipe_om_cost_5yr_df)

pv_om_offsh_pipe_df = pv_inv_offsh_pipe_df.copy()
pv_om_offsh_pipe_df[year_columns] = round(pv_om_offsh_pipe_df[year_columns] * 0.026, 6) # Multiply all those year columns by 0.026 to get O&M costs

# print(pv_om_offsh_pipe_df)



"""

IMPORTING REVENUES AND EXPORTING COSTS FOR FRANCE, MOROCCO AND ALGERIA

Importing revenues and exporting costs are assumed to be only dependent on the quantity of CO2 transported.
However, revenues and costs are not the same for all countries, depending on the distance to the Iberian Peninsula and the transport method.
Moreover, the revenues and costs are assumed to evolve with the HICP predictions.

"""

# Initial 2026 prices (€/t)
base_prices = {
    'France':  {'revenue': 30, 'cost': 32}, # €/t
    'Morocco': {'revenue': 45, 'cost': 48}, # €/t
    'Argelia': {'revenue': 50, 'cost': 55}, # €/t
}

years = np.arange(2026, 2051)
n_years = len(years)

# Helper to create a price vector grown with inflation
def price_series(base_value):
    # (1 + infl)^0, (1 + infl)^1, ... (1 + infl)^(n_years-1)
    factors = (1 + mean_inflation / 100) ** np.arange(n_years)
    return np.round(base_value * factors, 2)   # round to 2 decimals

# Assemble columns
data = {
    'Year': years,
    'Importing revenue from France [€/t]':  price_series(base_prices['France']['revenue']),
    'Exporting cost to France [€/t]':       price_series(base_prices['France']['cost']),
    'Importing revenue from Morocco [€/t]': price_series(base_prices['Morocco']['revenue']),
    'Exporting cost to Morocco[€/t]':       price_series(base_prices['Morocco']['cost']),
    'Importing revenue from Argelia[€/t]':  price_series(base_prices['Argelia']['revenue']),
    'Exporting cost to Argelia[€/t]':       price_series(base_prices['Argelia']['cost']),
}

trading_prices_df = pd.DataFrame(data)

# Helper: build new rows with 5-year means
rows = []
for yr in milestone_years:
    window_years = list(range(yr - 4, yr + 1))        # e.g. 2026-2030
    subset = trading_prices_df[trading_prices_df['Year'].isin(window_years)]

    # Compute column-wise mean (exclude 'Year'), round to 2 decimals
    mean_vals = subset.drop(columns='Year').mean().round(2)
    mean_vals['Year'] = yr                            # add label for the row
    rows.append(mean_vals)

# Create the reduced DataFrame, keeping original column order
cols_order = ['Year'] + [c for c in trading_prices_df.columns if c != 'Year']
trading_prices_5yr_df = (
    pd.DataFrame(rows)
      .loc[:, cols_order]           # enforce order
      .sort_values('Year')          # ensure chronological order
      .reset_index(drop=True)
)

trading_prices_5yr_df['Year'] = trading_prices_5yr_df['Year'].astype(int)

# print(trading_prices_5yr_df)

# Calculation of the Present Value

pv_trading_prices_df = trading_prices_5yr_df.copy()

# 1) Base year for discounting
base_year = pv_trading_prices_df['Year'].min()          # e.g. 2030

# 2) Row-wise discount factor: (1 + k)^(Year – base_year)
discount_factor = (1 + INTEREST_RATE) ** (pv_trading_prices_df['Year'] - base_year)

# 3) Identify numeric columns to be discounted (all except 'Year')
value_cols = pv_trading_prices_df.select_dtypes(include='number').columns.drop('Year')

# 4) Apply the discount factor (broadcast row-wise, no column renaming)
pv_trading_prices_df[value_cols] = pv_trading_prices_df[value_cols].div(discount_factor, axis=0)

# 5) Optional: round for neat display
pv_trading_prices_df[value_cols] = pv_trading_prices_df[value_cols].round(2)
pv_trading_prices_df = pv_trading_prices_df.rename(
    columns=lambda c: c.replace('[€/t]', '[M€/MtCO2]') if '[€/t]' in c else c
)

# print(pv_trading_prices_df)




"""

MAXIMUM RATE OF CO2 TRADING PER YEAR

The amount of CO2 that can be traded per year across borders between countries is calculated as a percentage of Iberia's total emissions.

In the first years (2026-2030), only projects on a national scale are developed. From 2031 exchanges are allowed with France (both imports and exports), from 2036 with Morocco and from 2041 with Algeria.

"""

max_trade_flows_df = pd.read_excel('model_parameters.xlsx', sheet_name='Max. trading flows') # Read max flows data from Excel file

# print(max_trade_flows_df)



"""

FLOW CAPACITY FOR A D-DIAMETER PIPELINE

Approach taken from: https://www.sciencedirect.com/science/article/pii/S1750583623001986

Flow capacity for pipelines ranging from 1 to 64 inches, in ton/y.

From the paper, which provides data on maximum flow rates for certain diameters, an approximation of maximum flow as a function of diameter squared is made.
This factor is then applied to all the diameters under study, thus obtaining the value of the maximum flow rate for each of them.

"""

paper_flow_df = pd.read_excel('model_parameters.xlsx', sheet_name='Max flows') # Read max flows data from Excel file

co2_density = 700  # kg/m³, approximate density of CO₂ at transport conditions
seconds_in_a_year = 365 * 24 * 3600  # s

paper_flow_df['Flow [kg/s]'] = paper_flow_df['Max. annual flow [Mt/year]'] * 1000000000 / seconds_in_a_year
paper_flow_df['Area [m²]'] = np.pi * (paper_flow_df['Diameter [m]'] / 2) ** 2
paper_flow_df['Velocity [m/s]'] = paper_flow_df['Flow [kg/s]'] / (co2_density * paper_flow_df['Area [m²]'])

avg_velocity = paper_flow_df['Velocity [m/s]'].mean()  # Average velocity across all diameters

# print(paper_flow_df)

pipe_diam_features_df = onshore_pipe_inv_cost_df.copy()

removed_columns = [col for col in pipe_diam_features_df if str(col).isdigit()]  # Columns to remove
pipe_diam_features_df.drop(columns=removed_columns, inplace=True)

pipe_diam_features_df['Diameter [m]'] = pipe_diam_features_df['Diameter [mm]'] / 1000  # Convert diameter from mm to m
pipe_diam_features_df['Area [m²]'] = np.pi * (pipe_diam_features_df['Diameter [m]'] / 2) ** 2  # Calculate cross-sectional area
pipe_diam_features_df['Flow capacity [kg/s]'] = pipe_diam_features_df['Area [m²]'] * co2_density * avg_velocity  # Calculate flow in kg/s
pipe_diam_features_df['Flow capacity [t/d]'] = pipe_diam_features_df['Flow capacity [kg/s]'] * 24 * 3600 / 1000  # Convert to t/d
pipe_diam_features_df['Assumed velocity [m/s]'] = avg_velocity * 0.6  # Calculate assumed velocity in m/s
pipe_diam_features_df['Assumed flow [MtCO2/5y]'] = pipe_diam_features_df['Flow capacity [t/d]'] * 0.6 * 365 * 5 / 1e6 # Assume 60% of max flow for pressure drop calculations

# print(pipe_diam_features_df)




"""

FRICTION PRESSURE DROP PER KM FOR A D-DIAMETER PIPELINE

"""

epsilon = 4.5e-5  # Roughness of the pipe in meters (for carbon steel)

fluid = "CO2"  # Fluid type
temp = 308.15  # Temperature in Kelvin (35 °C)
pressure = 125e5  # Pressure in Pa (125 bar)

avg_density = PropsSI('D', 'T', temp, 'P', pressure, fluid)  # Density in kg/m³
avg_viscosity = PropsSI('V', 'T', temp, 'P', pressure, fluid)  # Viscosity in Pa.s

# print (f"Average density of CO2 at {temp} K and {pressure/1e5} bar: {avg_density:.2f} kg/m³")
# print (f"Average viscosity of CO2 at {temp} K and {pressure/1e5} bar: {avg_viscosity:.6f} Pa.s")

pipe_diam_features_df['Reynolds number'] = pipe_diam_features_df['Assumed velocity [m/s]'] * pipe_diam_features_df['Diameter [m]'] * avg_density / avg_viscosity

def swamee_roughness(reynolds_number, diameter, epsilon):
    """
    Calculate the Swamee-Jain friction factor for turbulent flow.
    
    Parameters:
        reynolds_number (float): Reynolds number of the flow.
        diameter (float): Diameter of the pipe in meters.
        epsilon (float): Roughness of the pipe in meters.
        
    Returns:
        float: Friction factor.
    """
    return 0.25 / (np.log10(epsilon / (3.7 * diameter) + 5.74 / reynolds_number ** 0.9)) ** 2

pipe_diam_features_df['Friction factor'] = swamee_roughness(pipe_diam_features_df['Reynolds number'], pipe_diam_features_df['Diameter [m]'], epsilon)
pipe_diam_features_df['Friction pressure drop [bar/km]'] = (pipe_diam_features_df['Friction factor'] * avg_density * pipe_diam_features_df['Assumed velocity [m/s]']**2 / (2 * pipe_diam_features_df['Diameter [m]']) / 100000 * 1000 ) # Pa/m → bar/km

# print(pipe_diam_features_df)




"""

ELEVATION PRESSURE DROP CALCULATION

"""

def elevation_pressure_drop(density, height_1, height_2):
    """
    Calculate the pressure drop due to elevation change.
    
    Parameters:
        density (float): Density of the fluid in kg/m³.
        height_1 (float): Initial height in meters.
        height_2 (float): Final height in meters.
        
    Returns:
        float: Pressure drop in bar.
    """
    g = 9.81  # Acceleration due to gravity in m/s²
    return (density * g * (height_2 - height_1)) / 101325  # Convert Pa to bar




"""

CALCULATION OF ALL THE PRESSURE DROP PARAMETERS FOR THE PIPELINE CANDIDATES DATABASE

"""

candidate_pipelines_gdf = pd.read_excel('iberian_co2_network_data.xlsx', sheet_name='Pipeline candidates')  # Read candidate pipelines data from Excel file

candidate_pipelines_gdf["Node connection"] = candidate_pipelines_gdf["Node connection"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
candidate_pipelines_gdf["Connection type"] = candidate_pipelines_gdf["Connection type"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)  # Convert string representation of lists to actual lists
candidate_pipelines_gdf["Node heights"] = candidate_pipelines_gdf["Node heights"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
candidate_pipelines_gdf["geometry"] = candidate_pipelines_gdf["geometry"].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)

def compute_elev_pressure_drops(emission_proj_summary_df, avg_density):
    high_elev_drop = []
    fart_elev_drop = []

    for idx, row in emission_proj_summary_df.iterrows():
        height_1 = row["Node heights"][0]
        height_2_high = row["Max height (m)"]
        height_2_fart = row["Node heights"][1]

        high_elev_drop.append(round(elevation_pressure_drop(avg_density, height_1, height_2_high),2))
        fart_elev_drop.append(round(elevation_pressure_drop(avg_density, height_1, height_2_fart),2))

    return high_elev_drop, fart_elev_drop

high_elev_drop, fart_elev_drop = compute_elev_pressure_drops(candidate_pipelines_gdf, avg_density)

cand_pipe_features_gdf = candidate_pipelines_gdf[["Pipeline identifier", "Node connection", "Connection type", "Longitude (km)", "Max height distance (km)", "Transport method", "Countries", "Stage"]].copy() # Create a DataFrame with the relevant columns
cand_pipe_features_gdf.rename(columns={"Pipeline identifier": "Pipe ID", "Longitude (km)": "Longitude [km]", "Max height distance (km)": "Distance until heighest point [km]"}, inplace=True)

cand_pipe_features_gdf["High. elev. pres. drop [bar]"] = high_elev_drop # Add elevation pressure drop in the highest point of the pipeline
cand_pipe_features_gdf["Fart. elev. pres. drop [bar]"] = fart_elev_drop # Add elevation pressure drop in the farthest point of the pipeline

cand_pipe_features_gdf["geometry"] = candidate_pipelines_gdf["geometry"] # Add geometry column from the candidate pipelines DataFrame

cand_pipe_features_gdf = gpd.GeoDataFrame(cand_pipe_features_gdf, geometry="geometry") # Convert to GeoDataFrame for geographical operations

# print(cand_pipe_features_gdf.head(50))
# print(cand_pipe_features_gdf.tail(50))



"""

FRICTION-RELATED PRESSURE DROP CALCULATION

"""

def build_frict_pres_drop_gdf(
        pipe_diam_features_df: pd.DataFrame,
        candidate_pipelines_gdf: pd.DataFrame,
        geometry_col: str = 'geometry'
    ) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame that stores, for every candidate pipeline
    and every standard diameter, the friction pressure drop at the
    highest point and at the farthest point.

    ▸ Column order:
        0. Pipeline identifier
        1. Max height distance (km)          ← swapped here
        2. Longitude (km)                    ← swapped here
        3. One column per diameter in inches (tuples rounded to 2 decimals)
        4. geometry
    """

    # Base DataFrame with the first three columns in the requested order
    frict_pres_drop_df = candidate_pipelines_gdf[
        ['Pipeline identifier', 'Max height distance (km)', 'Longitude (km)', geometry_col]
    ].copy()

    # Loop through all standard diameters listed in pipe_diam_features_df
    for _, row in pipe_diam_features_df.iterrows():
        diam_inch  = str(row['Diameter [inch]'])                 # column name
        fpd_bar_km = row['Friction pressure drop [bar/km]']      # loss per km

        # Compute pressure losses and round to 2 decimals
        deltaP_max_height = (fpd_bar_km *
                             candidate_pipelines_gdf['Max height distance (km)']).round(2)
        deltaP_farthest   = (fpd_bar_km *
                             candidate_pipelines_gdf['Longitude (km)']).round(2)

        # Add the tuples column
        frict_pres_drop_df[diam_inch] = list(zip(deltaP_max_height, deltaP_farthest))

    # Convert to GeoDataFrame (if not already)
    frict_pres_drop_gdf = gpd.GeoDataFrame(
        frict_pres_drop_df,
        geometry=geometry_col,
        crs=getattr(candidate_pipelines_gdf, 'crs', None)
    )

    # Final column order: identifier ▸ max-height-dist ▸ longitude ▸ diameters ▸ geometry
    ordered_cols = (
        ['Pipeline identifier', 'Max height distance (km)', 'Longitude (km)'] +
        [str(d) for d in pipe_diam_features_df['Diameter [inch]']] +
        [geometry_col]
    )
    frict_pres_drop_gdf = frict_pres_drop_gdf[ordered_cols]

    return frict_pres_drop_gdf


# Exeute the function to build the friction pressure drop GeoDataFrame
frict_pres_drop_gdf = build_frict_pres_drop_gdf(
    pipe_diam_features_df,
    candidate_pipelines_gdf
)

# print(frict_pres_drop_gdf.head(50))
# print(frict_pres_drop_gdf.tail(50))




"""

CALCULATION OF THE CITY SURROUNDING PENALTY FACTOR FOR THE PIPELINE CANDIDATES DATABASE

The penalty factor is calculated in such a way that is equal to 1 for pipelines that do not pass through any city, and increases with the square root of the populations of the cities through which the pipeline passes.
For a 3 million population city, the penalty factor is 1.15. When several cities are crossed, the penalty factor increases, but it is capped anyway.

"""

gamma = 0.5 # Sublinear exponent
p_ref = 1000000 # Reference population for the penalty factor calculation
beta = 0.087

pop_lookup = (
    candidate_pipelines_gdf
    .set_index('Pipeline identifier')['Cities populations']
    .to_dict()
)

def to_numeric_list(cell):
    """Return a list[float] from cell content (list, tuple, str, number, NaN)."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    # already list/tuple
    if isinstance(cell, (list, tuple)):
        return [float(p) for p in cell]
    # string → try to parse "[...]" or "(...)" or single number
    if isinstance(cell, str):
        cell = cell.strip()
        try:
            parsed = ast.literal_eval(cell)
        except (ValueError, SyntaxError):
            parsed = cell  # treat as single number string
        if isinstance(parsed, (list, tuple)):
            return [float(p) for p in parsed]
        try:
            return [float(parsed)]
        except ValueError:
            return []
    # number
    try:
        return [float(cell)]
    except (TypeError, ValueError):
        return []

def city_penalty(pipe_id):
    """Compute detour-penalty factor for a given pipeline identifier."""
    pops = to_numeric_list(pop_lookup.get(pipe_id, []))
    pop_term = sum((p / p_ref) ** gamma for p in pops)
    return round(1.0 + beta * pop_term, 3)

cand_pipe_features_gdf['City detour penalty factor'] = (cand_pipe_features_gdf['Pipe ID'].apply(city_penalty))




"""

CALCULATION OF THE UNEVENNESS OF TERRAIN PENALTY FACTOR FOR THE PIPELINE CANDIDATES DATABASE

The penalty factor is calculated in such a way that is equal to 1 for candidate pipelines which cumulative elevation gain is 0, and increases linearly with the cumulative elevation gain divided by the longitude of the pipeline.

"""

coef   = 0.002

# Build a Series with the slope-penalty factor for each pipe
slope_factor = round((
    1.0 + coef *
    (candidate_pipelines_gdf.set_index('Pipeline identifier')['Cumul. pos. height (m)'] /
     candidate_pipelines_gdf.set_index('Pipeline identifier')['Longitude (km)'])),3)

# Map it into cand_pipe_features_gdf
cand_pipe_features_gdf['Slope penalty factor'] = (cand_pipe_features_gdf['Pipe ID'].map(slope_factor))
# print(cand_pipe_features_gdf.tail(50))




"""

CALCULATION OF THE SHIPPING COSTS FOR OFFSHORE CO2 TRANSPORTATION INSTEAD OF PIPELINES

The calculation of the shipping costs is divided into two parts:
1. The fixed costs per trip, which include liquefaction, loading, unloading, gasification and harbour fees.
2. The variable costs per km, which include fuel costs.

Costs are taken from: https://www.gov.uk/government/publications/shipping-carbon-dioxide-co2-uk-cost-estimation-study.
The costs are given in £/tCO2, so they are converted to €/tCO2 using the current exchange rate (1.16 € / £). Then, the costs from 2018 are scaled to the future market costs using the HICP predictions.

"""

complete_nodes_gdf = pd.read_excel('iberian_co2_network_data.xlsx', sheet_name='Nodes')  # Read nodes data from Excel file

complete_nodes_gdf['Total CO2 emissions'] = (complete_nodes_gdf['Total CO2 emissions'].astype(str).str.replace(',', '.', regex=False).astype(float))

complete_nodes_gdf = complete_nodes_gdf.rename(columns={'Total CO2 emissions': 'Total CO2 emissions [Mt/y]'})
complete_nodes_gdf = complete_nodes_gdf.rename(columns={'Total CO2 capacity': 'Total CO2 capacity [Mt]'})
complete_nodes_gdf = complete_nodes_gdf.rename(columns={'Annual CO2 utilization capacity': 'Annual CO2 utilization capacity [Mt/y]'})

# print(complete_nodes_gdf.head(50))



####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################
# Ship costs
####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################

# Parameters

# -------------------------------------------------------------------------
# CONSTANTS – 2018 baseline (central case, 1 Mt/a, 10 kt vessel)
# -------------------------------------------------------------------------
gbp_to_eur = 1.16           # 2018 £ → €
ship_capacity = 10_000      # tCO2 per voyage (round-trip capacity)

# -- Harbour fee (given per round trip) ------------------------------------
harbour_fee_trip_eur = 10_194 * gbp_to_eur   # €/cycle

# -- Liquefaction unit cost (£/t a) already annualised in BEIS -------------
lic_capex_gbp  = 9.8
lic_opex_gbp   = lic_capex_gbp * 0.10        # 10 % of CAPEX
lic_elec_gbp   = 24.6 * 0.08 / 0.86          # kWh/t · £/kWh / η
lic_unit_eur   = (lic_capex_gbp + lic_opex_gbp + lic_elec_gbp) * gbp_to_eur
lic_cycle_eur  = lic_unit_eur * ship_capacity

# -- Loading / unloading (symmetric) ---------------------------------------
load_capex_gbp = 1.4
load_opex_gbp  = load_capex_gbp * 0.03       # 3 % OPEX
load_unit_eur  = (load_capex_gbp + load_opex_gbp) * gbp_to_eur
load_cycle_eur = load_unit_eur * ship_capacity
unload_cycle_eur = load_cycle_eur            # same figure

# -- Gasification ----------------------------------------------------------
gas_capex_gbp  = 0.83
gas_opex_gbp   = gas_capex_gbp * 0.33        # 33 % OPEX (BEIS)
gas_unit_eur   = (gas_capex_gbp + gas_opex_gbp) * gbp_to_eur
gas_cycle_eur  = gas_unit_eur * ship_capacity

# -------------------------------------------------------------------------
# FIXED COST PER CYCLE – €2018
# -------------------------------------------------------------------------
fixed_ship_cost_per_cycle = (
    lic_cycle_eur
  + load_cycle_eur
  + unload_cycle_eur
  + gas_cycle_eur
  + harbour_fee_trip_eur
)

# -------------------------------------------------------------------------
# VARIABLE COST PER km – €2018
# -------------------------------------------------------------------------
fuel_price_eur_mwh = 20 * gbp_to_eur          # €/MWh
daily_fuel_mwh = 263                          # MWh per day @ 15 kn
hourly_fuel_cost_eur = daily_fuel_mwh * fuel_price_eur_mwh / 24

knots_to_kmh = 1.852
ship_speed_kmh = 15 * knots_to_kmh
variable_ship_cost_per_cycle = hourly_fuel_cost_eur / ship_speed_kmh  # €/km/cycle

####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################
# Creating the DataFrame with the ship fixed and variable costs for each year
####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################

# Get the cumulative inflation factor between 2018 and 2026

cumulative_inflation_factor_2026 = (
    cumulative_infl_df.loc[cumulative_infl_df["Year"] == 2026,
                           "Cumulative Inflation Prediction"].squeeze()              # converts 1-element Series → scalar
)

cumulative_inflation_factor_2018 = (
    cumulative_infl_df.loc[cumulative_infl_df["Year"] == 2018,
                            "Cumulative Inflation"].squeeze()                        # converts 1-element Series → scalar
)

# Scale the fixed and variable ship cost per cycle to 2026 prices

fixed_ship_cost_per_cycle = round(fixed_ship_cost_per_cycle * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2018/100), 2) # Fixed cost per cycle in 2026 prices
variable_ship_cost_per_cycle = round(variable_ship_cost_per_cycle * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2018/100), 2) # Variable cost per km in 2026 prices


# Create a DataFrame with the ship costs for each year

years = range(2026, 2051) # 2026 inclusive up to 2050 inclusive
infl_factor = 1 + mean_inflation/100  # yearly multiplier

# Lists to store the inflated values year by year
fixed_cost_list = [fixed_ship_cost_per_cycle / 1e6]   # start from 2026 price
variable_cost_list = [variable_ship_cost_per_cycle / 1e6]

for _ in range(1, len(years)):
    # Multiply previous year's value by the inflation factor and round to 2 dp
    fixed_cost_list.append(round(fixed_cost_list[-1] * infl_factor, 2))
    variable_cost_list.append(round(variable_cost_list[-1] * infl_factor, 9))

# Build the DataFrame
ship_costs_df = pd.DataFrame({
    "Year": years,
    "Fixed_cost [M€/cycle]": fixed_cost_list,
    "Fuel_cost [M€/km/cycle]": variable_cost_list
})

# print(ship_costs_df)

rows = []
for yr in milestone_years:
    window_years = list(range(yr - 4, yr + 1))           # e.g. 2026-2030
    subset = ship_costs_df[ship_costs_df['Year'].isin(window_years)]

    # Compute column-wise mean excluding 'Year'
    mean_vals = subset.drop(columns='Year').mean().round(10)
    mean_vals['Year'] = yr                        
    rows.append(mean_vals)

# Build the reduced DataFrame, preserving original column order
cols_order = ['Year'] + [c for c in ship_costs_df.columns if c != 'Year']
ship_costs_5yr_df = (
    pd.DataFrame(rows)
      .loc[:, cols_order]
      .sort_values('Year')
      .reset_index(drop=True)
)

ship_costs_5yr_df['Year'] = ship_costs_5yr_df['Year'].astype(int)

# print(ship_costs_5yr_df)

ship_costs_pv_df = ship_costs_5yr_df.copy()

# 1) Determine the base year (earliest year in the series)
base_year = ship_costs_pv_df['Year'].min()   # e.g. 2030

# 2) Compute the row-wise discount factor: (1 + k)^(Year - base_year)
discount_factor = (1 + INTEREST_RATE) ** (ship_costs_pv_df['Year'] - base_year)

# 3) Identify numeric columns to be discounted (all except 'Year')
value_cols = ship_costs_pv_df.select_dtypes(include='number').columns.drop('Year')

# 4) Convert nominal costs to present value — keep same column names & order
ship_costs_pv_df[value_cols] = ship_costs_pv_df[value_cols].div(discount_factor, axis=0)

# 5) Optional: round for cleaner display
ship_costs_pv_df[value_cols] = ship_costs_pv_df[value_cols].round(10)

# print(ship_costs_pv_df)




"""

CALCULATION OF THE REVENUES FOR SELLING CO2 FOR INDUSTRIAL UTILIZATION

The revenue associated to selling CO2 to companies is assumed to be of 100€/ton in 2026, and to grow with inflation afterwards.

"""

selling_revenue_2026 = 100 # €/tCO2

selling_revenues = [selling_revenue_2026]

for _ in range(1, len(years)):
    selling_revenues.append(round(selling_revenues[-1] * infl_factor, 2))

# Build the DataFrame
selling_revenues_df = pd.DataFrame({
    "Year": years,
    "Selling revenue [M€/MtCO2]": selling_revenues
})

rows = []                                           

for yr in milestone_years:
    window_years = list(range(yr - 4, yr + 1))      # e.g. 2026-2030
    mean_rev = (
        selling_revenues_df.loc[
            selling_revenues_df['Year'].isin(window_years),
            'Selling revenue [M€/MtCO2]'
        ]
        .mean()
        .round(2)
    )
    rows.append({"Year": yr, "Selling revenue [M€/MtCO2]": mean_rev})

# Clean DataFrame with only the desired columns
selling_revenues_5yr_df = (
    pd.DataFrame(rows)
      .astype({"Year": int})
      .sort_values('Year')
      .reset_index(drop=True)
)

# print(selling_revenues_5yr_df)

selling_revenues_pv_df = selling_revenues_5yr_df.copy()

# 1) Determine the base year (earliest in the series)
base_year = selling_revenues_pv_df['Year'].min()   # e.g. 2030

# 2) Compute the row-wise discount factor: (1 + k)^(Year - base_year)
discount_factor = (1 + INTEREST_RATE) ** (selling_revenues_pv_df['Year'] - base_year)

# 3) Identify numeric columns to be discounted (all except 'Year')
value_cols = selling_revenues_pv_df.select_dtypes(include='number').columns.drop('Year')

# 4) Convert nominal revenues to present value — keep same column names & order
selling_revenues_pv_df[value_cols] = selling_revenues_pv_df[value_cols].div(discount_factor, axis=0)

# 5) Optional: round for cleaner display
selling_revenues_pv_df[value_cols] = selling_revenues_pv_df[value_cols].round(2)

# print(selling_revenues_pv_df)


"""

CALCULATION OF THE COMPRESSION COSTS FOR BOTH INITIAL COMPRESSION STATIONS AND BOOSTING STATIONS

Compression costs based on the information provided in: https://www.stet-review.org/articles/stet/full_html/2025/01/stet20240102/stet20240102.html

The costs provided in the paper are associtated with the compression of 1,825 MtCO2/year.
Of course, each compression station has a different capacity, but to keep a simple approach the costs are assumed to be proportional to the capacity of the station.

Initial pressure raising costs are divided into three parts:
1. Machinery annualized CAPEX, which value equals to 3.65 €2005/tCO2/year according to the paper.
2. Machinery OPEX, which value equals to 0.97 €2005/tCO2 according to the paper.
3. Electricity costs, which value equals to 7.01 €2005/tCO2 according to the paper.
   This electricity cost stands for the energy needed to raise the pressure of the CO2 from 1 bar to 150 bar.

To calculate the costs of boosting stations, a similar approach is taken, but a cost estimation is done to consider the fact that the pressure is only raised from 100 bar to 150 bar.
Two empirical rules widely accepted in engineering are applied:

i) The power required to compress is proportional to the logarithm of the pressure ratio.
ii) Six-tenths rule: the cost of the equipment raises with the 0.6 power of the capacity.

Then, the costs are scaled to the 2026 prices using the HICP registers, and until 2050 the costs are assumed to grow with the average inflation rate.

"""

# Definition of the reference values for the compression costs in 2005 prices

initial_compr_ann_capex_2005 = 3.65  # Machinery CAPEX in €2005/tCO2/year
initial_compr_opex_2005 = 0.97       # Machinery OPEX in €2005/tCO2
initial_compr_elec_2005 = 7.01       # Electricity costs in €2005/tCO2

INTEREST_RATE = 0.08
COMPR_LIFESPAN = 20

CAPITAL_RECOVERY_FACTOR = INTEREST_RATE * (1 + INTEREST_RATE) ** COMPR_LIFESPAN / ((1 + INTEREST_RATE) ** COMPR_LIFESPAN - 1)

initial_compr_capex_2005 = initial_compr_ann_capex_2005/CAPITAL_RECOVERY_FACTOR

initial_compr_rate = 150 / 1  # Pressure ratio for initial compression (from 1 bar to 150 bar)
boosting_compr_rate = 150 / 100  # Pressure ratio for boosting compression (from 100 bar to 150 bar)

energetic_factor = np.log(boosting_compr_rate) / np.log(initial_compr_rate)  # Electricity costs for boosting compression in €2005/tCO2

boosting_compr_ann_capex_2005 = initial_compr_ann_capex_2005 * (boosting_compr_rate / initial_compr_rate) ** 0.6  # Machinery annualized CAPEX for boosting compression in €2005/tCO2/year
boosting_compr_opex_2005 = initial_compr_opex_2005 * (boosting_compr_rate / initial_compr_rate) ** 0.6  # Machinery OPEX for boosting compression in €2005/tCO2
boosting_compr_elec_2005 = initial_compr_elec_2005 * energetic_factor  # Electricity costs for boosting compression in €2005/tCO2

boosting_compr_capex_2005 = boosting_compr_ann_capex_2005/CAPITAL_RECOVERY_FACTOR

inflation_factor_2005_2010 = 90.14/82.3 # Cumulative inflation factor from 2005 to 2010 (HICP data). Taken from: https://www.oenb.at/isawebstat/stabfrage/createReport;jsessionid=DC49927B975BFF48981D95298DB29887?lang=EN&original=false&report=6.3

# Scale the costs to 2026 prices using the cumulative inflation factor from 2005 to 2026

cumulative_inflation_factor_2026 = (
    cumulative_infl_df.loc[cumulative_infl_df["Year"] == 2026,
                            "Cumulative Inflation Prediction"].squeeze()                 # converts 1-element Series → scalar
)
cumulative_inflation_factor_2010 = (
    cumulative_infl_df.loc[cumulative_infl_df["Year"] == 2010,
                            "Cumulative Inflation"].squeeze()                            # converts 1-element Series → scalar
)

# Scale the costs to 2026 prices
initial_compr_ann_capex_2026 = round(initial_compr_ann_capex_2005 * inflation_factor_2005_2010 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2010/100), 2)  # Machinery annualized CAPEX in €2026/tCO2/year
initial_compr_capex_2026 = round(initial_compr_capex_2005 * inflation_factor_2005_2010 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2010/100), 2)  # Machinery CAPEX in €2026/tCO2
# print((initial_compr_ann_capex_2026, initial_compr_capex_2026))

initial_compr_opex_2026 = round(initial_compr_opex_2005 * inflation_factor_2005_2010 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2010/100), 2)  # Machinery OPEX in €2026/tCO2
initial_compr_elec_2026 = round(initial_compr_elec_2005 * inflation_factor_2005_2010 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2010/100), 2)  # Electricity costs in €2026/tCO2

boosting_ann_capex_2026 = round(boosting_compr_ann_capex_2005 * inflation_factor_2005_2010 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2010/100), 2)  # Machinery annualized CAPEX for boosting compression in €2026/tCO2/year
boosting_capex_2026 = round(boosting_compr_capex_2005 * inflation_factor_2005_2010 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2010/100), 2)  # Machinery CAPEX for boosting compression in €2026/tCO2
# print((boosting_ann_capex_2026, boosting_capex_2026))

boosting_opex_2026 = round(boosting_compr_opex_2005 * inflation_factor_2005_2010 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2010/100), 2)  # Machinery OPEX for boosting compression in €2026/tCO2
boosting_elec_2026 = round(boosting_compr_elec_2005 * inflation_factor_2005_2010 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2010/100), 2)  # Electricity costs for boosting compression in €2026/tCO2

# -------------- GROWTH WITH MEAN INFLATION --------------------
yrs, ic_capex, ic_opex, ic_elec = [], [], [], []
bs_capex, bs_opex, bs_elec      = [], [], []

cap1, op1, el1 = initial_compr_capex_2026, initial_compr_opex_2026, initial_compr_elec_2026
cap2, op2, el2 = boosting_capex_2026, boosting_opex_2026, boosting_elec_2026

for yr in range(2026, 2051):
    yrs.append(yr)
    ic_capex.append(cap1)
    ic_opex.append(op1)
    ic_elec.append(el1)
    bs_capex.append(cap2)
    bs_opex.append(op2)
    bs_elec.append(el2)

    # Update for next year
    cap1 *= (1 + mean_inflation / 100)
    op1  *= (1 + mean_inflation / 100)
    el1  *= (1 + mean_inflation / 100)
    cap2 *= (1 + mean_inflation / 100)
    op2  *= (1 + mean_inflation / 100)
    el2  *= (1 + mean_inflation / 100)

# ----------------- BUILD FINAL DATAFRAME ----------------------
compression_costs_df = pd.DataFrame({
    'Year': yrs,
    'Init. compr. CAPEX [M€/MtCO2]': np.round(ic_capex, 3),
    'Init. compr. OPEX [M€/MtCO2]'   : np.round(ic_opex,  3),
    'Init. compr. electr. [M€/MtCO2]'  : np.round(ic_elec,  3),
    'Boost. st. CAPEX [M€/MtCO2]'  : np.round(bs_capex, 3),
    'Boost. st. OPEX [M€/MtCO2]'     : np.round(bs_opex,  3),
    'Boost. st. electr. [M€/MtCO2]'    : np.round(bs_elec,  3)
})

# print(compression_costs_df)

# Helper: build new rows with 5-year means
rows = []
for yr in milestone_years:
    window_years = list(range(yr - 4, yr + 1))        # e.g. 2026-2030
    subset = compression_costs_df[compression_costs_df['Year'].isin(window_years)]

    # Compute column-wise mean (exclude 'Year'), round to 3 decimals
    mean_vals = subset.drop(columns='Year').mean().round(3)
    mean_vals['Year'] = yr                            # add label for the row
    rows.append(mean_vals)

# Create the reduced DataFrame, keeping original column order
cols_order = ['Year'] + [c for c in compression_costs_df.columns if c != 'Year']
compression_costs_5yr_df = (
    pd.DataFrame(rows)
      .loc[:, cols_order]           # enforce order
      .sort_values('Year')          # ensure chronological order
      .reset_index(drop=True)
)

compression_costs_5yr_df['Year'] = compression_costs_5yr_df['Year'].astype(int)

# print(compression_costs_5yr_df)

compression_costs_pv_df = compression_costs_5yr_df.copy()

# 1) Determine the base year (earliest in the series)
base_year = compression_costs_pv_df['Year'].min()   # e.g. 2030

# 2) Row-wise discount factor: (1 + k)^(Year - base_year)
discount_factor = (1 + INTEREST_RATE) ** (compression_costs_pv_df['Year'] - base_year)

# 3) Numeric columns to be discounted (everything except 'Year')
value_cols = compression_costs_pv_df.select_dtypes(include='number').columns.drop('Year')

# 4) Convert nominal costs to present value — keep names & order unchanged
compression_costs_pv_df[value_cols] = compression_costs_pv_df[value_cols].div(discount_factor, axis=0)

# 5) Optional: round for neat display
compression_costs_pv_df[value_cols] = compression_costs_pv_df[value_cols].round(3)

# print(compression_costs_pv_df)




"""

CALCULATION OF THE CARBON CAPTURE AND INJECTION COSTS

Both costs based on the information provided in: https://www.sciencedirect.com/science/article/pii/S1750583623001986

Costs for carbon capture are assumed to be 70 €2024/tCO2, which will be scaled to 2026-2050 prices using the HICP predictions.

Costs for injection of CO2 into the storage site are assumed to be 6 €2024/tCO2, which will be scaled to 2026-2050 prices using the HICP predictions.

"""

capture_cost_2024 = 70  # €2024/tCO2
injection_cost_2024 = 6  # €2024/tCO2

# Scale the costs to 2026 prices using the cumulative inflation factor from 2024 to 2026

cumulative_inflation_factor_2026 = (
    cumulative_infl_df.loc[cumulative_infl_df["Year"] == 2026,
                            "Cumulative Inflation Prediction"].squeeze()                 # converts 1-element Series → scalar
)

cumulative_inflation_factor_2024 = (
    cumulative_infl_df.loc[cumulative_infl_df["Year"] == 2024,
                            "Cumulative Inflation"].squeeze()                            # converts 1-element Series → scalar
)

# Scale the costs to 2026 prices
capture_cost_2026 = round(capture_cost_2024 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2024/100), 2)  # Capture cost in €2026/tCO2
injection_cost_2026 = round(injection_cost_2024 * (1 + cumulative_inflation_factor_2026/100) / (1 + cumulative_inflation_factor_2024/100), 2)  # Injection cost in €2026/tCO2

# Create a DataFrame with the capture and injection costs for each year
capture_injection_costs_df = pd.DataFrame({
    'Year': range(2026, 2051),  # 2026 inclusive up to 2050 inclusive
    'Capture cost [M€/MtCO2]': np.round([capture_cost_2026 * (1 + mean_inflation / 100) ** i for i in range(25)],2),
    'Injection cost [M€/MtCO2]': np.round([injection_cost_2026 * (1 + mean_inflation / 100) ** i for i in range(25)],2)
})

# print(capture_injection_costs_df)

# Helper: build new rows with 5-year means
rows = []
for yr in milestone_years:
    window_years = list(range(yr - 4, yr + 1))        # e.g. 2026-2030
    subset = capture_injection_costs_df[capture_injection_costs_df['Year'].isin(window_years)]

    # Compute column-wise mean (exclude 'Year'), round to 2 decimals
    mean_vals = subset.drop(columns='Year').mean().round(2)
    mean_vals['Year'] = yr                            # add label for the row
    rows.append(mean_vals)

# Create the reduced DataFrame, keeping original column order
cols_order = ['Year'] + [c for c in capture_injection_costs_df.columns if c != 'Year']
capture_injection_costs_5yr_df = (
    pd.DataFrame(rows)
      .loc[:, cols_order]           # enforce order
      .sort_values('Year')          # ensure chronological order
      .reset_index(drop=True)
)

capture_injection_costs_5yr_df['Year'] = capture_injection_costs_5yr_df['Year'].astype(int)

# print(capture_injection_costs_5yr_df)

capture_injection_costs_pv_df = capture_injection_costs_5yr_df.copy()

# 1) Determine the base year (earliest in the series)
base_year = capture_injection_costs_pv_df['Year'].min()   # e.g. 2030

# 2) Row-wise discount factor: (1 + k)^(Year - base_year)
discount_factor = (1 + INTEREST_RATE) ** (capture_injection_costs_pv_df['Year'] - base_year)

# 3) Numeric columns to be discounted (everything except 'Year')
value_cols = capture_injection_costs_pv_df.select_dtypes(include='number').columns.drop('Year')

# 4) Convert nominal costs to present value — keep names & order unchanged
capture_injection_costs_pv_df[value_cols] = capture_injection_costs_pv_df[value_cols].div(discount_factor, axis=0)

# 5) Optional: round for neat display
capture_injection_costs_pv_df[value_cols] = capture_injection_costs_pv_df[value_cols].round(2)

# print(capture_injection_costs_pv_df)




"""

CALCULATION OF THE CARBON UTILIZATION AND SEQUESTRATION OBJECTIVES FOR THE DIFFERENT TIME STEPS

The calculation of these objectives is based on several different scientific papers and reports. Essentially:

- The industrial emissions in 2022 in Spain were 48,3 MtCO2. Data taken from: https://climate.ec.europa.eu/eu-action/climate-strategies-targets/progress-climate-action_en
- The industrial emissions in 2022 in Portugal were 8,8 MtCO2. Data taken from: https://climate.ec.europa.eu/eu-action/climate-strategies-targets/progress-climate-action_en
- The industrial emissions in 2022 in the EU were 490 MtCO2. Data taken from: https://www.eionet.europa.eu/etcs/etc-cm/products/etc-cm-report-2023-07-1

Then, the emissions from Spain and Portugal represent the 11,65% of the total industrial emissions in the EU.

Concerning the CCUS objectives, all of the following data is taken from the report: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:52024DC0062

- In 2030, 50 MtCO2 should be captured in the EU, with very limited utilization. It is assumed that all of the captured CO2 is sequestered.
- In 2040, 280 MtCO2 should be captured in the EU, almost 30% of the captured CO2 should be used for utilization, and the rest for sequestration.
- In 2050, 450 MtCO2 should be captured in the EU, almost 45% of the captured CO2 should be used for utilization, and the rest for sequestration.

These objectives are then scaled to the Spanish and Portuguese emissions, assuming that the same percentage of emissions is captured in both countries.
For this scaling, it is assumed that the rate of captured emissions is the same as the rate of industrial emissions in Spain and Portugal compared to the total industrial emissions in the EU (11,65%). Then:

- In 2030, 5,825 MtCO2 should be captured in Spain and Portugal, with very limited utilization. It is assumed that all of the captured CO2 is sequestered.
- In 2040, 32,62 MtCO2 should be captured in Spain and Portugal, almost 30% of the captured CO2 should be used for utilization, and the rest for sequestration.
- In 2050, 52,425 MtCO2 should be captured in Spain and Portugal, almost 45% of the captured CO2 should be used for utilization, and the rest for sequestration.

This results in the following assumptions for the sequestration and utilization objectives:

1) For the case of the sequestration, it is assumed that these values are deterministic, as well as the sequestration calculated for the rest of the time steps.
2) For the case of the utilization, it is assumed that these values are stochastic, and they are sampled from a normal distribution with a mean equal to the objective
   and a standard deviation equal to 10% of the objective. Based on this, three different scenarios are generated:

- Scenario 1: Low utilization scenario, where the utilization is only 50% of the objective. The probability of this scenario is 25%.
- Scenario 2: Medium utilization scenario, where the utilization is equal to the objective. The probability of this scenario is 50%.
- Scenario 3: High utilization scenario, where the utilization is 150% of the objective. The probability of this scenario is 25%.

However, in the following code only the total amounts of CO2 to be captured, sequestered and utilized are calculated, not the stochastic scenarios.

"""

# ---------------------------------------------------------
# 0. Input data (2022 industrial emissions & EU objectives)
# ---------------------------------------------------------
# emissions_spain_2022 = 48.3   # MtCO2
# emissions_portugal_2022 = 8.8 # MtCO2
# emissions_eu_2022 = 490       # MtCO2 – industrial sector (EU-27 + EEA)

# capture_eu = {2030: 50, 2040: 280, 2050: 450}  # MtCO2
# util_eu = {
#     2030: 0,
#     2040: 0.3 * capture_eu[2040],
#     2050: 0.45 * capture_eu[2050],
# }

# # Share of Iberian industrial emissions
# share_iberia = (emissions_spain_2022 + emissions_portugal_2022) / emissions_eu_2022  # ≈ 0.1165

# # Scale EU objectives to Iberia
# capture_ib = {yr: val * share_iberia for yr, val in capture_eu.items()}
# util_ib_base = {yr: val * share_iberia for yr, val in util_eu.items()}
# seq_ib_base = {yr: capture_ib[yr] - util_ib_base[yr] for yr in capture_ib}

# ---------------------------------------------------------
# 0. Input data - Total CO2 emitted in 2050
# ---------------------------------------------------------

# capture_eu = {2030: 50, 2040: 280, 2050: 450}  # MtCO2
# util_eu = {
#     2030: 0,
#     2040: 0.3 * capture_eu[2040],
#     2050: 0.45 * capture_eu[2050],
# }

# # Share of Iberian industrial emissions
# share_iberia = (emissions_spain_2022 + emissions_portugal_2022) / emissions_eu_2022  # ≈ 0.1165

# # Scale EU objectives to Iberia
# capture_ib = {yr: val * share_iberia for yr, val in capture_eu.items()}
# util_ib_base = {yr: val * share_iberia for yr, val in util_eu.items()}
# seq_ib_base = {yr: capture_ib[yr] - util_ib_base[yr] for yr in capture_ib}

"""
In this version of the model, the amount of CO2 captured is decided as a function of the alpha parameter, with 0 <= alpha <= 1. 
When alpha = 0, no CO2 is captured, while when alpha = 1, all CO2 generated is captured by year 2050.
For the rest of the years, the CO2 captured increases following the trend stated by the EU: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:52024DC0062

"""

# The amount of CO2 generated in 2050 is 127.05 MtCO2

capture_peninsula = { # MtCO2
    2030: (50/450) * 127.05, 
    2040: (280/450) * 127.05, 
    2050: 127.05
    }

util_peninsula = { # MtCO2
    2030: 0,
    2040: 0.3 * capture_peninsula[2040],
    2050: 0.45 * capture_peninsula[2050],
}

seq_peninsula = { # MtCO2
    2030: capture_peninsula[2030] - util_peninsula[2030],
    2040: capture_peninsula[2040] - util_peninsula[2040],
    2050: capture_peninsula[2050] - util_peninsula[2050],
}

capture_ib = {yr: val for yr, val in capture_peninsula.items()}
util_ib_base = {yr: val for yr, val in util_peninsula.items()}
seq_ib_base = {yr: val for yr, val in seq_peninsula.items()}


# ---------------------------------------------------------
# 1. Helper: quadratic through three (x, y) points
# ---------------------------------------------------------
def quadratic_through(points):
    """
    Return coefficients (a, b, c) for y = a*x^2 + b*x + c
    that passes through three supplied points.
    """
    x, y = zip(*points)
    return np.polyfit(x, y, 2)            # [a, b, c]

# ---------------------------------------------------------
# 2. Common curve 2025-2030 (capt = seq)
# ---------------------------------------------------------
k_quad = capture_ib[2030] / 25            # y(2030) = target, y(2025) = 0

def early_curve(year):
    return k_quad * (year - 2025) ** 2

# ---------------------------------------------------------
# 3. Post-2030 quadratics (base scenario)
# ---------------------------------------------------------
cap_pts = [(2030, capture_ib[2030]),
           (2040, capture_ib[2040]),
           (2050, capture_ib[2050])]

seq_pts_base = [(2030, seq_ib_base[2030]),
                (2040, seq_ib_base[2040]),
                (2050, seq_ib_base[2050])]

a_cap, b_cap, c_cap = quadratic_through(cap_pts)
a_seq, b_seq, c_seq = quadratic_through(seq_pts_base)

def capture_curve(year):
    return a_cap * year**2 + b_cap * year + c_cap

def sequestration_base_curve(year):
    return a_seq * year**2 + b_seq * year + c_seq

# ---------------------------------------------------------
# 4. Build full yearly series (2025-2051) for plotting
# ---------------------------------------------------------
years_full = np.arange(2025, 2052)        # includes 2025 & 2051
capture_full = []
seq_base_full = []

for y in years_full:
    if y <= 2030:
        cap = seq = early_curve(y)        # identical until 2030
    else:
        cap = capture_curve(y)
        seq = sequestration_base_curve(y)
    capture_full.append(cap)
    seq_base_full.append(seq)

capture_full = np.array(capture_full)
seq_base_full = np.array(seq_base_full)
util_base_full = capture_full - seq_base_full   # S2 (base) utilization

# ---------------------------------------------------------
# 5. Three scenarios: S1 (50 %; probability 25%), S2 (100 %; probability 50%), S3 (150 %, probability 25%)
# ---------------------------------------------------------
scales = {"S1": 0.5, "S2": 1.0, "S3": 1.5}
seq_full = {}
util_full = {}

for key, factor in scales.items():
    util = np.minimum(util_base_full * factor, capture_full)  # cap at capture
    seq = capture_full - util
    util_full[key] = util
    seq_full[key] = seq

# ---------------------------------------------------------
# 6. DataFrame 2026-2050 (rounded to 2 decimals)
# ---------------------------------------------------------
mask_df = (years_full >= 2026) & (years_full <= 2050)
df_ccus = pd.DataFrame({
    "Year": years_full[mask_df],
    "Capture (MtCO₂/y)": np.round(capture_full[mask_df], 2),
    "Sequestration S1 [MtCO₂/y]": np.round(seq_full["S1"][mask_df], 2),
    "Utilization S1 [MtCO₂/y]": np.round(util_full["S1"][mask_df], 2),
    "Sequestration S2 [MtCO₂/y]": np.round(seq_full["S2"][mask_df], 2),
    "Utilization S2 [MtCO₂/y]": np.round(util_full["S2"][mask_df], 2),
    "Sequestration S3 [MtCO₂/y]": np.round(seq_full["S3"][mask_df], 2),
    "Utilization S3 [MtCO₂/y]": np.round(util_full["S3"][mask_df], 2),
})

# print(df_ccus)

bins   = [2025, 2030, 2035, 2040, 2045, 2050]   # right-closed intervals
labels = [2030, 2035, 2040, 2045, 2050]

df = df_ccus.copy()

# 2) Assign each row to its 5-year period
df['Target_Year'] = pd.cut(
    df['Year'],
    bins=bins,
    labels=labels,
    right=True,            # interval (2025, 2030] includes 2030
    include_lowest=True
)

# 3) Aggregate by 5-year period – keep last row, then ×5 ---------------
df_last = (
    df.sort_values('Year')            # ensure chronological order
      .groupby('Target_Year', as_index=False)
      .last()                         # pick the row for the final year
      .drop(columns='Year')           # drop the *original* Year column
)

# multiply every numeric column EXCEPT Target_Year by 5
num_cols = [c for c in df_last.columns if c != 'Target_Year']
df_last[num_cols] = df_last[num_cols].mul(5)

# ----------------------------------------------------------------------
# build final df_ccus_5yr ----------------------------------------------
# ----------------------------------------------------------------------
df_ccus_5yr = (
    df_last
      .rename(columns={'Target_Year': 'Year'})   # now we create Year
      .astype({'Year': int})                     # Year as int
)

# reorder columns exactly as in the original (no duplicates now)
cols_order = ['Year'] + [c for c in df_ccus.columns if c != 'Year']
df_ccus_5yr = df_ccus_5yr[cols_order]

# change unit label MtCO₂/y → MtCO₂/5y
df_ccus_5yr = df_ccus_5yr.rename(
    columns=lambda c: c.replace('MtCO₂/y', 'MtCO₂/5y') if 'MtCO₂/y' in c else c
)

# print(df_ccus_5yr)

# ---------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------
# plt.figure(figsize=(10, 6))

# # Sequestration curves
# plt.plot(years_full, seq_full["S1"], color="#082150", linestyle="--", label="Sequestration in Low utilization scenario (S1)")
# plt.plot(years_full, seq_full["S2"], color="#1e4b9e", linestyle="-", label="Sequestration in Expected utilization scenario (S2)")
# plt.plot(years_full, seq_full["S3"], color="#4074d4", linestyle="--", label="Sequestration in High utilization scenario (S3)")

# # Capture curve
# plt.plot(years_full, capture_full, color="black", linewidth=2, label="Capture")

# # Shaded areas for S2 (base) only
# plt.fill_between(years_full, 0, seq_full["S2"], alpha=0.3,
#                  label="Sequestration area (S2)")
# plt.fill_between(years_full, seq_full["S2"], capture_full, alpha=0.3, color="#e2446e",
#                  label="Utilization area (S2)")

# # Grid: horizontal lines only
# plt.grid(axis='y')

# # Axis limits
# plt.xlim(2025.5, 2050.5)
# plt.ylim(0, 55)

# plt.title("CCUS Scenarios – Spain & Portugal (2026-2050)")
# plt.xlabel("Year")
# plt.ylabel("Mt CO₂ per year")
# plt.legend()
# plt.tight_layout()

# plt.savefig("sequestration_and_utilization_scenarios.png", dpi=300, bbox_inches='tight')
# plt.show()




####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################
# Data exportation
####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################

output_excel = "PV_complete_parameters_definition.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    node_emissions_5yr_df.to_excel(writer, sheet_name="Emissions projections", index=False)
    carbon_price_pv_df.to_excel(writer, sheet_name="Carbon allowances costs", index=False)
    pv_inv_onsh_pipe_df.to_excel(writer, sheet_name="Onshore inv. cost", index=False)
    pv_om_onsh_pipe_df.to_excel(writer, sheet_name="Onshore O&M cost", index=False)
    pv_inv_offsh_pipe_df.to_excel(writer, sheet_name="Offshore inv. cost", index=False)
    pv_om_offsh_pipe_df.to_excel(writer, sheet_name="Offshore O&M cost", index=False)
    pv_trading_prices_df.to_excel(writer, sheet_name="Trading prices", index=False)
    max_trade_flows_df.to_excel(writer, sheet_name="Trading rates", index=False)
    pipe_diam_features_df.to_excel(writer, sheet_name="Flow capacity", index=False)
    cand_pipe_features_gdf.to_excel(writer, sheet_name="Candidate pipelines features", index=False)
    frict_pres_drop_gdf.to_excel(writer, sheet_name="Friction pressure drop", index=False)
    ship_costs_pv_df.to_excel(writer, sheet_name="Ship costs", index=False)
    selling_revenues_pv_df.to_excel(writer, sheet_name="Selling revenues", index=False)
    compression_costs_pv_df.to_excel(writer, sheet_name="Compression costs", index=False)
    capture_injection_costs_pv_df.to_excel(writer, sheet_name="Capture & injection costs", index=False)
    df_ccus_5yr.to_excel(writer, sheet_name="CCUS targets", index=False)






"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STRUCTURING DATA IN A FRIENDLY WAY FOR PYOMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# ———————————————————————————————————————————————————————————————
# Type aliases
# ———————————————————————————————————————————————————————————————
Node     = int
PipeID   = str
Diameter = str          # inches as string
Year     = int
Country  = str

###############################################################################
# build_input_dict – main public helper
###############################################################################
def build_input_dict(
    cand_pipe_features_gdf:     gpd.GeoDataFrame,
    pipe_diam_features_df:      pd.DataFrame,
    complete_nodes_gdf:         gpd.GeoDataFrame,
    node_emissions_5yr_df:      pd.DataFrame,
    frict_pres_drop_gdf:        gpd.GeoDataFrame,
    df_ccus_5yr:                pd.DataFrame,
    years:                      Iterable[Year],
    # ─── economic data ───────────────────────────────────
    carbon_price_pv_df:        pd.DataFrame,
    pv_inv_onsh_pipe_df:        pd.DataFrame,
    pv_inv_offsh_pipe_df:       pd.DataFrame,
    pv_om_onsh_pipe_df:         pd.DataFrame,
    pv_om_offsh_pipe_df:        pd.DataFrame,
    ship_costs_pv_df:           pd.DataFrame,
    capture_injection_costs_pv_df: pd.DataFrame,
    compression_costs_pv_df:    pd.DataFrame,
    pv_trading_prices_df:       pd.DataFrame,
    max_trade_flows_df:         pd.DataFrame,
    selling_revenues_pv_df:     pd.DataFrame,
    ship_capacity:              float = 0.01,     
    # ─── scenarios ───────────────────────────────────────
    scenarios_cfg:              dict | None = None,
) -> Dict[str, Any]:
    """Return a pure-Python dict with *all* parameters for Pyomo."""

    # 1) Node sets --------------------------------------------------------
    nodes = complete_nodes_gdf.copy()

    # If the geometry is in WKT string format, convert it to shapely Point
    if isinstance(nodes.geometry.iloc[0], str):        # ← detect WKT string
        nodes["geometry"] = nodes["geometry"].apply(wkt.loads)

    # Ensure GeoDataFrame with correct CRS
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")

    nodes["Node identifier"] = nodes["Node identifier"].astype(int)

    E = nodes.loc[nodes["Node type"] == "E", "Node identifier"].tolist()
    S = nodes.loc[nodes["Node type"] == "S", "Node identifier"].tolist()
    A = nodes.loc[nodes["Node type"] == "A", "Node identifier"].tolist()
    M = nodes.loc[nodes["Node type"] == "M", "Node identifier"].tolist()
    K = nodes.loc[nodes["Node type"] == "K", "Node identifier"].tolist()

    N = sorted(nodes["Node identifier"].tolist())

    node_type  = nodes.set_index("Node identifier")["Node type"].to_dict()
    country_n  = nodes.set_index("Node identifier")["Country"].to_dict()

    node_by_country = {country: node_id
                   for node_id, country in country_n.items()
                   if country in ["France", "Morocco", "Argelia"]}

    # 2) Pipeline sets & maps --------------------------------------------
    cand_pipe_features_gdf["Pipe ID"] = cand_pipe_features_gdf["Pipe ID"].astype(str)

    P_on  = cand_pipe_features_gdf.loc[
        cand_pipe_features_gdf["Transport method"] == "Onshore", "Pipe ID"
    ].tolist()

    P_off = cand_pipe_features_gdf.loc[
        cand_pipe_features_gdf["Transport method"] == "Offshore", "Pipe ID"
    ].tolist()

    _pipes = cand_pipe_features_gdf.set_index("Pipe ID")
    start  = {p: int(conn[0]) for p, conn in _pipes["Node connection"].items()}
    end_   = {p: int(conn[1]) for p, conn in _pipes["Node connection"].items()}

    # Geometry dict
    pipe_geom = {p: geom for p, geom in zip(cand_pipe_features_gdf["Pipe ID"],
                                            cand_pipe_features_gdf.geometry)}

    IN, OUT = defaultdict(list), defaultdict(list)
    for p in P_on + P_off:
        i, j = start[p], end_[p]
        OUT[i].append(p)
        IN[j].append(p)
    for n in N:
        IN.setdefault(n, [])
        OUT.setdefault(n, [])

    # 3) Other sets -------------------------------------------------------
    D = pipe_diam_features_df["Diameter [inch]"].astype(str).tolist()
    T = list(years)

    # 4) Diameter-dependent capacity (qmax) -------------------------------
    flows_year = (
        pipe_diam_features_df.set_index("Diameter [inch]")["Assumed flow [MtCO2/5y]"]
        .astype(float)
    )
    qmax = {str(d): val for d, val in flows_year.items()}  # MtCO2/y

    # 5) Node attributes (static) ----------------------------------------
    _nodes = nodes.set_index("Node identifier")

    stage_n   = _nodes["Stage"].to_dict()
    cap_store = (_nodes.loc[_nodes["Node type"] == "S", "Total CO2 capacity [Mt]"]
                .fillna(0).to_dict())
    height    = pd.to_numeric(_nodes["Height"], errors="coerce").fillna(0).to_dict()

    # --- coordinates (lon, lat) -----------------------------------------
    # using the original GeoDataFrame ('nodes'), not '_nodes'
    coords = {
        int(row["Node identifier"]): (row.geometry.x, row.geometry.y)
        for _, row in nodes.iterrows()
    }

    # 6) Node emissions (dynamic) ----------------------------------------

    emis: Dict[Tuple[Node, Year], float] = {}

    node_emissions_5yr_df = node_emissions_5yr_df.copy()
    node_emissions_5yr_df.rename(
        columns={node_emissions_5yr_df.columns[0]: "Source Cluster"}, inplace=True
    )

    # Builging a robust mapping: Source Cluster 0 corresponds to the first
    # node_id of type 'E', 1 to the second, etc.
    cluster_to_node = {idx: node_id for idx, node_id in enumerate(sorted(E))}

    for _, row in node_emissions_5yr_df.iterrows():
        cluster_id = int(row["Source Cluster"])
        n = cluster_to_node.get(cluster_id)
        if n is None:
            # The cluster has no equivalent 'E' node; we ignore it
            continue

        for t in years:
            col = t if t in row.index else str(t)
            if col not in row.index:
                raise KeyError(f"Year column {t} not found in node_emissions_5yr_df")
            emis[(n, t)] = float(row[col])   # MtCO2 emitted in the 5-year period

    # 7) Carbon allowance price ------------------------------------------
    allow_price = {int(r["Year"]): float(r.iloc[1])
                   for _, r in carbon_price_pv_df.iterrows() if int(r["Year"]) in years}

    # 8) Pipe ΔP (elevation & friction) ----------------------------------
    dP_elev_high_raw = _pipes["High. elev. pres. drop [bar]"].astype(float)
    dP_elev_high = dP_elev_high_raw.clip(lower=0).to_dict()
    dP_elev_far  = _pipes["Fart. elev. pres. drop [bar]"].to_dict()

    frict_idx = frict_pres_drop_gdf.set_index("Pipeline identifier")
    frict_high: Dict[Tuple[PipeID, Diameter], float] = {}
    frict_far:  Dict[Tuple[PipeID, Diameter], float] = {}
    for p, row in frict_idx.iterrows():
        for d in D:
            tup = row[str(d)]
            if isinstance(tup, (list, tuple)) and len(tup) == 2:
                frict_high[(p, d)] = float(tup[0])
                frict_far[(p, d)]  = float(tup[1])

    # 9) Pipe categorical attributes -------------------------------------
    tmethod_p = _pipes["Transport method"].to_dict()
    stage_p   = _pipes["Stage"].to_dict()
    country_i = {p: country_n[start[p]] for p in start}
    country_j = {p: country_n[end_[p]]   for p in end_}
    n1_type   = {p: node_type[start[p]] for p in start}
    n2_type   = {p: node_type[end_[p]]   for p in end_}

    # 10) Sequestration / utilization targets -----------------------------
    seq_target  = {int(r["Year"]): float(r["Sequestration S2 [MtCO₂/5y]"]) for _, r in df_ccus_5yr.iterrows()}
    util_target = {int(r["Year"]): float(r["Utilization S2 [MtCO₂/5y]"])   for _, r in df_ccus_5yr.iterrows()}

    # 11) Pipe costs (diameter-year) --------------------------------------
    def _df_to_costdict(df: pd.DataFrame) -> Dict[Tuple[Diameter, Year], float]:
        tmp = df.copy()
        tmp["Diameter [inch]"] = tmp["Diameter [inch]"].astype(str)
        d: Dict[Tuple[Diameter, Year], float] = {}
        for _, r in tmp.iterrows():
            d_in = str(r["Diameter [inch]"])
            for t in years:
                col = t if t in tmp.columns else str(t)
                if col in tmp.columns:
                    d[(d_in, t)] = float(r[col])
        return d
    cins_on  = _df_to_costdict(pv_inv_onsh_pipe_df)
    cins_off = _df_to_costdict(pv_inv_offsh_pipe_df)
    cop_on   = _df_to_costdict(pv_om_onsh_pipe_df)
    cop_off  = _df_to_costdict(pv_om_offsh_pipe_df)

    # 12) Shipping costs --------------------------------------------------
    ship_fixed_cost = {int(r["Year"]): float(r["Fixed_cost [M€/cycle]"])
                       for _, r in ship_costs_pv_df.iterrows() if int(r["Year"]) in years}
    ship_fuel_cost  = {int(r["Year"]): float(r["Fuel_cost [M€/km/cycle]"])
                       for _, r in ship_costs_pv_df.iterrows() if int(r["Year"]) in years}

    # 13) Capture & injection costs ---------------------------------------
    capture_cost   = {int(r["Year"]): float(r["Capture cost [M€/MtCO2]"])
                      for _, r in capture_injection_costs_pv_df.iterrows() if int(r["Year"]) in years}
    injection_cost = {int(r["Year"]): float(r["Injection cost [M€/MtCO2]"])
                      for _, r in capture_injection_costs_pv_df.iterrows() if int(r["Year"]) in years}

    # 14) Compression costs ----------------------------------------------
    comp_df = compression_costs_pv_df.copy()
    def _series_to_dict(sub):  # helper
        col = [c for c in comp_df.columns if sub in c][0]
        return {int(r["Year"]): float(r[col]) for _, r in comp_df.iterrows() if int(r["Year"]) in years}
    cins_init  = _series_to_dict("Init. compr. CAPEX [M€/MtCO2]")
    cop_init   = _series_to_dict("Init. compr. OPEX [M€/MtCO2]")
    cel_init   = _series_to_dict("Init. compr. electr. [M€/MtCO2]")
    cins_boost = _series_to_dict("Boost. st. CAPEX [M€/MtCO2]")
    cop_boost  = _series_to_dict("Boost. st. OPEX [M€/MtCO2]")
    cel_boost  = _series_to_dict("Boost. st. electr. [M€/MtCO2]")

    # 15) Selling revenue (industrial use) --------------------------------
    selling_revenue = {int(r["Year"]): float(r["Selling revenue [M€/MtCO2]"])
                       for _, r in selling_revenues_pv_df.iterrows() if int(r["Year"]) in years}

    # print(_pipes.head())

    # 16) Division between P1/P2: P2 if the pipeline connects with a K node ---
    def touches_K(p: str) -> bool:
        return (start[p] in K) or (end_[p] in K)
    P2_on  = [p for p in P_on  if touches_K(p)]
    P1_on  = [p for p in P_on  if p not in P2_on]
    P2_off = [p for p in P_off if touches_K(p)]
    P1_off = [p for p in P_off if p not in P2_off]

    # --- Scenarios from scenarios.py ---
    cfg = dict(scenarios_cfg or {})
    cfg.setdefault("nodes_uncertain", list(K))
    cfg.setdefault("time_steps", list(years))

    W, prob_dict, gcap_df = get_scenarios(cfg)   # W: list[str], prob: dict[str,float]
    # gcap_df: index (scenario,node,t), col "capacity"

    # DataFrame → dict plano (k,t,w) → float
    g_cap = {}
    series = gcap_df["capacity"] if "capacity" in gcap_df.columns else gcap_df.squeeze()
    for (w, k, t), val in series.items():
        if (k in K) and (t in years) and (w in W):
            g_cap[(int(k), int(t), str(w))] = float(val)

    # 17) Assemble dict ----------------------------------------------------
    data: Dict[str, Any] = {
        # Sets
        "E": E, "S": S, "A": A, "M": M, "K": K, "N": N,
        "P_on": P_on, "P_off": P_off, "D": D, "T": T,
        "P1_on": P1_on, "P2_on": P2_on, "P1_off": P1_off, "P2_off": P2_off,

        # Incidence
        "IN": dict(IN), "OUT": dict(OUT),

        # Node static
        "store_cap": cap_store, "height": height, "coords": coords, "country_n": country_n, "stage_n": stage_n,

        # Node dynamic
        "emission": emis,

        # Pipe static
        "start": start, "end": end_, "L": _pipes["Longitude [km]"].to_dict(), "Lh": _pipes["Distance until heighest point [km]"].to_dict(),
        "dP_elev_high": dP_elev_high, "dP_elev_far": dP_elev_far,
        "dP_frict_high": frict_high,  "dP_frict_far": frict_far,
        "pen_city": _pipes["City detour penalty factor"].to_dict(),
        "pen_slope": _pipes["Slope penalty factor"].to_dict(),
        "tmethod_p": tmethod_p, "stage_p": stage_p,
        "country_i": country_i, "country_j": country_j,
        "n1_type": n1_type, "n2_type": n2_type, "pipe_geom": pipe_geom,

        # Diameter capacity
        "qmax": qmax,

        # Pressure / flow constants
        "p_emit": 150.0, "delta_p_boost": 50.0, "p_min": 100.0,
        "M_flow": 150.0, "M_press": 250.0, "M_eur": 1500.0,

        # Targets
        "seq_target": seq_target, "util_target": util_target,

        # Prices & costs
        "allow_price": allow_price,
        "cins_on": cins_on, "cins_off": cins_off,
        "cop_on": cop_on,   "cop_off": cop_off,
        "ship_fixed_cost": ship_fixed_cost, "ship_fuel_cost": ship_fuel_cost,
        "capture_cost": capture_cost, "injection_cost": injection_cost,

        # Compression
        "cins_init": cins_init, "cop_init": cop_init, "cel_init": cel_init,
        "cins_boost": cins_boost, "cop_boost": cop_boost, "cel_boost": cel_boost,

        # Selling to industry
        "selling_revenue": selling_revenue,

        # Scenarios and uncertainty
        "scenarios": W,
        "scenario_prob": prob_dict,
        "g_cap": g_cap,

        # Constant
        "ship_capacity": ship_capacity,
        "trading_prices": trading_prices_5yr_df,
        "pv_trading_prices": pv_trading_prices_df,
    }


    return data


# YEARS = range(2026, 2051)

# try:
#     DATA = build_input_dict(
#         cand_pipe_features_gdf, pipe_diam_features_df, complete_nodes_gdf,
#         node_emissions_df, frict_pres_drop_gdf, df_ccus, YEARS,
#         predicted_values_df, onshore_pipe_ann_capex_df, offshore_pipe_ann_capex_df,
#         onshore_pipe_om_cost_df, offshore_pipe_om_cost_df,
#         ship_costs_df, capture_injection_costs_df, compression_costs_df,
#         trading_prices_df, max_trade_flows_df, selling_revenues_df,
#     )

#     print(data)

# except NameError as exc:
# #     missing = [v for v in (
#         "cand_pipe_features_gdf", "pipe_diam_features_df", "complete_nodes_gdf",
#         "node_emissions_df", "frict_pres_drop_gdf", "df_ccus", "predicted_values_df",
#         "onshore_pipe_ann_capex_df", "offshore_pipe_ann_capex_df",
#         "onshore_pipe_om_cost_df", "offshore_pipe_om_cost_df",
#         "ship_costs_df", "capture_injection_costs_df", "compression_costs_df",
#         "trading_prices_df", "max_trade_flows_df", "selling_revenues_df"
#     ) if v not in globals()]
#     raise RuntimeError(
#         f"⚠️  Faltan objetos para construir DATA: {', '.join(missing)}"
#     ) from exc




# ###############################################################################
# # Quick self-test
# ###############################################################################
# if __name__ == "__main__":
# REQUIRED = [
#     "cand_pipe_features_gdf", "pipe_diam_features_df", "complete_nodes_gdf",
#     "node_emissions_df", "frict_pres_drop_gdf", "df_ccus",
#     "predicted_values_df", "onshore_pipe_ann_capex_df", "offshore_pipe_ann_capex_df",
#     "onshore_pipe_om_cost_df", "offshore_pipe_om_cost_df",
#     "ship_costs_df", "capture_injection_costs_df", "compression_costs_df",
#     "trading_prices_df", "max_trade_flows_df", "selling_revenues_df",
# ]
# missing = [v for v in REQUIRED if v not in globals()]
# if missing:
#     raise RuntimeError("⚠️  Load these objects first: " + ", ".join(missing))

years = [2030, 2035, 2040, 2045, 2050]

DATA = build_input_dict(
    cand_pipe_features_gdf, pipe_diam_features_df, complete_nodes_gdf, node_emissions_5yr_df,
    frict_pres_drop_gdf, df_ccus_5yr, years,
    carbon_price_pv_df, pv_inv_onsh_pipe_df, pv_inv_offsh_pipe_df,
    pv_om_onsh_pipe_df, pv_om_offsh_pipe_df,
    ship_costs_pv_df, capture_injection_costs_pv_df, compression_costs_pv_df,
    pv_trading_prices_df, max_trade_flows_df, selling_revenues_pv_df,
)

# print("W =", DATA["scenarios"])
# print("prob =", DATA["scenario_prob"])
# print("g_cap sample =", list(DATA["g_cap"].items())[:5])

# print(DATA)
# print("✓ keys OK – selling revenue 2030:", DATA["selling_revenue"].get(2030, "NA"))

# print(complete_nodes_gdf)
