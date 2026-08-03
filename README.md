# Iberian CO<sub>2</sub> Transport Network

This repository contains the code used to conduct a study of a cost-optimal CO<sub>2</sub> transport network for CCUS in the Iberian Peninsula. To this end, a stochastic mixed-integer program has been developed that accounts for uncertainty in CO<sub>2</sub> demand for industrial utilization, identifying both the optimal network layout and its phased development over time. Three scenarios with different utilization forecasts are defined, and the impact of uncertainty on the net cost to the transport network operator is analyzed. The proposed approach explicitly incorporates pipeline pressure constraints, including friction and elevation losses, enabling more realistic and efficient infrastructure planning. The results help identify robust configurations under uncertainty and offer valuable insights for the development of the regional CCUS infrastructure. The study, developed in Python and optimized using Gurobi, uses source data from Excel sheets to generate new Excel sheets with results. Several additional Python programs are included to obtain some relevant figures.

## 1) Repository structure

- **data/** contains, one the one hand, the .xlsx files with the techno-economic parameters used for the model, and on the other hand, the .json files with the geography of the studied region.
- **results_processing/** contains the .py files required to generate different plots to analyze the results provided by the model execution.
- **iberian_co2_network/** contains the .py files required to execute the optimization models (the developed and the traditional one).
- **releases** contains the SRTM (Shuttle Radar Topography Mission) file, which includes an accurate elevation profile of the Iberian Peninsula, also required to obtain part of the data.

## 2) Requirements and environment

The project has been developed using:

- Python **v. 3.13**.
- **Libraries:** Pyomo, pandas, geopandas, numpy, matplotlib, CoolProp, shapely, collections, math, ast, future, typing, sys, os, pathlib, rasterio, pyproj, scipy.
- Solver: Gurobi **v. 11.0.3**.
- Computer operating system: **Windows 11**.

## 3) Installation

1. Create a virtual environment
2. Install all the required libraries
3. Run the model

## 4) Data

The optimization model creates a data dictionary, obtained from raw data and its processing. The raw data is distributed across several Excel documents, all of them contained in the 'data' folder of the repository, each one including different information:

- **Spanish and Portuguese emissions and sinks database.xlsx**: Contains information regarding:
  - The coordinates of all the emitters, sinks and utilization nodes
  - The emissions released by the emitters and the capacities of the sinks.
- **iberian_co2_network_data.xlsx**: Contains:
  - The coordinates of all the selected nodes, their type and their total emissions/capacity.
  - A list of all the original emitters and the cluster they belong to.
  - The selected set of candidate pipelines, indicating, among other parameters, which nodes do they connect, their length, the environment through which they flow (onshore/offshore), the maximum elevation between nodes, the amount of cities crossed by the pipelines and their geometry.
- **model_parameters.xlsx**: Includes the historical information required to calculate the average percentage increase in the HICP, which has then been used to calculate the updated project costs.
- **PV_complete_parameters_definition.xlsx**: Contains:
  - The emissions projection at each of the source nodes for the entire time horizon.
  - The updated pipeline CAPEX and OPEX as a function of the diameter for the entire time horizon.
- The rest of the relevant data can be found directly in the model code, in the **'data.py' file**. This includes CAPEX and OPEX for initial compressors and boosters, fixed and variable costs for transport by ship, electricity prices,  emission costs, and utilization node capacities, among others.
- The .json files contain the **geometry of country borders**, which is necessary for creating plots related to the project.
- The **merged_srtm.tif file**, available in the 'Releases' folder, includes the precise topography of the study region. The data were extracted from a NASA experiment that measured the elevation of the terrain over almost the entire planet, the Shuttle Radar Topography Mission. These data were used to find the difference in height between the nodes connected by a candidate pipeline, as well as the highest point among the former.

## 5) How to run

The files contained in the 'iberian_co2_network' folder correspond to the optimization models:

- On the one hand, the **developed model** is the one that explicitly accounts for the calculation of pressure drops through the transport network. It uses pressure-related constraints to calculate whether the pipeline requires boosters or not. In addition, it allows a potentially more expensive route to be ruled out due to these constraints and replaced by another that, despite being longer, does not require investment in additional boosters.

- On the other hand, the **traditional model** is identical to the developed one, but eliminates all types of pressure constraints. Booster costs are calculated outside the model, assuming that a booster is installed every 150 km, in accordance with the simplification assumed in most of the current literature.

### Developed model

The required files to run the developed model are the following:

- **data.py**: The file loads all the raw data from the aforementioned Excel files, and also loads additional data to obtain data that does not come from the files. After processing them, it creates the data dictionary required by the model.
- **scenarios.py**: Defines the three scenarios of the model for stochastic optimization: LUS (Low Utilization Scenario), EUS (Expected Utilization Scenario), and HUS (High Utilization Scenario).
- **developed.main.py**: It is the program that calls all the others to generate the data dictionary, solve the model, generate the results, and obtain the figures with a single command (running the file itself).
- **developed.model.py**: Contains the definition of all variables, parameters, constraints, and objective function of the optimization model.
- **developed.solution.py**: Processes the optimization results, generating an Excel file with the obtained output. The expected results are detailed in the following section.
- **developed.plots.py**: Generates plots representing the gradual deployment of the transport network, which can also be obtained from the Excel sheets resulting from the model execution.
- **init.py**, **config.py** and **utils.py**: Auxiliary files to allow the processing and correct running of the model.

### Traditional model

The files needed to run the traditional model are essentially the same, replacing the files 'developed_main.py', 'developed_model.py', 'developed_plots.py' and 'developed_solution.py' with 'traditional_main.py', 'traditional_model.py', 'traditional_plots.py' and 'traditional_solution.py', respectively. The main differences between these files and those of the developed model are that the costs of the boosters are calculated outside the optimization, and that the plots do not include pressure gradients, as these are not computed. The 'data.py', 'scenarios.py', 'init.py', 'config.py' and 'utils.py' are also required to run the traditional model, so it is recommended to save all of the files together in the same folder, both for the developed and the traditional models.

## 6) Expected output

The optimization model must be run for each value of theta. The resulting files are an Excel file and 15 figures representing the development of the CO<sub>2</sub> transport network: 

- The resulting Excel file, named "stochastic_results_theta_X.XX" or "stochastic_traditional_results_theta_X.XX" (depending of whether the developed or the traditional model has been run) is stored in a folder called “results_summaries” and contains 13 sheets of results. For each scenario (LUS, EUS and HUS), there are four sheets:
  - The first sheet contains the finally selected candidate pipes, their diameters, the pressure gradient through them, and the flow transported in each time step.
  - The second one includes the remaining capacity in all the sinks after each time step.
  - The thrid sheet shows the amount of CO<sub>2</sub> utilized along each time step in each one of the utilization nodes.
  - The forth one includes the breakdown of all costs considered by the model, as well as the profits from the sale of CO<sub>2</sub> for its industrial utilization.
  - The last sheet is common for all the scenarios and includes a comparison between the length of the transport network deployed in each scenario for each time step of the model.
 - The 15 figures obtained are named "developed_solution_XXX_theta_Y.YY_ZZZZ.png", where XXX is the scenario, Y.YY is the value of theta and ZZZZ is the time step of the plot. These 15 figures are saved in a folder called "output_developed_plots_theta_Y.YY", which in turn is stored in the folder "output_developed_plots". In the case of the traditional model, the 15 figures obtained are named "stochastic_traditional_solution_XXX_theta_Y.YY_ZZZZ.png". These 15 figures are saved in a folder called "output_stochastic_traditional_theta_Y.YY", which in turn is stored in the folder "output_traditional_plots".

## 7) Results post-processing and figures generation

The files contained in the folder 'results_processing' allow one to generate insightful figures from the results, which need to have been previously obtained for the desired value of theta.

- **plots_from_results.py**: The file generates 15 figures (3 scenarios x 5 time steps) representing the temporal deployment of the CO<sub>2</sub> transport network according to the scenario. The results obtained in the time steps 2030, 2035, 2040, 2045, and 2050 are shown for the LUS, EUS, and HUS scenarios. To change the value of theta, find the line “THETA = 1.00 # <-- Change this value manually each time you run the script” and modify the value.
- **scenario_plots_together.py**: Generates the same results as “plots_from_results.py,” but includes the entire time development for that scenario in a single figure (includes the plots of the 5 time steps in a single figure). Therefore, it generates 3 figures, one for each scenario. As in the previous case, to change the value of theta, find the line “THETA = 1.00 # <-- Change this value manually each time you run the script” and modify the value.
- **comparison_with_traditional_model_plots.py**: It generates, for the desired theta value and scenario (to be defined manually in the code itself), a figure comparing the transport network deployment obtained between the developed model and the traditional model for the time steps 2030, 2040, and 2050.
- **system_cost_per_ton_plot.py**: Generates, for a given scenario (to be selected manually in the code itself), a figure with the Levelized Cost of CO<sub>2</sub> Transport based on the value of theta. The total cost is displayed based on the contributions of each process (capture, pipeline transport, boosting, injection, etc.).
- **system_operator_costs_plot.py**: For a given scenario (to be selected manually in the code itself), it generates a figure containing, for each value of theta, the total deployed cost of the CO<sub>2</sub> network. It also shows the profits generated by the sale of CO<sub>2</sub> for industrial use, as well as the net cost of the system, resulting from subtracting the profits from the total cost.
- **theta_comparison_plots.py**: Given a scenario, a time step, and two values of theta (to be defined manually in the code), it generates a figure with two plots of the transport network deployment obtained for each selected value of theta.
- **traditional_vs_developed_costs_plot.py**: It generates, for the desired theta value and scenario (to be defined manually in the code itself), a figure comparing the transport network costs between the developed model and the traditional model.
