import cantera as ct
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PCAfold import preprocess
from PCAfold import analysis
from PCAfold import __version__ as PCAfold_version

import pickle

data_state_space = pd.read_csv("../data/freely-propagating-flame-oxyfuel-state-space.csv", header=None)
data_state_sources = pd.read_csv("../data/freely-propagating-flame-oxyfuel-state-space-sources.csv", header=None)
data_mf = pd.read_csv("../data/freely-propagating-flame-oxyfuel-mixture-fraction.csv", header=None)
# data_names = ("freely-propagating-flame-oxyfuel-state-space-names.csv")

file = open("../data/freely-propagating-flame-oxyfuel-state-space-names.csv", "r")
data_names = list(csv.reader(file, delimiter=","))
file.close()
# print("Data names:", data_names)



### Definition of the progress variable ###
# Pierce & Moin 
species_list_PM = ['H2O', 'CO2']
weight_list_PM = [1.0, 1.0]

# Ihme & Pitsch
species_list_IP = ['H2O', 'CO2', 'CO', 'H2']
weight_list_IP = [1.0, 1.0, 1.0, 1.0]

# Fiorina et al.
species_list_F = ['CO2', 'CO']
weight_list_F = [1.0, 1.0]

# Progress_variables = {
#     "Pierce_Moin": {
#         "species": ["H2O","CO2"],
#         "weights": [1,1]
#     },

#     "Ihme_Pitsch": {
#         "species": ["H2O","CO2","CO","H2"],
#         "weights": [1,1,1,1]
#     },

#     "Fiorina": {
#         "species": ["CO2","CO"],
#         "weights": [1,1]
#     }
# }
mf = data_mf.iloc[:,0]
T = data_state_space.iloc[:,0]

##############################    CALCULATION     #############################################
###############################################################################################
# Calcul of the PV_PM
PV_PM = pd.Series(0.0, index = data_state_space.index)
for index_species, species in enumerate(species_list_PM) : # iterate species_list
        for index, value in enumerate(data_names):
            if species != value[0] :
                pass
            else : 
                selected_column = data_state_space.iloc[:,index]
                PV_PM = PV_PM + selected_column*weight_list_PM[index_species]

# Calcul of the PV source term PM
PV_ST_PM = pd.Series(0.0, index = data_state_sources.index)
for index_species, species in enumerate(species_list_PM) :
    for index, value in enumerate(data_names):
                if species != value[0] :
                    pass
                else :
                    selected_column = data_state_sources.iloc[:,index]
                    PV_ST_PM = PV_ST_PM + selected_column*weight_list_PM[index_species]


###############################################################################################
# Calcul of the PV_IP
PV_IP = pd.Series(0.0, index = data_state_space.index)
for index_species, species in enumerate(species_list_IP)
        for index, value in enumerate(data_names):
            if species != value[0] :
                pass
            else : 
                selected_column = data_state_space.iloc[:,index]
                PV_IP = PV_IP + selected_column*weight_list_IP[index_species]

# Calcul of the PV source term IP
PV_ST_IP = pd.Series(0.0, index = data_state_sources.index)
for index_species, species in enumerate(species_list_IP) :
    for index, value in enumerate(data_names):
                if species != value[0] :
                    pass
                else :
                    selected_column = data_state_sources.iloc[:,index]
                    PV_ST_IP = PV_ST_IP + selected_column*weight_list_IP[index_species]


###############################################################################################
# Calcul of the PV_F
PV_F = pd.Series(0.0, index = data_state_space.index)
for index_species, species in enumerate(species_list_F)
        for index, value in enumerate(data_names):
            if species != value[0] :
                pass
            else : 
                selected_column = data_state_space.iloc[:,index]
                PV_F = PV_F + selected_column*weight_list_F[index_species]

# Calcul of the PV source term IP
PV_ST_F = pd.Series(0.0, index = data_state_sources.index)
for index_species, species in enumerate(species_list_F) :
    for index, value in enumerate(data_names):
                if species != value[0] :
                    pass
                else :
                    selected_column = data_state_sources.iloc[:,index]
                    PV_ST_F = PV_ST_F + selected_column*weight_list_F[index_species]

#################################

mf = mf.to_numpy().reshape(-1,1)
VD_target_variables_names = ['T','H','O2','OH','O','H2','H2O','CO','CO2', 'PV_ST']
position_VD_target_variables = [0,1,2,3,4,5,6,9,10]
VD_target_variables = data_state_space.iloc[:, position_VD_target_variables].to_numpy()
bandwidth_values = np.logspace(-6,2,200)

# Hyper-parameters of the cost function
power = 4
vertical_shift = 1
penalty_function = 'log-sigma-over-peak'

fontsize = 20
fontsize_axes = 16

##############################  Pierce & Moin ###################################################################
depvars_PM = np.column_stack((VD_target_variables,PV_ST_PM))
PV_PM = PV_PM.to_numpy().reshape(-1,1)
indepvars_PM = np.hstack((mf, PV_PM))


VarianceData_PM = analysis.compute_normalized_variance(
    indepvars_PM,
    depvars_PM,
    depvar_names=VD_target_variables_names,
    scale_unit_box=True,
    bandwidth_values=bandwidth_values
)

pickle.dump(VarianceData_PM, open("VarianceData_PM.pkl", "wb"))



##############Analysis#############
# VarianceData_PM = pickle.load(open("VarianceData_PM.pkl", "rb"))

# D_hat_PM = analysis.normalized_variance_derivative(VarianceData_PM)
# sigmas_PM = VarianceData_PM.bandwidth_values

# costs_PM = analysis.cost_function_normalized_variance_derivative(
#     VarianceData_PM,
#     penalty_function=penalty_function,
#     norm=None,
#     power=power,
#     vertical_shift=vertical_shift,
#     integrate_to_peak=False
# )

# L2_norm_cost_PM = np.linalg.norm(costs_PM) / len(costs_PM)

# print("Cost of Pierce & Moin:", L2_norm_cost_PM)
# for name, cost in zip(VD_target_variables_names, costs_PM):
#     print(name, cost)

# plt.figure()
# plt.semilogx(sigmas_PM[1:-1], D_hat_PM[0]['PV_ST'], lw=4, c='k', zorder=10)
# plt.xticks([10**-6, 10**-4, 10**-2, 10**0, 10**2], fontsize=fontsize_axes)
# plt.yticks([0,0.5,1], ['0', '0.5', '1'], fontsize=fontsize_axes)
# plt.xlim([10**-6,10**2])
# plt.xlabel('$\sigma$ [$-$]', fontsize=14)
# plt.ylabel('$\\hat{\\mathcal{D}}(\\sigma)$ for $\\dot{\\omega}_{PV}/\\rho$ [$-$]',fontsize=14)
# plt.title('Pierce & Moin PV\n'+ '$\mathcal{L}_{\dot{\omega}_{PV} / \\rho} = $' + str(round(costs_PM[-1], 1)))
# plt.savefig("D_hat_PM_PV_source.png", dpi=300, bbox_inches="tight")
# plt.grid(True, which="both")
# plt.show()




################################## Ihme & Pitsch ###############################################################
depvars_IP = np.column_stack((VD_target_variables,PV_ST_IP))
PV_IP = PV_IP.to_numpy().reshape(-1,1)
indepvars_IP = np.hstack((mf, PV_IP))


VarianceData_IP = analysis.compute_normalized_variance(
    indepvars_IP,
    depvars_IP,
    depvar_names=VD_target_variables_names,
    scale_unit_box=True,
    bandwidth_values=bandwidth_values
)


pickle.dump(VarianceData_IP, open("VarianceData_IP.pkl", "wb"))




##############Analysis#############
# VarianceData_IP = pickle.load(open("VarianceData_IP.pkl", "rb"))

# D_hat_IP = analysis.normalized_variance_derivative(VarianceData_IP)
# sigmas_IP = VarianceData_IP.bandwidth_values


# costs_IP = analysis.cost_function_normalized_variance_derivative(
#     VarianceData_IP,
#     penalty_function=penalty_function,
#     norm=None,
#     power=power,
#     vertical_shift=vertical_shift,
#     integrate_to_peak=False
# )

# L2_norm_cost_IP = np.linalg.norm(costs_IP) / len(costs_IP)

# print("Cost of Ihme & Pitsch:", L2_norm_cost_IP)
# for name, cost in zip(VD_target_variables_names, costs_IP):
#     print(name, cost)

# plt.figure()
# plt.semilogx(sigmas_IP[1:-1], D_hat_IP[0]['PV_ST'], lw=4, c='k', zorder=10)
# plt.xticks([10**-6, 10**-4, 10**-2, 10**0, 10**2], fontsize=fontsize_axes)
# plt.yticks([0,0.5,1], ['0', '0.5', '1'], fontsize=fontsize_axes)
# plt.xlim([10**-6,10**2])
# plt.xlabel('$\sigma$ [$-$]', fontsize=14)
# plt.ylabel('$\\hat{\\mathcal{D}}(\\sigma)$ for $\\dot{\\omega}_{PV}/\\rho$ [$-$]',fontsize=14)
# plt.title('Ihme & Pitsch PV\n'+ '$\mathcal{L}_{\dot{\omega}_{PV} / \\rho} = $' + str(round(costs_IP[-1], 1)))
# plt.savefig("D_hat_IP_PV_source.png", dpi=300, bbox_inches="tight")
# plt.grid(True, which="both")
# plt.show()




##################################  Fiorina & al. ###############################################################
depvars_F = np.column_stack((VD_target_variables,PV_ST_F))
PV_F = PV_F.to_numpy().reshape(-1,1)
indepvars_F = np.hstack((mf, PV_F))

VarianceData_F = analysis.compute_normalized_variance(
    indepvars_F,
    depvars_F,
    depvar_names=VD_target_variables_names,
    scale_unit_box=True,
    bandwidth_values=bandwidth_values
)

pickle.dump(VarianceData_F, open("VarianceData_F.pkl", "wb"))

##############Analysis#############
# VarianceData_F = pickle.load(open("VarianceData_F.pkl", "rb"))

# D_hat_F = analysis.normalized_variance_derivative(VarianceData_F)
# sigmas_F = VarianceData_F.bandwidth_values


# costs_F = analysis.cost_function_normalized_variance_derivative(
#     VarianceData_F,
#     penalty_function=penalty_function,
#     norm=None,
#     power=power,
#     vertical_shift=vertical_shift,
#     integrate_to_peak=False
# )

# L2_norm_cost_F = np.linalg.norm(costs_F) / len(costs_F)

# print("Cost of Fiorina et al:", L2_norm_cost_F)
# for name, cost in zip(VD_target_variables_names, costs_F):
#     print(name, cost)

# plt.figure()
# plt.semilogx(sigmas_F[1:-1], D_hat_F[0]['PV_ST'], lw=4, c='k', zorder=10)
# plt.xticks([10**-6, 10**-4, 10**-2, 10**0, 10**2], fontsize=fontsize_axes)
# plt.yticks([0,0.5,1], ['0', '0.5', '1'], fontsize=fontsize_axes)
# plt.xlim([10**-6,10**2])
# plt.xlabel('$\sigma$ [$-$]', fontsize=14)
# plt.ylabel('$\\hat{\\mathcal{D}}(\\sigma)$ for $\\dot{\\omega}_{PV}/\\rho$ [$-$]',fontsize=14)
# plt.title('Fiorina et al. PV\n'+ '$\mathcal{L}_{\dot{\omega}_{PV} / \\rho} = $' + str(round(costs_F[-1], 1)))
# plt.savefig("D_hat_F_PV_source.png", dpi=300, bbox_inches="tight")
# plt.grid(True, which="both")
# plt.show()


################################## Comparison of the progress variables #############################################################
# names_neat = ['$T$', '$Y_{H}$', '$Y_{O_2}$', '$Y_{OH}$', '$Y_{O}$', '$Y_{H_2}$', '$Y_{H_2O}$', '$Y_{CO}$', '$Y_{CO_2}$', '$\\frac{\dot{\omega}_{PV}}{\\rho}$']
# figure = plt.figure(figsize=(8, 4))
# spec = figure.add_gridspec(ncols=1, nrows=1, width_ratios=[1], height_ratios=[1])

# x_range = [i for i in range(0,len(names_neat))]
# fontsize_axes = 12
# fontsize = 14
# s = 2

# figure_a = figure.add_subplot(spec[0,0])
# plt.scatter(x_range, costs_PM, c='#ffafaf', s=60, zorder=30, label='PV $= Y_{CO_2} + Y_{H_2O} $')
# plt.scatter(x_range, costs_IP, c='#ff5d5d', s=100, zorder=40, label='PV $ = Y_{CO_2} + Y_{H_2O} + Y_{CO} + Y_{H_2}$')
# plt.scatter(x_range, costs_F, c='#d70040', s=140, zorder=20, label='PV $ = Y_{CO_2} + Y_{CO}$')
# # plt.scatter(x_range, costs_optimized, c='k', s=220, zorder=10, label='Optimized PV')
# figure_a.xaxis.grid(True, alpha=1, zorder=1)
# figure_a.tick_params(which='minor', labelsize=fontsize_axes)
# figure_a.tick_params(which='major', labelsize=fontsize_axes)
# plt.xticks(x_range, names_neat, fontsize=fontsize)
# plt.ylabel('$\mathcal{L}_i$ [$-$]', fontsize=fontsize)
# plt.yscale('log')
# plt.legend(frameon=False, fontsize=fontsize, ncol=2, bbox_to_anchor=(0.91,1.3))
# plt.savefig("PV_comparison.png", dpi=500, bbox_inches='tight')





# # Plot PV vs mixture fraction with color bar temperature
# plt.figure()
# scat = plt.scatter(mf, PV, c=T, s=2)
# plt.xlabel('Mixture fraction $f$')
# plt.ylabel('Progress Variable')
# plt.title('Progress Variable Fiorina et al.')
# cbar = plt.colorbar(scat, aspect=15)
# cbar.set_label('Temperature (K)', rotation=90)
# plt.show()

# # print(PV_ST.head())
# # Plot PV vs mixture fraction with color bar PV source term
# plt.figure()
# scat = plt.scatter(mf, PV, c=PV_ST, s=2)
# plt.xlabel('Mixture fraction $f$')
# plt.ylabel('Progress Variable')
# plt.title('Progress Variable Fiorina et al.')
# cbar = plt.colorbar(scat, aspect=15)
# cbar.set_label('$\dot{\omega}_{PV}/\\rho$ [$1/s$]', rotation=90)
# plt.show()






















#     # Create PV_source_term = series of 0
#     PV_source_term = pd.Series(0.0, index = data.index)
#     for i, species in enumerate(species_list):
#         component = weight_list[i]*data[f"Molecular weight of {species} [kg/kmol]"]*data[f"Net production rate of {species} [kmol/m3/s]"]  # weight for each species * molecular weight of each species * net production rate of every species
#         PV_source_term = PV_source_term + component

#     return PV, PV_source_term


# # Creation of an empty database
# results = pd.DataFrame()

# for name, definition in Progress_variables.items():
#     # PV calculation
#     species_list = definition["species"]
#     weight_list = definition["weights"]
#     PV,PV_source_term = Progress_Variable(species_list, weight_list, data)      # Call the function

#     temporary_dataframe = pd.DataFrame({"Equivalence ratio" : data["Equivalence ratio"], "Progress variable": PV})

    
    

# #print(results.head())
# #print(results.columns)
# data = pd.concat([data, results], axis=1)    
# #data.to_csv("Thermochemical_data_with_PV.csv", index=False)

# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------

# ### Plot temperature, species mass fractions and Progress variable Source term versus PV ###
# # Separate the dataframe into several parts according to their equivalence ratio 
# # Select all the rows for which the equivalence ratio in the same and save it as a new data frame  

# groups = data.groupby("Equivalence ratio")


# #plt.figure()
# #for name, group in groups :     # iteration over each group of equivalence ratio : name = equivalence ratio, group = dataframe containing the rows for that equivalence ratio
#     # Plot temperature versus PV
#     #plt.plot(group["PV_Pierce_Moin"], group["Temperature (K)"], label=name)
#     #plt.xlabel("Progress Variable of Pierce & Moin")
#     #plt.ylabel("Temperature (K)")
#     #plt.title("Temperature (K) vs PV of Pierce & Moin")
#     #plt.legend()
# #plt.legend(title="Equivalence ratio")
# #plt.show()


# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------


# PV_columns = {
#     "Progress Variable of Pierce & Moin": "PV_Pierce_Moin",
#     "Progress Variable of Ihme and Pitsch": "PV_Ihme_Pitsch",
#     "Progress Variable of Fiorina": "PV_Fiorina"
# }

# # Plot the Temperature (K) vs PV
# for legend_PV, column_PV in PV_columns.items():
#     plt.figure()
#     for name, group in groups :     # iteration over each group of equivalence ratio : name = equivalence ratio, group = dataframe containing the rows for that equivalence ratio
#         #Plot temperature versus PV
#         plt.plot(group[column_PV], group["Temperature (K)"], label=f"$\phi$ = {name}")

#     plt.xlabel(f"{legend_PV}")
#     plt.ylabel("Temperature (K)")
#     plt.title(f"Temperature (K) vs {legend_PV}")
#     plt.legend()
#     plt.legend(title="Equivalence ratio")
#     plt.show()

# # PV_source_terms_columns = {
# #     "Progress Variable source term of Pierce & Moin": "Source_Pierce_Moin",
# #     "Progress Variable source term of Ihme and Pitsch": "Source_Ihme_Pitsch",
# #     "Progress Variable source term of Fiorina": "Source_Fiorina"
# # }

# # Plot the PV Source Term vs PV
# # for legend_PV, column_PV in PV_columns.items():
# #     for legend_PV_source_term, column_PV_source_term in PV_source_terms_columns.items():
# #         plt.figure()
# #         for name, group in groups :
# #             #Plot PV source term versus PV
# #             plt.plot(group[column_PV], group[column_PV_source_term], label=f"$\phi$ = {name}")

# #         plt.xlabel(f"{legend_PV}")
# #         plt.ylabel(f"{legend_PV_source_term}")
# #         plt.title(f"{legend_PV_source_term} vs {legend_PV}")
# #         plt.legend()
# #         plt.legend(title="Equivalence ratio")
# #         plt.show()

# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------


# # Plot PV source term versus PV
# # Store the definitions of the progress variables
# Progress_variables = {
#     "Pierce_Moin": {
#         "PV": "PV_Pierce_Moin",
#         "Source": "Source_Pierce_Moin"
#     },

#     "Ihme_Pitsch": {
#         "PV": "PV_Ihme_Pitsch",
#         "Source": "Source_Ihme_Pitsch"
#     },

#     "Fiorina": {
#         "PV": "PV_Fiorina",
#         "Source": "Source_Fiorina"
#     }
# }
# for name, definition in Progress_variables.items():
#     PV_column = definition["PV"]
#     Source_column = definition["Source"]

#     plt.figure()
#     for name, group in groups :
#     # Plot PV source term versus PV
#         plt.plot(group[PV_column], group[Source_column], label=f"$\phi$ = {name}")
    
#     plt.xlabel(PV_column)
#     plt.ylabel(Source_column)
#     plt.title(f"{Source_column} vs {PV_column}")
#     plt.legend(title="Equivalence ratio")
#     plt.grid()
#     plt.show()

# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------
# ##-------------------------------------------------------------------------------------


# # Plot species mass fractions versus PV
# Species_mass_fractions_columns = {
#     "Mass fraction of O2": "Mass fraction of O2",
#     "Mass fraction of CH4": "Mass fraction of CH4",
#     "Mass fraction of H2": "Mass fraction of H2",
#     "Mass fraction of H2O": "Mass fraction of H2O",
#     "Mass fraction of CO2": "Mass fraction of CO2",
#     "Mass fraction of CO": "Mass fraction of CO"
# }

# for legend_PV, column_PV in PV_columns.items():
#     for Species_mass_fractions_legend, Species_mass_fractions_column in Species_mass_fractions_columns.items():
#         plt.figure()
#         for name, group in groups :
#             #Plot PV source term versus PV
#             plt.plot(group[column_PV], group[Species_mass_fractions_column], label=f"$\phi$ = {name}")

#         plt.xlabel(f"{legend_PV}")
#         plt.ylabel(f"{Species_mass_fractions_legend}")
#         plt.title(f"{Species_mass_fractions_legend} vs {legend_PV}")
#         plt.legend()
#         plt.legend(title="Equivalence ratio")
#         plt.show()

