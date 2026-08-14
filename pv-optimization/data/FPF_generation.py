import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

## Define the composition of the fuel and oxidant

# fuel_composition = 'H2:1'
fuel_composition = 'CH4:0.5, H2:0.5'

# oxidant_composition = 'O2:0.21, N2:0.79' 
oxidant_composition = 'O2:1.0, N2:0.0'

data_tag = 'oxyfuel'

## Define the equivalence ratio 
equivalence_ratio = np.linspace(0.6, 1.5, 200)
# equivalence_ratio = np.linspace(0.6, 1.5, 50)
# equivalence_ratio = [0.6, 0.8, 1.0]

## Inlet temperature 
T_in = 300  # K

## Pressure
P_in = 101325  # Pa

## Select the mechanism file

## H2 mixtures
# mechanism = 'UC-SD_H2'
# mechanism = 'C3Mech_H2'
# mechanism = 'C3Mech_H2_NOx'

## CH4 - H2 mixtures
# mechanism = 'UC-SD_R26S'
# mechanism = 'C3Mech_CH4'

## CH4 - H2 - NH3 mixtures
mechanism = 'C3Mech_CH4_NH3_NOx'

# =================================================================================================
# =================================================================================================
# =================================================================================================
## Functions

def freeFlameSimulation(gas, fuel, oxidant, phi, T_in, P_in):

    # Set mixture temperature and pressure
    gas.TP = T_in, P_in

    # Set equivalence ratio
    gas.set_equivalence_ratio(phi, fuel, oxidant)

    # set the width of the flame domain
    width = 0.02

    # Create a FreeFlame object
    f = ct.FreeFlame(gas, width=width)

    ##-------------------------------------------------------------------------------------
    ## Set numerical parameters

    f.set_initial_guess()

    #Set tolerance properties
    tol_ss    = [1.0e-5, 1.0e-8]        # [rtol atol] for steady-state problem
    tol_ts    = [1.0e-5, 1.0e-8]        # [rtol atol] for time stepping

    loglevel  = 1                       # amount of diagnostic output (0 to 5)
                        
    refine_grid = True                  # True to enable refinement, False to disable

    f.flame.set_steady_tolerances(default=tol_ss)
    f.flame.set_transient_tolerances(default=tol_ts)

    #Max number of times the Jacobian will be used before it must be re-evaluated
    f.set_max_jac_age(50, 50)

    #Set time steps whenever Newton convergence fails
    f.set_time_step(5.e-06, [10, 20, 80]) #s

    f.max_time_step_count = 10000 # 1000

    f.set_max_grid_points(f.domains[f.domain_index("flame")],10000)

    ##-------------------------------------------------------------------------------------
    # Flame simulation starts here
    ##-------------------------------------------------------------------------------------
    # First flame:
    # No energy for starters
    f.energy_enabled = False

    f.solve(loglevel, auto = True)

    ##-------------------------------------------------------------------------------------
    # Second flame:

    #Energy equation enabled
    f.energy_enabled = True

    f.solve(loglevel, auto = True)

    ##-------------------------------------------------------------------------------------
    ## Last flame
    # Set the grid refinement criteria (https://cantera.org/dev/reference/onedim/grid-refinement.html)
    # f.set_refine_criteria(ratio = 2.0, slope = 0.03, curve = 0.01, prune = 0.0)
    f.set_refine_criteria(ratio = 10.0, slope = 0.1, curve = 0.1, prune = 0.02)
    f.set_max_grid_points(f.domains[f.domain_index("flame")],500)


    ## Set diffusion medel
    # f.transport_model = 'UnityLewis'
    # f.transport_model = 'Mix'
    f.transport_model = 'Multi'

    ## Set Soret effect - Thermal mass diffusion
    f.soret_enabled  = True

    f.solve(loglevel, refine_grid)

    return f

#
# Mixture fraction calculation using Bilger's definition
#
def Bilger_mixture_fraction(f, gas, fuel_g, oxi_g):
    """
    Computes the mixture fraction Z using Bilger's definition from a solved Cantera 1D flame.
    
    f: Cantera 1D flame object (e.g., FreeFlame, BurnerFlame)
    gas: Cantera Solution object used in the simulation
    fuel_g: Fuel gas composition string (e.g., 'CH4:1')
    oxi_g: Oxidizer gas composition string (e.g., 'O2:0.29, N2:0.71')
    """
    
    comp = [sp.composition for sp in gas.species()]             #Atomic composition, gas.species() returns all the species of the mecanism
    nC   = np.fromiter((c.get('C',0) for c in comp), float)     #Total quantity of carbon in the mixture, otherwise return 0 if there is no C
    nH   = np.fromiter((c.get('H',0) for c in comp), float)     #Total quantity of hydrogen in the mixture
    nO   = np.fromiter((c.get('O',0) for c in comp), float)     #Total quantity of oxygen in the mixture

    
    speW = gas.molecular_weights            # molecular weight of the species
    WC, WH, WO = 12.0107, 1.00784, 15.999   # atomic weights
    bC, bH, bO = 2/WC, 1/(2*WH), -1/WO      # computation of the coeffcicient which are needed to calculate the Bilger fraction

    w = (bC*WC * nC +
         bH*WH * nH +
         bO*WO * nO) / speW                 # per-unit-mass contribution

    # Pure oxidizer baseline
    gas.set_equivalence_ratio(0, fuel_g, oxi_g)     # only oxidizer
    B_oxi = w.dot(gas.Y)                            # B_oxi=sum(w*Y) with Y the mass fraction of the species

    # Pure fuel baseline
    gas.set_equivalence_ratio(1e10, fuel_g, oxi_g)  # only fuel
    B_fuel = w.dot(gas.Y)                           # B_fuel=sum(w*Y) with Y the mass fraction of the species

    # f.Y has shape (n_species, n_points)
    Beta = w.dot(f.Y)  
    Z    = (Beta - B_oxi) / (B_fuel - B_oxi)

    return Z




##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------

# Define the path to the mechanism file
mechanism_path = os.path.join('..','kineticMechanisms', mechanism)

# Create gas object
gas = ct.Solution(mechanism_path+'.yaml')

# grid
grid_all = []
# mixture fraction computed along the flame
mixture_fraction_along_flame_all = []
# mixture fraction at the inlet boundary
mixture_fraction_inlet_all = []
# temperature
temperature_all = []
# dictionary that will store all mass fractions
mass_fraction_all = {f"Mass fraction of {sp}": [] for sp in gas.species_names}
# temperature source term 
temperature_source_term = []
# dictionary that will store all species source terms
species_source_term = {f"Species source term {sp}": [] for sp in gas.species_names}

# Loop over the equivalence ratio
for phi in equivalence_ratio:

    gas = ct.Solution(mechanism_path + '.yaml')

    # Flame simulation
    f = freeFlameSimulation(gas, fuel_composition, oxidant_composition, phi, T_in, P_in)

    # Mixture fraction 
    Z_bilger = Bilger_mixture_fraction(f, gas, fuel_composition, oxidant_composition)
    Z_inlet = Z_bilger[0] # mixture fraction at the inlet boundary

    ## Grid
    # Add the grid of this flame to the list
    grid_all.extend(f.grid)

    ## Mixture fraction computed along the flame
    mixture_fraction_along_flame_all.extend(Z_bilger)

    ## Mixture fraction at the inlet boundary
    mixture_fraction_inlet_all.extend([Z_inlet] * len(f.grid))

    ## Temperature
    temperature_all.extend(f.T)

    ## Species mass fraction
    for i, species in enumerate(gas.species_names):
        mass_fraction_all[f"Mass fraction of {species}"].extend(f.Y[i])

    ## Temperature source term
    temperature_source_term.extend(f.heat_release_rate/f.cv_mass)

    ## Pre-calcultation of the PV source term
    for j in range(len(f.grid)):
        # set gas state at grid point j
        gas.TPY = f.T[j], f.P, f.Y[:,j]
        # compute source terms
        omega = (gas.net_production_rates * gas.molecular_weights)/gas.density
        for i, species in enumerate(gas.species_names):
            species_source_term[f"Species source term {species}"].append(omega[i])

# Create dataframe of the grid with one single column
results_grid = pd.DataFrame(grid_all)
results_grid.to_csv("freely-propagating-flame-"+ data_tag +"-grid.csv", index=False, header=False)

# Create dataframe of the mixture fraction at the inlet boundary with one single column
results_mixture_fraction_inlet = pd.DataFrame(mixture_fraction_inlet_all)
results_mixture_fraction_inlet.to_csv("freely-propagating-flame-"+ data_tag +"-mixture-fraction.csv", index=False, header=False)

# Save the mixture fraction along the flame
results_mixture_fraction_along_flame = pd.DataFrame(mixture_fraction_along_flame_all)
results_mixture_fraction_along_flame.to_csv("freely-propagating-flame-"+ data_tag +"-mixture-fraction-along-flame.csv", index=False, header=False)

# Save temperature and species mass fractions
results_state_space = {"Temperature": temperature_all}
results_state_space.update(mass_fraction_all)
results_state_space = pd.DataFrame(results_state_space)
results_state_space.to_csv("freely-propagating-flame-"+ data_tag +"-state-space.csv", index=False, header=False)

# Save temperature source term and species source term
results_state_space_sources = {"Temperature source term": temperature_source_term}
results_state_space_sources.update(species_source_term)
results_state_space_sources = pd.DataFrame(results_state_space_sources)
results_state_space_sources.to_csv("freely-propagating-flame-"+ data_tag +"-state-space-sources.csv", index=False, header=False)


print("Species names:", gas.species_names)

