#################################################################################################################################
## Load combustion data generated using the oxyfuel chemical mechanism UC-SD_R26S

data_type = 'FPF'
file_prefix = '../data/freely-propagating-flame-'
grid = pd.read_csv(file_prefix + data_tag + '-grid.csv', sep = ',', header=None).to_numpy()

state_space = pd.read_csv(file_prefix + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()
state_space_sources = pd.read_csv(file_prefix + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()
state_space_names = pd.read_csv(file_prefix + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()
# mf = pd.read_csv(file_prefix + data_tag + '-mixture-fraction.csv', sep = ',', header=None).to_numpy()
mf = pd.read_csv(file_prefix + data_tag + '-mixture-fraction-along-flame.csv', sep = ',', header=None).to_numpy()

target_variables = state_space[:,target_variables_indices]
target_variables_names = list(state_space_names[target_variables_indices])
(_, n_target_variables) = np.shape(target_variables)
print('\nUsing: ' + ', '.join(target_variables_names) + ' as target state variables at the decoder output.\n')

try:
    VD_target_variables = state_space[:,VD_target_variables_indices]
    VD_target_variables_names = list(state_space_names[VD_target_variables_indices])
    print('\nUsing: ' + ', '.join(VD_target_variables_names) + ' as target state variables for VarianceData computation.\n')
except:
    pass

if  pure_streams:
    pure_streams_prefix = 'tps' # Trainable Pure Streams
else:
    species_to_remove_list = ['H2', 'CH4', 'O2']
    pure_streams_prefix = 'ntps' # Non-Trainable Pure Streams

tex_names = pd.read_csv('../data/oxyfuel-tex-names.csv', sep = ',', header=None).to_numpy().ravel()
    
idx_trim, _ =  np.where((grid>0.005)&(grid <0.015))
state_space = state_space[idx_trim,:]
state_space_sources = state_space_sources[idx_trim,:]
target_variables = target_variables[idx_trim,:]
grid = grid[idx_trim,:]
mf = mf[idx_trim,:]

try:
    VD_target_variables = VD_target_variables[idx_trim,:]
except:
    pass

#################################################################################################################################
## Print data shape
#################################################################################################################################

(n_observations, n_variables) = np.shape(state_space)

print(str(n_observations) + ' observations')
print(str(n_variables) + ' state variables')
    
#################################################################################################################################