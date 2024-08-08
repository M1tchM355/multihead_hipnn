"""
    Example of Multi-Fidelity Learning for ani-1x dataset.
    
    NOTE
        - Only trains to subset of configurations for which all levels of theory are present!
"""
# command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("tag",type=str,help='name tag for run')
parser.add_argument("gpu",type=int,help='which GPU to run on')
args=parser.parse_args()

import sys, os
import numpy as np
import ase

# Import PyaniTools
anitools_loc="/vast/home/smatin/Code_Repo_22/Hipnn_Fit-Zn/readers/" 
sys.path.append(anitools_loc)
import pyanitools

import torch
torch.set_default_dtype(torch.float32)
torch.cuda.set_device("cuda:0")

# Random Seed tied to GPU tags.
hashed= hash(args.gpu)
seed = hash(str(hashed))
seed = seed%1_000_000
torch.manual_seed(seed)
print("SEED ::", seed)

# Set Correct Back-end for matplotlib. 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
import numpy as np

# Hippynn imports. 
import hippynn
# hippynn.settings.PROGRESS=None
hippynn.settings.WARN_LOW_DISTANCES=False
hippynn.custom_kernels.set_custom_kernels("triton")
from hippynn.graphs import inputs, networks, targets, physics

# Disable Numba Warnings.
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def load_db(db_info, energy_force_names, seed, anidata_location, n_workers, debug=False):
    """
    Load the configurations for ANI-1ccx for the `e_name` and `force_name` arguments. 
    """
    # Self-Energy constants -> wb97x-6-31g*, G16. Doesn't need to be exact for most models.
    SELF_ENERGY_APPROX = {'C': -37.764142, 'H': -0.4993212, 'N': -54.4628753, 'O': -74.940046}
    SELF_ENERGY_APPROX = {k: SELF_ENERGY_APPROX[v] for k, v in zip([6, 1, 7, 8], 'CHNO')}
    SELF_ENERGY_APPROX[0] = 0

    # Load DB using hipnn internal tools 
    from hippynn.databases.h5_pyanitools import PyAniFileDB
    torch.set_default_dtype(torch.float64) # Ensure total energies loaded in float64.
    database = PyAniFileDB(
        file=anidata_location,
        species_key='atomic_numbers',
        seed=seed,
        num_workers=n_workers,
        allow_unfound=True,
        **db_info
    )
    
    for en_name, force_name in energy_force_names.items():
        assert en_name in database.arr_dict
        # Compute (approximate) atomization energy by subtracting self energies
        self_energy = np.vectorize(SELF_ENERGY_APPROX.__getitem__)(database.arr_dict['atomic_numbers'])
        self_energy = self_energy.sum(axis=1)  # Add up over atoms in system.
        database.arr_dict[en_name] = (database.arr_dict[en_name] - self_energy)
        kcalpmol = (ase.units.kcal/ase.units.mol)
        conversion = ase.units.Ha/kcalpmol
        database.arr_dict[en_name] = database.arr_dict[en_name].astype(np.float32)*conversion
        if force_name and force_name in database.arr_dict:
            database.arr_dict[force_name] = database.arr_dict[force_name]*conversion
        torch.set_default_dtype(torch.float32)
        database.arr_dict['atomic_numbers']=database.arr_dict['atomic_numbers'].astype(np.int64)
   
    # Ensure overlapping indices for levels of theory (TODO clean up)
    indices_en = {}
    for en_name in energy_force_names:
        indices_en[en_name] = ~np.isnan(database.arr_dict[en_name])
        
    idx_all = np.array([indices_en[en_name] for en_name in energy_force_names])
    found_indices = idx_all[0]
    for i in range(1,len(idx_all)):
        found_indices = found_indices & idx_all[i] 
    
    database.arr_dict = {k: v[found_indices] for k, v in database.arr_dict.items()}

    # Split database
    if debug:
        database.make_trainvalidtest_split(0.8, 0.1) # 80% goes to test set. 
    else:
        database.make_trainvalidtest_split(0.1, 0.1) 
    
    return database


network_params = {
    "possible_species": [0,1,6,7,8],   # Z values of the elements
    "n_features": 64,                     # Number of neurons at each layer
    "n_sensitivities": 20,                # Number of sensitivity functions in an interaction layer
    "dist_soft_min": 0.75,  # qm7 1.7  qm9 .85  AL100 .85
    "dist_soft_max": 5.5,  # qm7 10.  qm9 5.   AL100 5.
    "dist_hard_max": 6.5,  # qm7 15.  qm9 7.5  AL100 7.5
    "n_interaction_layers": 1,            # Number of interaction blocks
    "n_atom_layers": 4,                   # Number of atom layers in an interaction block
    "resnet": True,
    "cusp_reg": 1e-8, 
}

# Define Network
species = inputs.SpeciesNode(db_name="atomic_numbers")
positions = inputs.PositionsNode(db_name="coordinates")
network = networks.HipnnVec("hipnn", (species,positions), module_kwargs=network_params)    

# List of target nodes etc
energy_force_dict = {
    "wb97x_dz.energy": "wb97x_dz.forces",
    "wb97x_tz.energy": "wb97x_tz.forces",
    "ccsd(t)_cbs.energy": None,  # No Forces for CC
}
HEnergy_Nodes = {}
Hier_Nodes = {} # TODO : An exercise for the reader to add hierarchicality losses to training!
Force_Nodes = {}
Loss_Dict = {}

# Create Energy and Force nodes (when applicable.)
for energy_name, force_name in energy_force_dict.items():
    print(energy_name, force_name)
    HEnergy_Nodes[energy_name] = targets.HEnergyNode(f"HNode_{energy_name}", network, db_name=energy_name) 
    if force_name: 
        Force_Nodes[force_name] = physics.GradientNode(
            f"GradNode_{force_name}",
            (HEnergy_Nodes[energy_name], positions), 
            sign=-1, 
            db_name=force_name
        )


### Create Loss graph
from hippynn.graphs import loss
L2_reg =  1e-4 * loss.l2reg(network)
loss_err = 0.0

# TODO allow different weights for different levels of theories.
w_E = 1.0
w_F = 1.0

# TODO Combine loss graph and validation loss dictionary creation into 1 step!
for energy_name, force_name in energy_force_dict.items():
    Loss_Dict[f"{energy_name}_loss"] = loss.MSELoss.of_node(HEnergy_Nodes[energy_name])**(1/2)  + loss.MAELoss.of_node(HEnergy_Nodes[energy_name])
    loss_err = loss_err + w_E*Loss_Dict[f"{energy_name}_loss"] 
    if force_name:
        Loss_Dict[f"{force_name}_loss"] = loss.MSELoss.of_node(Force_Nodes[force_name])**(1/2)  + loss.MAELoss.of_node(Force_Nodes[force_name])
        loss_err = loss_err + w_F*Loss_Dict[f"{force_name}_loss"]
    
loss_train = L2_reg + loss_err

# Validation losses
validation_losses = {"L2_reg": L2_reg, "Loss_Err":loss_err, "Loss_Train":loss_train}
for energy_name, force_name in energy_force_dict.items():
    validation_losses[f"{energy_name}"] = Loss_Dict[f"{energy_name}_loss"]
    if force_name:
        validation_losses[f"{force_name}"] = Loss_Dict[f"{force_name}_loss"]
###

### Set up plotting. 
from hippynn import plotting

plots_to_make = (
    plotting.SensitivityPlot(network.torch_module.sensitivity_layers[0], saved="Sensitivity0.pdf",shown=False),
)

# energy and forces histograms.
# TODO Energy hierarchicality plots.
for energy_name, force_name in energy_force_dict.items():
    plots_to_make = plots_to_make + (plotting.Hist2D.compare(HEnergy_Nodes[energy_name], saved=True),)
    if force_name:
        plots_to_make = plots_to_make + (
            plotting.Hist2D(Force_Nodes[force_name].true,Force_Nodes[force_name].pred, saved=f"{force_name}.pdf", xlabel="True_Force",ylabel="Pred_Force"),
        )
    
if network_params["n_interaction_layers"] > 1:
    plots_to_make = plots_to_make + (
        plotting.SensitivityPlot(network.torch_module.sensitivity_layers[1], saved="Sensitivity1.pdf",shown=False),
    )
    
plot_maker = plotting.PlotMaker(*plots_to_make, plot_every=100)
###

# Assemble Pytorch Model to be trained. 
training_modules, db_info = hippynn.experiment.assemble_for_training(
    loss_train, 
    validation_losses, 
    plot_maker=plot_maker
)

print(db_info)
database = load_db(db_info,
    energy_force_names=energy_force_dict,
    n_workers=1,
    seed=seed, 
    anidata_location="/usr/projects/ml4chem/internal_datasets/ANI1x_official/ani1x-release.h5",
    debug=False,
)

# Fit the non-interacting energies by examining the database.
from hippynn.pretraining import hierarchical_energy_initialization
for energy_name in energy_force_dict:
    hierarchical_energy_initialization(HEnergy_Nodes[energy_name], database, peratom=False, energy_name=energy_name, decay_factor=1e-2)
    
# Training parameters
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController
optimizer = torch.optim.Adam(training_modules.model.parameters(),lr=5e-4)
batch_size = 256
patience = 15
max_batch_size=256
scheduler =  RaiseBatchSizeOnPlateau(
    optimizer=optimizer,
    max_batch_size=max_batch_size,
    patience=patience,
    factor=0.5,
)
controller = PatienceController(
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=batch_size,
    eval_batch_size=max_batch_size,
    max_epochs=200,
    # max_epochs=5,
    stopping_key="Loss_Err",
    termination_patience=2*patience
)
experiment_params = hippynn.experiment.SetupParams(
    controller=controller,
    device=0,
)

# Network Name
netname = f"{args.tag}_GPU{args.gpu}"
dirname = netname

# Training
with hippynn.tools.active_directory(dirname):
    
    # Construct GraphViz (helps with debugging.)
    print("Constructing Graph")
    from hippynn.graphs.viz import visualize_graph_module, visualize_connected_nodes
    # Graph Vizualization object. 
    viz_name = f"Multi-Fidelity_Ani-1x_{args.gpu}"
    model = training_modules.model
    vgm = visualize_graph_module(model)
    graphviz_name = f"{viz_name}.dot"
    vgm.save(graphviz_name) 
    os.system("dot -Tpng %s -o %s.png"%(graphviz_name, viz_name))
    
    # Training happens here!
    with hippynn.tools.log_terminal("training_log.txt", 'wt'):
            print("Data Loaded and Network set up! Just need to train... ")
            # sys.exit()
            from hippynn.experiment import setup_and_train
            setup_and_train(
                training_modules=training_modules,
                database=database,
                setup_params=experiment_params,
            )