"""
Example script for training HIP-NN directly from the ANI1x_datasets h5 file.

This script was designed for an external dataset available at
https://doi.org/10.6084/m9.figshare.c.4712477
pyanitools reader available at
https://github.com/aiqm/ANI1x_datasets

For info on the dataset, see the following publication:
Smith, J.S., Zubatyuk, R., Nebgen, B. et al.
The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional
theory properties for molecules. Sci Data 7, 134 (2020).
https://doi.org/10.1038/s41597-020-0473-z

"""

import argparse
import numpy as np
import torch
import hippynn
import ase.units

import sys
sys.path.append("../../datasets/ani-al/readers/lib/")

#import pyanitools

def make_model(network_params,tensor_order):
    """
    Build the model graph for energy and potentially force prediction.
    """
    from hippynn.graphs import inputs, networks, targets, physics

    net_class={
            0:networks.Hipnn,
            1:networks.HipnnVec,
            2:networks.HipnnQuad,
            }[tensor_order]
    species = inputs.SpeciesNode(db_name="atomic_numbers")
    positions = inputs.PositionsNode(db_name="coordinates")
    network = net_class("hipnn_model", (species, positions), module_kwargs=network_params)
    henergy1 = targets.HEnergyNode("HEnergy1", network)
    force1 = physics.GradientNode("forces1", (henergy1, positions), sign=-1)
    henergy2 = targets.HEnergyNode("HEnergy2", network)
    force2 = physics.GradientNode("forces2", (henergy2, positions), sign=-1)

    return henergy1, force1, henergy2, force2


def make_loss(henergy1,henergy2,force1,force2,force_training1,force_training2):
    """
    Build the loss graph for energy and force error.
    """
    from hippynn.graphs.nodes.loss import MSELoss, MAELoss, Rsq, Mean

    losses1 = {
        "T-RMSE": MSELoss.of_node(henergy1) ** (1 / 2),
        "T-MAE": MAELoss.of_node(henergy1),
        "T-RSQ": Rsq.of_node(henergy1),
        "T-Hier": Mean.of_node(henergy1.hierarchicality),
    }

    losses2 = {
        "T-RMSE": MSELoss.of_node(henergy2) ** (1 / 2),
        "T-MAE": MAELoss.of_node(henergy2),
        "T-RSQ": Rsq.of_node(henergy2),
        "T-Hier": Mean.of_node(henergy2.hierarchicality),
    }

    force_losses1 = {
        "F-RMSE1": MSELoss.of_node(force1) ** (1 / 2),
        "F-MAE1": MAELoss.of_node(force1),
        "F-RSQ1": Rsq.of_node(force1),
    }

    force_losses2 = {
        "F-RMSE2": MSELoss.of_node(force2) ** (1 / 2),
        "F-MAE2": MAELoss.of_node(force2),
        "F-RSQ2": Rsq.of_node(force2),
    }

    losses1["EnergyTotal"] = losses1['T-RMSE'] + losses1["T-MAE"] 
    losses2["EnergyTotal"] = losses2['T-RMSE'] + losses2["T-MAE"] 
    losses1["LossTotal"] = losses1["EnergyTotal"] + losses1["T-Hier"]
    losses2["LossTotal"] = losses2["EnergyTotal"] + losses2["T-Hier"]

    losses = {
        "T-RMSE": losses1['T-RMSE'] + losses2['T-RMSE'], 
        "T-MAE": losses1['T-MAE'] + losses2['T-MAE'],
        "T-RSQ": losses1['T-RSQ'] + losses2['T-RSQ'],
        "T-Hier": losses1['T-RSQ'] + losses2['T-RSQ'],
        "EnergyTotal": losses1['EnergyTotal'] + losses2['EnergyTotal'],
        "LossTotal": losses1['LossTotal'] + losses2['LossTotal'],
        "ForceTotal": 0
    }

    if force_training1:
       losses.update(force_losses1)
       losses["ForceTotal"] = losses["ForceTotal"] + losses["F-RMSE1"] + losses["F-MAE1"]
       losses["LossTotal"] = losses["LossTotal"] + losses["ForceTotal"]

    if force_training2:
       losses.update(force_losses2)
       losses["ForceTotal"] = losses["ForceTotal"] + losses["F-RMSE2"] + losses["F-MAE2"]
       losses["LossTotal"] = losses["LossTotal"] + losses["ForceTotal"]

    return losses


# wb97x-6-31g*, G16. Doesn't need to be exact for most models.
SELF_ENERGY_APPROX = {'C': -37.764142, 'H': -0.4993212, 'N': -54.4628753, 'O': -74.940046}
SELF_ENERGY_APPROX = {k: SELF_ENERGY_APPROX[v] for k, v in zip([6, 1, 7, 8], 'CHNO')}
SELF_ENERGY_APPROX[0] = 0


def load_db(db_info, en_name1, force_name1, en_name2, force_name2, seed, anidata_location1, anidata_location2, n_workers):
    """
    Load the database.
    """

    from hippynn.databases.h5_pyanitools import PyAniFileDB

    # Ensure total energies loaded in float64.
    torch.set_default_dtype(torch.float64)
    import os
    database = PyAniFileDB(
        file=anidata_location1,
        species_key='atomic_numbers',
        seed=seed,
        num_workers=n_workers,
        **db_info
    )

    # compute (approximate) atomization energy by subtracting self energies
    self_energy = np.vectorize(SELF_ENERGY_APPROX.__getitem__)(database.arr_dict['atomic_numbers'])
    self_energy = self_energy.sum(axis=1)  # Add up over atoms in system.
    database.arr_dict[en_name1] = (database.arr_dict[en_name1] - self_energy)
    database.arr_dict[en_name2] = (database.arr_dict[en_name2] - self_energy)
    kcalpmol = (ase.units.kcal/ase.units.mol)
    conversion = ase.units.Ha/kcalpmol
    database.arr_dict[en_name1] = database.arr_dict[en_name1].astype(np.float32)*conversion
    database.arr_dict[en_name2] = database.arr_dict[en_name2].astype(np.float32)*conversion
    if force_name1 in database.arr_dict:
        database.arr_dict[force_name1] = database.arr_dict[force_name1]*conversion
    if force_name2 in database.arr_dict:
        database.arr_dict[force_name2] = database.arr_dict[force_name2]*conversion
    torch.set_default_dtype(torch.float32)
    database.arr_dict['atomic_numbers']=database.arr_dict['atomic_numbers'].astype(np.int64)

    # Drop indices where computed energy not retrieved.
    found_indices1 = ~np.isnan(database.arr_dict[en_name1])
    found_indices2 = ~np.isnan(database.arr_dict[en_name2])
    database.arr_dict = {k: v[found_indices1] for k, v in database.arr_dict.items()}
    database.arr_dict = {k: v[found_indices2] for k, v in database.arr_dict.items()}

    database.make_trainvalidtest_split(0.1, 0.1)
    return database


def setup_experiment(training_modules, device, batch_size, init_lr, patience, max_epochs, stopping_key):
    """
    Set up the training run.
    """
    from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController

    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=init_lr)
    scheduler = RaiseBatchSizeOnPlateau(optimizer=optimizer,
                                        max_batch_size=batch_size,
                                        patience=patience,
                                        factor=0.5,
                                        )

    controller = PatienceController(optimizer=optimizer,
                                    scheduler=scheduler,
                                    batch_size=batch_size,
                                    eval_batch_size=batch_size,
                                    max_epochs=max_epochs,
                                    stopping_key=stopping_key,
                                    termination_patience=2 * patience)

    setup_params = hippynn.experiment.SetupParams(
        controller=controller,
        device=device,
    )
    return setup_params


ANI1X_DSETS_KEYS = ['hf_tz.energy', 'coordinates', 'tpno_ccsd(t)_dz.corr_energy', 'wb97x_dz.hirshfeld_charges',
                    'wb97x_tz.mbis_charges', 'wb97x_tz.forces', 'mp2_tz.corr_energy', 'npno_ccsd(t)_tz.corr_energy',
                    'wb97x_tz.mbis_volumes', 'wb97x_tz.energy', 'wb97x_tz.dipole', 'wb97x_tz.mbis_octupoles',
                    'wb97x_tz.mbis_quadrupoles', 'mp2_qz.corr_energy', 'wb97x_tz.mbis_dipoles', 'wb97x_dz.cm5_charges',
                    'path', 'atomic_numbers', 'hf_qz.energy', 'mp2_dz.corr_energy', 'wb97x_dz.dipole',
                    'npno_ccsd(t)_dz.corr_energy', 'wb97x_dz.energy', 'hf_dz.energy', 'wb97x_dz.quadrupole',
                    'ccsd(t)_cbs.energy', 'wb97x_dz.forces']

AVAIL_METHODS = ['hf', 'wb97x', 'ccsd(t)', 'mp2']
AVAIL_BASIS = ['dz', 'tz', 'qz', 'cbs']


def get_data_names(qm_method, basis_set):
    assert qm_method in AVAIL_METHODS, f"Method not found: {qm_method}"
    assert basis_set in AVAIL_BASIS, f"Basis set not found: {basis_set}"
    data_spec = f"{qm_method}_{args.basis_set}"
    en_name = f"{data_spec}.energy"
    force_name = f"{data_spec}.forces"
    assert en_name in ANI1X_DSETS_KEYS, f"Method-basis combination not available: {data_spec}"
    if args.force_training:
        assert f"{data_spec}.forces" in ANI1X_DSETS_KEYS, f"Force training not available for data spec: {data_spec}"
    return en_name, force_name


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.set_default_dtype(torch.float32)

    hippynn.settings.WARN_LOW_DISTANCES = False
    if args.noprogress:
        hippynn.settings.PROGRESS = None 

    netname = f"{args.tag}_GPU{args.gpu}"
    network_parameters = {
        "possible_species": [0, 1, 6, 7, 8],
        'n_features': args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.lower_cutoff,
        "dist_soft_max": args.cutoff_distance - 1,
        "dist_hard_max": args.cutoff_distance,
        "n_interaction_layers": args.n_interactions,
        "n_atom_layers": args.n_atom_layers,
    }

    with hippynn.tools.active_directory(netname):
        with hippynn.tools.log_terminal("training_log.txt", 'wt'):
            henergy1, force1, henergy2, force2 = make_model(network_parameters,tensor_order=args.tensor_order)

            en_name1, force_name1 = get_data_names(args.qm_method1, args.basis_set1)
            en_name2, force_name2 = get_data_names(args.qm_method2, args.basis_set2)

            henergy1.mol_energy.db_name = en_name1
            force1.db_name = force_name1
            henergy2.mol_energy.db_name = en_name2
            force2.db_name = force_name2

            validation_losses = make_loss(henergy1, henergy2, force1, force2, force_training=args.force_training)

            train_loss = validation_losses["LossTotal"]

            from hippynn.experiment import assemble_for_training

            training_modules, db_info = assemble_for_training(train_loss, validation_losses)

            database = load_db(db_info,
                               en_name1,
                               force_name1,
                               en_name2,
                               force_name2,
                               n_workers=args.n_workers,
                               seed=args.seed,
                               anidata_location=args.anidata_location)

            from hippynn.pretraining import hierarchical_energy_initialization

            hierarchical_energy_initialization(henergy1, database, trainable_after=False)

            setup_params = setup_experiment(training_modules,
                                            device=args.gpu,
                                            batch_size=args.batch_size,
                                            init_lr=args.init_lr,
                                            patience=args.patience,
                                            max_epochs=args.max_epochs,
                                            stopping_key=args.stopping_key,
                                            )

            from hippynn.experiment import setup_and_train

            setup_and_train(training_modules=training_modules,
                            database=database,
                            setup_params=setup_params,
                            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag", type=str, default="TEST_MODEL_ANI1X", help='name for run')
    parser.add_argument("--gpu", type=int, default=0, help='which GPU to run on')
    parser.add_argument("--seed", type=int, default=0, help='random seed for init and split')

    parser.add_argument("--n_interactions", type=int, default=2)
    parser.add_argument("--n_atom_layers", type=int, default=3)
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--n_sensitivities", type=int, default=20)
    parser.add_argument("--cutoff_distance", type=float, default=6.5)
    parser.add_argument("--lower_cutoff",type=float,default=0.75,
            help="Where to initialize the shortest distance sensitivty")
    parser.add_argument("--tensor_order",type=int,default=0)

    parser.add_argument("--anidata_location1", type=str, default='../../../datasets/ani1x_release/ani1x-release.h5') #change this to the $PWD
    parser.add_argument("--anidata_location2", type=str, default='../../../datasets/ani1x_release/ani1x-release.h5')
    parser.add_argument("--qm_method1", type=str, default='wb97x') #change to ccsd(t)
    parser.add_argument("--qm_method2", type=str, default='"ccsd(t)"')
    parser.add_argument("--basis_set1", type=str, default='dz') #change to cbs
    parser.add_argument("--basis_set2", type=str, default='cbs')

    parser.add_argument("--force_training", action='store_true', default=False) #change default to True

    parser.add_argument("--batch_size",type=int, default=1024)
    parser.add_argument("--init_lr",type=float, default=1e-3)
    parser.add_argument("--patience",type=int, default=5)
    parser.add_argument("--max_epochs",type=int, default=500)
    parser.add_argument("--stopping_key",type=str, default="T-RMSE")

    parser.add_argument("--noprogress", action='store_true', default=False, help='suppress progress bars')
    parser.add_argument("--n_workers", type=int, default=2, help='workers for pytorch dataloaders')
    args = parser.parse_args()

    main(args)
