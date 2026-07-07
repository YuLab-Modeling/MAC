import pandas as pd
import numpy as np
import os
os.environ['PYTHONNOUSERSITE'] = '1'

# Read the Dataset
dataset = pd.read_csv("dataset/mols-energies.csv", index_col=3)
ids = dataset.index.to_list()

print("\n-- Parameters " + "-"*100)
print(f"-- Ids : {', '.join([str(i) for i in ids])}")
print("-"*100+"\n")

mlip_name = input("MLIP name: ")

if mlip_name == "M3GNet":
    import matgl
    from matgl.ext.ase import PESCalculator
    matgl.set_backend('DGL')

elif mlip_name == "Mattersim":
    from mattersim.forcefield.potential import MatterSimCalculator

elif mlip_name == "UMA":
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

elif mlip_name == "eSCN":
    from fairchem.core import OCPCalculator

else:
    raise ValueError("MLIP name invalid")

def get_potential(mlip_name):

    if mlip_name == "M3GNet":
        model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        mlip_pot = PESCalculator(potential=model)

    elif mlip_name == "Mattersim":
        mlip_pot = MatterSimCalculator()

    elif mlip_name == "UMA":
        predictor = load_predict_unit("uma-s-1p1.pt", device="cuda")
        mlip_pot = FAIRChemCalculator(predictor, task_name = "oc20")

    elif mlip_name == "eSCN":
        mlip_pot = OCPCalculator(
            checkpoint_path="escn_l6_m3_lay20_all_md_s2ef.pt",
            cpu=False
        )

    return mlip_pot


# updated functions with runtype input

from ase import io
from ase.optimize import BFGS
from ase.constraints import FixAtoms

def calculate_molecule_energy(mlip_name, mlip_pot, runtype, id : int, fmax = 0.02): #---------------------------------------------------------------------------------------------------------------------

    assert runtype in ['opt', 'spc'], "invalid run type"

    # load the molecules
    molecule = io.read(f"xyz-s/all/{id}.xyz")

    # explicitly set charge and spin
    molecule.info.update({"spin": 1, "charge": 0})

    # set the potential for optimization
    molecule.calc = mlip_pot

    # set PBC
    molecule.center(vacuum=10)
    molecule.pbc=True
    # print(molecule.cell)

    if runtype == 'spc':
        return molecule.get_potential_energy()

    opt = BFGS(molecule, logfile=None)
    opt.run(fmax=fmax, steps=300)

    molecule.wrap()

    mol_energy = molecule.get_potential_energy()

    # write the optimized structure
    io.write(f'{mlip_name}/opt-mols/{id}_{mlip_name}_opt.xyz', molecule)

    return mol_energy

def calculate_slab_plus_molecule_energy(mlip_name : str, mlip_pot, id : int,  n_calcs : int, i_calc : int, runtype : str, fmax = 0.02): #---------------------------

    assert runtype in ['opt', 'spc'], "invalid run type"

    # load the slab plus molecule
    slab_plus_molecule = io.read(f"abcluster/id{id}_{n_calcs}/{i_calc}.xyz")

    # explicitly set charge and spin
    slab_plus_molecule.info.update({"spin": 1, "charge": 0})

    # set the potential for optimization
    slab_plus_molecule.calc = mlip_pot
    
    # set PBC
    slab_plus_molecule.set_cell([9.405, 9.405, 20., 90, 90, 60])
    #slab_plus_molecule.center(vacuum=7.5, axis=2)
    slab_plus_molecule.pbc=True
    #print(slab_plus_molecule.cell)

    if runtype == "spc":
        return slab_plus_molecule.get_potential_energy()

    # fix the bottom half
    n_atoms_to_fix = 24      #half of all atoms
    constraint = FixAtoms(indices=np.arange(n_atoms_to_fix))
    slab_plus_molecule.set_constraint(constraint)

    opt = BFGS(slab_plus_molecule, logfile=None)                      
    opt.run(fmax=fmax, steps=300)

    energy = slab_plus_molecule.get_potential_energy()

    # slab_plus_molecule.wrap()
    # slab_plus_molecule.center(axis=2)

    # write the optimized structure
    io.write(f'{mlip_name}/opt-configurations/id{id}/id{id}-{i_calc}.xyz', slab_plus_molecule)

    return energy

# Run

from helper_functions import *
from tqdm import tqdm
import glob
import time

n_calcs=500

runtype = 'spc' # opt

if os.path.exists(f"{mlip_name}/{mlip_name}_{runtype}_timecost.csv"):
    df_exists = True
    df = pd.read_csv(f"{mlip_name}/{mlip_name}_{runtype}_timecost.csv", index_col=0)
else:
    df_exists = False
    timesDict = {}

for id in tqdm(ids, desc="Molecule count", position=0, ascii=True, leave=True):

    # initializing potential every
    # predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")

    timeStart = time.perf_counter()
    mlip_pot = get_potential(mlip_name)

    # energy of the molecule
    molecule_energy = calculate_molecule_energy(mlip_name=mlip_name, mlip_pot=mlip_pot, id=id, runtype=runtype)

    xyz_files = np.array(glob.glob(f'abcluster/id{id}_{n_calcs}/*xyz'))
    i_calcs = sorted([file.split('/')[2].split('.')[0] for file in xyz_files])

    # run one conf optimization to get the approximate computational cost
    i_calc = i_calcs[0]
    slab_plus_molecule_energy = calculate_slab_plus_molecule_energy(mlip_name=mlip_name, mlip_pot=mlip_pot, id=id, n_calcs=n_calcs, i_calc=i_calc, runtype=runtype)

    timeEnd = time.perf_counter()    

    if df_exists:
        df.loc[:, id] = round(timeEnd - timeStart, 3)
    else:
        timesDict[id] = round(timeEnd - timeStart, 3)
        df = pd.DataFrame(timesDict, index=[[0]])

    print(df)
    df.to_csv(f"{mlip_name}/{mlip_name}_{runtype}_timecost.csv")

