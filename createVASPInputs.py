import numpy as np
import pandas as pd

import os
import glob
import shutil

path = "/home/sakengali/Desktop/research/SIB-additives/interaction-energies/side-proj-E-int/MLIPs"

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Get MLIP Name
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

mlip_name = input("\nCreating VASP inputs for MLIP-optimized structures\nMLIP name: ")

src = 'distinct-top-configurations'
print(f"\npulling values from /{src}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Get the list of ids for which MLIP has been succesfully run
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

enerInfo = os.listdir(f"{mlip_name}/{src}/")
ids = [fileName.split('_interaction')[0][2:] for fileName in enerInfo]

# Read the Dataset
dataset = pd.read_csv("dataset/mols-energies.csv", index_col=3)
ids = dataset.index.to_list()

print(f"\n-- Found {mlip_name} calculations for the following ids (N = {len(ids)}): " + "-"*50)
print(f"-- Ids : {', '.join([str(i) for i in ids])}")
print("-"*100+"\n")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Read top configs for each id
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

top_configs_dict = {}

print("-- Configurations with the lowest E_int:")
print("\tId\t|\t[configs]")
print("\t--\t|\t---------")
for id in ids:
    energies_df = pd.read_csv(f"{mlip_name}/{src}/id{id}_interaction_energies.csv", index_col=0)
    top_configs = list(energies_df.index[:5])
    top_configs_dict[id] = top_configs
    print(f"\t{id}\t|\t{top_configs}")

vasp_inp = input("\n -- Do you want to proceed with the creation of VASP inputs (POSCAR & INCAR files)? (Yes / No) ")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create inputs
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.io.xyz import XYZ

from ase.io import read, write

from vasp_inputs import *

from tqdm import tqdm

if vasp_inp.lower() == "yes":

    # Inputs for individual molecules

    molLattice = Lattice.from_parameters(a=20, b=20, c=20., alpha=90, beta=90, gamma=90)

    for id, configs in tqdm(top_configs_dict.items(), desc="Molecule count", position=0, ascii=True, leave=True):
        
        os.makedirs(f"{mlip_name}/dft-inputs/id{id}-mol", exist_ok=True)

        shutil.copy(f"{mlip_name}/dft-inputs/createPOTCAR.sh", f"{mlip_name}/dft-inputs/id{id}-mol/")
        shutil.copy(f"{mlip_name}/dft-inputs/KPOINTS", f"{mlip_name}/dft-inputs/id{id}-mol/")

        # write POSCAR
        atoms = read(f"{path}/{mlip_name}/opt-mols/{id}_{mlip_name}_opt.xyz")
        print(atoms.cell)
        atoms.set_pbc=True
        atoms.wrap()
        write(f"{mlip_name}/dft-inputs/id{id}-mol/POSCAR", atoms)


        # mol = XYZ.from_file(f"{path}/{mlip_name}/opt-mols/{id}_{mlip_name}_opt.xyz").molecule
        # structure = Structure(molLattice, mol.species, mol.cart_coords, coords_are_cartesian=True)
        # Poscar(structure).write_file(f"{mlip_name}/dft-inputs/id{id}-mol/POSCAR")     

        # write INCAR
        writeIncar(mlip_name=mlip_name, str_name=f"id{id}-mol", structType="mol")

    # creating the lattice (same for all structures)
    configLattice = Lattice.from_parameters(a=9.405, b=9.405, c=20., alpha=90, beta=90, gamma=60)

    for id, configs in tqdm(top_configs_dict.items(), desc="Configs: Molecule count", position=0, ascii=True, leave=True):

        for config in configs:

            os.makedirs(f"{mlip_name}/dft-inputs/id{id}-{config}", exist_ok=True)

            shutil.copy(f"{mlip_name}/dft-inputs/createPOTCAR.sh", f"{mlip_name}/dft-inputs/id{id}-{config}/")
            shutil.copy(f"{mlip_name}/dft-inputs/KPOINTS", f"{mlip_name}/dft-inputs/id{id}-{config}/")

            # write POSCAR
            atoms = read(f"{path}/{mlip_name}/opt-configurations/id{id}/id{id}-{config}.xyz")
            atoms.wrap()
            atoms.center(axis=2)
            write(f"{mlip_name}/dft-inputs/id{id}-{config}/POSCAR", atoms)

            # mol = XYZ.from_file(f"{path}/{mlip_name}/opt-configurations/id{id}/id{id}-{config}.xyz").molecule
            # structure = Structure(configLattice, mol.species, mol.cart_coords, coords_are_cartesian=True)
            # Poscar(structure).write_file(f"{mlip_name}/dft-inputs/id{id}-{config}/POSCAR")

            # write INCAR
            writeIncar(mlip_name=mlip_name, str_name=f"id{id}-{config}", structType="config")


# separate the runs into 6 batches

batch_inp = input("\n -- Do you want to proceed with the creation of batches and their corresponding SLURM scripts? (Yes/No) ")

anvil = True

if anvil:

    if batch_inp.lower() == "yes":

        n = len(ids)
        nbatches = 6
        nb =n // nbatches + 1

        for i in range(nbatches):
            if i == nbatches - 1:
                batch = ids[i*nb : ]
            else:
                batch = ids[i*nb : (i+1)*nb]

            print(i, len(batch))

            with open(f"{mlip_name}/dft-inputs/submit_anvil_template.shr", 'r') as f:
                slurm = f.readlines()

            slurm[2] = slurm[2].replace("N", str(i))
            slurm[3] = slurm[3].replace("N", str(i))
            slurm[4] = slurm[4].replace("N", str(i))
            
            slurm[17] = slurm[17].replace("()", "("+" ".join(batch)+")")

            with open(f"{mlip_name}/dft-inputs/submit_anvil_batch{i}.sh", 'w') as f:
                f.writelines(slurm)
else:

    if batch_inp.lower() == "yes":

        n = len(ids)
        nbatches = 6
        nb =n // nbatches + 1

        for i in range(nbatches):
            if i == nbatches - 1:
                batch = ids[i*nb : ]
            else:
                batch = ids[i*nb : (i+1)*nb]

            print(i, len(batch))

            with open(f"{mlip_name}/dft-inputs/submit_template.shr", 'r') as f:
                slurm = f.readlines()

            slurm[2] = slurm[2].replace("N", str(i))
            slurm[3] = slurm[3].replace("N", str(i))
            slurm[4] = slurm[4].replace("N", str(i))
            
            slurm[16] = slurm[16].replace("()", "("+" ".join(batch)+")")

            with open(f"{mlip_name}/dft-inputs/submit_batch{i}.sh", 'w') as f:
                f.writelines(slurm)