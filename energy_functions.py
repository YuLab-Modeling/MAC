from ase import io
from ase.optimize import BFGS
from ase.constraints import FixAtoms
#import matgl

#M3GNet
#from matgl.ext.ase import M3GNetCalculator

#MACE-MP: https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html#foundation-models
#from mace.calculators import mace_mp

#Mattersim: https://microsoft.github.io/mattersim/examples/relax_example.html | 2024
#from mattersim.forcefield.potential import MatterSimCalculator

#META FairChem Core: https://github.com/facebookresearch/fairchem | 2024
#from fairchem.core import pretrained_mlip, FAIRChemCalculator

import numpy as np
from helper_functions import x_dim_dict, y_dim_dict

n_layers = 8

def calculate_slab_energy(mlip_name : str, mlip_pot, fmax = 0.02): #-------------------------------------------------------------------------------------------------------------

    #load the slab
    slab = io.read(f"xyz-s/all/zn.xyz")

    # explicitly set charge and spin
    slab.info.update({"spin": 1, "charge": 0})

    # set the potential for optimization
    slab.calc = mlip_pot

    #set PBC
    slab.set_cell([9.405, 9.405, 20., 90, 90, 60])
    #slab.center(vacuum=7.5, axis=2)
    slab.pbc=True

    #fix the bottom half
    n_atoms_to_fix = 24       #half of all atoms
    constraint = FixAtoms(indices=np.arange(n_atoms_to_fix))
    slab.set_constraint(constraint)

    opt = BFGS(slab, logfile=None)
    opt.run(fmax=fmax, steps=300)
    slab_energy = slab.get_potential_energy()

    # write the optimized structure
    io.write(f'{mlip_name}/opt-slabs/zn_opt.xyz', slab)

    return slab_energy

#calculate_M3GNet_slab_energy(2,2)


def calculate_molecule_energy(mlip_name, mlip_pot, id : int, fmax = 0.02): #---------------------------------------------------------------------------------------------------------------------

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

    opt = BFGS(molecule, logfile=None)
    opt.run(fmax=fmax, steps=300)

    molecule.wrap()

    mol_energy = molecule.get_potential_energy()

    # write the optimized structure
    io.write(f'{mlip_name}/opt-mols/{id}_{mlip_name}_opt.xyz', molecule)

    return mol_energy

#calculate_M3GNet_molecule_energy(1)


def calculate_slab_plus_molecule_energy(mlip_name : str, mlip_pot, id : int,  n_calcs : int, i_calc : int, fmax = 0.02): #---------------------------

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


