import numpy as np
from helper_functions import get_min_dimension_size, get_radius, n_layers

def write_abcluster_input(id : str, num_cals: 500):
    
    """generate {num_calcs} initial configurations of the molecules on a slab"""

    with open("abcluster/template.inp", 'r') as f:
        lines = f.readlines()

    lines[0] = lines[0].replace('LM', f"id{id}_{num_cals}")         #output dir
    lines[1] = lines[1].replace('500', str(num_cals))               #number of calcs
    lines[15] = lines[15].replace('sample', id)                #molecule id
    lines[16] = f'random  -0.5 -0.5 5 0.5 0.5 7\n'

    with open(f"abcluster/{id}.inp", 'w') as f:
        f.writelines(lines)

import os

def run_abcluster_input(base_folder : str, id : str, n_calcs : int):
    os.chdir(base_folder)
    os.chdir('abcluster/')

    os.system(f'rm -rf id{id}*{n_calcs}')    #delete the previous locals so that the geom doesn't restart

    os.system(f'geom {id}.inp > {id}.out')
    os.chdir('..')


import glob

def get_min_z_dist(xyz):
    """ Input: xyz file
        Output: minimum distance of molecule's atoms to the slab in z-direction
    """

    with open(xyz, 'r') as f:
        lines = f.readlines()

    lines_mol = [line for line in lines[2:] if not 'Zn' in line]
    lines_slab = [line for line in lines[2:] if 'Zn' in line]
    slab_z = np.max([float(line.split()[-1]) for line in lines_slab])

    z_coords = [float(line.split()[-1]) for line in lines_mol]
    
    min_z_coord = np.min(z_coords)

    return min_z_coord - slab_z

def filter_configurations(id, n_calcs, z_low = 1.0, z_upp = 3.0):

    xyzs = sorted(glob.glob(f"abcluster/id{id}_{n_calcs}/*xyz"))
    n_fit = 0
    for xyz in xyzs:
        #print(xyz)

        min_dist_z = get_min_z_dist(xyz)

        if not z_low <= min_dist_z <= z_upp:        # lower and upper bounds for z
            os.system(f'rm {xyz}')
        else:
            n_fit += 1

    return n_fit
