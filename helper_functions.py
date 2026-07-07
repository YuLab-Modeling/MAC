from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

n_layers = 8

x_dim_dict = {
    1 : 4.230,
    2 : 8.460,
    3 : 12.690,
    4 : 16.920,
    5 : 21.150,
    6 : 25.380,
}

y_dim_dict = {
    2 : 5.982,
    4 : 11.964,
    6 : 17.946,
    8 : 23.928,
}


def get_radius(smiles : str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())         #generate 3D ocordinates
    AllChem.UFFOptimizeMolecule(mol)  

    conf = mol.GetConformer()

    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    centroid = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)

    return distances.max()

def get_min_dimension_size(smiles : str):

    x_dim_num_opt = y_dim_num_opt = 0

    delta_d = 6
    min_dim = get_radius(smiles)*2 + delta_d
    #print(min_dim)

    for x_dim_num, x_dim_length in x_dim_dict.items():
        if x_dim_length > min_dim:
            x_dim_num_opt = x_dim_num
            break

    for y_dim_num, y_dim_length in y_dim_dict.items():
        if y_dim_length > min_dim:
            y_dim_num_opt = y_dim_num
            break

    if x_dim_num_opt and y_dim_num_opt:
        return x_dim_num_opt, y_dim_num_opt
    else:
        raise ValueError(f'Dimensions were not found for: {smiles} (R = {min_dim-6})')
    

#convergence counter
class ConvergenceCounter:
    def __init__(self, mlip : str, id : int):

        self.mlip = mlip
        self.id = id

        with open(f'{mlip}/convergence-counters/convergence_counter_id{self.id}.txt', 'w') as f:
            f.write('')

    def write(self, n_total, n_converged):

        perc = round(n_converged/n_total*100, 3)

        with open(f'{self.mlip}/convergence-counters/convergence_counter_id{self.id}.txt', 'a') as f:
            f.write(f'id : {self.id}, convergence : {n_converged}/{n_total} ({perc} %)\n')



# write slab dimensions for each mol
def writeDims(ids):

    molecules_to_skip = np.array([176, 256, 78])
    ids = np.array([id for id in ids if id not in molecules_to_skip])

    for id in ids:
        with open(f"abcluster/{id}.inp", "r") as f:
            lines = f.readlines()
            dims_str = lines[11].split('/')[-1].split('_')[2]
            x, y = int(dims_str[0]), int(dims_str[1])
            x_dim, y_dim, z_dim = x_dim_dict[x], y_dim_dict[y], 50.0

        with open(f"abcluster/dimensions.txt", "a+") as f:
            f.write(f"id{id}\tx{x}x{y}x8\n")

        print("Done")