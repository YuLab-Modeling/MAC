

C6_vals = {
    "Zn" : 3.419,
    "H" : 0.140,
    "C" : 1.750,
    "N" : 1.230,
    "O" : 0.7,
    "S" : 5.570,
}

R0_vals = {
    "Zn" : 1.531,
    "H" : 1.001,
    "C" : 1.452,
    "N" : 1.397,
    "O" : 1.342,
    "S" : 1.683,
}

def writeIncar(mlip_name, str_name, structType):
    with open(f"{mlip_name}/dft-inputs/{str_name}/POSCAR", 'r') as f:
        elements = f.readlines()[5].split()

    with open(f"{mlip_name}/dft-inputs/INCAR_template", "r") as f:
        incar = f.readlines()

    incar[42] = incar[42][:-1]            
    incar[43] = incar[43][:-1]
    incar[44] = incar[44][:-1]

    for element in elements:
        incar[42] += f" {element}"
        incar[43] += f" {C6_vals[element]}"
        incar[44] += f" {R0_vals[element]}"

    incar[42] += "\n"
    incar[43] += "\n"
    incar[44] += "\n"

    if structType == "mol":
        incar[47] = incar[47].replace("3", "0")

    with open(f"{mlip_name}/dft-inputs/{str_name}/INCAR", "w") as f:
        f.writelines(incar)