#!/bin/bash

PBEPath="/apps/VASP/potentials/potpaw_PBE.54"

elements_str=$(sed -n '6p' POSCAR)
IFS=' ' read -r -a elements_arr <<< $elements_str

for el in ${elements_arr[@]}; do
    echo $PBEPath/$el/POTCAR 
    cat $PBEPath/$el/POTCAR >> ./POTCAR
done