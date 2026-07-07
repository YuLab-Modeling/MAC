# MAC
## Benchmarking Universal Machine-Learning Interatomic Potentials for Accelerated Adsorption Energy Evaluation with DFT Single-Point Calculations

{ DOI placeholder }

This repository contains the code that we used to run the MLIP-based Adsorption Calculation (MAC) workflow. It is uploaded to be used as a reference for your implementation of the MAC workflow. We recommend using separate conda environments for different MLIPs. 

The main scripts are:
	
	run_mattersim.py
		requires mattersim 1.2.0 library
		
	run_uma.py
		requires ==fairchem-core 2.7.1==  library
		requires uma-s-1p1.pt file from https://fair-chem.github.io
		
	run_escn.py
        requires fairchem-core 1.10.0  library
        requires escn_l6_m3_lay20_all_md_s2ef.pt file from https://fair-chem.github.io
        
	run_m3gnet.py

	createVASPInputs.py
	
A typical workflow:
	
	conda activate MLIP_Mattersim
	run_mattersim.py
	echo "Mattersim" | python3 createVASPInputs.py
	
	<run DFT>

The requirements.txt file is not present, the remaining required python libraries cam be inferred.

ABCluster (http://zhjun-sci.com/abcluster.html) is required to run the scripts.
