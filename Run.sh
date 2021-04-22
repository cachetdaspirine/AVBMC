#!/bin/bash

for SimNum in {0..5}
do
	rm -rf "Res/Sim$SimNum"
	mkdir "Res/Sim$SimNum"
	sed "3s/.*/SimNum =+$SimNum/" Parameter.py > Res/Sim$SimNum/Parameter.py
	sbatch CanonicalAnnealing.pbs $SimNum
done
