#!/bin/bash

SimNum=$1

rm -rf "Res/Sim$SimNum"
mkdir "Res/Sim$SimNum"

cp Parameter.py "Res/Sim$SimNum/Parameter.py"



sbatch CanonicalAnnealing.pbs $SimNum
