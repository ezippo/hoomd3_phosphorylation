#!/bin/bash

arr=()
#while read -r line
#do
#    arr+=(${line})
#done < seeds.txt

for INDEX in 0
do
    mkdir -p sim_${INDEX}
    
    NSTEPS=5000
    TEMP=300
    BOXN=200
    START=0
#    FILEST="sim_${INDEX}/thermalized_ck1d-rigid_multi-tdp43_start${INDEX}.gsd"
#    FILELOG="sim_${INDEX}/therm${INDEX}_ck1d-rigid_multi-tdp43"
    #SEED=${arr[${INDEX}-1]}
    SEED=$RANDOM
    DTDUMP=1000
    DTLOG=2500
    DTBCKP=1000
    DTT=1000
    sed "s/SEED/${SEED}/" template | sed "s/NSTEPS/${NSTEPS}/" | sed "s/TEMP/${TEMP}/" | sed "s/BOXN/${BOXN}/" | sed "s/START/${START}/" | sed "s/INDEX/${INDEX}/" | sed "s/INDEX/${INDEX}/" | sed "s/DTDUMP/${DTDUMP}/" | sed "s/DTLOG/${DTLOG}/" | sed "s/DTBCKP/${DTBCKP}/" | sed "s/DTT/${DTT}/"  > sim_${INDEX}/input${INDEX}_hoomd3_sim.in
done
