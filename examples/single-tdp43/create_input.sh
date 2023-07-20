#!/bin/bash

arr=()
while read -r line
do
    arr+=(${line})
done < seeds.txt

for INDEX in 1 2 3 4 5 6
do
    mkdir -p sim_${INDEX}
    
    NSTEPS=2000
    TEMP=300
    BOXN=200
    START=1
#    FILEST="sim_${INDEX}/thermalized_ck1d-rigid_multi-tdp43_start${INDEX}.gsd"
#    FILELOG="sim_${INDEX}/therm${INDEX}_ck1d-rigid_multi-tdp43"
    SEED=${arr[${INDEX}-1]}
    DTDUMP=200
    DTLOG=5000000
    DTBCKP=10000000
    DTT=1000000
    sed "s/SEED/${SEED}/" template | sed "s/NSTEPS/${NSTEPS}/" | sed "s/TEMP/${TEMP}/" | sed "s/BOXN/${BOXN}/" | sed "s/START/${START}/" | sed "s/INDEX/${INDEX}/" | sed "s/INDEX/${INDEX}/" | sed "s/DTDUMP/${DTDUMP}/" | sed "s/DTLOG/${DTLOG}/" | sed "s/DTBCKP/${DTBCKP}/" | sed "s/DTT/${DTT}/"  > sim_${INDEX}/input${INDEX}_sim.in
done
