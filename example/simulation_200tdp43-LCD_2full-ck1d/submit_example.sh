#!/bin/bash

# create configuration
python3 main_hps_phosphorylation.py -i simulation_200tdp43-LCD_2full-ck1d/input_300K_create_conf.in -c -r 30

# run simulation
CUDA_VISIBLE_DEVICES=1 python3 main_hps_phosphorylation.py -i simulation_200tdp43-LCD_2full-ck1d/input_300K.in -m HPS_cp -r 30 >> example.out



