#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import MDAnalysis as mda


u = mda.Universe("ck1d-rigid_tdp43_ex6_dump.gsd")
ag = u.atoms
print(ag.types)

types = []
for ts in u.trajectory:
    types.append(ag.types[149])

print(types)
