#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gsd, gsd.hoomd

traj = gsd.hoomd.open("therm1_ck1d-rigid_multi-tdp43restart1.gsd", 'rb')

print(len(traj))

