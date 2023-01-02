#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gsd, gsd.hoomd
print('a')
traj = gsd.hoomd.open("therm1_ck1d-rigid_multi-tdp43restart1.gsd", 'rb')
print('b')
print(len(traj))

