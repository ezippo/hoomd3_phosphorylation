#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gsd, gsd.hoomd

traj = gsd.hoomd.open("ck1d-rigid_tdp43_ex7_dump.gsd", 'rb')

print(traj[0].log.keys())
print(traj[0].particles.typeid)
print(traj[13].particles.typeid)
