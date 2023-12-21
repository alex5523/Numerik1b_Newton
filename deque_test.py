#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:37:24 2023

@author: alexandrajohansen
"""
import numpy as np
from collections import deque

x = 0.0  # or any other initial value
x_deque = deque(maxlen=3)

while x < 10:
    x_deque.append(x)
    x += 1

print(list(x_deque))

try:
    if x_deque[-3] < x_deque[-2] < x_deque[-1]:
        raise ValueError("X is increasing!")
except ValueError as e:
    print(e)
else:
    print(x_deque)
