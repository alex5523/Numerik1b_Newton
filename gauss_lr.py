"""
# -*- coding: utf-8 -*-
Created: 11.10.2023
@Author: Alexandra Johansen
Numerik 1a, project 1
"""

import numpy as np

# Gauss-function
def gauss(A, b):
    # LU-Decomposition
    L, U = lu(A)

    # Forwards substitution
    z = forw_sub(L,b)

    # Backwards substitution
    x = back_sub(U, z)

    return x

# LU-decomposition
def lu(A):
    m, n = np.shape(A)
    if m != n:
        raise ValueError("Error: A not squared!")

    U = np.copy(A)
    L = np.eye(n)

#for every column in the range, (n=1,2,..,10), j starts at 0, so we have to start one before where n starts, other not defined 
    for j in range (n-1):                   # the last column is not needed <false => next row does that 
        for i in range(j+1, n):             # i = j+1 (row number one higher than column number)
            if U[i][j] != 0.0:
                L[i][j] = U[i][j]/U[j][j]
                U[i,:] -= L[i][j]*U[j,:]
    return L, U

# Forwards-substitution
def forw_sub(L, b,):
    m, n = np.shape(L)

    mb = len(b)
    if n != mb:
        raise ValueError("Error: A and b are not of the same dimension")

    z = np.zeros_like(b, dtype=float)
    z[0] = b[0]

    for i in range(1,n):                    # first column is not needed
        if np.allclose(abs(L[i][i]), 0.0) == True:
            raise ValueError("Error! Divide by zero error")

        sum = 0.0    # scalar factor
        for j in range(i):
            sum += (L[i,j] * z[j])          # assuming our algorithm works so far, we don't divide by the pivot element, which should be 1  
        z[i] = (b[i] - sum)
    return z


# Backwards-substitution
def back_sub(U,z):
    m, n = np.shape(U)

    x = np.zeros_like(z,dtype=float) 
    x[0] = z[0]

    for i in reversed(range(n)):            # from the bottom and up
        if np.allclose(abs(U[i][i]), 0.0) == True:
            raise ValueError("Error! Divide by zero error")

        sum = 0.0
        for j in range(i+1, n):             # for every row in the range, start one row down, go to the last
           sum += (U[i,j] * x[j])           # build scaling factor by multiplying element under pivot, with pivot element
        x[i] = (z[i] - sum)/U[i,i]          # xi = zi - (u_i,j*x_j)/u_i,i
    return x








