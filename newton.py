#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:33:56 2023

@author: alexandrajohansen

Newton with more variables

Num1b P1, WiSe 23/24

"""
import numpy as np
from scipy.linalg import qr, LinAlgError
from scipy.optimize import fsolve
from gauss_lr import back_sub



############ Newton method for a system of equations #########################

def newtonsys(F_expr, J_expr, x_init, tol = 1e-6, iter_max = 50, x_max = 1e3):
    
    """
    Input: 
        F_expr: A string of equations (vector as a list)
        J_expr: A more dimensional string of equations? (matrix as a list)  
        x_inti: Initial guess
        tol: Tolerance
        iter_max: Maximum number of iterations
    
    Increment is in x
    Residual is in y 
    """
    
    # Check dimension of System 
    n_eq = len(F_expr)
    
    if np.ndim(x_init) == 1:
        n_unknowns = len(x_init)
    else:
        raise ValueError("x_init must be a 1D array")
        
    if n_eq != n_unknowns:
        raise ValueError('The number of equations must be equal to the number of unknowns')
        
    
    # Initialisations
    x = x_init      # Solution vector to be updated and iterated on in while loop
    n = n_eq        # Dimension of system to be solved
    iter_count = 0  # Iteration counter to be incremented in while loop

    # error calculation?
    err = 1 + tol   # Makes sure loop runs at least once

    # Vectors and matrices to be evaluated to be evaluated by functions
    Fxn = np.zeros(n)
    Jxn = np.zeros((n,n))
    
    
    # Define functions based on string inputs
    def F_func(x):
        variables = {f'x{i+1}': x[i] for i in range(n)}    
        return np.array([eval(expr, variables) for expr in F_expr])
    
    def J_func(x):
            variables = {f'x{i+1}': x[i] for i in range(n)}
            return np.array([[eval(expr, variables) for expr in row] for row in J_expr])


    # While loop
    while iter_count < iter_max and err > tol and np.all(np.abs(x)) < x_max:
        
        """
        STOP if 
            
        DIVERGING
        2) Absolute value of x is getting out of bounds    
        1) Max iterations have been reached

        CONVERGED
        1) err > tol
    
        OTHERWISE continue        
        """   
        
        # Evaluate 
        Fxn = F_func(x)
        Jxn = J_func(x)
            
        # Solve LSE
        try:
            Q, R = qr(Jxn)
            deltax = back_sub(R, Q.T @ - Fxn)  # Increment?

        except Exception:
            print("Matrix ist Singulär")
            raise ValueError('Method may not converge, Jacobian is singular')
        
        # Increment x
        x = x + deltax
        
        r = np.linalg.norm(F_func(x))
                
        # Increment the counter
        iter_count += 1
        
        # Calculate norm of error/increment?
        err = np.linalg.norm(deltax)
        
        
        # Look at x values
        print(x)

        
        if iter_count == iter_max:
            raise ValueError("Method did not converge within the maximum amount of iterations")
            # Oder ValueError?

    # Return estimated x solution and reached iterations        
    print("Method converged")
    return x, iter_count


# Test
# 1
F1_expr = ['x1**2 + x2**2 - 2', 'x1**2 - x2**2 - 1']
J1_expr = [['2*x1', '2*x2'], ['2*x1', '-2*x2']]
x_init = np.array([1.0, 1.0])

result = newtonsys(F1_expr, J1_expr, x_init)
print("1:", result)

# 2
F2_expr = ['x1**2 + x2**2 - 9', 'x1**2 + (x2-5)**2 - 9']
J2_expr = [['2*x1', '2*x2'], ['2*x1', '2*x2 - 10']]

#x_init = np.array([1.0, 4.0])   #
x_init = np.array([1.0, 1.0])   #

result = newtonsys(F2_expr, J2_expr, x_init)
print("2:",result)

# 3
F3_expr = ['x1**2 - 9', '(-2*x1)**2']
J3_expr = [['2*x1'], ['-2*x1']]

#x_init = np.array([1.0, 4.0])   #
x_init = np.array([1.0, 2.0])   #

result = newtonsys(F3_expr, J3_expr, x_init)
print("3:", result)









