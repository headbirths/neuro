
##########################################
# Approximate a Normal (Gaussian) distribution function
# with a piecewise quadratic function.
# Perform with minimal integer arithmetic (16-bit):
#  *  3 table look-ups
#  *  one condition
#  *  one shift-left
#  *  4 multiplications
#  *  2 additions, and
#  *  1 subtraction
#
# See https://headbirths.wordpress.com/2020/03/23/approximating-normal/
#
# Put onto github after it had got corrupted when editting with new Wordpress editor.
#
##########################################


# Based on optimization example at
#    https://www.youtube.com/watch?v=iSnTtV6b0Gw

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize



# Calculation of fixed point coefficients for
# 11-segment quadratic approximation
# of the Normal (Gaussian) Distribution function
# using the Python 'scipy.optimize.minimize' optimization function
# for fast, simple calculation on e.g. microcontrollers

# Optimization (fitting) conditions
# Global: normalized when doing fitting.
the_mean = 0
the_reciprocal_of_std_dev = 1
# Using the reciprocal means there aren't any division operations

# Naming of threshold and coefficients: segment n covers n-1.5 to n-0.5
segment_thresholds = np.linspace(0,5.5,12)
num_segments = len(segment_thresholds)-1
print('# Segment partitioning (%d segments):' % (num_segments))
for i in range(num_segments):
    print('# Segment #%d from %.3f to %.3f' % (i, segment_thresholds[i], segment_thresholds[i+1]))

# Constant:
one_over_sqrt_2pi = 0.3989422804014327 # 1/(np.sqrt(2*np.pi))

def pdf_vector(x_vector, mean: float, reciprocal_of_std_dev: float):
    s1 = one_over_sqrt_2pi * reciprocal_of_std_dev
    y = []
    for x in x_vector:
        s2 = np.exp(-(np.square((x - mean)*reciprocal_of_std_dev)/2))
        y.append(s1 * s2)
    return y

# For a single segment, find the sum of squares of errors.
def cost_function(xrange, a, b, c, mean, reciprocal_of_std_dev):
    # Wanting to *minimize* cost
    step_size = 1/128 # x resolution:step size (100 steps per half standard deviation)
    cost    = 0.0
    x = xrange[0]
    while x < xrange[1]:
        y = pdf_vector([x], mean, reciprocal_of_std_dev)
        y_est = a*x**2 +b*x + c
        cost += (y_est - y)**2 # Sum of squares of errors
        x += step_size
    return cost

# coeff is an array of the quadratic coefficients
def calcCost(coeff):
    cost  = 0
    for i in range(num_segments):
        cost += cost_function([ segment_thresholds[i], segment_thresholds[i+1]], coeff[3*i],  coeff[3*i+1],  coeff[3*i+2],  the_mean, the_reciprocal_of_std_dev)
    return cost # single value: sum of squares of errors over all segments

def objective(x):
    # Minimize the error matching sections of quadratics to a normal function
    return calcCost(x)

def report_coefficients(coeff):
    print('# Segment quadratics:')
    for i in range(num_segments):
        print('# Segment #%d y=(%.8f)x^2 + (%.8f)x + (%.8f)' % (i, coeff[3*i], coeff[3*i+1],  coeff[3*i+2]))

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def approx_normal_vector(x_vector, coeffs, mean, reciprocal_of_std_dev):
    # x_vector is a vector: enclose '[x]' for a single value
    y_vector = []
    for i in range(len(x_vector)):
        norm = abs(x_vector[i]-mean) * reciprocal_of_std_dev # normalized
        segment = math.floor(norm * 2) # which 1/2-std-dev segment does the item belong to?
        if (segment >= num_segments):
            y_vector.append(0) # Result is zero beyond the last segment
        else:
            y_vector.append(quadratic(norm, coeffs[3*segment],  coeffs[3*segment+1],  coeffs[3*segment+2])  * reciprocal_of_std_dev)
    return y_vector

def find_largest_errors(coeffs):
    bins = np.linspace(-10.0,10.0,201)
    reference = pdf_vector(bins, the_mean, the_reciprocal_of_std_dev)
    approx    = approx_normal_vector(bins, coeffs, the_mean, the_reciprocal_of_std_dev)
    # https://www.geeksforgeeks.org/python-maximum-minimum-elements-position-list/
    err = [m - n for m,n in zip(approx, reference)] # Subtract *list* elements
    for sign in ['positive', 'negative']:
        if sign=='positive' : yerr = max(err)
        if sign=='negative' : yerr = min(err)
        xval = bins[err.index(yerr)]
        yref = pdf_vector([xval], the_mean, the_reciprocal_of_std_dev)
        yact = approx_normal_vector([xval], coeffs, the_mean, the_reciprocal_of_std_dev)
        print("# Maximum %s error %.8f at %.3f is %.8f instead of %.8f." % (sign, yerr, xval, yact[0], yref[0]))

def report_list(x_array, name, precision=6):
    print('# %s = [' % (name), end='')
    for x in x_array:
        if precision==0:
            print('%4.0f, ' % (x), end='')
        else:
            print('%10.6f, ' % (x), end='')
    print(']')

def split_up(coeffs):
    a, b, c = [], [], []
    for i in range(len(coeffs)):
        if i % 3 == 0: a.append(coeffs[i])
        if i % 3 == 1: b.append(coeffs[i])
        if i % 3 == 2: c.append(coeffs[i])
    return a, b, c

def plot_results(coeffs, show = 'pdf'):
    bins = np.linspace(-10.0,10.0,201)
    plt.figure(figsize=(10,7))
    axes = plt.gca()
    plt.xlabel("$x$")
    reference = pdf_vector(bins, the_mean, the_reciprocal_of_std_dev)
    approx    = approx_normal_vector(bins, coeffs, the_mean, the_reciprocal_of_std_dev)
    if show == 'pdf':
        plt.ylabel("pdf")
        plt.plot(bins, reference, color='red',  label="True pdf")
        plt.plot(bins, approx,    color='navy', label="Approx pdf")
    elif show == 'absolute error':
        err = [m - n for m,n in zip(approx, reference)] # Subtract *list* elements
        plt.ylabel("abs error")
        plt.plot(bins, err,       color='navy', label="Error")
    plt.legend()
    plt.show()

def quantize_coefficients(coeffs, bits):
    y = []
    print("# [", end='')
    for x in coeffs:
        integer = round(x*(2**bits))
        y.append( integer/(2**bits) )
        print("%d," % (integer), end='')
    print("]")
    return y

print('##### BEFORE #####')
# Initial Guess, pre-optimization
# Lazy here: just set every coefficient set to zero
x0 = np.zeros([3*num_segments]) # 3 coefficients (a, b and c) per segment  
report_coefficients(x0)
print('# Cost function: %.8f' % (calcCost(x0)))
plot_results(x0, show='absolute error')

print('##### OPTIMIZING #####')
# Non-linear: use Sequential Least SQuares Programming (SLSQP)
sol = minimize(objective, x0, method='SLSQP', options={'disp':True})
xOpt = sol.x
errOpt = sol.fun

print('##### AFTER #####')
find_largest_errors(xOpt)
report_coefficients(xOpt)
print('# Cost function: %.8f' % (calcCost(xOpt)))
a, b, c = split_up(xOpt)
report_list(a, 'a')
report_list(b, 'b')
report_list(c, 'c')

print('##### Quantized to 8-bit signed fixed point #####')
quantized_xOpt = quantize_coefficients(xOpt, 8-1)   
find_largest_errors(quantized_xOpt)

print('##### Quantized to 16-bit signed fixed point #####')
quantized_xOpt = quantize_coefficients(xOpt, 16-1)   
find_largest_errors(quantized_xOpt)
a, b, c = split_up(quantized_xOpt)
report_list([n*32768 for n in a], 'a', precision=0)
report_list([n*32768 for n in b], 'b', precision=0)
report_list([n*32768 for n in c], 'c', precision=0)

print('##### USAGE #####')
# Example usage:
the_mean = 3
the_reciprocal_of_std_dev = 1/2 # i.e. variance=4
plot_results(quantized_xOpt)
plot_results(quantized_xOpt, show='absolute error')
find_largest_errors(quantized_xOpt)



# Segment partitioning (11 segments):
# Segment #0 from 0.000 to 0.500
# Segment #1 from 0.500 to 1.000
# Segment #2 from 1.000 to 1.500
# Segment #3 from 1.500 to 2.000
# Segment #4 from 2.000 to 2.500
# Segment #5 from 2.500 to 3.000
# Segment #6 from 3.000 to 3.500
# Segment #7 from 3.500 to 4.000
# Segment #8 from 4.000 to 4.500
# Segment #9 from 4.500 to 5.000
# Segment #10 from 5.000 to 5.500
##### BEFORE #####
# Segment quadratics:
# Segment #0 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #1 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #2 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #3 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #4 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #5 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #6 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #7 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #8 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #9 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Segment #10 y=(0.00000000)x^2 + (0.00000000)x + (0.00000000)
# Cost function: 18.13364415
##### OPTIMIZING #####
#> Optimization terminated successfully.    (Exit mode 0)
#>             Current function value: 0.00018880714614915984
#>             Iterations: 36
#>             Function evaluations: 1313
#>             Gradient evaluations: 36
##### AFTER #####
# Maximum positive error 0.00189853 at -1.000 is 0.24386926 instead of 0.24197072.
# Maximum negative error -0.00252195 at -2.000 is 0.05146902 instead of 0.05399097.
# Segment quadratics:
# Segment #0 y=(-0.17973425)x^2 + (-0.00505443)x + (0.39916185)
# Segment #1 y=(-0.07666328)x^2 + (-0.10744328)x + (0.42505809)
# Segment #2 y=(0.08668947)x^2 + (-0.44298249)x + (0.60016227)
# Segment #3 y=(0.03994489)x^2 + (-0.29089544)x + (0.47401580)
# Segment #4 y=(0.00659829)x^2 + (-0.10214379)x + (0.22936344)
# Segment #5 y=(-0.00053429)x^2 + (-0.02287305)x + (0.07667299)
# Segment #6 y=(-0.00066900)x^2 + (-0.00259672)x + (0.01775457)
# Segment #7 y=(-0.00032765)x^2 + (0.00101441)x + (0.00121242)
# Segment #8 y=(-0.00025578)x^2 + (0.00180205)x + (-0.00297611)
# Segment #9 y=(-0.00014274)x^2 + (0.00173502)x + (-0.00501367)
# Segment #10 y=(-0.00028692)x^2 + (0.00272845)x + (-0.00640868)
# Cost function: 0.00018881
# a = [ -0.179734,  -0.076663,   0.086689,   0.039945,   0.006598,  -0.000534,  -0.000669,  -0.000328,  -0.000256,  -0.000143,  -0.000287, ]
# b = [ -0.005054,  -0.107443,  -0.442982,  -0.290895,  -0.102144,  -0.022873,  -0.002597,   0.001014,   0.001802,   0.001735,   0.002728, ]
# c = [  0.399162,   0.425058,   0.600162,   0.474016,   0.229363,   0.076673,   0.017755,   0.001212,  -0.002976,  -0.005014,  -0.006409, ]
##### Quantized to 8-bit signed fixed point #####
# [-23,-1,51,-10,-14,54,11,-57,77,5,-37,61,1,-13,29,0,-3,10,0,0,2,0,0,0,0,0,0,0,0,-1,0,0,-1,]
# Maximum positive error 0.01439278 at 3.400 is 0.01562500 instead of 0.00123222.
# Maximum negative error -0.00782848 at -4.500 is -0.00781250 instead of 0.00001598.
##### Quantized to 16-bit signed fixed point #####
# [-5890,-166,13080,-2512,-3521,13928,2841,-14516,19666,1309,-9532,15533,216,-3347,7516,-18,-750,2512,-22,-85,582,-11,33,40,-8,59,-98,-5,57,-164,-9,89,-210,]
# Maximum positive error 0.00189524 at -1.000 is 0.24386597 instead of 0.24197072.
# Maximum negative error -0.00253833 at -2.000 is 0.05145264 instead of 0.05399097.
# a = [    -0,     -0,      0,      0,      0,     -0,     -0,     -0,     -0,     -0,     -0, ]
# b = [    -0,     -0,     -0,     -0,     -0,     -0,     -0,      0,      0,      0,      0, ]
# c = [     0,      0,      1,      0,      0,      0,      0,      0,     -0,     -0,     -0, ]
##### USAGE #####
# Maximum positive error 0.00094762 at 1.000 is 0.12193298 instead of 0.12098536.
# Maximum negative error -0.00126916 at -1.000 is 0.02572632 instead of 0.02699548.




