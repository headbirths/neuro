##########################################################################################


# "Predator-Prey Relationships - The Lotka-Volterra Model" Aubrey Moore
#    https://aubreymoore.github.io/ALBI345F17/pdfs/Lotka-Volterra-Model.html

# "Coupling in predator-prey dynamics: Ratio-Dependence" Roger Arditi Lev R.Ginzburg

# "The Robust Volterra Principle" Michael Weisberg & Kenneth Reisman
#    http://www.phil.upenn.edu/~weisberg/documents/VolterraPrinciple3g.pdf

# "Evaluating the symbiosis status of tourist towns: The case of Guizhou Province, China"
#    https://dro.dur.ac.uk/25629/1/25629.pdf

# Term notes "Lotka-Volterra ( Predator prey)" Dr Christopher Prior
#    https://www.maths.dur.ac.uk/~ktch24/term1Notes(10).pdf

# Coming up with a predator-prey model that accounts for predator satiation
#    https://math.stackexchange.com/questions/3610291/coming-up-with-a-predator-prey-model-that-accounts-for-predator-satiation


from random import *
from numpy import *

# declare...
xA = xB = f24A = f2A = f24B = f2B = 0
t_list = []; xA_list = []; xB_list = []; f24A_list = []; f2A_list = []; f24B_list = []; f2B_list = []

# model parameters
k1A = 5;  c24 = 24/24; c2 = 2/24 

def report_state(s):
    if s>0: # results to report
        lo = 100000*s-99999
        hi = 100000*s+1
        t_num = len(t_list[lo:hi])
        #print("   Samp = %d" % (t_num), end='')
        #debug#print("   Lim = [%d:%d]" % (lo, hi), end='')
        fA_list = array(f24A_list[lo:hi])+array(f2A_list[lo:hi])
        yB_list = array(f24B_list[lo:hi])+array(f2B_list[lo:hi])
        print("Max[%6d:%6d] xA,xB|f24A,f2A,f24B,f2B = %6.3f, %6.3f " % (lo, hi, max(xA_list[lo:hi]), max(xB_list[lo:hi])), end='')
        print("| %6.3f %6.3f, %6.3f, %6.3f" % (max(f24A_list[lo:hi]), max(f2A_list[lo:hi]), max(f24B_list[lo:hi]), max(f2B_list[lo:hi])), end='')
        print("")
    #print("Params(set %d): %.3f   %.3f   %.3f / %.3f   %.3f : " % ((s+1), a, b, c24, c2, d), end='')
    #print("Set %d" % (s+1), end='')

def ratio(a, b):                       return a/max(b, 1e-6) # Prevent divide by zero
def limit_malthusian_growth(x, limit): return (1-x/limit)    # prevent negative

TIME_UNITS = 1000; dt = 0.001; max_time = 700
# Initial populations of prey (x) and predators (f24, f2) in regions A and B
xA = 0.5; xB = 0.1; f24A = 0.5; f2A = 0.0; f24B = 0.0; f2B = 0.0;

# empty lists in which to store time and populations
time = 0;
t_list = []; xA_list = []; xB_list = []; f24A_list = []; f2A_list = []; f24B_list = []; f2B_list = []
# initialize lists
t_list.append(time); xA_list.append(xA); xB_list.append(xB);
f24A_list.append(f2A); f2A_list.append(f2A); f24B_list.append(f24B); f2B_list.append(f2B)
state = 0
k1A = k1B = 1 # Default: negligible effect (prey)

a = 0.5; b = d = 0.5 # For 'F1' flapfish, set
b = d = 0.25
b = d = 0.05

for t in range(max_time*TIME_UNITS): # Simulation loop
    if False and t>=216000 and t<=216020:
        # dxB = ( a*xB*(1-xB/k1B)  - b*xB*yB)*dt # Prey on region B
        print("Debug: @%8d : %8.5f | %8.5f %8.5f %8.3f%%" % (t, xA, f24A, f2A, 100*f2A/f24A))

    if (t == 0*TIME_UNITS):    # Initialize: Simplistic classic Lotka-Volterra
        k1A = 999;  k1B = 999  # No carrying capacity limitation
        c24 = 24/24; c2 = 2/24 # Decay parameter: risk death for 24 or 2 hours per day 
    if (t == 120*TIME_UNITS):  # Introduce prey Carrying Capacity to stabilize:
        k1A = 5; k1B = 2       # Less capacity in region B
    if (t > 220*TIME_UNITS) and (t%TIME_UNITS == 0): # Shift 0.1% every time unit
        f24B += 0.001*f24A; f24A -= 0.001*f24A; f2B += 0.001*f2A; f2A -= 0.001*f2A
    if (t == 300*TIME_UNITS):  # Introduce mutation (f2 supplants f24 in both regions A and B
        f2A += 0.01*f24A; f24A -= 0.01*f24A # 1% of samples have mutated

    if (t%100000 == 0) and (t>0): state += 1; report_state(state)

    # Calculate new values for time t, prey x and predators f2 and f24
    dxA   = dt*( a*xA*limit_malthusian_growth(xA, k1A) - b*xA*(f24A + f2A)) # Prey, region A
    dxB   = dt*( a*xB*limit_malthusian_growth(xB, k1B) - b*xB*(f24B + f2B)) # Prey, region B
    df24A = dt*(-c24*f24A + d*xA*f24A) # F24 predators in region A
    df2A  = dt*(-c2 *f2A  + d*xA*f2A ) # F2  predators in region A
    df24B = dt*(-c24*f24B + d*xB*f24B) # F24 predators in region B
    df2B  = dt*(-c2 *f2B  + d*xB*f2B ) # F2  predators in region B
    # Update populations
    xA = xA + dxA;    f24A = f24A + df24A;    f2A = f2A + df2A
    xB = xB + dxB;    f24B = f24B + df24B;    f2B = f2B + df2B
    # Store new values in lists
    time  = time + dt
    t_list.append(time);  xA_list.append(xA);   xB_list.append(xB);
    f24A_list.append(f24A); f2A_list.append(f2A); f24B_list.append(f24B); f2B_list.append(f2B)
    
#### end of simulation loop ####################################
report_state(state+1)
print("")


############################################################################
# Plotting results
############################################################################

import matplotlib.pyplot as plt

p_color = [ 'red', 'orange', 'lawngreen', 'forestgreen', 'dodgerblue', 'rebeccapurple', 'deeppink'   ]
def choose_color(n):
    if n >= 7:
        return 'black'
    else:
        return p_color[n]
    
num_segments = max_time//100

if True:
    title = "Populations in region A"
    fig1 = plt.figure(title, figsize=[10, 4])
    plt.plot(t_list, xA_list,  color='darkgray',    label="prey") 
    for s in range(num_segments): 
        lo = 100000*(s+1)-99999
        hi = 100000*(s+1)+1
        plt.plot(t_list[lo:hi], f24A_list[lo:hi],  color=choose_color(s), label="cathemeral")
        plt.plot(t_list[lo:hi], f2A_list[lo:hi],  color='black',         label="matutinal") 
    plt.xlabel('time')
    plt.ylabel('population')
    plt.title(title)
    #plt.legend(loc='upper left')
    plt.show(block=False)

if False:
    title = "Populations in region A"
    fig1 = plt.figure(title, figsize=[10, 4])
    plt.plot(t_list, xA_list,  color='darkgray',    label="prey") 
    for s in range(num_segments): 
        lo = 100000*(s+1)-99999
        hi = 100000*(s+1)+1
        #plt.plot(t_list[lo:hi], f24A_list[lo:hi],  color=choose_color(s), label="cathemeral")
        #plt.plot(t_list[lo:hi], f2A_list[lo:hi],  color='black',         label="matutinal") 
    plt.xlabel('time')
    plt.ylabel('population')
    plt.title(title)
    #plt.legend(loc='upper left')
    plt.show(block=False)

if True:
    title = "Populations in region B"
    fig1 = plt.figure(title, figsize=[10, 4])
    for s in range(num_segments): 
        lo = 100000*(s+1)-99999
        hi = 100000*(s+1)+1
        plt.plot(t_list[lo:hi], f2B_list[lo:hi],  color='black',         label="matutinal") 
        plt.plot(t_list[lo:hi], f24B_list[lo:hi],  color=choose_color(s), label="cathemeral") 
    #plt.plot(t_list, xB_list,  color='darkgray',    label="prey") 
    plt.xlabel('time')
    plt.ylabel('population')
    plt.title(title)
    #plt.legend(loc='upper left')
    plt.show(block=False)

if True:
    fig2 = plt.figure("Phase plots, region A", figsize=[10, 4])
    #plt.plot(xA_list, f24B_list,  color='black',    label=("set %d" % (s))) 
    for s in range(num_segments): 
        lo = 100000*(s+1)-99999
        hi = 100000*(s+1)+1
        plt.plot(xA_list[lo:hi], f2A_list[lo:hi],  color='black',            label=("set %d" % (s))) 
        plt.plot(xA_list[lo:hi], f24A_list[lo:hi],  color=choose_color(s),    label=("set %d" % (s))) 
    plt.xlabel('prey')
    plt.ylabel('predator')
    plt.title("Phase")
    plt.show(block=False)

plt.show()



############################################################################
# Irrelevant
############################################################################

if False:
    # Quiver plot example
    # https://scipy-cookbook.readthedocs.io/items/LoktaVolterraTutorial.html
    ymax = p.ylim(ymin=0)[1]                        # get axis limits
    xmax = p.xlim(xmin=0)[1]
    nb_points   = 20

    x = linspace(0, xmax, nb_points)
    y = linspace(0, ymax, nb_points)

    X1 , f24A  = meshgrid(x, y)                       # create a grid
    DX1, Df24A = dX_dt([X1, f24A])                      # compute growth rate on the gridt
    M = (hypot(DX1, Df24A))                           # Norm of the growth rate 
    M[ M == 0] = 1.                                 # Avoid zero division errors 
    DX1 /= M                                        # Normalize each arrows
    Df24A /= M

    #-------------------------------------------------------
    # Draw direction fields, using matplotlib 's quiver function
    # I choose to plot normalized arrows and to use colors to give information on
    # the growth speed
    p.title('Trajectories and direction fields')
    Q = p.quiver(X1, f24A, DX1, Df24A, M, pivot='mid', cmap=p.cm.jet)
    p.xlabel('Number of rabbits')
    p.ylabel('Number of foxes')
    p.legend()
    p.grid()
    p.xlim(0, xmax)
    p.ylim(0, ymax)
    f2.savefig('rabbits_and_foxes_2.png')




# Demonstration in phases:
# 1. 'Classic' Lotka-Volterra oscillations, unstable (no homeostasis)
# 2. Introduce large Malthusian limit for practical purposes of plotting
# 3. Real scenario: prey population is limited by prey's satiation limit. Limits satiation.



# The wind is like a second predator



# FREE WILL AND GUILT
# ===================
# Free will: the combination of intention followed by action leading to the sensation of will
# can be mapped across to the legal concept fo guilt: the combination of 'mens rea' and 'actus reus':
# a guilty mind  AND a guilty act'; the combination of a bad intention followed by a bad act.



# The classic Lotka-Volterra equations neatly fit into what has been said about entropy beforehand.
# The collective entropic decay of the predator population
# is counteracted by the negentropy (or 'free energy') gained from the collective prey.
# The collective Malthusian exponential growth of the prey is tempered
# by being that very same negentropy supplied to the predators.


# Holling type II: 

# General Gause-Kolmogorov equation:
#   dx/dt =  a.x.f(x) - y.g(x, y)
#   dy/dt = -c.y      + y.g(x, y).d
# where  
#   a = Malthusian growth rate.
#   c = mortality: here scaled by 1, 1/2, 1/12, 1/24 depending on critter type.
#   d = efficiency, here 100%
# using a, b, c, d used so far.
#
# The classic Lotka-Volterra equation is achieved with f(x) = 1; g(x, y)=x.
# This can be converted to the *stable* form of the Lotka-Volterra equation by setting logistic f(x) = (1 - x/k)

# Popular simple realistic: g(x, y) = s.x/(1+s.h.x)
#   s = searching efficiency
#   h = handling time (holding, stifling, devouring, digesting)
# Try: g(x, y) = x/(1+x)

# Hence negentropy = y.x/(1+x):
#   Proportional to y
#   When x gets high, the consumption rategoes down
#   When x=1, consumption is halved
# Want to have a system with as few parameters as possible. Here just a and c()

# Here: there is no handling time. Probabilistic competition between predators.

# How about:
#   dx/dt =  a.x - g(x, y)
#   dy/dt = -c.y + g(x, y)
# where g(x, y) = x.y.(1 - y/k)
# As prey gets 


###################################
# TODO
###################################

#

# Region A negentropy production...
#Max[     1:100001] xA,xB|f24A,f2A,f24B,f2B|H = 10.000,  2.000 |  7.073  0.000,  0.000,  0.000 | 1284348.165
#Max[100001:200001] xA,xB|f24A,f2A,f24B,f2B|H = 10.000,  2.000 |  7.073  0.000,  0.000,  0.000 | 1300210.052
#Max[200001:300001] xA,xB|f24A,f2A,f24B,f2B|H = 10.000,  2.000 |  7.073  0.000,  0.000,  0.000 | 1370013.560
##
#Max[300001:400001] xA,xB|f24A,f2A,f24B,f2B|H =  6.607,  2.000 |  4.388  0.000,  0.003,  0.000 | 605503.695
#Max[400001:500001] xA,xB|f24A,f2A,f24B,f2B|H =  4.798,  1.999 |  1.308  0.000,  0.003,  0.000 | 627243.021
#Max[500001:600001] xA,xB|f24A,f2A,f24B,f2B|H =  4.798,  1.999 |  1.294  5.742,  0.003,  2.870 | 147044.878
#Max[600001:700001] xA,xB|f24A,f2A,f24B,f2B|H =  0.665,  0.396 |  0.000  3.216,  0.000,  2.254 | 107339.785

# Region B negentropy production...
#Max[     1:100001] xA,xB|f24A,f2A,f24B,f2B|H = 10.000,  2.000 |  7.073  0.000,  0.000,  0.000 |  0.000
#Max[100001:200001] xA,xB|f24A,f2A,f24B,f2B|H = 10.000,  2.000 |  7.073  0.000,  0.000,  0.000 |  0.000
#Max[200001:300001] xA,xB|f24A,f2A,f24B,f2B|H = 10.000,  2.000 |  7.073  0.000,  0.000,  0.000 |  0.000
##
#Max[300001:400001] xA,xB|f24A,f2A,f24B,f2B|H =  6.607,  2.000 |  4.388  0.000,  0.003,  0.000 | 179.114
#Max[400001:500001] xA,xB|f24A,f2A,f24B,f2B|H =  4.798,  1.999 |  1.308  0.000,  0.003,  0.000 | 373.036
#Max[500001:600001] xA,xB|f24A,f2A,f24B,f2B|H =  4.798,  1.999 |  1.294  5.742,  0.003,  2.870 | 84482.127
#Max[600001:700001] xA,xB|f24A,f2A,f24B,f2B|H =  0.665,  0.396 |  0.000  3.216,  0.000,  2.254 | 88650.226




########### compete_b.py SCENARIO ###############

# Prey x and predators f24 (pre-putation) and f2 (post-mutation) within separate regions A and B.
# f2 are 'fitter': they have a lower decay:c2<c24.
# A small fraction of f24 and f2 continually move across from A to B every iteration.
#

# Start with no predators on region B.
# Start with only predator f24A on region A.
# Switch 2% of f24A to f2A after a time.

# Should see: f2A make f24A extinct => Better predictor (higher complexity) emerges (arms race for high-level predator-predators)
# If the conditions are right:
#       f24B NEVER get established on B.
#       f2B grows on B => Better predictor finding a new channel of negentropy production


####################################

# Problem:
# Reducing c (predator decay) isn't increasing y (because of?... k2?)

# See the problem as: predator is competing with the wind.
# As it gets better at predicting when there is food
# it takes a greater proportion of the food.
# BUT: that's not true - when there *is* food, it takes the same amount as if it wasn't

# Next:
# 3. Use alpha, beta, gamma, k1A, k2 instead of a, b, c, d, k1A, k2 so that k1A and k2 are the actual saturatino values.

# Done:
# 1. Extend time by 100; inject one-off food demand in at t=100; see effect (of random passing explorer)
# To do:
# 2. Make b and d the same: represents the energy (negentropy)


####################################


# The Izhikevich model is a dynamical system; but it is a *model* of a neuron and that doesn't mean it *is*
# but the depolarization and repolarization *is* a dynamical system.


#while t < max_time:
#if ((t % 1000)==0): print("t = %d" % (t))
#       t<100: show volatility without inter-species competition (c.f. Izhikevich)
# Red: classic Lotka-Volterra interaction between predator and prey e.g. foxes and rabbits.
# Orange: intruder makes a change of short duration (single timestep) (Dr. Carruthers eats whilst on the islands) - lasting effect
#           No homeostasis - returning to its original orbit
# Light green: Shift to more realistic scenario: competition *between predators*. Leads to steady state.
# Green: reduced predator decay; ...?
# Blue
# Purple
# t>200: stable coupling 
# t>300: shift decay 


# Lotka-Volterra general form:
#   dx = ( a*x - b*x*y)*dt # Prey
#   dy = (-c*y + d*x*y)*dt # Predator
# has been modified by factor of 1-x/k1A or 1-y/k2 to include carrying capacities:
# Predator types 1 and 2 only get a proportion of the prey based on their relative populations


############################################################################
# Separately change parameters a, b, c, d and see the effect
############################################################################

# https://aubreymoore.github.io/ALBI345F17/pdfs/Lotka-Volterra-Model.html
# The following is adapted from Python Programming in OpenGL:
# A Graphical Approach to Programming by Stan Blank, Ph.D.,
# Wayne City High School, Wayne City, Illinois, 62895.
# http://new.math.uiuc.edu/public198/ipython/stanblank/PyOpenGL.pdf



