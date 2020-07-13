import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, square, pi, exp, zeros, linspace
from copy  import deepcopy
 
 
##############################################################
# Input: samples from a sum of Gaussian distributions
##############################################################
 
K = 3 # Select 2 or 3
 
if K==3: # Main 3-cluster example
  dist_set1 = [ # Gaussians clearly separable (relatively small variances)
    {'mean': -8, 'variance': 1, 'weight': 0.2},
    {'mean':  0, 'variance': 1, 'weight': 0.2},
    {'mean':  8, 'variance': 1, 'weight': 0.6}]
  dist_set2 = [ # Gaussians spread and merging
    {'mean': -7, 'variance': 3, 'weight': 0.2},
    {'mean': -2, 'variance': 2, 'weight': 0.6},
    {'mean':  6, 'variance': 1, 'weight': 0.2}]
  # Compared with dist_set1, dist_set2 has:
  # * the means closer together,
  # * larger variances and
  # * weighting drifting
  num_samples = 400
 
if K==2: # 2-cluster cross-over example
  # A very small variance distribution crosses over a much wider one
  dist_set1 = [
    {'mean': 2,  'variance': 0.1,  'weight': 0.5},
    {'mean': -3, 'variance': 4,    'weight': 0.5}]
  dist_set2 = [
    {'mean': -3, 'variance': 0.1,  'weight': 0.5},
    {'mean': 2,  'variance': 4,    'weight': 0.5}]
  num_samples = 1000
 
# Note: The weights must add up to 1.0  
K = len(dist_set1) # Number of input and K-means clusters
num_iterations = num_samples
 
# Input means, variances and weights will shift over time. Use these vectors for this...
m_vec, v_vec, w_vec = zeros((K, num_samples)), zeros((K, num_samples)), zeros((K, num_samples))
# Samples will be drawn from a distribution that gradually shifts from dist1 to dist2.
def generate_distribution_vectors(num_samples, dist1, dist2):
  # Parallel interpolation using linspace...
  for i in range(K):
    m_vec[i]  = linspace(dist1[i]['mean'],     dist2[i]['mean'],     num_samples)
    v_vec[i]  = linspace(dist1[i]['variance'], dist2[i]['variance'], num_samples)
    w_vec[i]  = linspace(dist1[i]['weight'],   dist2[i]['weight'],   num_samples)
 
def normal_pdf(data, mean, variance):
  return 1/(sqrt(2*pi*variance)) * exp(-(square(data - mean)/(2*variance)))
 
def online_update_mean_and_variance(new_data, weight, mean, variance, hyperparameter_a):
  # 1. Convert from mean and variance to sums, and
  # 2. Effectively remove 1 sample from the existing distribution...
  #    (keeping the mean and variance the same)
  rolling_samples = 1/hyperparameter_a
  num_items      = rolling_samples - weight
  sum_x          = mean * num_items
  sum_x_squared  = num_items * (square(mean) + variance)
  # 3. Add the new item (taking the number of samples back to the original
  num_items      = rolling_samples
  sum_x         += weight * new_data
  sum_x_squared += weight * square(new_data)
  # 4. Convert back to mean and variance...
  new_mean       = sum_x/num_items
  new_variance   = sum_x_squared/num_items - square(new_mean)
  # 'rolling_samples' controls how quickly old values are forgotten.
  # e.g rolling_samples=10 implies new mean is the weighted sum of 90% old + 10% new.
  return new_mean, new_variance
 
def online_update_weights(previous_weights, winner, hyperparameter_a):
  new_weights = []
  # e.g. a=0.1 (rolling_samples=10)
  # 1. Reduce the membership from rolling_samples to rolling_samples-1 members
  #    All the weights are multiplied by 0.9=1-a
  for k in range(K):
    new_weights.append( (1-hyperparameter_a) * previous_weights[k] )
  # 2. Add the winner to the membership
  new_weights[winner] += hyperparameter_a
  # The weights still sum to 1
  return new_weights
 
 
##############################################################
# 3-D plotting of sum of Gaussian distributions
##############################################################
 
# To produce a sum of Gaussians distribution for every time step
def weighted_sum_of_gaussians(x, t):
  result = 0
  for k in range(K):
    result += w_vec[k][t] * normal_pdf(x, m_vec[k][t], v_vec[k][t])
  return result
 
if True:
  print("# Plotting surface plot of input distribution")
  x_max = 12 # x sample input is over range -xmax to +xmax
  steps_per_unit_x = 10 # enough to get good surface plot 
  total_x_steps = (2 * x_max * steps_per_unit_x)
 
  # Produce Z[t, x] matrix (over all x input values and all iterations)
  # of sum-of-Gaussians
  generate_distribution_vectors(num_samples, dist_set1, dist_set2)
  x_range = np.arange(-x_max, x_max, 1/steps_per_unit_x)
  t_range = np.arange(1, num_samples+1)
  Z = zeros((num_samples, total_x_steps))
  for x in range(total_x_steps):
    for t in range(num_samples):
      Z[t, x] = weighted_sum_of_gaussians(x_range[x], t)
  X, T = np.meshgrid(x_range, t_range)
 
  # Plot the surface.
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
 
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, T, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.25)
 
  # Customize the z axis.
  ax.set_zlim(0.0, 0.4)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
 
  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
 
  ax.set_xlabel('x')
  ax.set_ylabel('iterations')
  ax.set_zlabel('PDF')
 
  plt.show()
 
 
###############################
# Options
###############################
 
# devs: the variance bars are plotted this no. of std deviations away from mean
devs = 2 # (1 => 68%, 2 => 95%, 3 => 99.7%)
 
np.random.seed(0) # for repeatable pseudo-random results
 
# Uses m_vec, v_vec and w_vec vectors of mean, variance and weight.
def generate_samples(num_samples):
  for t in range(num_samples):
    source = np.random.choice(a=list(range(K)), size=1, p = list(w_vec[:, t]))
    mean   =      m_vec[source, t]
    stddev = sqrt(v_vec[source, t])
    sample_history.append( np.random.normal(mean, stddev))
    # Also generate vectors for plotting results
    centr_history.append( mean )
    lower_history.append( mean - stddev*devs )
    upper_history.append( mean + stddev*devs )
 
# Output: Graphical plot
def scatter_plot(title, iterations):
  plt.figure(figsize=(10,6))
  axes = plt.gca()
  plt.xlabel("samples / iterations")
  plt.ylabel("x")
  plt.title(title)
  steps = list(range(len(centr_history)))
  plt.scatter(steps, upper_history,   color='navy',  s=30, marker="1", label=("μ+%dσ" % (devs)))
  plt.scatter(steps, lower_history,   color='navy',  s=30, marker="2", label=("μ-%dσ" % (devs)))
  for k in range(K):
    plt.scatter(sorted_sample_t[k], sorted_sample_x[k],  color=palette[6*k+7], s=30, marker=".", label=("k=%d % (k))"))
  # '6*k+7': Plot the (max K=3) clusters in 'lime green', 'cyan' and 'rebeccapurple' 
  steps = list(range(iterations+1))
  for k in range(K):
    # Convert variance to 2 standard deviations...
    plus2sd_history = mean_history + 2*sqrt(variance_history)
    less2sd_history = mean_history - 2*sqrt(variance_history)
    plt.scatter(steps,     mean_history[0:iterations+1,k], color='red',     s=30, marker="_", label=("μ[%d][t]"     %       (k)))
    plt.scatter(steps,  plus2sd_history[0:iterations+1,k], color='orange',  s=30, marker="_", label=("μ+%dσ[%d][t]" % (devs, k)))
    plt.scatter(steps,  less2sd_history[0:iterations+1,k], color='orange',  s=30, marker="_", label=("μ+%dσ[%d][t]" % (devs, k)))
  plt.show()
 
# Plot multiple Gaussians using a spectrum of colors...
palette = ['red', 'orangered', 'darkorange', 'orange', 'gold', 'yellow', 'greenyellow', 'limegreen', 'green',
  'mediumseagreen', 'mediumaquamarine', 'mediumturquoise', 'paleturquoise', 'cyan', 'deepskyblue', 'dodgerblue',
  'royalblue', 'navy', 'slateblue', 'rebeccapurple', 'darkviolet', 'violet', 'magenta'] 
 
def plot_gaussians(iterations, plot_end_distribution=False):
  bins = np.linspace(-15,15,300)
 
  plt.figure(figsize=(7,5)) # (10,7) may be too large
  plt.xlabel("$x$")
  plt.ylabel("pdf")
  plt.scatter(sample_history, [-0.005] * len(sample_history), color='navy', s=30, marker=2, label="Sample data")
 
  for t in range(iterations+1): # early timesteps
    for k in range(K):
      plt.plot(bins, weight_history[t,k]*normal_pdf(bins, mean_history[t,k], variance_history[t,k]), color=palette[(t%8)+9])
  for k in range(K): # Initial (or constant) input distribution
    plt.plot(bins, dist_set1[k]['weight']*normal_pdf(bins, dist_set1[k]['mean'], dist_set1[k]['variance']), color='red')
  if plot_end_distribution:
    for k in range(K): # Final input distribution
      plt.plot(bins, dist_set2[k]['weight']*normal_pdf(bins, dist_set2[k]['mean'], dist_set1[k]['variance']), color='magenta')
 
  plt.legend()
  plt.plot()
  plt.show()
 
# In order to plot samples allocated to different clusters in different colors...
sorted_sample_x = [ [], [], [] ]
sorted_sample_t = [ [], [], [] ]
 
def find_closest_cluster(x, t, metric='distance'):
  if metric == 'distance': # Find MINIMUM distance
    closest_cluster, closest_value = 999999, 999999 # initially invalid
    for k in range(K): # no. clusters
      distance = (x - mean_history[t-1,k])**2 # Euclidean distance
      if distance  closest_value:
        closest_cluster  = k
        closest_value    = likelihood
  # Record result for later plotting
  sorted_sample_x[closest_cluster].append(x)
  sorted_sample_t[closest_cluster].append(t)
  return closest_cluster
 
def kullback_leibler_divergence(m1, v1, m2, v2): # KL(P||Q)
  # m1, v1: mean and variance of the posterior P
  # m2, v2: mean and variance of the prior Q
  sd1, sd2 = np.sqrt(v1), np.sqrt(v2) # standard deviations
  return (v1-v2+(m1-m2)**2)/v2/2 + np.log(sd2/sd1)
 
 
##################################################################
# Algorithms: K-means
##################################################################
 
def naive_k_means(num_iterations):
  for t in range(1, num_iterations+1):
    # Assignment phase: Assign each x_ix i to nearest cluster by calculating its distance to each centroid.
    sums     = zeros((K)) # sum of x's for each cluster
    counts   = zeros((K)) # count of number of samples in each cluster
    for sample in sample_history:
      allocated_cluster = find_closest_cluster(sample, mean_history[t-1])
      counts[allocated_cluster] += 1
      sums[allocated_cluster]   += sample
    # Update phase: Find new cluster center by taking the average of the assigned points.
    print("# Iteration %2d centroids:" % (t), end='')
    for k in range(K):
      mean_history[t,k] = sums[k]/counts[k]
      variance_history[t,k] = 2 # !!!!!!!!!!!!! Temp !!!!!!!!
      print(" %6.3f" % (mean_history[t,k]), end='')
    print("")
 
def sequential_k_means(num_samples, forgetful=False, a=0):
  sample_count = zeros((K))
  for t in range(1, num_samples+1):
    closest_k = find_closest_cluster(sample_history[t-1], t)
    sample_count[closest_k] += 1
    mean_history[t] = deepcopy(mean_history[t-1]) # carry over from previous ...
    # ... but then adjust the *closest* centroid ...
    if forgetful:
      # c[t]=c[t-1]+a(x[t]-c[t-1]) = (1-a).c[t-1]+a.x[t]
      mean_history[t,closest_k] += (sample_history[t-1] - mean_history[t-1,closest_k])*a
    else: # normal sequential K-means...
      mean_history[t,closest_k] += (sample_history[t-1] - mean_history[t-1,closest_k])/sample_count[closest_k]
    print("# Iteration %2d centroids:" % (t), end='')
    for k in range(K):
      print(" %6.3f" % (mean_history[t,k]), end='')
    print("")
 
 
##################################################################
# Algorithms: Gaussian Mixture Models
##################################################################
 
eps=1e-8 # Small value to prevent division by zero
def batch_gmm(num_iterations):
  for t in range(1, num_iterations+1):
    # Expectation step: calculate the maximum likelihood of each observation xi
    likelihood = []
    for k in range(K):
      likelihood.append(normal_pdf(sample_history, mean_history[t-1,k], variance_history[t-1,k]))
    likelihood = np.array(likelihood)
       
    # Maximization step 
    b = []
    for k in range(K):
      # Use the current values for the parameters to evaluate the posterior
      # probabilities of the data to have been generated by each Gaussian    
      b.append((likelihood[k] * weight_history[t-1,k]) / (np.sum([likelihood[i] * weight_history[t-1,i] for i in range(K)], axis=0)+eps))
     
      # update mean, variance and weight
      mean_history[t,k]     = np.sum(b[k] * sample_history) / (np.sum(b[k]+eps))
      variance_history[t,k] = np.sum(b[k] * square(sample_history - mean_history[t,k])) / (np.sum(b[k]+eps))
      weight_history[t,k]   = np.mean(b[k])
 
    # Reporting... 
    print("# Iteration %2d mvw:" % (t), end='') # mvw = mean, variance, weight in that order
    for k in range(K):
      print("(%6.3f,%6.3f,%6.3f)" % (mean_history[t,k],variance_history[t,k],weight_history[t,k]), end='')
    print("")
  for x in range(len(sample_history)): # Not color-coding Batch samples
    sorted_sample_x[0].append(sample_history[x])
    sorted_sample_t[0].append(x)
 
 
def forgetful_online_gmm(num_samples, hyperparameter_a):
  for t in range(1, num_samples+1):
    # Expectation step: calculate the maximum likelihood of each observation xi
    winner = find_closest_cluster(sample_history[t-1], t, metric='probability')
 
    #likelihood = []
    #for k in range(K):
    #  likelihood.append(normal_pdf(sample_history[t-1], mean_history[t-1,k], variance_history[t-1,k]))
    #likelihood = np.array(likelihood)
       
    # Maximization step 
    #membership = [] # represents the degree of membership to each cluster
    #for k in range(K):
    #  # Use the current values for the parameters to evaluate the posterior
    #  # probabilities of the data to have been generated by each Gaussian    
    #  membership.append((likelihood[k] * weight_history[t-1,k]) / (np.sum([likelihood[i] * weight_history[t-1,i] for i in range(K)], axis=0)+eps))
     
    # copy from previously...
    mean_history[t]     = deepcopy(mean_history[t-1])
    variance_history[t] = deepcopy(variance_history[t-1])
    weight_history[t]   = deepcopy(  weight_history[t-1])
    # update mean, variance and weight of the winning cluster
    mean_history[t, winner], variance_history[t, winner] = online_update_mean_and_variance(sample_history[t-1], weight_history[t-1, winner], mean_history[t-1, winner], variance_history[t-1, winner], hyperparameter_a)
    weight_history[t] = online_update_weights(weight_history[t-1], winner, hyperparameter_a)
 
    # Reporting normal PDF, mean, variance and weight of each cluster at each iteration... 
    print("# Iteration %2d pmvw:" % (t), end='') # pmvw = PDF, mean, variance, weight in that order
    for k in range(K):
      print("(%6.3f" % (normal_pdf(sample_history[t-1], mean_history[t-1,k], variance_history[t-1,k])), end='')
      if k == winner:
        print("*", end='')
      print(",%6.3f,%6.3f,%6.3f)" % (mean_history[t,k],variance_history[t,k],weight_history[t,k]), end='')
    # Reporting the Kullback-Leibler divergence and the input sample value... 
    print("KL=%.4f" % (kullback_leibler_divergence(mean_history[t,winner], variance_history[t,winner], mean_history[t-1,winner], variance_history[t-1,winner])), end='')
    print(" x=%6.3f" % (sample_history[t-1]), end='')
    print("")
 
 
 
##################################################################
# Run numerous combinations of distribution and algorithm
##################################################################
 
# Uncomment to include...
test_runs = [ #"Naive K-means: constant distribution",
              #"Naive K-means: drifting distribution",
              #"Sequential K-means: constant distribution",
              #"Sequential K-means: drifting distribution",
              #"Forgetful K-means (a=0.01): constant distribution",
              #"Forgetful K-means (a=0.1): drifting distribution" ]
              #"Batch GMM: constant distribution",
              #"Batch GMM: drifting distribution" ,
              #"Forgetful GMM (a=0.1): constant distribution",
              #"Forgetful GMM (a=0.1): drifting distribution",
              #"Forgetful GMM (a=0.05): drifting distribution",
              "Forgetful GMM (a=0.1): drifting distribution" ]
 
for test_name in test_runs:
  print("# Running %s #######################" % (test_name))
  if "a=0.01" in test_name: hyperparameter_a = 0.01 # hyper-parameter for Forgetful Sequential K-means
  if "a=0.05" in test_name: hyperparameter_a = 0.05
  if "a=0.1"  in test_name: hyperparameter_a = 0.1
  if "a=0.2"  in test_name: hyperparameter_a = 0.2
  print("# Generate input sample values #######################")
  centr_history, upper_history, lower_history, sample_history = [ ], [ ], [ ], [ ]
  if "constant" in test_name: # input sequence from distribution dist_set1  
    generate_distribution_vectors(num_samples, dist_set1, dist_set1)
  elif "drifting" in test_name: # input sequence drifting from distribution dist_set1 to dist_set2
    generate_distribution_vectors(num_samples, dist_set1, dist_set2)
  generate_samples(num_samples)
 
  # This is the container for the results, to be able to plot the position of the centroids over time
  mean_history     = zeros((1+num_iterations, K)) # for cluster k at iteration t
  variance_history = zeros((1+num_iterations, K)) # for cluster k at iteration t
  weight_history   = zeros((1+num_iterations, K)) # for cluster k at iteration t
  # Size is 'num_iterations+1' rather than 'num_iterations' because centroids[0,] is the *initial* guess...
  print("# Setting initial values of parameters #######################")
  if K==2: # Start estimate very similar to actual start
    mean_history[0], variance_history[0], weight_history[0] = [2,-3], [0.1,2], [0.5,0.5] # Adds up to 1
  elif K==3:
    mean_history[0], variance_history[0], weight_history[0] = [-11, 3, 11], [2,2,2], [0.34,0.33,0.33]
 
  print("# Run algorithm #######################")
  if "Naive K-means" in test_name: # standard (batch) K-means
    naive_k_means(num_iterations) # Will actually converge in about 2 iterations
  elif "Forgetful K-means" in test_name:
    sequential_k_means(num_samples, forgetful=True, a=hyperparameter_a)
  elif "Sequential K-means" in test_name:
    sequential_k_means(num_samples)
  elif "Batch GMM" in test_name:
    batch_gmm(num_iterations) # Will actually converge in a few iterations
  elif "Forgetful GMM" in test_name:
    forgetful_online_gmm(num_samples, hyperparameter_a)
 
  if True:
    print("# Plot output scatter plot #######################")
    if ("Naive" in test_name) or ("Batch" in test_name):
      scatter_plot(test_name, num_iterations)
    else:
      scatter_plot(test_name, num_samples)
 
 
##############################################################
# 3-D plotting of output Gaussian distributions
##############################################################
 
def weighted_sum_of_output_gaussians(x, t):
  result = 0
  for k in range(K):
    result += weight_history[t,k] * normal_pdf(x, mean_history[t,k], variance_history[t,k])
  return result
 
if True:
  print("# 3-D Plot of output Gaussians #######################")
  x_max = 15 # x sample input is over range -xmax to +xmax
  steps_per_unit_x = 10 # enough to get good surface plot 
  total_x_steps = (2 * x_max * steps_per_unit_x)
 
  # Produce Z[t, x] matrix (over all x input values and all iterations)
  # of sum-of-Gaussians
  x_range = np.arange(-x_max, x_max, 1/steps_per_unit_x)
  t_range = np.arange(1, num_samples+1)
  Z = zeros((num_samples, total_x_steps))
  for x in range(total_x_steps):
    for t in range(num_samples):
      Z[t, x] = weighted_sum_of_output_gaussians(x_range[x], t)
  X, T = np.meshgrid(x_range, t_range)
 
  # Plot the surface.
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
 
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, T, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.25)
 
  # Customize the z axis.
  ax.set_zlim(0.0, 1)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
 
  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
 
  ax.set_xlabel('x')
  ax.set_ylabel('iterations')
  ax.set_zlabel('PDF')
 
  plt.show()


