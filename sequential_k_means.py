import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, square, pi, exp, zeros, linspace
from copy  import deepcopy
 
num_samples = 400
# For sequential learning, an iteration is performed on every new sample.
# For batch learning, K-means will converge in a much smaller number of iterations.
# Here, we just use the same number regardless.
num_iterations = num_samples
 
##############################################################
# Input: samples from a sum of Gaussian distributions
##############################################################
 
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
# Note: The weights must add up to 1.0
K = len(dist_set1) # Number of input and K-means clusters
 
# Mean, variances and weights will shift over time. Use these vectors for this...
m_vec, v_vec, w_vec = zeros((3, num_samples)), zeros((3, num_samples)), zeros((3, num_samples))
# Samples will be drawn from a distribution that gradually shifts from dist1 to dist2.
def generate_distribution_vectors(num_samples, dist1, dist2):
  # Parallel interpolation using linspace...
  # means, variances and weights
  for i in range(K):
    m_vec[i]  = linspace(dist1[i]['mean'],     dist2[i]['mean'],     num_samples)
    v_vec[i]  = linspace(dist1[i]['variance'], dist2[i]['variance'], num_samples)
    w_vec[i]  = linspace(dist1[i]['weight'],   dist2[i]['weight'],   num_samples)
 
##############################################################
# 3-D plotting of sum of Gaussian distributions
##############################################################
 
if False:
  # 'Normal' distribution function, a.k.a 'Gaussian'
  def normal_pdf(data, mean, variance):
    return 1/(sqrt(2*pi*variance)) * exp(-(square(data - mean)/(2*variance)))
 
  # To produce a sum of Gaussians distribution for every time step
  def sum_of_gaussians(x, t):
    result = 0
    for k in range(K):
      result += normal_pdf(x, m_vec[k][t], v_vec[k][t]) * w_vec[k][t]
    return result
 
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
      Z[t, x] = sum_of_gaussians(x_range[x], t)
  X, T = np.meshgrid(x_range, t_range)
 
  # Plot the surface.
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import cm
  from matplotlib.ticker import LinearLocator, FormatStrFormatter
 
  fig = plt.figure(figsize=(10,6))
  ax = fig.add_subplot(111, projection='3d')
  ############surf = ax.plot_surface(X, T, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.25)
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
  exit()
 
###############################
# Options
###############################
 
# devs: the variance bars are plotted this no. of std deviations away from mean
devs = 2 # (1 =&gt; 68%, 2 =&gt; 95%, 3 =&gt; 99.7%)
 
np.random.seed(0) # for repeatable results
 
# Uses m_vec, v_vec and w_vec vectors of mean, variance and weight.
def generate_samples(num_samples):
  for t in range(num_samples):
    prob = list(w_vec[:, t])
    source = np.random.choice(a=list(range(K)), size=1, p = list(w_vec[:, t]))
    mean   =      m_vec[source, t]
    stddev = sqrt(v_vec[source, t])
    means_history.append( mean )
    lower_history.append( mean - stddev*devs )
    upper_history.append( mean + stddev*devs )
    sample_history.append( np.random.normal(mean, stddev))
 
# Output: Graphical plot
def scatter_plot(title, iterations):
  plt.figure(figsize=(10,6))
  axes = plt.gca()
  plt.xlabel("samples / iterations")
  plt.ylabel("x")
  plt.title(title)
  steps = list(range(len(means_history)))
  plt.scatter(steps, upper_history,   color='navy',  s=30, marker="1", label=("μ+%dσ" % (devs)))
  plt.scatter(steps, lower_history,   color='navy',  s=30, marker="2", label=("μ-%dσ" % (devs)))
  plt.scatter(steps, sample_history,  color='green', s=30, marker=".", label="x[t]")
  steps = list(range(iterations+1))
  for k in range(K):
    plt.scatter(steps, centroid_history[0:iterations+1,k], color='red',  s=30, marker="_", label=("c%d[t]" % (k)))
  #plt.legend(loc='upper left')
  plt.show()
 
##################################################################
# Algorithm: K-means
##################################################################
 
def find_closest_cluster(x, centroids):
  closest_cluster, closest_distance = 999999, 999999 # initially invalid
  for k in range(K): # no. clusters
    distance = (x - centroids[k])**2 # Euclidean distance
    if distance <span id="mce_SELREST_start" style="overflow:hidden;line-height:0;"></span><span id="mce_SELREST_start" style="overflow:hidden;line-height:0;"></span>&lt; closest_distance:
      closest_cluster  = k
      closest_distance = distance
  return closest_cluster
 
def naive_k_means(num_iterations):
  for t in range(1, num_iterations+1):
    # Assignment phase: Assign each x_ix i to nearest cluster by calculating its distance to each centroid.
    sums     = zeros((K)) # sum of x&#039;s for each cluster
    counts   = zeros((K)) # count of number of samples in each cluster
    for sample in sample_history:
      allocated_cluster = find_closest_cluster(sample, centroid_history[t-1])
      counts[allocated_cluster] += 1
      sums[allocated_cluster]   += sample
    # Update phase: Find new cluster center by taking the average of the assigned points.
    print(&quot;# Iteration %2d centroids:&quot; % (t), end=&#039;&#039;)
    for k in range(K):
      centroid_history[t,k] = sums[k]/counts[k]
      print(&quot; %6.3f&quot; % (centroid_history[t,k]), end=&#039;&#039;)
    print(&quot;&quot;)
 
def sequential_k_means(num_samples, a=0, forgetful=False):
  sample_count = zeros((K))
  for t in range(1, num_samples+1):
    closest_k = find_closest_cluster(sample_history[t-1], centroid_history[t-1])
    sample_count[closest_k] += 1
    centroid_history[t] = deepcopy(centroid_history[t-1]) # carry over from previous ...
    # ... but then adjust the *closest* centroid ...
    if forgetful:
      # c[t]=c[t-1]+a(x[t]-c[t-1]) = (1-a).c[t-1]+a.x[t]
      centroid_history[t,closest_k] += (sample_history[t-1] - centroid_history[t-1,closest_k])*a
    else: # normal sequential K-means...
      centroid_history[t,closest_k] += (sample_history[t-1] - centroid_history[t-1,closest_k])/sample_count[closest_k]
    print(&quot;# Iteration %2d centroids:&quot; % (t), end=&#039;&#039;)
    for k in range(K):
      print(&quot; %6.3f&quot; % (centroid_history[t,k]), end=&#039;&#039;)
    print(&quot;&quot;)
 
##################################################################
# Run numerous combinations of distribution and algorithm
##################################################################
 
test_runs = [ &quot;Naive K-means: constant distribution&quot;,
              &quot;Sequential K-means: constant distribution&quot;,
              &quot;Sequential K-means: drifting distribution&quot;,
              # &quot;Naive K-means: drifting distribution&quot;,
              &quot;Forgetful Sequential K-means (a=0.01): constant distribution&quot;,
              &quot;Forgetful Sequential K-means (a=0.1): drifting distribution&quot; ]
 
for test_name in test_runs:
  print(&quot;# Running %s&quot; % (test_name))
  if &quot;a=0.01&quot; in test_name: hyperparameter_a = 0.01 # hyper-parameter for Forgetful Sequential K-means
  if &quot;a=0.1&quot;  in test_name: hyperparameter_a = 0.1
  # Initialize...
  means_history, upper_history, lower_history, sample_history = [ ], [ ], [ ], [ ]
  if &quot;constant&quot; in test_name: # input sequence from distribution dist_set1
    generate_distribution_vectors(num_samples, dist_set1, dist_set1)
  elif &quot;drifting&quot; in test_name: # input sequence drifting from distribution dist_set1 to dist_set2
    generate_distribution_vectors(num_samples, dist_set1, dist_set2)
  generate_samples(num_samples)
 
  # This is the container for the results, to be able to plot the position of the centroids over time
  centroid_history = zeros((1+num_iterations, K)) # for cluster k at iteration t
  # &#039;num_iterations+1&#039; because centroids[0,] is the *initial* guess...
  centroid_history[0] = [ -11, 3, 11 ]
 
  if &quot;Naive&quot; in test_name: # standard (batch) K-means
    naive_k_means(num_iterations) # Will actually converge in about 2 iterations
  elif &quot;Forgetful&quot; in test_name:
    sequential_k_means(num_samples, hyperparameter_a, forgetful=True)
  elif &quot;Sequential&quot; in test_name:
    sequential_k_means(num_samples)
 
  if &quot;Naive&quot; in test_name: # standard (batch) K-means
    scatter_plot(test_name, num_iterations)
  else: # sequential
    scatter_plot(test_name, num_samples)

