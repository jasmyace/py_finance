import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BaseMinimizer():
  """A class to house basic results from minimizing functions."""

  def __init__(self, flavor):
    """Initialize attributes."""
    self.flavor = 'basic'


class Options(BaseMinimizer):
  """A class to house options."""

  def __init__(self):
    """Initialize attributes."""
    self.flavor = 'options'

  def call(self, S, K):
    self.call = max(K - S, 0)
    return self
  
  def put(self, S, K):
    self.put = max(S - K, 0)
    return self


class BinomialTree(BaseMinimizer):
  """A class to house Binomial Tree option-price estimates."""

  def __init__(self):
    """Initialize attributes."""
    self.flavor = 'binomial_tree'

  def binomial_tree(self, ticker = 'UNSP', K = 50, r = 0.10, S0 = 50, sigma = 0.40, h = 5, T = 5/12, american = True:

    # ticker = "UNSP"
    # K = 50
    # r = 0.10
    # S0 = 50
    # sigma = 0.40
    # h = 30
    # T = 5/12
    # american = True 

    # Calculate helpful quantities. 
    dt = T / h
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-1 * sigma * np.sqrt(dt))
    # u = 1.1224
    # d = 0.8909
    a = np.exp(r * dt)
    p = (a - d) / (u - d)

    # Create holding structures. 
    S_grid = np.zeros((h + 1, h + 1))  # Stock price.
    f_grid = np.zeros((h + 1, h + 1))  # European option price.  
    g_grid = np.zeros((h + 1, h + 1))  # American (early exercise) option price. 

    # Times i go left to right (0, 1, ...); prices j up to down (i, i - 1, ...).
    i_list = list(range(h, -1, -1))
    for i in i_list:
      j_list = list(range(i, -1, -1))
      for j in j_list:
        # print("(" + str(i) + ", " + str(j) + ")")
        
        # Mathematically, think of stock prices as moving forward in time, but 
        # options estimate backwards in time.  Let's try to prevent two passes 
        # by considering stock prices backwards in time too.  
        S_grid[h - j, i] = S0 * u**(j) * d**(i - j)  
        
        # f_grid provides options prices assuming no exercise at the node.  So, 
        # the option price is the present value of the expected option price 
        # one step later.  
        if i == h:
          f_grid[h - j, i] = max(K - S_grid[h - j, i], 0)
          g_grid[h - j, i] = max(K - S_grid[h - j, i], 0)
        else: 
          
          # Note the np.exp calculates the present value of the expected price 
          # one time step later.  So, technically we are calculating 1 dt unit. 
          f_grid[h - j, i] = np.exp(-1 * r * dt) * (p * f_grid[h - (j + 1), i + 1] + (1 - p) * f_grid[h - j, i + 1])
          g_grid[h - j, i] = max(K - S_grid[h - j, i], 0)

          # Check if early exercise is the better option.  If so, exercise. 
          if american: 
            f_grid[h - j, i] = max(f_grid[h - j, i], g_grid[h - j, i])
            
    self.S = S_grid
    self.f = f_grid
    self.f_today = self.f[:, 0][-1]
    self.K = K
    self.r = r
    self.S0 = S0
    self.sigma = sigma
    self.h = h
    self.T = T
    self.ticker = ticker
    return self
  
  def __repr__(self):
    return ""
    
all_df = []
for h in range(2, 500):
  bin_tree = BinomialTree()
  bin_tree.binomial_tree(ticker = "MSFT", K = 50, r = 0.10, S0 = 50, sigma = 0.40, h = h, T = 5/12)
  df = pd.DataFrame({'h': bin_tree.h, 'f_today': bin_tree.f_today}, index = [h])
  all_df.append(df)
all_df = pd.concat(all_df)
plt.scatter(all_df.h, all_df.f_today)
plt.show()
