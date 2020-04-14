import pandas as pd
import numpy as np
import os
import scipy
os.chdir("C:/Users/naz/Ozel/EDHEC/Intro/data/")
def annualized_ret(r,period):
    """
    Calculate annualized return for a given period. I.e. for quarterly data give 4 monthly 12 etc.
    r = return series
    period = For Monthly give 12 and quarterly 4 etc.
    """
    compounded_return = (1+r).prod()
    n = r.shape[0]
    com_ret = compounded_return**(period/n)-1
    return com_ret

def annualized_vola(r, period):
    """
    Calculate annualized Volatility for a given period.
    r = return series
    period = For Monthly give 12 and quarterly 4 etc.
    """
    comp_ret = r.std()
   
    ann_vola = comp_ret*np.sqrt(period)
    
    return ann_vola

def sharpe_ratio(r, rf, period):
    """
    Calculate Sharpe Ratio of given Risk Free rate and period for return series.
    Make sure that risk free rate also has same period as return.
    Give risk free rate as a decimal for 3% give 0.03
    r = return series
    rf = risk free rate
    period = For Monthly give 12 and quarterly 4 etc.
    """
    rf_period = (1+rf)**(1/period)-1
    excess_ret = r - rf_period
    ret = annualized_ret(excess_ret,period)
    vola = annualized_vola(r,period)
    sharpe_ratio = (ret)/vola
    return sharpe_ratio

def drawdown_cal(r):
    
    """
    This function will calculate drawdown of Panda Series or DataFrame
    and Return a DataFrame back
    """
    
    if isinstance(r,pd.DataFrame): 
        return r.aggregate(drawdown_cal)
    elif isinstance(r, pd.Series):
        wealth = 1000* (1+r).cumprod()
        prev_peak = wealth.cummax()
        drawdown = (wealth -prev_peak)/prev_peak
        return pd.DataFrame({
            "Drawdown" : drawdown,
             "Wealth": wealth,
            "Peak": prev_peak
        })
         
def get_ffme_return(col):
         """
         Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
         """
         returns = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",
                              index_col = 0, header =0, parse_dates =True,
                              na_values = -99.99)
         
         returns = returns[col]
         returns.columns = ["SmallCap","LargeCap"]
         returns.index = pd.to_datetime(returns.index,format ="%Y%m").to_period("M")
         return returns
    
def get_ind_return():
    """
    Load Index Returns
"""
    ind = pd.read_csv("ind30_m_vw_rets.csv", header =0, index_col =0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format ="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    Load Index Sizes
"""
    ind = pd.read_csv("ind30_m_size.csv", header =0, index_col =0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format ="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind
def get_ind_nfirms():
    ind = pd.read_csv("ind30_m_nfirms.csv", header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format ="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_hfi():
    """
    Give back Hedge Fund indices back
    """
    rets = pd.read_csv("edhec-hedgefundindices.csv", index_col =0, header= 0)
    return rets
def skewness(s):
    """
    Find the skewness of a series and return it back a value or series
    """
    de_mean = s -s.mean()
    nominator = (de_mean**3).mean()
    denominator = (s.std(ddof=0))**3
    return nominator/denominator
def kurtosis(s):
    """
    Find the kurtosis of a series and return it back a value or series
    """
    de_mean = s-s.mean()
    nominator = (de_mean**4).mean()
    sigma = s.std(ddof=0)
    denominator = sigma**4
    return nominator/denominator

def is_normal(r, level= 0.01):
    """
    Imply Jarque Bera Test from scipy
    and test the series normality with given level
    """
    import scipy
    if isinstance(r,pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        stat, p = scipy.stats.jarque_bera(r)
        return p>level
    
def var_historic(r,level):
    """
    Calculate historical VaR at 95% confidence interval
    """
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("Expect r either panda Series or DataFrame")
        
        
def cvar_historic(r,level=5):
    """
    Calculate conditional VaR
    """
    if isinstance (r, pd.DataFrame):
        return r.aggregate(cvar_historic, level =level)
    elif isinstance (r, pd.Series):
        indx= -var_historic(r,level=level)
        cr =r[r<=indx]
        return -cr.mean()
        
    else:
        raise TypeError("Expected panda DataFrame or Series")
from scipy.stats import norm   
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def port_ret(w,r):
    """
    Calculate portfolio return with given assets weights and returns
    w = weight
    r = return ( it should be annualized return)
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    p_r = w.T @ r #T is transpose @ is matrix multiplication
    return p_r

def port_vola(w,cov):
    """
    Calculate portfolio volatility with given weights and covariance matrix
    w= weights
    v = volatility
    cov = covariance matrix
    """
    vol = (w.T@ cov @w)**0.5 # We have to transpose the weight martix then multiple it to
    #covariance matrix and againt multiple it to weight and take square to find std. dev
    return vol

def plot_ef2(r,n,cov):
    """
    Plot Efficient Frontier for 2 Assets
    n = number of points
    r = return series
    cov = covariance Matrix
    """
    weight = [np.array([w,1-w]) for w in np.linspace(0,1,n)]
    portfolio_ret = [port_ret(w,r) for w in weight]
    portfolio_vola = [port_vola(w, cov) for w in weight]
    data = pd.DataFrame({"R": portfolio_ret, "Vol": portfolio_vola})
    
    return data.plot.scatter(x="Vol",y="R") 

def plot_ef(n,r,cov, rf,show_cml =False, show_ew = False, show_gmv=False):
    """
    Plot Efficient Frontier for n Assets
    n = number of points
    r = return series
    cov = covariance Matrix
    """
    weight_msr = msr(rf, r, cov)
    weight = optimal_weights(n, r, cov)
    portfolio_ret = [port_ret(w,r) for w in weight]
    portfolio_vola = [port_vola(w, cov) for w in weight]
    data = pd.DataFrame({"Return": portfolio_ret, "Volatility": portfolio_vola})
    ax = data.plot(x = "Volatility", y = "Return", style =".-")
    ax.set_xlim(left = 0)
    
    if show_ew:
        # Will show equally weightes portfolio point
        n_point = r.shape[0]
        w_ew = np.repeat(1/n_point,n_point) #weights of ew portfolio
        r_ew = port_ret(w_ew,r) #Return of EW Portfolio
        vol_ew = port_vola(w_ew,cov) # Vola of EW Portfolio
        ax.plot(vol_ew,r_ew, marker ="o", markersize =10, color ="green")
    if show_gmv:
        #Will show Global Minimum Variance Portfolio
        w_gmv = gmv(cov)
        r_gmv = port_ret(w_gmv,r)
        vol_gmv = port_vola(w_gmv,cov)
        ax.plot(vol_gmv, r_gmv,marker ="o", markersize=10, color ="midnightblue")
        
    if show_cml:
        return_msr = port_ret(weight_msr, r)
        vola_msr = port_vola(weight_msr, cov)

        #Add Capital Market Line
        cml_x = [0, vola_msr]
        cml_y =[rf, return_msr]
        ax.plot(cml_x,cml_y, color ="red", marker = "o", linestyle ="dashed")
        
    return ax

def gmv(cov):
    n_point = cov.shape[0]
    return msr(0,np.repeat(1,n_point),cov)


def optimal_weights(n_points,er,cov):
    '''
    it generates list of weights to run the optimizer on to minimize the volatility
    '''
    target_returns = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(tr,er,cov) for tr in target_returns]
    return weights
    
from scipy.optimize import minimize

def minimize_vol(target_ret, er, cov):
    """
    target_ret -> w
    The way the optmizer works you needs to give some constrains,
    you need to give target return and some initial guess
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0, 1.0),)*n # For every element of vector you have to give a bounders that is why we have to give tuple in tuple and n times
    # Constrain is to minimize volatility with target return
    return_is_target = {
        #return_is_target is input for scipy minize function 
        'type' :'eq', #Constraint type: 'eq' for equality, 'ineq' for inequality
        'args': (er,), #args : tuple, optional Extra arguments passed to the objective function and its derivatives (`fun`, `jac` and `hess` functions). 
        'fun': lambda weights, er: target_ret - port_ret(weights,er) #optimazer function will find if target return is met in other words target ret minus port return should equal to zero
        
    }
    weights_sum_to_1 = {
     #This is a second constraint for minize function which tell sum of weights should be 1
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    
    results = minimize(port_vola, init_guess,
                       args= (cov), #args are extra arguments that port_vola funct needs
                       method ="SLSQP",
                       options = {'disp': False},
                       constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds
                      
                      )
    return results.x


def msr(rf, er, cov):
    """
    minimum sharpe return calculation.
    The way the optmizer works you needs to give some constrains,
    you need to give target return and some initial guess
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0, 1.0),)*n # For every element of vector you have to give a bounders that is why we have to give tuple in tuple and n times
   
    
    def neg_sharpe_ratio(weights, rf, er, cov):  
        # This time we do not need return_constrain what we need to maximize sharpe ratio
        # so because we minimze that is why we need to min negative sharpe ratio to obtain
        #maximized sharpe ratio
            ret = port_ret(weights,er)
            vol = port_vola(weights,cov)
            sr = - (ret-rf)/vol
            return sr
 
    weights_sum_to_1 = {
     #This is a second constraint for minize function which tell sum of weights should be 1
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }

    results = minimize(neg_sharpe_ratio, init_guess,
                       args= (rf,er,cov), 
                       method ="SLSQP",
                       options = {'disp': False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      
                      )
    return results.x

def run_cppi(risky_r, safe_r = None, m = 3, start =1000, floor = 0.8, riskfree = 0.03, drawdown = None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky  Weight History
    """
    #set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns = ["R"])
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree/12
    #set up the DataFrame for saving intermediate values        
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)


    for step in range(n_steps):
        
        if drawdown is not None:
            peak = np.maximum(account_value,peak)
            floor_value = peak*(1-drawdown)
            
        cushion = (account_value - floor_value)/account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1) # We want to strict leverage so max 100% can be invested
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value * safe_w
    
    ## Update the account value for this time step
        account_value =(risky_alloc * (1+risky_r.iloc[step])) +safe_alloc*(1+safe_r.iloc[step])
        #save the values so I can look at he history and plot it
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth" : account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m" :m,
        "start" : start,
        "floor": floor,
        "risky_r" : risky_r,
        "safe_r": safe_r
    }
        
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualized_ret, period=12)
    ann_vol = r.aggregate(annualized_vola, period=12)
    ann_sr = r.aggregate(sharpe_ratio, rf=riskfree_rate, period=12)
    dd = r.aggregate(lambda r: drawdown_cal(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    
    
    """
        Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices
        :param n_years:  The number of years to generate data for
        :param n_paths: The number of scenarios/trajectories
        :param mu: Annualized Drift, e.g. Market Return
        :param sigma: Annualized Volatility
        :param steps_per_year: granularity of the simulation
        :param s_0: initial value
        :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)
   
    
    #Normally we write code like below but it will slow the speed because of the loop in sigma*xi*np.sqrt(dt). To fasten the speed we rewrite the code with rets_plus_1
    # xi = np.random.normal(size = (n_steps, n_scenarios))
    #rets = mu*dt + sigma*np.sqrt(dt)*xi
    #prices = s_0*(1+rets).cumprod()
    
    
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    #loc is location i.e mean value and scale is std 
    
    prices = s_0*pd.DataFrame(rets_plus_1).cumprod()
    return prices

def show_gbm(n_sce, mean, std, S0):
    """
    Show interactive plot of Brownian Motion
    """
    prices = gbm(n_scenarios = n_sce, mu = mean, sigma = std, s_0 =S0)
    ax = prices.plot(legend=False, color = "indianred", alpha = 0.5, linewidth = 2, figsize = (12,6))
    ax.axhline(y= S0, ls = ":", color= "black")
    ax.plot(0,S0, marker='o',color='darkred', alpha=0.2)
  
