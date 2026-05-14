#---------------------------------------
#----------------
# GALACTIC DISC CODE 
#---------------------------------------
#----------------

# Original galactic disc code modified to compare with Piyush's code
    # - Takes Mh0 as an input instead of Mhz_init and adds additional input for z_final
    # - Calculate beta and fsf for intermediate-mass galaxies using linear ramp
    # - Using original expressions for MdotSF, Mdottrans, sigmaSF
    # - Not recalculating sigmag if phint < 0 or if Fsigma < 0 when solving mass equilibrium equation
    # - Not calculating MdotSF with recalculated sigmag if Fsigma < 0
    # - Do not recalculate phiQ, Q after phint check. Just recalculate sigmag and phint
    # - Only recalculate sigmag and phint after Fsigma check

# Including Mdotacc and etaw from different cosmological simulations given by Ruby
# - Inputs for interpolator functions are logMh and z
    # - TNG
    # - SIMBA
    # - EAGLE

import numpy as np

# Importing astropy
from astropy import units as u
import astropy.constants as cons
from astropy.constants import G
from astropy.cosmology import FlatLambdaCDM # Import cosmology
cosmo = FlatLambdaCDM(H0=71, Om0=0.27)
Omega_m_0 = cosmo.Om0
Omega_Lam_0 = 1 - Omega_m_0
H0 = cosmo.H0

# Importing scipy
from scipy.integrate import solve_ivp, odeint
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d, LinearNDInterpolator

# Import warnings
import warnings

# Defining functions
def halo_hist(Mh0, z=None):
    """
    Generates the halo mass versus redshift for a halo of mass Mh0 at
    z = 0

    Parameters:
       Mh0 : float, array, or astropy.units.quantity.Quantity
          halo mass at z = 0; if not an astropy Quantity, should
          be in units of Msun
       z : arraylike
          grid of redshifts at which to compute halo properties starting from lowest to highest redshift; if
          left as None, a grid is created automatically

    Returns:
       z : array
          array of output redshifts
       Mh : astropy.units.quantity.Quantity
          halo mass at each redshift in Msun; if given more than one Mh, this
          will be a 2D array, otherwise it will be a 1D array
    """

    # Get halo mass in units of 10^12 Msun, which we will use as our
    # integration variable
    if type(Mh0) is u.quantity.Quantity:
        Mh12 = Mh0.to(u.Msun).value / 1e12 
    else:
        Mh12 = np.array(Mh0) / 1e12
    
    # Make grid in z if not provided
    if z is None:
        z = np.linspace(0, 3)
   
    # Integrate to get Mh12 vs time, then return
    if np.size(Mh12) == 1:
        Mh12_z = np.array(
                odeint(_halo_hist_dMdz, np.asarray(Mh12), z)).flatten()
        return z, Mh12_z * 1e12 * u.Msun
    else:
        Mh12_z = []
        for Mh12_ in Mh12:
            Mh12_z.append(np.array(
                odeint(_halo_hist_dMdz, Mh12_, z)).flatten())
        Mh12_z = np.array(Mh12_z) * 1e12 * u.Msun
        return z, Mh12_z 
    
def _halo_hist_dMdz(Mh12, z):
    deriv = (Mdot_h(Mh12*1e12, z) * dtUdz(z)).to(u.Msun).value / 1e12
    return deriv

def Mdot_h(Mh, z):
    """
    Function that returns the halo dark matter accretion rate

    Parameters:
       Mh : float, array, or astropy.units.quantity.Quantity
          halo mass; if not an astropy Quantity, should have be in
          units of Msun
       z : float or array
          redshift

    Returns:
       Mdot_g : Float or array - astropy.units.quantity.Quantity
          halo gas accretion rate for given Mh and z
    """
    # Add units if needed
    if type(Mh) is not u.quantity.Quantity:
        Mh_ = Mh * u.Msun 
    else:
        Mh_ = Mh

    # Compute fit
    return (1e12*u.Msun * (-0.628) * (Mh_/(1e12*u.Msun))**1.14 * 
            omega_dot(z)).to(u.Msun/u.yr) # Eqn 3 (Sharda2024)

def omega_dot(z):
    """
    Neistein & Dekel approximation to time derivative of EPS formalism
    time variable

    Parameters:
       z : float or array
          redshift

    Returns:
       omega_dot : float or array
          self-similar time variable derivative
    """
    return -0.0476 * (1.0 + z + 0.093 * (1.0+z)**-1.22)**2.5 / \
        u.Gyr # Eqn 4 (Sharda)

def dtUdz(z):
    """
    Function to return dt/dz, where t = age of the universe

    Parameters:
       z : float or array
          redshift

    Returns:
       dtUdz : astropy.units.quantity.Quantity
          dt/dz
    """

    return -1.0 / (H0*(1.0+z) *
                  np.sqrt((1.0+z)**3*Omega_m_0+Omega_Lam_0))

def Mdot_gasacc(Mh, z):
    """
    Calculates gas mass accretion rate using Eqn. 8 of Sharda+2024
    
    Parameters: 
      Mh: float
        Halo mass in Msun, can take value only or with units
      z: float
        Redshift

   Returns:
      Mdotacc: float
        Gas accretion rate in g/s
    """ 
    # Ensure that Mh is in Msun
    if type(Mh) is not u.quantity.Quantity:
        Mh = Mh * u.Msun 
    else:
        Mh = Mh

    f_B = 0.17 
    Mdot_h_val = Mdot_h(Mh, z)
    eps_in = min(0.31*(Mh / (1e12*u.Msun))**(-0.25) * (1+z)**0.38, 1) # Eqn 17 (Ginzburg2022)
    Mdot_gasacc_val = Mdot_h_val*f_B*eps_in # Eqn 8 (Sharda2024)

    return Mdot_gasacc_val.to('g/s')

# # Varying c as a function of Mh using Fig. 16 Zhao2009
# Mh_arr = np.loadtxt("Mh_vs_c.txt", usecols = 0)
# c_arr = np.loadtxt("Mh_vs_c.txt", usecols = 1)
# c_Mh = interp1d(Mh_arr, c_arr) 
def find_halo_conc(Mh, z):
    """
    Calculates halo concentration, c, using Fig. 16 of Zha0+2009

    Parameters:
        Mh: float
            Halo mass in g, but value only
        z: float
            Redshift

    Returns:
        c: float
            Halo concentration
    """
    #return halo concentration given halo mass
    #figure 15 of Zhao, D et al. 2009
    if z <= 0.5:
        mhalo = np.array([11605267956,19444335388,48455695061,1.00997e11,1.76072e11,3.13105e11,
                          6.02799e11,1.03022e12,2.41892e12,5.56788e12,1.07194e13,1.65893e13,
                          2.51688e13,4.47571e13,7.35156e13,1.23173e14,1.90622e14,3.19382e14,
                          5.14289e14,7.95907e14,1.25643e15,2.14732e15])

        chalo = np.array([19.3174755,18.50755126,17.17098649,16.10267866,15.26362195,14.39121683,
                          13.49616515,12.72492683,11.55688586,10.3847185,9.558679074,8.964750624,
                          8.452751547,7.759839037,7.200313929,6.627840704,6.133660596,5.60098976,
                          5.15573281,4.796839007,4.510778576,4.275761576])
        
    elif z > 0.5 and z <= 1.5:
        mhalo = np.array([12810554614.09,16094446811.90,24173556293.37,45615575708.62,82728347870.66,
                          130573675209.34,227588560793.70,381240333771.29,638641638330.10,
                          1205077222871.76,2665006165554.10,5336887325653.84,7780571499420.01,14109699584211.10,
                          26101153641608.40,47335930250456.80,85850352935187.60,125170063000626.00,
                          209704982102530.00,331028131350302.00,533030100486917.00,875506282911291.00,
                          1936788931542690.00])

        chalo = np.array([9.061593131,8.918437866,8.638752274,8.235642427,7.893212103,7.584902664,7.250098469,
                          6.911549147,6.624021201,6.264637548,5.800038403,5.398438208,5.201298814,4.892898803,
                          4.639758579,4.423202849,4.26194622,4.183632201,4.117882012,4.074756254,4.053654061,
                          4.032684314,4.022875514])
        
    elif z > 1.5 and z <= 2.5:
        mhalo = np.array([11837932575.70,22790750182.27,55678772880.21,130731218873.01,289205732142.38,
                          652612176167.35,1137717616431.19,1906217331147.72,3323163614766.95,5041806883892.57,
                          8616776482670.85,15943593275816.60,29500378367232.90,65261217616734.90,
                          172611197503464.00,627211183012804.00,1726111975034630.00])

        chalo = np.array([5.857139793,5.626241122,5.318194201,5.037132861,4.809457262,4.592072474,4.455621562,
                          4.358141332,4.25422995,4.194762951,4.144453286,4.094747008,4.061941126,4.013224453,
                          4.005162003,4.005162003,4.005162003])
        
    else:
        mhalo = np.logspace(10,16,40)

        chalo = np.ones(40)*4.2
        
    return np.interp(Mh/cons.M_sun.cgs.value, mhalo, chalo)

# Making linear ramp function to find beta and fsf for galactic disc code and ZCGM for metallicity code
def line_func(x1, y1, x2, y2):
    """
    Function to calculate the slope and y-intercept of a line given two points
    
    Parameters:
        x1: Float - x-coordinate of first point
        y1: Float - y-coordinate of first point
        x2: Float - x-coordinate of second point
        y2: Float - y-coordinate of second point

    Returns:
        m: Float - Slope of line
        b: Float - y-intercept of line
    """

    m = (y2 - y1) / (x2 - x1) # Slope
    c = y1 - m*x1 # y-intercept
    return m, c

# Function to find index closest to desired value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# SHM relation
def Moster_SHM(Mh, z):
    """
    Calculates stellar mass from halo mass using SHM relation from Moster+2012

    Parameters:
        Mh: float or array
            Halo mass in Msun, but value only
        z: float or array
            Redshift

    Returns: 
        Mstar: float or array
            Stellar mass in Msun, but value only
    """
    # Calculating stellar mass from Moster et al. 2012 - parameter values from Table 1, Moster+2013
    M1 = 10**(11.59 + 1.195*(z / (z+1))) # Eqn 11 (Moster2013)
    N  = 0.0351 - 0.0247*(z / (z+1)) # Eqn 12 (Moster2013)
    beta  = 1.376 - 0.826*(z / (z+1)) # Eqn 13 (Moster2013)
    gamma  = 0.608 + 0.329*(z / (z+1)) # Eqn 14 (Moster2013)
    Mstar_val = (2*N*Mh * ((Mh/M1)**(-beta) + (Mh/M1)**gamma)**(-1)) # Eqn 2 (Moster+2013)

    return Mstar_val

# Using snippet of Piyush's code to calculate fHI and fH2 using Fig. 4 of Saintonge & Catinella 2022
def find_HIH2_SC22(mstar):
    #find f_HI from Saintonger & Catinella 2022 review fig 4
    #we dont really have HI beyond z ~ 0.1/1
    logmstar = np.array([9.139037433155082, 9.431372549019608, 9.727272727272728, 10.03565062388592, 
                         10.577540106951872, 10.86096256684492, 11.11764705882353])
    logfHI = np.array([-0.23199999999999954, -0.5759999999999994, -0.8479999999999992, -0.9999999999999991,
                       -1.3439999999999985, -1.5919999999999987, -1.7199999999999989])
    #fitting with a 3rd order polynomial is good enough
    fit_fHI_SC22 = np.polyfit(logmstar, logfHI, 3)

    #H2
    logmstar = np.array([9.153161175422975, 9.425645592163848, 9.726625111308994, 10.025823686553874,
                        10.29652715939448, 10.577916295636687, 10.852181656277828, 11.1193232413179])
    logfH2 = np.array([-1.092233009708738, -1.2888349514563113, -1.4854368932038833, -1.3834951456310682,
                       -1.660194174757282, -2.009708737864078, -2.133495145631068, -2.446601941747573])
    #fitting with a 6th order polynomial is good enough
    fit_fH2_SC22 = np.polyfit(logmstar, logfH2, 6)

    logmstar = np.log10(mstar / (1*u.Msun).to('g').value)
    fHI = 10**(np.poly1d(fit_fHI_SC22)(logmstar))
    fH2 = 10**(np.poly1d(fit_fH2_SC22)(logmstar))
    return fHI, fH2

# # Making function that calculates array of redshifts and Mhz_init when given Mhz_final
# def z_arr_Mhz_init(z_init, z_final, logMhz_final):
#     """
#     Function to create array of redshifts given initial and final redshift and to calculate Mhz_init that will give Mhz_final at z_final

#     Parameters:
#         z_init: Float - Initial redshift
#         z_final: Float - Final redshift
#         logMhz_final: Float - Log10 of the final halo mass at z = z_final in units of Msun astropy.units

#     Returns:
#         z_arr: Arraylike - Array of redshifts with reasonable timesteps
#         Mhz_init: Float - Halo mass at z = z_init in units of Msun astropy.units
#     """

#     # Finding z_arr based on given z_init, z_final
#     z_arr = [z_init]
#     z_current = z_init

#     dt_min = 1e4 * u.yr
#     dt_max = 1e9 * u.yr

#     while z_current > z_final:
#         dz_min = (dt_min / np.abs(dtUdz(z_current))).value
#         dz_max = (dt_max / np.abs(dtUdz(z_current))).value

#         # Choose reasonable dz within limits
#         dz = min(0.05, max(dz_min, min(dz_max, 0.1)))

#         # Update redshift
#         z_next = max(z_final, z_current - dz) # To ensure I don't go below z = 0
#         z_arr.append(z_next)
#         z_current = z_next

#     z_arr = np.array(z_arr)    
#     z_arr = z_arr[np.abs(z_arr) > 1e-12] # Removing floating point noise
#     z_arr = np.append(z_arr, z_final) # Ensuring z_final is last element 

#     # Calculating Mhz_init that will give Mhz_final at z_final
#     Mhz_init_arr = np.logspace(logMhz_final - 2, logMhz_final) # Array of Mh(z_final) values
#     Mhz_final_arr = [] # Empty list to put all Mhz_final values
#     for i in range(len(Mhz_init_arr)):
#         Mhz_final_val = halo_hist(Mhz_init_arr[i] * u.Msun, z = [z_final, z_init])[1][1] # To determine halo mass at z_final
#         Mhz_final_arr.append(Mhz_final_val.value)

#     Mhz_final_interp = interp1d(np.log10(u.Quantity(Mhz_init_arr)), np.log10(Mhz_final_arr)) 
#     logMhz_initial = Mhz_final_interp(logMhz_final) # Using an interpolator to find Mhz_final

#     Mhz_init = 10**logMhz_initial # Mh(z=z_initial) value that returns Mh(z=z_final)=1e12

#     return z_arr, Mhz_init

# Function to import and interpolate Mdotacc data from Ruby
z_header = [0, 1, 2, 4]
cosmo_sims = ['TNG', 'SIMBA', 'EAGLE']
def Mdotacc_sim(sim):
    """
    Takes Mdotacc data from given cosmo sim and interpolates in 2D (logMh, z) returning an interpolator function

    Parameters:
        sim: 'TNG', 'SIMBA', 'EAGLE'
            Cosmological simulation to get Mdotacc data from
    Returns:
        Interpolator function: Takes log10(Mh/Msun) and z as inputs and returns log10Mdotacc
    """

    # Load accretion data
    logMdotacc_data = np.genfromtxt(f"Data/Mdot_acc/m200_inflow_{sim}.csv", skip_header = 1, delimiter = ",", filling_values = np.nan)
    
    # Use LinearNDInterpolator to 2D interpolate Mh, z for Mdotacc
    logMh_arr = logMdotacc_data[:, 0]
    logMdotacc_arr = logMdotacc_data[:, 1:]

    # Flattening Mdotacc and etaw into 1D arrays to input into LinearNDInterpolator
    logMh_flat = []
    z_flat = []
    logMdotacc_flat = []
    for i, Mh in enumerate(logMh_arr):
        for j, z in enumerate(z_header):
            logMh_flat.append(Mh)
            z_flat.append(z)
            logMdotacc_flat.append(logMdotacc_arr[i, j])
            
    logMdotacc_interpfunc = LinearNDInterpolator(list(zip(logMh_flat, z_flat)), logMdotacc_flat, fill_value = np.nan)
    
    return logMdotacc_interpfunc

# Function to import and interpolate etaw data from Ruby
def etaw_sim(sim):
    """
    Takes etaw data from given cosmo sim and interpolates in 2D (logMh, z) returning an interpolator function

    Parameters:
        sim: 'TNG', 'SIMBA', 'EAGLE'
            Cosmological simulation to get Mdotacc data from
    Returns:
        Interpolator function: Takes log10(Mh/Msun) and z as inputs and returns logetaw
    """

    # Load accretion data
    logetaw_data = np.genfromtxt(f"Data/Mdot_out/m200_eta_{sim}.csv", skip_header = 1, delimiter = ",", filling_values = np.nan)
    
    # Use LinearNDInterpolator to 2D interpolate Mh, z for Mdotacc
    logMh_arr = logetaw_data[:, 0]
    logetaw_arr = logetaw_data[:, 1:]

    # Flattening Mdotacc and etaw into 1D arrays to input into LinearNDInterpolator
    logMh_flat = []
    z_flat = []
    logetaw_flat = []
    for i, Mh in enumerate(logMh_arr):
        for j, z in enumerate(z_header):
            logMh_flat.append(Mh)
            z_flat.append(z)
            logetaw_flat.append(logetaw_arr[i, j])
            
    logetaw_interpfunc = LinearNDInterpolator(list(zip(logMh_flat, z_flat)), logetaw_flat, fill_value = np.nan)
    
    return logetaw_interpfunc

def find_etaw(Mh, z, etaw_option, etaw):
    """
    Calculate mass-loading factor, etaw, using values from TNG, TNG (Ruby), HH17, FIRE-2, EAGLE,
    EAGLE (Ruby), or SIMBA (Ruby)

    Parameters:
        Mh: float or array
            Halo mass in Msun, but value only
        z: float or array
            Redshift
        etaw_option: 'constant', 'TNG', 'TNG (Ruby)', 'HH17', 'FIRE-2', 'EAGLE', 'EAGLE (Ruby)', 'SIMBA (Ruby)'
            Values of etaw to use
        etaw_const: float
            Constant etaw value to use for etaw_option='constant'

    Returns:
        etaw: float or array
            Mass-loading factor
    """
    mstar = Moster_SHM(Mh, z) # Returns Mstar in Msun, but value only
    logMh = np.log10(Mh) # logMh required as input for logetaw data from Ruby
    
    if etaw_option=='TNG':
        #Pillepich et al. 2018a
        #Joop - TNG puts this scaling by hand?!
        etaw = 1.5*(Mh/1e12)**(-5./6.)

    elif etaw_option=='TNG (Ruby)':
        logetaw_interpfunc = etaw_sim('TNG')
        if z == 2:
            z = 1.9999999999 # To avoid floating point precision error: Avoid interpolator returning nan
        etaw = 10**(logetaw_interpfunc(logMh, z)) # Data is given in log(etaw)
    
    elif etaw_option=='HH17':
        #Hayward & Hopkins 2017 appendix
        f0_HH17 = 1.0/(1 + (mstar/10**9.15)**0.4)
        tauz_HH17 = cosmo.lookback_time(z).value/cosmo.lookback_time(np.inf).value
        fg_HH17 = f0_HH17*(1 - tauz_HH17*(1 - f0_HH17**1.5))**(-2./3.)
        etaw = 14*((fg_HH17*mstar/1e10)**-0.23)*np.exp(-0.75/fg_HH17)
    
    elif etaw_option=='FIRE-2':
        #Pandya et al. 2021, equation 15
        etaw = (10**4.3)*(mstar)**-0.45
        
    elif etaw_option=='EAGLE':
        #Mitchell et al. 2020 fig 15
        mstar_array = np.array([8.2935, 8.3935, 8.4548, 8.5516, 8.6419, 8.7516, 8.8742, 8.9839,
                                9.0871, 9.1806, 9.2871, 9.3839, 9.5161, 9.6548, 9.7774, 9.9161,
                                10.0484, 10.1581, 10.2677, 10.3710, 10.4613, 10.5548, 10.6677,
                                10.7710, 10.8806, 10.9806])
        neta_array = np.array([0.4212, 0.3788, 0.3545, 0.3333, 0.3152, 0.2879, 0.2788, 0.2636, 0.2515,
                               0.2303, 0.2152, 0.2000, 0.1727, 0.1485, 0.1182, 0.0879, 0.0667, 0.0545,
                               0.0545, 0.0545, 0.0606, 0.0818, 0.1152, 0.1455, 0.1788, 0.2030])
        
        etaw = 10**(interp1d(mstar_array, neta_array, fill_value='extrapolate')(np.log10(mstar)))

    elif etaw_option=='EAGLE (Ruby)':
        logetaw_interpfunc = etaw_sim('EAGLE')
        if logMh == 13.5:
            logMh = 13.49999 # To avoid floating point precision error: Avoid interpolator returning nan
        elif logMh == 12:
            logMh = 11.99999 # To avoid floating point precision error: Avoid interpolator returning nan
        elif z == 2:
            z = 1.9999999999 # To avoid floating point precision error: Avoid interpolator returning nan
        etaw = 10**(logetaw_interpfunc(logMh, z)) # Data is given in log(etaw)
    
    elif etaw_option == 'SIMBA (Ruby)':
        logetaw_interpfunc = etaw_sim('SIMBA')
        etaw = 10**(logetaw_interpfunc(logMh, z)) # Data is given in log(etaw)

    return etaw

def dMgdz(z, Mg, z_init, Mhz_init, etaw_const, etaw_option, beta_const, beta_option, xi_const, xi_option, Mdotacc_option):
    """
    Function to integrate mass equilibrium equation which includes accretion, star formation, outflows, 
    and gas transport, to calculate Mg

    Parameters:
        z: Float - Redshift to integrate over from high redshift to low redshift
        Mg: Float - Gas mass in Msun astropy.units, or cgs, at redshift z. If dimensionless,
            have in value of Msun
        z_init: Float - Initial redshift
        Mhz_init: Float - Log10 of the final halo mass at z = z_final in units of Msun astropy.units
        etaw_const: Float - Mass loading factor at all redshifts if 'constant' eta is chosen
        etaw_option: 'constant', 'TNG', 'HH17', 'FIRE-2', 'EAGLE', "TNG (Ruby)", "SIMBA (Ruby)", "EAGLE (Ruby)" -
                      Specify what values of eta to use
        beta_const: Float - Constant for rotation curve index of galaxy
        beta_option: 'constant', 'varying' - Choose if beta is chosen constant value or varies with stellar mass
        fsf_const: Float - Fixed value for fsf, fraction of star-forming molecular gas
        fsf_option: 'constant', 'varying' - Choose if beta is fixed or varies with stellar mass
        xi_const: Float - Constant that describes fraction of accretion kinetic energy that goes to turbulence
        xi_option: 'constant', 'varying' - Choose if xi_a is chosen constant value or varies with redshift as 0.2(1+z)
        Mdotacc_option: 'original', 'TNG', 'SIMBA', 'EAGLE' - Choose which cosmological simulation to use for Mdotacc or to use original expression

    Returns:
        dMgdz: Float - Total gas mass rate of change in units of Msun/z at given redshift 
    """
    # print(f'z:{z}, Mg: {Mg}')

    # Ensure units is correct on Mg
    Mg = Mg * u.g

    # Calculating Mh at given z
    Mh_val = halo_hist((Mhz_init*u.g).to('Msun'), z = [z_init, z])[1][1].to('g') # Halo mass at redshift z in g
    logMh_val = np.log10(Mh_val.to('Msun').value) # log10(Mh/Msun)

    # Finding halo concentration
    c = find_halo_conc(Mh_val.value, z) 

    # Calculating Mstar using Moster+2012
    Mstar_val = (Moster_SHM(Mh_val.to('Msun').value, z) * u.Msun).to('g') # Mstar in g
    logMstar = np.log10(Mstar_val.to('Msun').value) # Finding log10(Mstar/Msun)

    ### Calculating gas accretion rate ###
    # Calculating parameters for original Mdotacc
    Mdoth_val = Mdot_h(Mh_val.to('Msun'), z).to('g/s') # Halo growth rate in g/s # Halo mass accretion rate - Eqn 3 (Sharda2024)
    f_B = 0.17 # Universal baryonic fraction
    eps_in = min(0.31 * (Mh_val.to('Msun') / (1e12*u.Msun))**(-0.25) * (1+z)**0.38, 1) # Eqn 17 (Ginzburg2022)
    if Mdotacc_option == "original":
        Mdot_gasacc_val = Mdoth_val*f_B*eps_in # In g/s - Eqn 8 (Sharda2024)
    
    else: # If using Mdotacc from cosmo sims
        if Mdotacc_option == "TNG":
            logMdotacc_interpfunc = Mdotacc_sim(Mdotacc_option)
            if z == 2:
                z = 1.9999999999 # To avoid floating point precision error: Avoid interpolator returning nan
            Mdot_gasacc_val = (10**(logMdotacc_interpfunc(logMh_val, z)) * u.Msun/u.Gyr).cgs # Data is given in log(Msun/Gyr) - convert to g/s
    
        elif Mdotacc_option == "EAGLE":
            logMdotacc_interpfunc = Mdotacc_sim(Mdotacc_option)
            if logMh_val == 13.5:
                logMh_val = 13.49999 # To avoid floating point precision error: Avoid interpolator returning nan
            elif logMh_val == 12:
                logMh_val = 11.99999 # To avoid floating point precision error: Avoid interpolator returning nan
            elif z == 2:
                z = 1.9999999999 # To avoid floating point precision error: Avoid interpolator returning nan
            Mdot_gasacc_val = (10**(logMdotacc_interpfunc(logMh_val, z)) * u.Msun/u.Gyr).cgs # Data is given in log(Msun/Gyr) - convert to g/s

        elif Mdotacc_option == "SIMBA":
            logMdotacc_interpfunc = Mdotacc_sim(Mdotacc_option)
            Mdot_gasacc_val = (10**(logMdotacc_interpfunc(logMh_val, z)) * u.Msun/u.Gyr).cgs # Data is given in log(Msun/Gyr) - convert to g/s

    ### Calculating star formation rate ###
    # Calculating v_phi, R, t_orb, Omega, kappa
    # v_phi = (75.17 * np.sqrt(c / (np.log(1+c) - (c/(1+c)))) * (Mh_val / (1e12*u.Msun).to('g'))**(1/3) * (1+z)**(0.5) * (u.km/u.s)).to('cm/s') # Eqn 5 (Sharda2024)
    Vv = ((200* u.km/u.s) * (Mh_val.to('Msun') / (1e12*u.Msun))**(1/3) * ((1+z)/3)**(0.5)).to('cm/s') # Eqn 2 Ginzburg+2022
    Vmax = 0.465 * Vv * np.sqrt(c / (np.log(1+c) - (c/(1+c)))) # Eqn 4 Ginzburg+2022
    v_phi = 1.4*Vmax
    
    # R = (1 * 1.89 * (Mh_val / (1e12*u.Msun).to('g'))**(1/3) * ((1+z)/ 3)**(-1) * u.kpc).to('cm') # Eqn 6 (Sharda2024)
    Rv = ((100*0.54*u.kpc) * (Mh_val.to('Msun') / (1e12*u.Msun))**(1/3) * ((1+z)/3)**(-1)).to('cm') # Eqn 1 Ginzburg+2022
    R = 2*0.035*Rv # Eqn 6 Sharda+2024

    # Toomre Q parameter
    Q_min = 1 # Toomre criterion for stability
    Q_val = Q_min # Without transport considered
    G_const = G.to('cm^3 / (g s^2)')

    # Calculating beta and f_gQ = f_gP
    fH2_Tacconi = 10**(0.06 + -3.33*(np.log10(1+z) - 0.65)**2 - 0.41*(logMstar - 10.7)) # fH2 from Tacconi
    fgz = fH2_Tacconi / (1 + fH2_Tacconi)
    # fgQ_highM = 0.26*fgz + 0.48 # Eqn 42 Ginzburg+2022 - Piyush uses different coefficients
    fgQ_highM = 0.39*fgz + 0.38 # Eqn 42 Ginzburg+2022
    fgQ_lowM = 0.0125*z + 0.9 # Following Piyush's code

    if logMstar <= 9:
        f_gQ = fgQ_lowM
        if beta_option == "constant":
            beta = beta_const
        else:
            beta = 0.5

    elif logMstar >= 10.5:
        f_gQ = fgQ_highM
        if beta_option == "constant":
            beta = beta_const
        else:
            beta = 0

    else:
        line_params = line_func(9, fgQ_lowM, 10.5, fgQ_highM)
        f_gQ = line_params[0]*logMstar + line_params[1]
        line_params = line_func(9, 0.5, 10.5, 0) # Getting slope and y-intercept of line
        if beta_option == "constant":
            beta = beta_const
        else:
            beta = line_params[0]*logMstar + line_params[1]

    # Calcutating fsf using fH2 from Tacconi and fH1 from Saintonge & Catinella 2022 review Fig. 4
    fH1_SC22 = find_HIH2_SC22(Mstar_val.value)[0] # Give Mstar in units of Msun
    f_sf = fH2_Tacconi / (fH2_Tacconi + fH1_SC22)

    # Calculating sigmag
    sigmag_val = np.sqrt(1+beta)*Mg*Q_val*G_const/ (2*np.sqrt(2)*f_gQ*v_phi*R)
    # Sigmag_val = Mg.cgs / (np.pi * R**2) # In g / cm^2 - Eqn 6 (Sharda2024)
    # sigmag_val = (np.pi*G_const*Sigmag_val) / (kappa*f_gQ) # From setting Toomre Q = 1 and solving for sigma_g in cgs

    # Calculating star formation rate
    phi_mp = 1.4 # Table 1 Sharda+2024
    eps_ff = 0.015 # Table 1 Sharda+2024
    t_SFmax = (2*u.Gyr).to('s') # Table 1 Sharda+2024
    t_orb = 2*np.pi*R / v_phi  # In s - Eqn 5 (Ginzburg2022) - with factor 2pi to see GMC regime at low z
    Omegarot = v_phi/R # In /s - Eqn 5 (Ginzburg2022)
    kappa = np.sqrt(2*(1+beta)) * Omegarot # Eqn 1 (Sharda2024) 

    Toomre_fac = np.sqrt(2*(1+beta) / (3*f_gQ*phi_mp)) * (8*eps_ff*f_gQ / Q_val) # Toomre term
    GMC_fac = t_orb/t_SFmax # GMC term
    if GMC_fac > Toomre_fac: # Different phi_a values based on text in 3.1.2 of Krumholz+2018
        phi_a = 1
    else:
        phi_a = 2

    Mdot_SF_Toomre = (np.sqrt(2/(1+beta)) * (phi_a*f_sf*f_gQ*(v_phi**2)*sigmag_val / (np.pi*G_const*Q_val)) * Toomre_fac)
    Mdot_SF_GMC = (np.sqrt(2/(1+beta)) * (phi_a*f_sf*f_gQ*(v_phi**2)*sigmag_val / (np.pi*G_const*Q_val)) * GMC_fac)
    Mdot_SF_val = max(Mdot_SF_Toomre, Mdot_SF_GMC) # Eqn 9 (Sharda2024) - In g/s    

    ## Calculating mass transport rate ###
    # Calculating phint 
    sigmath = (f_sf*0.2*(u.km/u.s)).to('cm/s') + ((1 - f_sf)*5.4*(u.km/u.s)).to('cm/s') # In cm/s - Section 2.1.3 Sharda+2024
    phint = 1 - (sigmath/sigmag_val)**2 # Eqn 12 Sharda+2024

    if phint < 0:
        Mdot_trans_val = 0

    else: # Calculate mass transport rate
        phi_Q = 2.0 # Table 1 Sharda+2024 
        eta = 1.5 # Table 1 Sharda+2024
        pmstar = (3e3 * u.km/u.s).to('cm/s') # Table 1 Sharda+2024 - In cm/s

        # Calculating sigmaSF
        t_orb_prime = t_orb/2 # Calculate torb using characteristic disc radius, i.e., without factor of 2
        GMC_term = np.sqrt(3*f_gQ / (8*(1+beta))) * (Q_min*phi_mp / (4*f_gQ*eps_ff)) * (t_orb_prime/t_SFmax) # GMC term 
        sigmaSF_Toomre = (4*f_sf*eps_ff / (np.sqrt(3*f_gQ)*np.pi*eta*phi_mp*phi_Q * phint**1.5)) * pmstar
        sigmaSF_GMC = (4*f_sf*eps_ff / (np.sqrt(3*f_gQ)*np.pi*eta*phi_mp*phi_Q * phint**1.5)) * pmstar * GMC_term
        sigmaSF_val = max(sigmaSF_Toomre, sigmaSF_GMC) # Eqn 39 KBFC18

       # Calculating sigmaacc
        if xi_option == "constant": # Giving option to have constant xi_a for all z
            xi_a = xi_const 
        else:
            xi_a = 0.2*(1+z) # Sharda+2024 section 2.1.3

        sigmaacc_val = (((2+beta)*xi_a*G_const*Mdot_gasacc_val / (8*(1+beta)*eta*phi_Q * phint**1.5)) * (Q_min/f_gQ)**2)**(1/3) # In cm/s

        # Calculating Fsigma
        Fsigma = 1 - (sigmaSF_val/sigmag_val) - (sigmaacc_val/sigmag_val)**3    
        if Fsigma < 0:
            Mdot_trans_val = 0
        else:
            Mdot_trans_val = ((4*eta*phi_Q*phint**1.5) * (f_gQ/Q_min)**2 * (1+beta / (1-beta)) * ((sigmag_val**3) / G_const) * Fsigma)

    ### Calculating outflows term ###
    etaw = find_etaw(Mh_val.to('Msun').value, z, etaw_option, etaw_const)
    
    Mdot_out_val = etaw*Mdot_SF_val

    ### Calculating Mdotg ###
    Mdotg_val = Mdot_gasacc_val-Mdot_trans_val-Mdot_SF_val-Mdot_out_val

    # print(f'z:{z}, log10Mh_val: {np.log10(Mh_val.to("Msun").value)}, Mdotg: {Mdotg_val}, Mdotacc: {Mdot_gasacc_val}, Mdotout:{Mdot_out_val}')
    
    return (Mdotg_val)*dtUdz(z).cgs.value # Return Mdotg in g/s

def Mg_outputs(z, Mg, z_init, Mhz_init, etaw_const, etaw_option, beta_const, beta_option, xi_const, xi_option, Mdotacc_option):
    """
    Function that returns all the relevant parameters used in mass equilibrium equation in a dictionary when inputting Mghist 
    obtained by integrating mass equilibrium equation

    Parameters:
        z: float
            Redshift
        Mg: float 
            Gas mass in g, at redshift z.
        z_init: float
            Initial redshift
        Mhz_init: float 
            Halo mass at z_init in units of g, but value only
        etaw_const: float
            Mass-loading constant for etaw_option='const'
        etaw_option: 'constant', 'TNG', 'HH17', 'FIRE-2', 'EAGLE', "TNG (Ruby)", "SIMBA (Ruby)", "EAGLE (Ruby)" -
                      Specify what values of eta to use
        beta_const: float
            beta constant for beta_option='constant'
        beta_option: 'constant', 'varying' 
            Choose if beta is chosen constant value or varies with stellar mass
        xi_const: float
            xi value for xi_const='constant'
        xi_option: 'constant', 'varying'
            Choose if xi_a is chosen constant value or varies with redshift as 0.2(1+z)
        Mdotacc_option: "original", 'TNG', 'SIMBA', 'EAGLE'
            Choose which cosmological simulation to use for Mdotacc or to use original expression

    Returns:
        z_arr: Arraylike - Redshift
        Mdotg: Arraylike - Total gas mass rate of change in g/s
        Mdotacc: Arraylike - Gas mass rate of change due to accretion in g/s
        MdotSF: Arraylike - Gas mass rate of change due to star formation in g/s
        Mdotout: Arraylike - Gas mass rate of change due to stellar outflows in g/s
        etaw: Arraylike - Mass loading factor
        Mdottrans: Arraylike - Gas mass rate of change due to gas transport in g/s
        Fsigma: Arraylike - Parameterises how much radial transport contributes to turbulence
        sigma_g: Arraylike - Gas velocity dispersion in cm/s
        sigma_SF: Arraylike - Stellar feedback contribution to gas velocity dispersion in cm/s
        zSF_Toomre: Arraylike - Redshifts where sigmaSF is in Toomre regime
        sigmaSF_Toomre: Arraylike - Gas velocity dispersion from SF in Toomre regime in cm/s
        zSF_GMC: Arraylike - Redshifts where sigmaSF is in GMC regime
        sigmaSF_GMC: Arraylike - Gas velocity dispersion from SF in GMC regime in cm/s
        sigma_acc: Arraylike - Accretion contribution to gas velocity dispersion in cm/s
        z_Toomre: Arraylike - Redshift where SF is in Toomre regime
        MdotSF_Toomre: Arraylike - SFR in Toomre regime in g/s
        z_GMC: Arraylike - Redshift where SF is in GMC regime
        MdotSF_GMC: Arraylike - SFR in GMC regime in g/s
        Mh_val: Arraylike - Halo mass at redshift z in Msun
        Mstar_val: Arraylike - Galaxy's stellar mass at redshift z in Msun
        Mstarint: Arraylike - Stellar mass from integrating MdotSF
        beta: Arraylike - Rotation curve index 
        fgQ: Arraylike - Effective fraction of gas
        vphi: Arraylike - Rotational velocity at outer disc edge in cm/s
        R: Arraylike - Galactic disc radius in cm
        torb: Arraylike - Orbital timescale at outer galactic disc in s
        Omega: Arraylike - Angular frequency at outer disc in /s
        c: Arraylike - Halo concentration
        kappa: Arraylike - Epicyclic frequency in /s
        phi_nt: Float - Describes contribution of thermal motions to turbulence
        phi_Q: Arraylike 
        Q_val: Arraylike - Toomre parameter
        fsf: Arraylike - Fraction of gas in star-forming molecular phase
        xia: Arraylike - Fraction of energy from accreted gas that goes to ISM
        epsin: Arraylike - Efficiency of gas accretion
        Mdoth: Arraylike - Halo growth rate in g/s
    """

    # Ensure units is correct on Mg
    Mg = Mg * u.g

    # Calculating Mh at given z
    Mh_val = halo_hist((Mhz_init*u.g).to('Msun'), z = [z_init, z])[1][1].to('g') # Halo mass at redshift z in g
    logMh_val = np.log10(Mh_val.to('Msun').value) # log10(Mh/Msun)

    # Finding halo concentration
    c = find_halo_conc(Mh_val.value, z) 

    # Calculating Mstar using Moster+201
    Mstar_val = (Moster_SHM(Mh_val.to('Msun').value, z) * u.Msun).to('g') # Mstar in g
    logMstar = np.log10(Mstar_val.to('Msun').value) # Finding log10(Mstar/Msun)

    ### Calculating gas accretion rate ###
    # Calculating parameters for original Mdotacc
    Mdoth_val = Mdot_h(Mh_val.to('Msun'), z).to('g/s') # Halo growth rate in g/s # Halo mass accretion rate - Eqn 3 (Sharda2024)
    f_B = 0.17 # Universal baryonic fraction
    eps_in = min(0.31 * (Mh_val.to('Msun') / (1e12*u.Msun))**(-0.25) * (1+z)**0.38, 1) # Eqn 17 (Ginzburg2022)
    if Mdotacc_option == "original":
        Mdot_gasacc_val = (Mdoth_val*f_B*eps_in).cgs # In cgs - Eqn 8 (Sharda2024)
    
    else: # If using Mdotacc from cosmo sims
        if Mdotacc_option == "TNG":
            logMdotacc_interpfunc = Mdotacc_sim(Mdotacc_option)
            if z == 2:
                z = 1.9999999999 # To avoid floating point precision error: Avoid interpolator returning nan
            Mdot_gasacc_val = (10**(logMdotacc_interpfunc(logMh_val, z)) * u.Msun/u.Gyr).cgs # Data is given in log(Msun/Gyr) - convert to g/s
    
        elif Mdotacc_option == "EAGLE":
            logMdotacc_interpfunc = Mdotacc_sim(Mdotacc_option)
            if logMh_val == 13.5:
                logMh_val = 13.49999 # To avoid floating point precision error: Avoid interpolator returning nan
            elif logMh_val == 12:
                logMh_val = 11.99999 # To avoid floating point precision error: Avoid interpolator returning nan
            elif z == 2:
                z = 1.9999999999 # To avoid floating point precision error: Avoid interpolator returning nan
            Mdot_gasacc_val = (10**(logMdotacc_interpfunc(logMh_val, z)) * u.Msun/u.Gyr).cgs # Data is given in log(Msun/Gyr) - convert to g/s

        elif Mdotacc_option == "SIMBA":
            logMdotacc_interpfunc = Mdotacc_sim(Mdotacc_option)
            Mdot_gasacc_val = (10**(logMdotacc_interpfunc(logMh_val, z)) * u.Msun/u.Gyr).cgs # Data is given in log(Msun/Gyr) - convert to g/s

    ### Calculating star formation rate ###
    # Calculating v_phi, R, t_orb, Omega, kappa
    # v_phi = (75.17 * np.sqrt(c / (np.log(1+c) - (c/(1+c)))) * (Mh_val / (1e12*u.Msun).to('g'))**(1/3) * (1+z)**(0.5) * (u.km/u.s)).to('cm/s') # Eqn 5 (Sharda2024)
    Vv = ((200* u.km/u.s) * (Mh_val.to('Msun') / (1e12*u.Msun))**(1/3) * ((1+z)/3)**(0.5)).to('cm/s') # Eqn 2 Ginzburg+2022
    Vmax = 0.465 * Vv * np.sqrt(c / (np.log(1+c) - (c/(1+c)))) # Eqn 4 Ginzburg+2022
    v_phi = 1.4*Vmax
    
    # R = (1 * 1.89 * (Mh_val / (1e12*u.Msun).to('g'))**(1/3) * ((1+z)/ 3)**(-1) * u.kpc).to('cm') # Eqn 6 (Sharda2024)
    Rv = ((100*0.54*u.kpc) * (Mh_val.to('Msun') / (1e12*u.Msun))**(1/3) * ((1+z)/3)**(-1)).to('cm') # Eqn 1 Ginzburg+2022
    R = 2*0.035*Rv # Eqn 6 Sharda+2024

    # Toomre Q parameter
    Q_min = 1 # Toomre criterion for stability
    Q_val = Q_min # Without transport considered
    G_const = G.to('cm^3 / (g s^2)')

    # Calculating beta and f_gQ = f_gP
    fH2_Tacconi = 10**(0.06 + -3.33*(np.log10(1+z) - 0.65)**2 - 0.41*(logMstar - 10.7)) # fH2 from Tacconi
    fgz = fH2_Tacconi / (1 + fH2_Tacconi)
    # fgQ_highM = 0.26*fgz + 0.48 # Eqn 42 Ginzburg+2022 - Piyush uses different coefficients
    fgQ_highM = 0.39*fgz + 0.38 # Eqn 42 Ginzburg+2022
    fgQ_lowM = 0.0125*z + 0.9 # Following Piyush's code

    if logMstar <= 9:
        f_gQ = fgQ_lowM
        if beta_option == "constant":
            beta = beta_const
        else:
            beta = 0.5

    elif logMstar >= 10.5:
        f_gQ = fgQ_highM
        if beta_option == "constant":
            beta = beta_const
        else:
            beta = 0

    else:
        line_params = line_func(9, fgQ_lowM, 10.5, fgQ_highM)
        f_gQ = line_params[0]*logMstar + line_params[1]
        line_params = line_func(9, 0.5, 10.5, 0) # Getting slope and y-intercept of line
        if beta_option == "constant":
            beta = beta_const
        else:
            beta = line_params[0]*logMstar + line_params[1]

    # Calcutating fsf using fH2 from Tacconi and fH1 from Saintonge & Catinella 2022 review Fig. 4
    fH1_SC22 = find_HIH2_SC22(Mstar_val.value)[0] # Give Mstar in units of Msun
    f_sf = fH2_Tacconi / (fH2_Tacconi + fH1_SC22)

    # Calculating sigmag
    sigmag_val = np.sqrt(1+beta)*Mg*Q_val*G_const/ (2*np.sqrt(2)*f_gQ*v_phi*R)
    sigmag_init = sigmag_val
    # Sigmag_val = Mg.cgs / (np.pi * R**2) # In g / cm^2 - Eqn 6 (Sharda2024)
    # sigmag_val = (np.pi*G_const*Sigmag_val) / (kappa*f_gQ) # From setting Toomre Q = 1 and solving for sigma_g in cgs

    # Calculating star formation rate
    phi_mp = 1.4 # Table 1 Sharda+2024
    eps_ff = 0.015 # Table 1 Sharda+2024
    t_SFmax = (2*u.Gyr).to('s') # Table 1 Sharda+2024
    t_orb = 2*np.pi*R / v_phi  # In s - Eqn 5 (Ginzburg2022) - with factor 2pi to see GMC regime at low z
    Omegarot = v_phi/R # In /s - Eqn 5 (Ginzburg2022)
    kappa = np.sqrt(2*(1+beta)) * Omegarot # Eqn 1 (Sharda2024) 

    Toomre_fac = np.sqrt(2*(1+beta) / (3*f_gQ*phi_mp)) * (8*eps_ff*f_gQ / Q_val) # Toomre term
    GMC_fac = t_orb/t_SFmax # GMC term
    if GMC_fac > Toomre_fac: # Different phi_a values based on text in 3.1.2 of Krumholz+2018
        phi_a = 1
    else:
        phi_a = 2

    Mdot_SF_Toomre = (np.sqrt(2/(1+beta)) * (phi_a*f_sf*f_gQ*(v_phi**2)*sigmag_val / (np.pi*G_const*Q_val)) * Toomre_fac)
    Mdot_SF_GMC = (np.sqrt(2/(1+beta)) * (phi_a*f_sf*f_gQ*(v_phi**2)*sigmag_val / (np.pi*G_const*Q_val)) * GMC_fac)
    Mdot_SF_val = max(Mdot_SF_Toomre, Mdot_SF_GMC) # Eqn 9 (Sharda2024) - In g/s  
    
    ### Calculating mass transport rate ###
    phi_Q = 2.0 # Table 1 Sharda+2024
    Qstar = (Q_val/f_gQ) / (phi_Q - 1) # Rearrange phiQ to get Qstar - Eqn 11 Sharda+2024
    eta = 1.5 # Table 1 Sharda+2024
    pmstar = (3e3 * u.km/u.s).to('cm/s') # Table 1 Sharda+2024 - In cm/s
    t_orb_prime = t_orb/2 # For calculating sigmaSF - only use characteristic disc radius, i.e., R without 2 factor

    # Calculating phint 
    sigmath = (f_sf*0.2*(u.km/u.s)).to('cm/s') + ((1 - f_sf)*5.4*(u.km/u.s)).to('cm/s') # In cm/s - Section 2.1.3 Sharda+2024
    phint = 1 - (sigmath/sigmag_val)**2 # Eqn 12 Sharda+2024
    phint_init = phint # Track original phint

    # Calculating xia
    if xi_option == "constant": # Giving option to have constant xi_a for all z
        xia = xi_const
    else:
        xia = 0.2*(1+z) # Sharda+2024 section 2.1.3

    # sigmag_postphint = 0 * u.cm/u.s # Define sigmag post phintcheck
    if phint < 0.2: # In this case, find sigmag from solving six-order polynomial        
        # Calculating sigmaSF coefficient
        GMC_term = np.sqrt(3*f_gQ / (8*(1+beta))) * (Q_min*phi_mp / (4*f_gQ*eps_ff)) * (t_orb_prime/t_SFmax) # GMC term 
        sigmaSF_Toomre_coeff = (4*f_sf*eps_ff / (np.sqrt(3*f_gQ)*np.pi*eta*phi_mp*phi_Q)) * pmstar
        sigmaSF_GMC_coeff = (4*f_sf*eps_ff / (np.sqrt(3*f_gQ)*np.pi*eta*phi_mp*phi_Q)) * pmstar * GMC_term
        SF_coeff = max(sigmaSF_Toomre_coeff, sigmaSF_GMC_coeff) # Eqn 39 KBFC18

        # Calculating sigmaacc coefficient
        acc_coeff = (((2+beta)*xia*G_const*Mdot_gasacc_val / (8*(1+beta)*eta*phi_Q))) * (Q_min/f_gQ)**2 # In cm^3/s^3 

        # Calculating polynomial coefficients
        order4_coeff = - (3*sigmath**2 + SF_coeff**2)
        order2_coeff = 3*sigmath**4 - 2*SF_coeff*acc_coeff
        constant_coeff = - (acc_coeff**2 + sigmath**6)
        
        # Solving Fsigma = 0 using np.roots #
        coeff = [1, 0, order4_coeff.value, 0, order2_coeff.value, 0, constant_coeff.value]
        roots = np.roots(coeff)
        sigmag_val = np.real(roots[np.logical_and(np.imag(roots) == 0.0, np.real(roots) > 0.0)])[0] * u.cm/u.s
        # sigmag_postphint = sigmag_val

        # Recalculate Q, phiQ, phint
        phint = 1 - (sigmath/sigmag_val)**2 # Recalculating phi_nt with new sigma_g
        Q_val = 2*np.sqrt(2)*f_gQ*v_phi*R*sigmag_val / (np.sqrt(1+beta)*Mg*G_const)
        phi_Q = 1 + (Q_val/f_gQ)/Qstar

    # Calculating sigma_SF and when it is in Toomre and GMC - all Q are set to Qmin
    GMC_term = np.sqrt(3*f_gQ / (8*(1+beta))) * (Q_min*phi_mp / (4*f_gQ*eps_ff)) * (t_orb_prime/t_SFmax) # GMC term 
    sigmaSF_Toomre = (4*f_sf*eps_ff / (np.sqrt(3*f_gQ)*np.pi*eta*phi_mp*phi_Q * phint**1.5)) * pmstar
    sigmaSF_GMC = (4*f_sf*eps_ff / (np.sqrt(3*f_gQ)*np.pi*eta*phi_mp*phi_Q * phint**1.5)) * pmstar * GMC_term
    sigmaSF = max(sigmaSF_Toomre, sigmaSF_GMC) # Eqn 39 KBFC18

    # Calculating sigma_acc
    sigmaacc = (((2+beta)*xia*G_const*Mdot_gasacc_val / (8*(1+beta)*eta*phi_Q * phint**1.5)) * (Q_min/f_gQ)**2)**(1.0/3.0) # In cm/s

    # Calculating F(sigma_g)
    Fsigma = 1 - (sigmaSF/sigmag_val) - (sigmaacc/sigmag_val)**3

    # sigmag_postFsigma = 0 * u.cm/u.s # Define recalculated sigmag post Fsigma check
    # sigmag_postboth = 0 * u.cm/u.s
    if Fsigma < 0:
        GMC_term = np.sqrt(3*f_gQ / (8*(1+beta))) * (Q_min*phi_mp / (4*f_gQ*eps_ff)) * (t_orb_prime/t_SFmax) # GMC term 
        sigmaSF_Toomre_coeff = (4*f_sf*eps_ff / (np.sqrt(3*f_gQ)*np.pi*eta*phi_mp*phi_Q)) * pmstar
        sigmaSF_GMC_coeff = (4*f_sf*eps_ff / (np.sqrt(3*f_gQ)*np.pi*eta*phi_mp*phi_Q)) * pmstar * GMC_term
        SF_coeff = max(sigmaSF_Toomre_coeff, sigmaSF_GMC_coeff) # Eqn 39 KBFC18

        # Calculating sigmaacc coefficient
        acc_coeff = (((2+beta)*xia*G_const*Mdot_gasacc_val / (8*(1+beta)*eta*phi_Q))) * (Q_min/f_gQ)**2 # In cm^3/s^3 
        
        # Solving six-order polynomial
        order4_coeff = - (3*(sigmath**2) + SF_coeff**2)
        order2_coeff = 3*(sigmath**4) - 2*SF_coeff*acc_coeff
        constant_coeff = - (acc_coeff**2 + sigmath**6)
        
        # Solving Fsigma = 0 using np.roots #
        coeff = [1, 0, order4_coeff.value, 0, order2_coeff.value, 0, constant_coeff.value]
        roots = np.roots(coeff)
        sigmag_val = np.real(roots[np.logical_and(np.imag(roots) == 0.0, np.real(roots) > 0.0)])[0] * u.cm/u.s
        # sigmag_postFsigma = sigmag_val

        # if sigmag_postphint != 0: # If sigmag was recalculated following phint check
        #     sigmag_postboth = sigmag_val
        #     sigmag_postFsigma = 0

        # Set Fsigma = 0
        Fsigma = 0

    # Calculating Mdot_trans 
    Mdottrans_val = (4*eta*phi_Q * phint**1.5 * f_gQ**2 / (Q_min**2)) * (1+beta / (1-beta)) * ((sigmag_val**3) / G_const) * Fsigma

    ### Calculating outflows term ###
    etaw = find_etaw(Mh_val.to('Msun').value, z, etaw_option, etaw_const)

    Mdotout_val = etaw*Mdot_SF_val

    ### Calculating Mdotg ###
    Mdotg = Mdot_gasacc_val - Mdot_SF_val - Mdotout_val - Mdottrans_val # Eqn 7 (Sharda2024) - In g/s

    ### Calculating parameters for metallicity code ###
    # Generate dimensionless disc radii values
    r0 = (1 * u.kpc).to('cm') # Sharda+2024 Eqn. 18
    xmin = 1
    xmax = R/r0 # Disc normalised by r0 - Sharda+2024 Sec. 2.3

    # Calculating x_b # 
    x_b = (4*np.sqrt(2*(1+beta))*f_gQ*eps_ff*v_phi*t_SFmax/(np.pi*Q_val*np.sqrt(3*f_gQ*phi_mp)) * \
           ((r0/R)**beta)*(1/r0))**(1/(1-beta)) # Disc radius where Toomre = GMC

    # Calculating dimensionless ratios
    P = 6*eta*(phi_Q**2)*((phint)**(1.5))*(f_gQ**2)/(Q_min**2) * (1+beta / (1-beta)) * Fsigma # Sharda+2024 Eqn. 23
    Sprime_Toomre =  (24*phi_Q*(f_gQ**2)*eps_ff*f_sf / (np.pi*(Q_val**2)*np.sqrt(3*f_gQ*phi_mp))) * (1+beta) * (v_phi/sigmag_val * (r0/R)**beta)**2 # Sharda+2024 Eqn. 24 - without phiy*y/solarZ factor
    Sprime_GMC = (3*np.sqrt(2*(1+beta))*f_sf*f_gQ*phi_Q / (Q_val*t_SFmax)) * (r0**(beta+1) / (R**beta)) * (v_phi / (sigmag_val**2)) # Without phiy*y/solarZ factor
    A = 3*G_const*phi_Q*Mdot_gasacc_val / (2 * sigmag_val**3 * np.log(xmax)) 
    T = (3*np.sqrt(2*(1+beta))*phi_Q*f_gQ/Q_val) * (v_phi/sigmag_val * (r0/R)**beta)**2 # Sharda+2024 Eqn. 22
    
    # Dict does not return any recalculated phint, sigmaSF, sigmaacc after Fsigma check
    # Dict returns recalculated phiQ and Q after phint check

    # "sigmag_postboth": sigmag_postboth,  "sigmag_postphint": sigmag_postphint, "sigmag_postFsigma": sigmag_postFsigma, 
    return {"Mdotg": Mdotg, 'Mdotacc': Mdot_gasacc_val, 'MdotSF': Mdot_SF_val, "MdotSF_Toomre": Mdot_SF_Toomre, "Toomre_fac": Toomre_fac,
            "GMC_fac": GMC_fac, "MdotSF_GMC": Mdot_SF_GMC, 'Mdotout': Mdotout_val, 'Mdottrans': Mdottrans_val, 'etaw': etaw, 
            'Mstar': Mstar_val, 'Mh': Mh_val, "Mg": Mg, "phint_init": phint_init, "phint": phint, "phiQ": phi_Q, "Q": Q_val, 'Fsigma': Fsigma, 
            "sigmag_init": sigmag_init, 'sigmag': sigmag_val, 'sigmaSF': sigmaSF, "sigmaSF_Toomre": sigmaSF_Toomre, "sigmaSF_GMC": sigmaSF_GMC,
            'sigmaacc': sigmaacc, 'xia': xia, 'epsin': eps_in, 'Mdoth': Mdoth_val, 'omegadot': omega_dot, 'kappa': kappa, 'Omega': Omegarot, 
            'torb': t_orb, 'R': R, 'vphi': v_phi, 'c': c, 'beta': beta, 'fgQ': f_gQ, 'fsf': f_sf, "beta": beta, "sigmath": sigmath, "r0": r0, 
            "x_b": x_b, "xmin": xmin, "xmax": xmax, "P": P, "Sprime_Toomre": Sprime_Toomre, "Sprime_GMC": Sprime_GMC, "A": A, "T": T,
            'Mdotacc_option': Mdotacc_option
            }
    
def disc_code_outputs(z_init, z_final, logMhz_final, etaw_const, etaw_option, beta_const, beta_option, xi_const, xi_option, Mdotacc_option):
    """
    Function that integrates dMgdz using the BDF method from solve_ivp to get Mg and returns all relevant parameters/quantities
    to determine the gas mass history of a galaxy for given Mhz_init at z = z_init

    Parameters:
        Parameters:
        z_init: float
            Initial redshift
        z_infal: float 
            Final redshift
        logMhz_final: float
            log10 value of Mh at z_final
        etaw_const: float
            Mass-loading constant for etaw_option='const'
        etaw_option: 'constant', 'TNG', 'HH17', 'FIRE-2', 'EAGLE', "TNG (Ruby)", "SIMBA (Ruby)", "EAGLE (Ruby)" -
                      Specify what values of eta to use
        beta_const: float
            beta constant for beta_option='constant'
        beta_option: 'constant', 'varying' 
            Choose if beta is chosen constant value or varies with stellar mass
        xi_const: float
            xi value for xi_const='constant'
        xi_option: 'constant', 'varying'
            Choose if xi_a is chosen constant value or varies with redshift as 0.2(1+z)
        Mdotacc_option: "original", 'TNG', 'SIMBA', 'EAGLE'
            Choose which cosmological simulation to use for Mdotacc or to use original expression
    
    Returns:
        z: Array - Array of redshift
        Mg: Array - Evolution of gas mass for given Mh0 at all redshifts
            in units of g 
        Same outputs as disc_code_outputs
    """

    # Finding z_arr based on given z_init, z_final and finding Mhz_init given Mhz_final
    # z_arr, Mhz_init = z_arr_Mhz_init(z_init, z_final, logMhz_final) 
    z_arr = np.linspace(z_final, z_init) # halo_hist function takes redshift in ascending order
    Mhz0_halohist = halo_hist(10**logMhz_final, z_arr)[1]
    Mhz_init = (Mhz0_halohist[-1]).to('g').value # Convert from Msun to g

    # Mhz_init = (2.81e11 * u.Msun).to('g').value

    # Initial guess for Mg
    Mgz_init = [0.01 * Mhz_init] # Initial guess for Mg at z_init in units of g
                                 # Mg_init needs to be array-like

    # Integrating over new redshift array 
    # Mg_integrate = solve_ivp(dMgdz, t_span = [z_init, z_final], y0 = Mgz_init, t_eval = z_arr, method = "BDF", atol = 1e-4, rtol = 1e-4, 
    #                          args = (z_init, Mhz_init, etaw_const, etaw_option, beta_const, beta_option, xi_const, xi_option))
    Mg_integrate = solve_ivp(dMgdz, t_span = [z_init, z_final], y0 = Mgz_init, method = "BDF", atol = 1e-4, rtol = 1e-4, 
                             args = (z_init, Mhz_init, etaw_const, etaw_option, beta_const, beta_option, xi_const, xi_option, Mdotacc_option))

    if Mg_integrate.status != 0:
        print("Unsuccessful integration")

    # Calculate parameters for all redshift
    disc_outputs = [Mg_outputs(Mg_integrate.t[i], Mg_integrate.y[0][i], z_init, Mhz_init, etaw_const, etaw_option, beta_const, \
                               beta_option, xi_const, xi_option, Mdotacc_option) for i in range(0, len(Mg_integrate.t))]

    # Saving parameters at each redshift
    z_arr = Mg_integrate.t

    Mdotg = np.array([dict["Mdotg"].value for dict in disc_outputs]) * u.g/u.s
    Mdotacc = np.array([dict["Mdotacc"].value for dict in disc_outputs]) * u.g/u.s
    MdotSF = np.array([dict["MdotSF"].value for dict in disc_outputs]) * u.g/u.s
    MdotSF_Toomre = np.array([dict["MdotSF_Toomre"].value for dict in disc_outputs]) * u.g/u.s
    MdotSF_GMC = np.array([dict["MdotSF_GMC"].value for dict in disc_outputs]) * u.g/u.s
    Mdottrans = np.array([dict["Mdottrans"].value for dict in disc_outputs]) * u.g/u.s
    Mdotout = np.array([dict["Mdotout"].value for dict in disc_outputs]) * u.g/u.s

    # Calculating Mstar by integrating MdotSF_arr
    dMstardz = (u.Quantity(MdotSF) * dtUdz(z_arr)).to(u.Msun)
    Mstarint = (cumtrapz(u.Quantity(dMstardz).value, z_arr, initial = 0) * u.Msun).to('g') # 'initial' can only be 0 or None
    Mstar = np.array([dict["Mstar"].value for dict in disc_outputs]) * u.g
    Mh = np.array([dict["Mh"].value for dict in disc_outputs]) * u.g 
    Mg = np.array([dict["Mg"].value for dict in disc_outputs]) * u.g

    Mdoth = np.array([dict["Mdoth"].value for dict in disc_outputs]) * u.g/u.s
    epsin = np.array([dict["epsin"] for dict in disc_outputs]) 
    Toomre_fac = np.array([dict["Toomre_fac"] for dict in disc_outputs]) 
    GMC_fac = np.array([dict["GMC_fac"] for dict in disc_outputs]) 
    etaw = np.array([dict["etaw"] for dict in disc_outputs])

    c = np.array([dict["c"] for dict in disc_outputs])
    vphi = np.array([dict["vphi"].value for dict in disc_outputs]) * u.cm/u.s
    R = np.array([dict["R"].value for dict in disc_outputs]) * u.cm
    torb = np.array([dict["torb"].value for dict in disc_outputs]) * u.s
    Omega = np.array([dict["Omega"].value for dict in disc_outputs]) * 1/u.s

    sigmag = np.array([dict["sigmag"].value for dict in disc_outputs]) * u.cm/u.s
    sigmag_init = np.array([dict["sigmag_init"].value for dict in disc_outputs]) * u.cm/u.s
    # sigmag_postphint = np.array([dict["sigmag_postphint"].value for dict in disc_outputs]) * u.cm/u.s
    # sigmag_postFsigma = np.array([dict["sigmag_postFsigma"].value for dict in disc_outputs]) * u.cm/u.s
    # sigmag_postboth = np.array([dict["sigmag_postboth"].value for dict in disc_outputs]) * u.cm/u.s
    sigmaSF = np.array([dict["sigmaSF"].value for dict in disc_outputs]) * u.cm/u.s
    sigmaSF_Toomre = np.array([dict["sigmaSF_Toomre"].value for dict in disc_outputs]) * u.cm/u.s
    sigmaSF_GMC = np.array([dict["sigmaSF_GMC"].value for dict in disc_outputs]) * u.cm/u.s
    sigmaacc = np.array([dict["sigmaacc"].value for dict in disc_outputs]) * u.cm/u.s
    sigmath = np.array([dict["sigmath"].value for dict in disc_outputs]) * u.cm/u.s

    phint = np.array([dict["phint"] for dict in disc_outputs])
    phint_init = np.array([dict["phint_init"] for dict in disc_outputs])
    fsf = np.array([dict["fsf"] for dict in disc_outputs])
    fgQ = np.array([dict["fgQ"] for dict in disc_outputs])
    beta = np.array([dict["beta"] for dict in disc_outputs])
    Q = np.array([dict["Q"] for dict in disc_outputs])
    phiQ = np.array([dict["phiQ"] for dict in disc_outputs])
    Fsigma = np.array([dict["Fsigma"] for dict in disc_outputs])
    xia = np.array([dict["xia"] for dict in disc_outputs])

    r0 = np.array([dict["r0"].value for dict in disc_outputs]) * u.cm
    xmin = np.array([dict["xmin"] for dict in disc_outputs])
    xmax = np.array([dict["xmax"] for dict in disc_outputs])
    x_b = np.array([dict["x_b"] for dict in disc_outputs])
    P = np.array([dict["P"] for dict in disc_outputs])
    Sprime_Toomre = np.array([dict["Sprime_Toomre"] for dict in disc_outputs])
    Sprime_GMC = np.array([dict["Sprime_GMC"] for dict in disc_outputs])
    A = np.array([dict["A"] for dict in disc_outputs])
    T = np.array([dict["T"] for dict in disc_outputs])

    # Returns dict where each key returns values of given parameter over redshifts specified

    #  "sigmag_postphint": sigmag_postphint, "sigmag_postFsigma": sigmag_postFsigma, "sigmag_postboth": sigmag_postboth
    return {"z": z_arr, "Mstar": Mstar, "Mh": Mh, "Mg": Mg, "Mdotg": Mdotg, "Mdotacc": Mdotacc, "MdotSF": MdotSF, 
            "Toomre_fac": Toomre_fac, "GMC_fac": GMC_fac, "MdotSF_Toomre": MdotSF_Toomre, "MdotSF_GMC": MdotSF_GMC, 
            "Mdottrans": Mdottrans, "Mdotout": Mdotout, "c": c, "vphi": vphi, "R": R, "torb": torb, "Omega": Omega, 
            "sigmag": sigmag, "sigmag_init": sigmag_init, "sigmaSF": sigmaSF, "sigmaSF_Toomre": sigmaSF_Toomre, "sigmaSF_GMC": sigmaSF_GMC, 
            "sigmaacc": sigmaacc, "phint": phint, "phint_init": phint_init, "fsf": fsf, "fgQ": fgQ, "beta": beta, "Fsigma": Fsigma, 
            "xia": xia, "Q": Q, "phiQ": phiQ, "r0": r0, "x_b": x_b, "xmin": xmin, "xmax": xmax, "P": P, "Sprime_Toomre": Sprime_Toomre, 
            "Sprime_GMC": Sprime_GMC, "A": A, "T": T, "etaw": etaw, "Mdoth": Mdoth, "epsin": epsin, "sigmath": sigmath, 
            'Mdotacc_option': Mdotacc_option, 'etaw_option': etaw_option, 'Mstarint': Mstarint
            }

#-------------------------------------------------------------
#--------------------------
# METALLICITY CODE 
#-------------------------------------------------------------
#--------------------------

# Using outputs from galactic disc equilibrium model to calculate galaxy metallicity gradient using an equilibrium metallicity model 
# - Using cases for Zr0 given in Appendix C of Sharda+2024 and the corresponding expressions for c1 from Piyush's code
# - Follow logic from Piyush's code
#   - This ensures that the c1 I use corresponds to the correct Zr0
#   - Retain the outputs as arrays
# - Have separate S for Toomre and GMC regime - result in different sstar for GMC part
# - Accounting for discontinuity between Toomre and GMC regime:
#   - First solve GMC part. 
#   - Use solution of GMC part at x_b to constrain c1 values for Toomre regime
#   - Zr0 for Toomre part set the same as GMC part
# - Calculating metallicity gradient using np.polyfit
# - Fix phiy even if outflows is on

def normZ_func(z_init, z_final, z_given, logMhz_final, etaw_const, etaw_option, beta_const, beta_option, xi_const, xi_option, Mdotacc_option, phiy):
    """
    Calculates metallicity at given galactic radius and equilibration time to see if equilibrium model applies

    Parameters:
        Parameters:
        z: float
            Redshift
        Mg: float 
            Gas mass in g, at redshift z.
        z_init: float
            Initial redshift
        Mhz_init: float 
            Halo mass at z_init in units of g, but value only
        etaw_const: float
            Mass-loading constant for etaw_option='const'
        etaw_option: 'constant', 'TNG', 'HH17', 'FIRE-2', 'EAGLE', "TNG (Ruby)", "SIMBA (Ruby)", "EAGLE (Ruby)" -
            Specify what values of eta to use
        beta_const: float
            beta constant for beta_option='constant'
        beta_option: 'constant', 'varying' 
            Choose if beta is chosen constant value or varies with stellar mass
        xi_const: float
            xi value for xi_const='constant'
        xi_option: 'constant', 'varying'
            Choose if xi_a is chosen constant value or varies with redshift as 0.2(1+z)
        Mdotacc_option: "original", 'TNG', 'SIMBA', 'EAGLE'
            Choose which cosmological simulation to use for Mdotacc or to use original expression
        phiy: float 
            Yield reduction factor, i.e., how much of the metals are ejected by outflows before mixing with ISM, between 0 and 1

    Returns:
        Dictionary of metallicity code parameters
    """
    # Calculate parameters from galactic disc model
    disc_outputs = disc_code_outputs(z_init, z_final, logMhz_final, etaw_const, etaw_option, beta_const, beta_option, xi_const, xi_option, Mdotacc_option)

    # Extract important parameters and their value at given z
    z_arr = disc_outputs["z"]
    z_index = find_nearest(z_arr, z_given)

    # print(f'Calculated z={z_arr[z_index]} for given z={z_given}')

    y = 0.028 # Yield factor, i.e., how much ISM is enriched with metals by SNII - Sharda+2024 Eqn. 26 
    solarZ = 0.0134 # Solar metallicity
    P = disc_outputs["P"][z_index]
    A = disc_outputs["A"][z_index]
    T = disc_outputs["T"][z_index]
    Sprime_Toomre = disc_outputs["Sprime_Toomre"][z_index]
    S_Toomre = Sprime_Toomre * (phiy*y/solarZ)
    Sprime_GMC = disc_outputs["Sprime_GMC"][z_index]
    S_GMC = Sprime_GMC * (phiy*y/solarZ)
    
    beta_z = disc_outputs["beta"][z_index]
    Fsigma_z = disc_outputs["Fsigma"][z_index]

    vphi_z = disc_outputs["vphi"][z_index] # In cm/s
    R_z = disc_outputs["R"][z_index] # In cm
    r0 = disc_outputs["r0"][z_index] # In cm
    Omega0 = vphi_z/r0 * ((r0/R_z)**beta_z) # Angular frequency at r0 in s^-1
    x_b = disc_outputs["x_b"][z_index]
    xmin = disc_outputs["xmin"][z_index]
    xmax = disc_outputs["xmax"][z_index] 
    x_arr = np.linspace(xmin, xmax, 200) # Generate array of x values

    Mstar_z = disc_outputs["Mstar"][z_index].to('Msun') # Convert from g to Msun
    logMstar = np.log10(Mstar_z.value)

    # # Print Mh to find where code goes wrong
    # log10Mh = np.log10(disc_outputs["Mh"][-1].to('Msun').value) # Want Mh at z=0
    # print(f'Mh at z=0: {log10Mh}')
    
    # Establishing Zmin and ZCGM
    if logMstar <= 9: # For low-mass galaxies
        ZCGM = 0.05 # Sharda+2024 Sec. 2.2.3
    elif logMstar >= 10.5:
        ZCGM = 0.2 # Sharda+2024 Sec. 2.2-0.7
    else: # For intermediate-mass galaxies, interpolate value of ZCGM
        line_params = line_func(9, 0.05, 10.5, 0.2) # Getting slope and y-intercept of line
        ZCGM = line_params[0]*logMstar + line_params[1]

    Zmin = 0.01
    # Zmin = ZCGM

    # Defining common groups of terms
    Toomre_denom = A - 2*beta_z*(P + 2*beta_z)
    GMC_denom = A - (1+beta_z)*(1+P+beta_z)
    c1power = -0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower1 = 0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower2 = 0.5*(-np.sqrt(P**2 + 4*A) - P)  

    # Empty lists store maximum and minimum values of c1 and Zr0
    c1_min = []
    c1_max = []
    Zr0_min = []
    Zr0_max = []

    if xmin <= x_b and xmax <= x_b: # Entire galactic disc in Toomre regime
        regime = "Toomre"
        print("Entire galactic disc in Toomre regime")

        # Return S used to calculate Zr0
        S = S_Toomre

        # Calculating lower bound for c1
        c1_lowbound_Toomre = (Zmin - (S_Toomre*(xmax**(2*beta_z)) / Toomre_denom)) * xmax**c1power
        
        # Currently do not have a case for P = 0 - need to solve equation derived from boundary condition
        # if P == 0:
        #     print("No implemented P = 0 case for entire galactic disc in Toomre regime - no upper bound for c1")
        if logMstar > 10.5: # Calculating upper bound using c1 from Piyush's code and Zr0 from Sharda+2024
            print("Zr0 set by source and accretion - Massive galaxy")

            # Calculate c1
            c1_uppbound_Toomre = (A*S_Toomre*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                 (-2.*P - 4.*beta_z) - 4.*np.sqrt(4.*A + P**2)*S_Toomre*beta_z**2 + \
                                 A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                 (2.*A - 8.*beta_z**2 - 4.*beta_z*P) - \
                                 2.*np.sqrt(4.*A + P**2)*S_Toomre*beta_z*P + P*S_Toomre*(4.*beta_z**2 + 2.*beta_z*P))/\
                                 (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                                 (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*(A - 4.*beta_z**2 - 2.*beta_z*P))
            
            # Calculate Zr0
            Zr0_lowbound_Toomre = S_Toomre/A
            Zr0_uppbound_Toomre = S_Toomre/A

        else:
            if P == 0:
                print("No implemented P = 0 case for Zr0 set by diffusion and advection - Low-mass galaxy")
                # Use the expressions for low-mass case with P != 0 for now since I don't have P = 0 case
                # for Toomre regime

                # Calculate c1
                c1_uppbound_Toomre = (2.*A**2*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM - \
                                    4.*np.sqrt(4.*A + P**2)*S_Toomre*beta_z**2 + \
                                    P*S_Toomre*beta_z*(2.*np.sqrt(4.*A + P**2) - 4.*np.sqrt(4.*A + P**2)*\
                                    xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z) + 4.*beta_z)\
                                    + P**2*(S_Toomre*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                    (-2.*np.sqrt(4.*A + P**2) - 4.*beta_z) - 2.*S_Toomre*beta_z - \
                                    8.*np.sqrt(4.*A + P**2)*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**3*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-2.*S_Toomre*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P) + \
                                    A*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (2.*P**3*ZCGM + 2.*P**2*np.sqrt(4.*A + P**2)*ZCGM - \
                                    4.*S_Toomre*xmax**(2.*beta_z)*beta_z + \
                                    P*(-2.*S_Toomre*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P)))/\
                                    (A**2*(1.*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(-5. + 5.*xmax**(1.*np.sqrt(4.*A + P**2)))) + \
                                    A*(2.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(-2. + 2.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(20. - 20.*xmax**(1.*np.sqrt(4.*A + P**2)))*(1.*beta_z**2 + 0.5*beta_z*P) + \
                                    np.sqrt(4.*A + P**2)*(-4. - 4.*xmax**(1.*np.sqrt(4.*A + P**2)))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**2*(np.sqrt(4.*A + P**2)*\
                                    (-8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))) + P*(8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P))
                
                # Calculate Zr0
                Zr0_lowbound_Toomre = ((P**2)*S_Toomre + A*(2*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2) + S_Toomre) - \
                                    4*S_Toomre*(beta_z**2) + P*S_Toomre*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                Zr0_uppbound_Toomre = ((P**2)*S_Toomre + A*(2*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2) + S_Toomre) - \
                                    4*S_Toomre*(beta_z**2) + P*S_Toomre*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                
            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculate c1
                c1_uppbound_Toomre = (2.*A**2*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM - \
                                    4.*np.sqrt(4.*A + P**2)*S_Toomre*beta_z**2 + \
                                    P*S_Toomre*beta_z*(2.*np.sqrt(4.*A + P**2) - 4.*np.sqrt(4.*A + P**2)*\
                                    xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z) + 4.*beta_z)\
                                    + P**2*(S_Toomre*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                    (-2.*np.sqrt(4.*A + P**2) - 4.*beta_z) - 2.*S_Toomre*beta_z - \
                                    8.*np.sqrt(4.*A + P**2)*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**3*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-2.*S_Toomre*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P) + \
                                    A*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (2.*P**3*ZCGM + 2.*P**2*np.sqrt(4.*A + P**2)*ZCGM - \
                                    4.*S_Toomre*xmax**(2.*beta_z)*beta_z + \
                                    P*(-2.*S_Toomre*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P)))/\
                                    (A**2*(1.*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(-5. + 5.*xmax**(1.*np.sqrt(4.*A + P**2)))) + \
                                    A*(2.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(-2. + 2.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(20. - 20.*xmax**(1.*np.sqrt(4.*A + P**2)))*(1.*beta_z**2 + 0.5*beta_z*P) + \
                                    np.sqrt(4.*A + P**2)*(-4. - 4.*xmax**(1.*np.sqrt(4.*A + P**2)))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**2*(np.sqrt(4.*A + P**2)*\
                                    (-8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))) + P*(8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P))
                
                # Calculate Zr0
                Zr0_lowbound_Toomre = ((P**2)*S_Toomre + A*(2*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2) + S_Toomre) - \
                                    4*S_Toomre*(beta_z**2) + P*S_Toomre*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                Zr0_uppbound_Toomre = ((P**2)*S_Toomre + A*(2*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2) + S_Toomre) - \
                                    4*S_Toomre*(beta_z**2) + P*S_Toomre*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                    
        if c1_uppbound_Toomre < c1_lowbound_Toomre: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for Toomre - decrease Zmin")
        
        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_Toomre)
        c1_max.append(c1_uppbound_Toomre)
        Zr0_max.append(Zr0_uppbound_Toomre)
        Zr0_min.append(Zr0_lowbound_Toomre)

        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_min = (S_Toomre*(x_arr**(2*beta_z)) / Toomre_denom) + c1_lowbound_Toomre*(x_arr**solpower1) + \
                            (Zr0_lowbound_Toomre - (S_Toomre/Toomre_denom) - c1_lowbound_Toomre) * (x_arr**solpower2) # Sharda+2024 Eqn 29
        normZ_profile_max = (S_Toomre*(x_arr**(2*beta_z)) / Toomre_denom) + c1_uppbound_Toomre*(x_arr**solpower1) + \
                            (Zr0_uppbound_Toomre - (S_Toomre/Toomre_denom) - c1_uppbound_Toomre) * (x_arr**solpower2) # Sharda+2024 Eqn 29

        # Finding teqbm
        dnormZdx_min = (2*beta_z*S_Toomre*(x_arr**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_lowbound_Toomre*(x_arr**(solpower1 - 1)) + \
                        solpower2*(Zr0_lowbound_Toomre - S_Toomre/Toomre_denom - c1_lowbound_Toomre)*(x_arr**(solpower2 - 1))
        dnormZdx_max = (2*beta_z*S_Toomre*(x_arr**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_uppbound_Toomre*(x_arr**(solpower1 - 1)) + \
                        solpower2*(Zr0_uppbound_Toomre - S_Toomre/Toomre_denom - c1_uppbound_Toomre)*(x_arr**(solpower2 - 1))
        
        adv_term_min = np.abs(P/x_arr + dnormZdx_min)
        adv_term_max = np.abs(P/x_arr + dnormZdx_max)
        
        d2normZdx2_min = (2*beta_z*(2*beta_z - 1)*S_Toomre*(x_arr**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_lowbound_Toomre*(x_arr**(solpower1 - 2)) + \
                         solpower2*(solpower2 - 1)*(Zr0_lowbound_Toomre - S_Toomre/Toomre_denom - c1_lowbound_Toomre)*(x_arr**(solpower2 - 2))
        d2normZdx2_max = (2*beta_z*(2*beta_z - 1)*S_Toomre*(x_arr**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_uppbound_Toomre*(x_arr**(solpower1 - 2)) + \
                         solpower2*(solpower2 - 1)*(Zr0_uppbound_Toomre - S_Toomre/Toomre_denom - c1_uppbound_Toomre)*(x_arr**(solpower2 - 2))
        
        diffusion_term_min = np.abs(dnormZdx_min/x_arr + d2normZdx2_min)
        diffusion_term_max = np.abs(dnormZdx_max/x_arr + d2normZdx2_max)
       
        sstar = x_arr**(2*(beta_z - 1))
        source_term = S_Toomre * sstar

        cstar = 1 / (x_arr**2)
        acc_term_min = normZ_profile_min*A*cstar
        acc_term_max = normZ_profile_max*A*cstar
        
        sg = (x_arr**beta_z) / x_arr
        teqbm_denom_min = normZ_profile_min*T*sg / Omega0 
        teqbm_denom_max = normZ_profile_max*T*sg / Omega0 

        teqbm_min = ((adv_term_min + diffusion_term_min + source_term + acc_term_min)/teqbm_denom_min)**(-1)
        teqbm_max = ((adv_term_max + diffusion_term_max + source_term + acc_term_max)/teqbm_denom_max)**(-1)

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1)

    elif xmin > x_b and xmax > x_b: # Entire galactic disc in GMC regime
        regime = "GMC"
        print("Entire galactic disc in GMC regime")
        
        # Return S used to calculate Zr0
        S = S_GMC

        # Calculating lower of c1
        c1_lowbound_GMC = (Zmin - (S_GMC*(xmax**(1+beta_z)) / GMC_denom)) * xmax**c1power

        # Using c1 upper bound values from Piyush's code
        # Mstar check first
        if logMstar > 10.5:
            print("Zr0 set by source and accretion - Massive galaxy")
            
            # Calculating c1
            c1_uppbound_GMC = (-1.*np.sqrt(4.*A + P**2)*S_GMC - 1.*P*np.sqrt(4.*A + P**2)*S_GMC + \
                            A*S_GMC*xmax**(1. + 0.5*P + 0.5*np.sqrt(4.*A + P**2) + beta_z)*\
                            (-2. - 2.*P - 2.*beta_z) - 2.*np.sqrt(4.*A + P**2)*S_GMC*beta_z - \
                            1.*P*np.sqrt(4.*A + P**2)*S_GMC*beta_z - 1.*np.sqrt(4.*A + P**2)*S_GMC*beta_z**2 + \
                            P*S_GMC*(1. + 1.*beta_z)*(1. + P + 1.*beta_z) + \
                            A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                            (2.*A + P*(-2. - 2.*beta_z) - 2.0000000000000004*(1. + 1.*beta_z)**2))/\
                            (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                            (-1. + 1.*A - 1.*P - 2.*beta_z - 1.*P*beta_z - 1.*beta_z**2))
            
            # Accounting for nan values for c1 max
            if np.isnan(c1_uppbound_GMC):
                g = np.sqrt(4.0*A + P*P)
                logx = np.log(xmax)

                # Choose sign so that the exponent in exp() is always <= 0 (prevents overflow)
                s = 1.0 if logx >= 0.0 else -1.0

                # Common, always-safe factors
                invT = np.exp(-s*g*logx)  # = xmax**(-s*g) ∈ (0, 1] in floating-point; may underflow to 0, which is fine

                # These are xmax**(E - s*g) evaluated in log space
                # E1 = 1 + beta_z + 0.5*(P + g)
                # E2 = 0.5*(P + g)
                x_pow_E1_minus_sg = np.exp((1.0 + beta_z + 0.5*(P + g) - s*g) * logx)
                x_pow_E2_minus_sg = np.exp((0.5*(P + g) - s*g) * logx)

                # Scaled numerator (already multiplied by xmax**(-s*g))
                const_part = (
                    -g*S_GMC
                    - P*g*S_GMC
                    - 2.0*g*S_GMC*beta_z
                    - P*g*S_GMC*beta_z
                    - g*S_GMC*beta_z**2
                    + P*S_GMC*(1.0 + beta_z)*(1.0 + P + beta_z)
                )

                num_scaled = (
                    invT*const_part
                    + A*S_GMC*x_pow_E1_minus_sg*(-2.0 - 2.0*P - 2.0*beta_z)
                    + A*P*x_pow_E2_minus_sg*ZCGM*(2.0*A + P*(-2.0 - 2.0*beta_z) - 2.0*(1.0 + beta_z)**2)
                )

                # Scaled denominator (also multiplied by xmax**(-s*g); the x**(±g) terms collapse to 1 ± invT)
                den_scaled = (
                    A*(P*(-invT + 1.0) + g*(invT + 1.0))
                    * (-1.0 + A - 1.0*P - 2.0*beta_z - 1.0*P*beta_z - 1.0*beta_z**2)
                )

                c1_uppbound_GMC = num_scaled/den_scaled

            # Calculating Zr0
            Zr0_lowbound_GMC = S_GMC/A
            Zr0_uppbound_GMC = S_GMC/A

        else:
            if Fsigma_z == 0:
                print("Zr0 set by diffusion and source - No transport")
                # Calculating c1

                c1_uppbound_GMC = ((2.*A + P*(1.*P + 1.*np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                (-1.*P*ZCGM + (0.5*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (4.*A + P**2 + P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - \
                                4.*(1. + beta_z)**2))/((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + (0.25*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*(4.*A + P**2 + \
                                P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - 4.*(1. + beta_z)**2))/\
                                ((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) - \
                                (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*P*S_GMC*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*S_GMC*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z))))/\
                                (A*P*(3. - 3.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*A*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))

                # Calculate Zr0
                Zr0_lowbound_GMC = (0.5 * (S_GMC*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                Zr0_uppbound_GMC = (0.5 * (S_GMC*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     
            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculating c1                    
                c1_uppbound_GMC = ((A + P*(P + np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-1.*P*ZCGM - (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                    (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                    xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S_GMC*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*S_GMC*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + \
                                    (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2))))/\
                                    (A*P*(2.5 - 2.5*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 0.5*A*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 1.*P**2*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
                
                # Include code that accounts for overflow error in xmax**(..P..) for really high P
                if np.isnan(c1_uppbound_GMC):
                    g = np.sqrt(4.0*A + P**2) # = sqrt(4A + P^2)
                    logx = np.log(xmax)

                    den1 = A - (1.0 + beta_z)*(1.0 + P + beta_z)
                    den2 = (A + P*(P + g)) * (A + P*(-1.0 - beta_z) - (1.0 + beta_z)**2)

                    # Stable powers (only small or negative exponents)
                    invT = np.exp(-g*logx)                    # = xmax**(-g) ~ 0 (safe)
                    x_pow_E_minus_g = np.exp((0.5*(P + g) - g)*logx)  # = xmax**(0.5*P - 0.5*g) ~ O(1)
                    x_pow_negE = np.exp(-0.5*(P + g)*logx)    # = xmax**(-0.5P - 0.5g) ~ 0 (safe)
                    x_pow_1p_beta = np.exp((1.0 + beta_z)*logx)       # = xmax**(1+beta) ~ moderate

                    # Bracket term (no huge positive powers remain)
                    B = (
                        -P*ZCGM
                        - (P*S_GMC*x_pow_negE)/den1
                        - (0.5*(-P - g)*S_GMC*x_pow_negE)/den1
                        + (P*S_GMC*x_pow_1p_beta)/den1
                        + (S_GMC*x_pow_1p_beta*(1.0 + beta_z))/den1
                        + (P*S_GMC*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                        + (0.5*(-P - g)*S_GMC*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                    )

                    # Numerator and denominator divided by xmax**g (so no overflow)
                    num_scaled = (A + P*(P + g)) * x_pow_E_minus_g * B # Numerator after beign multipled across by xmax**g
                    den_scaled = ( # Numerator after being multiplied by xmax**(-g)
                        A*P*(2.5*invT - 2.5)
                        + P**3*(invT - 1.0)
                        - 0.5*A*g*(invT + 1.0)
                        - P**2*g*(invT + 1.0)
                    ) 
                    
                    c1_uppbound_GMC = num_scaled/den_scaled

                 # Calculating Zr0
                Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S_GMC*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S_GMC*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
        
        if c1_uppbound_GMC < c1_lowbound_GMC: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for GMC - decrease Zmin")

        # print(f'c1 max: {c1_uppbound_GMC} for logMhz0 = {disc_outputs['Mh'][-1]} and P = {P} at z = {z_given}')

        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_GMC)
        c1_max.append(c1_uppbound_GMC)
        Zr0_max.append(Zr0_uppbound_GMC)
        Zr0_min.append(Zr0_lowbound_GMC)

        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_min = (S_GMC*(x_arr**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_arr**solpower1) + \
                            (Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*(x_arr**solpower2) # Sharda+2024 Eqn 30
        normZ_profile_max = (S_GMC*(x_arr**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_arr**solpower1) + \
                            (Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*(x_arr**solpower2) # Sharda+2024 Eqn 30

        # Finding teqbm
        dnormZdx_min = ((S_GMC*(1+beta_z)*x_arr**(beta_z)) / GMC_denom) + solpower1*c1_lowbound_GMC*x_arr**(solpower1 - 1) + \
                       solpower2*(Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*x_arr**(solpower2 - 1)
        dnormZdx_max = ((S_GMC*(1+beta_z)*x_arr**(beta_z)) / GMC_denom) + solpower1*c1_uppbound_GMC*x_arr**(solpower1 - 1) + \
                       solpower2*(Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*x_arr**(solpower2 - 1)
        
        adv_term_min = np.abs((P/x_arr) * dnormZdx_min)
        adv_term_max = np.abs((P/x_arr) * dnormZdx_max) 
        
        d2normZdx2_min = ((S_GMC*beta_z*(1+beta_z)*x_arr**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_lowbound_GMC*x_arr**(solpower1 - 2) + \
                         solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*x_arr**(solpower2 - 2)
        d2normZdx2_max = ((S_GMC*beta_z*(1+beta_z)*x_arr**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_uppbound_GMC*x_arr**(solpower1 - 2) + \
                         solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*x_arr**(solpower2 - 2)
        
        diffusion_term_min = np.abs((dnormZdx_min/x_arr + d2normZdx2_min))
        diffusion_term_max = np.abs((dnormZdx_max/x_arr + d2normZdx2_max))
        
        sstar = x_arr**(beta_z-1)
        source_term = S_GMC * sstar

        cstar = 1 / (x_arr**2)
        acc_term_min = normZ_profile_min*A*cstar
        acc_term_max = normZ_profile_max*A*cstar
        
        sg = (x_arr**beta_z) / x_arr
        teqbm_denom_min = normZ_profile_min*T*sg / Omega0 
        teqbm_denom_max = normZ_profile_max*T*sg / Omega0 

        teqbm_min = ((adv_term_min + diffusion_term_min + source_term + acc_term_min)/teqbm_denom_min)**(-1)
        teqbm_max = ((adv_term_max + diffusion_term_max + source_term + acc_term_max)/teqbm_denom_max)**(-1)

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1)

    elif xmin <= x_b and xmax > x_b: # Inner disc in Toomre and outer disc in GMC
        regime = "Toomre and GMC" 
        print("Inner disc is Toomre, outer disc is GMC")

        # Calculating GMC part first which is the outer part of the disc
        # Calculating lower bound of c1
        c1_lowbound_GMC = (Zmin - (S_GMC*(xmax**(1+beta_z)) / GMC_denom)) * xmax**c1power
        
        # Using c1 upper bound values from Piyush's code
        # Mstar check first
        if logMstar > 10.5:
            print("Zr0 set by source and accretion - Massive galaxy")

            # Calculating c1
            c1_uppbound_GMC = (-1.*np.sqrt(4.*A + P**2)*S_GMC - 1.*P*np.sqrt(4.*A + P**2)*S_GMC + \
                            A*S_GMC*xmax**(1. + 0.5*P + 0.5*np.sqrt(4.*A + P**2) + beta_z)*\
                            (-2. - 2.*P - 2.*beta_z) - 2.*np.sqrt(4.*A + P**2)*S_GMC*beta_z - \
                            1.*P*np.sqrt(4.*A + P**2)*S_GMC*beta_z - 1.*np.sqrt(4.*A + P**2)*S_GMC*beta_z**2 + \
                            P*S_GMC*(1. + 1.*beta_z)*(1. + P + 1.*beta_z) + \
                            A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                            (2.*A + P*(-2. - 2.*beta_z) - 2.0000000000000004*(1. + 1.*beta_z)**2))/\
                            (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                            (-1. + 1.*A - 1.*P - 2.*beta_z - 1.*P*beta_z - 1.*beta_z**2))

            if np.isnan(c1_uppbound_GMC):
                g = np.sqrt(4.0*A + P*P)
                logx = np.log(xmax)

                # Choose sign so that the exponent in exp() is always <= 0 (prevents overflow)
                s = 1.0 if logx >= 0.0 else -1.0

                # Common, always-safe factors
                invT = np.exp(-s*g*logx)  # = xmax**(-s*g) ∈ (0, 1] in floating-point; may underflow to 0, which is fine

                # These are xmax**(E - s*g) evaluated in log space
                # E1 = 1 + beta_z + 0.5*(P + g)
                # E2 = 0.5*(P + g)
                x_pow_E1_minus_sg = np.exp((1.0 + beta_z + 0.5*(P + g) - s*g) * logx)
                x_pow_E2_minus_sg = np.exp((0.5*(P + g) - s*g) * logx)

                # Scaled numerator (already multiplied by xmax**(-s*g))
                const_part = (
                    -g*S_GMC
                    - P*g*S_GMC
                    - 2.0*g*S_GMC*beta_z
                    - P*g*S_GMC*beta_z
                    - g*S_GMC*beta_z**2
                    + P*S_GMC*(1.0 + beta_z)*(1.0 + P + beta_z)
                )

                num_scaled = (
                    invT*const_part
                    + A*S_GMC*x_pow_E1_minus_sg*(-2.0 - 2.0*P - 2.0*beta_z)
                    + A*P*x_pow_E2_minus_sg*ZCGM*(2.0*A + P*(-2.0 - 2.0*beta_z) - 2.0*(1.0 + beta_z)**2)
                )

                # Scaled denominator (also multiplied by xmax**(-s*g); the x**(±g) terms collapse to 1 ± invT)
                den_scaled = (
                    A*(P*(-invT + 1.0) + g*(invT + 1.0))
                    * (-1.0 + A - 1.0*P - 2.0*beta_z - 1.0*P*beta_z - 1.0*beta_z**2)
                )

                c1_uppbound_GMC = num_scaled/den_scaled
            
            # Calculating Zr0
            # # Using S_GMC
            # Zr0_lowbound_GMC = S_GMC/A
            # Zr0_uppbound_GMC = S_GMC/A
            # Using S_Toomre
            Zr0_lowbound_GMC = S_Toomre/A # Using S_Toomre gives reasonable Zr0 for massive galaxies
                                          # Using S_GMC returns low Zr0 and high metallicity gradients
            Zr0_uppbound_GMC = S_Toomre/A

            # Return S used to calculate Zr0
            S = S_Toomre

        else:
            if Fsigma_z == 0:
                print("Zr0 set by diffusion and source - No transport")
                # Calculating c1
                c1_uppbound_GMC = ((2.*A + P*(1.*P + 1.*np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                (-1.*P*ZCGM + (0.5*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (4.*A + P**2 + P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - \
                                4.*(1. + beta_z)**2))/((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + (0.25*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*(4.*A + P**2 + \
                                P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - 4.*(1. + beta_z)**2))/\
                                ((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) - \
                                (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*P*S_GMC*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*S_GMC*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z))))/\
                                (A*P*(3. - 3.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*A*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
                
                # Calculate Zr0
                # Using S_GMC - S_GMC is used because when calculating MZGR I get negative Zr0 values when using S_Toomre
                Zr0_lowbound_GMC = (0.5 * (S_GMC*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                   4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                   ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                Zr0_uppbound_GMC = (0.5 * (S_GMC*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                   4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                   ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     
                # # Using S_Toomre
                # Zr0_lowbound_GMC = (0.5 * (S_Toomre*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                #                     4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                #                     ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                # Zr0_uppbound_GMC = (0.5 * (S_Toomre*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                #                     4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                #                     ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     

                # Return S used to calculate Zr0
                S = S_GMC

            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculating c1
                c1_uppbound_GMC = ((A + P*(P + np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-1.*P*ZCGM - (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                    (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                    xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S_GMC*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*S_GMC*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + \
                                    (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2))))/\
                                    (A*P*(2.5 - 2.5*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 0.5*A*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 1.*P**2*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))

                # Account for getting nan values for c1 max
                if np.isnan(c1_uppbound_GMC):
                    g = np.sqrt(4.0*A + P**2) # = sqrt(4A + P^2)
                    logx = np.log(xmax)

                    den1 = A - (1.0 + beta_z)*(1.0 + P + beta_z)
                    den2 = (A + P*(P + g)) * (A + P*(-1.0 - beta_z) - (1.0 + beta_z)**2)

                    # Stable powers (only small or negative exponents)
                    invT = np.exp(-g*logx)                    # = xmax**(-g) ~ 0 (safe)
                    x_pow_E_minus_g = np.exp((0.5*(P + g) - g)*logx)  # = xmax**(0.5*P - 0.5*g) ~ O(1)
                    x_pow_negE = np.exp(-0.5*(P + g)*logx)    # = xmax**(-0.5P - 0.5g) ~ 0 (safe)
                    x_pow_1p_beta = np.exp((1.0 + beta_z)*logx)       # = xmax**(1+beta) ~ moderate

                    # Bracket term (no huge positive powers remain)
                    B = (
                        -P*ZCGM
                        - (P*S_GMC*x_pow_negE)/den1
                        - (0.5*(-P - g)*S_GMC*x_pow_negE)/den1
                        + (P*S_GMC*x_pow_1p_beta)/den1
                        + (S_GMC*x_pow_1p_beta*(1.0 + beta_z))/den1
                        + (P*S_GMC*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                        + (0.5*(-P - g)*S_GMC*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                    )

                    # Numerator and denominator divided by xmax**g (so no overflow)
                    num_scaled = (A + P*(P + g)) * x_pow_E_minus_g * B # Numerator after beign multipled across by xmax**g
                    den_scaled = ( # Numerator after being multiplied by xmax**(-g)
                        A*P*(2.5*invT - 2.5)
                        + P**3*(invT - 1.0)
                        - 0.5*A*g*(invT + 1.0)
                        - P**2*g*(invT + 1.0)
                    ) 
                    
                    c1_uppbound_GMC = num_scaled/den_scaled

                # Calculating Zr0
                # Using S_GMC - S_GMC is used because when calculating MZGR I get negative Zr0 values when using S_Toomre
                Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S_GMC*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S_GMC*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                
                # # Using S_Toomre
                # Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                #                     S_Toomre*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                #                     ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                # Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                #                     S_Toomre*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                #                     ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C

                # Return S used to calculate Zr0
                S = S_GMC

        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_GMC)
        c1_max.append(c1_uppbound_GMC)
        Zr0_max.append(Zr0_uppbound_GMC)
        Zr0_min.append(Zr0_lowbound_GMC)

        # Finding metallicity profiles corresponding to minimum and maximum of c1_GMC for x_b < x < xmax
        normZ_profile_GMC_min = (S_GMC*(x_arr[x_arr > x_b]**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_arr[x_arr > x_b]**solpower1) + \
                                (Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*(x_arr[x_arr > x_b]**solpower2) # Sharda+2024 Eqn 30
        normZ_profile_GMC_max = (S_GMC*(x_arr[x_arr > x_b]**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_arr[x_arr > x_b]**solpower1) + \
                                (Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*(x_arr[x_arr > x_b]**solpower2) # Sharda+2024 Eqn 30

        # Finding teqbm
        dnormZdx_GMC_min = ((S_GMC*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z)) / GMC_denom) + solpower1*c1_lowbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 1) + \
                           solpower2*(Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 1)
        dnormZdx_GMC_max = ((S_GMC*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z)) / GMC_denom) + solpower1*c1_uppbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 1) + \
                           solpower2*(Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 1)
        
        adv_term_GMC_min = np.abs((P/x_arr[x_arr > x_b]) * dnormZdx_GMC_min)
        adv_term_GMC_max = np.abs((P/x_arr[x_arr > x_b]) * dnormZdx_GMC_max) 
        
        d2normZdx2_GMC_min = ((S_GMC*beta_z*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_lowbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 2) + \
                             solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 2)
        d2normZdx2_GMC_max = ((S_GMC*beta_z*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_uppbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 2) + \
                             solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 2)
        
        diffusion_term_GMC_min = np.abs((dnormZdx_GMC_min/x_arr[x_arr > x_b] + d2normZdx2_GMC_min))
        diffusion_term_GMC_max = np.abs((dnormZdx_GMC_max/x_arr[x_arr > x_b] + d2normZdx2_GMC_max))
        
        sstar = x_arr[x_arr > x_b]**(beta_z-1)
        source_term_GMC = S_GMC * sstar

        cstar = 1 / x_arr[x_arr > x_b]**2
        acc_term_GMC_min = normZ_profile_GMC_min*A*cstar
        acc_term_GMC_max = normZ_profile_GMC_max*A*cstar
        
        sg = (x_arr[x_arr > x_b]**beta_z) / x_arr[x_arr > x_b]
        teqbm_denom_GMC_min = normZ_profile_GMC_min*T*sg 
        teqbm_denom_GMC_max = normZ_profile_GMC_max*T*sg 

        teqbm_GMC_min = (Omega0*(adv_term_GMC_min + diffusion_term_GMC_min + source_term_GMC + acc_term_GMC_min)/teqbm_denom_GMC_min)**(-1)
        teqbm_GMC_max = (Omega0*(adv_term_GMC_max + diffusion_term_GMC_max + source_term_GMC + acc_term_GMC_max)/teqbm_denom_GMC_max)**(-1)

        # Finding value of normZ at x = x_b from GMC part
        normZ_GMC_xb_min = (S_GMC*(x_b**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_b**solpower1) + \
                           (Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*(x_b**solpower2) # normZ at x = x_b for c1_GMC_min   
        normZ_GMC_xb_max = (S_GMC*(x_b**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_b**solpower1) + \
                           (Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*(x_b**solpower2) # normZ at x = x_b

        # Set Zr0_Toomre to be the same as Zr0_GMC
        Zr0_lowbound_Toomre = Zr0_lowbound_GMC
        Zr0_uppbound_Toomre = Zr0_uppbound_GMC  

        # # At x_b, normZ from Toomre and GMC part must be the same - use this to calculate c1 for Toomre part
        c1_Toomre_fromGMC_min = (normZ_GMC_xb_min - (S_Toomre*(x_b**(2*beta_z)) / Toomre_denom) - (Zr0_lowbound_Toomre - S_Toomre/Toomre_denom)*(x_b**solpower2)) \
                                / (x_b**solpower1 - x_b**solpower2)
        c1_Toomre_fromGMC_max = (normZ_GMC_xb_max - (S_Toomre*(x_b**(2*beta_z)) / Toomre_denom) - (Zr0_uppbound_Toomre - S_Toomre/Toomre_denom)*(x_b**solpower2)) \
                                / (x_b**solpower1 - x_b**solpower2)

        # Ensure that normZ > normZmin at x_b
        c1_lowbound_xb_Toomre = (Zmin - (S_Toomre*(x_b**(2*beta_z)) / Toomre_denom)) * x_b**c1power # Lower bound for c1 in Toomre part at x_b
        if c1_lowbound_xb_Toomre > c1_Toomre_fromGMC_max or  c1_lowbound_xb_Toomre > c1_Toomre_fromGMC_min:
            warnings.warn("New c1 from Toomre part is invalid", UserWarning)
            print(f'Toomre c1 min at xb: {c1_lowbound_xb_Toomre}')
            print(f'Toomre c1 min from GMC: {c1_Toomre_fromGMC_min}')
            print(f'Toomre c1 max from GMC: {c1_Toomre_fromGMC_max}')
            print(f'Error occured for z={z_arr[z_index]}, logMstar={logMstar}, logMh={np.log10(disc_outputs['Mh'][z_index].to('Msun').value)}')
            
        c1_min.append(c1_Toomre_fromGMC_min)
        c1_max.append(c1_Toomre_fromGMC_max)
        Zr0_min.append(Zr0_lowbound_Toomre)
        Zr0_max.append(Zr0_uppbound_Toomre)

        # Solve normZ in Toomre part
        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_Toomre_min = (S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z)) / Toomre_denom) + c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**solpower1) + \
                                   (Zr0_lowbound_Toomre - (S_Toomre/Toomre_denom) - c1_Toomre_fromGMC_min) * (x_arr[x_arr <= x_b]**solpower2) # Sharda+2024 Eqn 29
        normZ_profile_Toomre_max = (S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z)) / Toomre_denom) + c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**solpower1) + \
                                   (Zr0_uppbound_Toomre - (S_Toomre/Toomre_denom) - c1_Toomre_fromGMC_max) * (x_arr[x_arr <= x_b]**solpower2) # Sharda+2024 Eqn 29

        # Finding teqbm
        dnormZdx_Toomre_min = (2*beta_z*S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**(solpower1 - 1)) + \
                              solpower2*(Zr0_lowbound_Toomre - S_Toomre/Toomre_denom - c1_Toomre_fromGMC_min)*(x_arr[x_arr <= x_b]**(solpower2 - 1))
        dnormZdx_Toomre_max = (2*beta_z*S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**(solpower1 - 1)) + \
                              solpower2*(Zr0_uppbound_Toomre - S_Toomre/Toomre_denom - c1_Toomre_fromGMC_max)*(x_arr[x_arr <= x_b]**(solpower2 - 1))
        
        adv_term_Toomre_min = np.abs(P/x_arr[x_arr <= x_b] + dnormZdx_Toomre_min)
        adv_term_Toomre_max = np.abs(P/x_arr[x_arr <= x_b] + dnormZdx_Toomre_max)
        

        d2normZdx2_Toomre_min = (2*beta_z*(2*beta_z - 1)*S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**(solpower1 - 2)) + \
                                solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S_Toomre/Toomre_denom - c1_Toomre_fromGMC_min)*(x_arr[x_arr <= x_b]**(solpower2 - 2))
        d2normZdx2_Toomre_max = (2*beta_z*(2*beta_z - 1)*S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**(solpower1 - 2)) + \
                                solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S_Toomre/Toomre_denom - c1_Toomre_fromGMC_max)*(x_arr[x_arr <= x_b]**(solpower2 - 2))
        
        diffusion_term_Toomre_min = np.abs(dnormZdx_Toomre_min/x_arr[x_arr <= x_b] + d2normZdx2_Toomre_min)
        diffusion_term_Toomre_max = np.abs(dnormZdx_Toomre_max/x_arr[x_arr <= x_b] + d2normZdx2_Toomre_max)


        sstar = x_arr[x_arr <= x_b]**(2*(beta_z - 1))
        source_term_Toomre = S_Toomre * sstar

        cstar = 1 / (x_arr[x_arr <= x_b]**2)
        acc_term_Toomre_min = normZ_profile_Toomre_min*A*cstar
        acc_term_Toomre_max = normZ_profile_Toomre_max*A*cstar
        
        sg = (x_arr[x_arr <= x_b]**beta_z) / x_arr[x_arr <= x_b]
        teqbm_denom_Toomre_min = normZ_profile_Toomre_min*T*sg 
        teqbm_denom_Toomre_max = normZ_profile_Toomre_max*T*sg

        teqbm_Toomre_min = (Omega0*(adv_term_Toomre_min + diffusion_term_Toomre_min + source_term_Toomre + acc_term_Toomre_min)/teqbm_denom_Toomre_min)**(-1)
        teqbm_Toomre_max = (Omega0*(adv_term_Toomre_max + diffusion_term_Toomre_max + source_term_Toomre + acc_term_Toomre_max)/teqbm_denom_Toomre_max)**(-1)
        
        # Combine parts from Toomre and GMC denom
        adv_term_min = np.array(list(adv_term_Toomre_min) + list(adv_term_GMC_min))
        adv_term_max = np.array(list(adv_term_Toomre_max) + list(adv_term_GMC_max))

        diffusion_term_min = np.array(list(diffusion_term_Toomre_min) + list(diffusion_term_GMC_min))
        diffusion_term_max = np.array(list(diffusion_term_Toomre_max) + list(diffusion_term_GMC_max))

        source_term = np.array(list(source_term_Toomre) + list(source_term_GMC))

        acc_term_min = np.array(list(acc_term_Toomre_min) + list(acc_term_GMC_min))
        acc_term_max = np.array(list(acc_term_Toomre_max) + list(acc_term_GMC_max))

        teqbm_min = np.array(list(teqbm_Toomre_min.value) + list(teqbm_GMC_min.value)) * u.s
        teqbm_max = np.array(list(teqbm_Toomre_max.value) + list(teqbm_GMC_max.value)) * u.s

        normZ_profile_min = np.array(list(normZ_profile_Toomre_min) + list(normZ_profile_GMC_min))
        normZ_profile_max = np.array(list(normZ_profile_Toomre_max) + list(normZ_profile_GMC_max))

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting

    return {"regime": regime, "x": x_arr, "x_b": x_b, "metgrad_min": metgrad_min, "metgrad_max": metgrad_max, "log10Zr0_min": log10Zr0_min,
            "log10Zr0_max":log10Zr0_max, "normZ_profile_min": normZ_profile_min, "normZ_profile_max": normZ_profile_max, "teqbm_min": teqbm_min, 
            "teqbm_max": teqbm_max, "adv_term_min": adv_term_min,"adv_term_max": adv_term_max, "diffusion_term_min": diffusion_term_min, 
            "diffusion_term_max": diffusion_term_max, "source_term": source_term, "acc_term_min": acc_term_min, "acc_term_max": acc_term_max, 
            "c1_min": c1_min, "c1_max": c1_max, "Zr0_min": Zr0_min, "Zr0_max": Zr0_max, "T":T, "P": P, "A": A, 'S':S, "S_Toomre": S_Toomre, "S_GMC": S_GMC, 
            "logMstar": logMstar, "beta": beta_z, "ZCGM": ZCGM, "Zmin": Zmin, "vphi": vphi_z, "R": R_z, "Omega0": Omega0, "disc_outputs": disc_outputs
            }

def normZ_func_shortcut(disk_code_outputs, z_given, phiy):
    """
    Calculates metallicity gradients for galaxy with parameters obtained from disk code output at given redshift

    Parameters:
        disk_code_outputs: dictionary
            dictionary output from disk_code_outputs
        z_given: float
            given redshift
        phiy: float 
            Yield reduction factor, i.e., how much of the metals are ejected by outflows before mixing with ISM, between 0 and 1

    Returns:
        Dictionary of metallicity code parameters
    """
    # Calculate parameters from galactic disc model
    disc_outputs = disk_code_outputs
    
    # Check log10Mh0, Mstar, Mdotacc, etaw
    logMhz_final = np.log10(disc_outputs['Mh'][-1].to('Msun').value)
    logMstarz_final = np.log10(disc_outputs['Mstar'][-1].to('Msun').value)
    z_init = disc_outputs['z'][0]
    z_final = disc_outputs['z'][-1]
    Mdotacc = disc_outputs['Mdotacc_option']
    etaw = disc_outputs['etaw_option']
    print(f'log10Mhz_final = {logMhz_final} ({logMstarz_final}) integrated from {z_init} to {z_final} with Mdotacc = {Mdotacc} and etaw = {etaw}')

    # Extract important parameters and their value at given z
    z_arr = disc_outputs["z"]
    z_index = find_nearest(z_arr, z_given)

    # print(f'Calculated z={z_arr[z_index]} for given z={z_given}')

    y = 0.028 # Yield factor, i.e., how much ISM is enriched with metals by SNII - Sharda+2024 Eqn. 26 
    solarZ = 0.0134 # Solar metallicity
    P = disc_outputs["P"][z_index]
    A = disc_outputs["A"][z_index]
    T = disc_outputs["T"][z_index]
    Sprime_Toomre = disc_outputs["Sprime_Toomre"][z_index]
    S_Toomre = Sprime_Toomre * (phiy*y/solarZ)
    Sprime_GMC = disc_outputs["Sprime_GMC"][z_index]
    S_GMC = Sprime_GMC * (phiy*y/solarZ)
    
    beta_z = disc_outputs["beta"][z_index]
    Fsigma_z = disc_outputs["Fsigma"][z_index]

    vphi_z = disc_outputs["vphi"][z_index] # In cm/s
    R_z = disc_outputs["R"][z_index] # In cm
    r0 = disc_outputs["r0"][z_index] # In cm
    Omega0 = vphi_z/r0 * ((r0/R_z)**beta_z) # Angular frequency at r0 in s^-1
    x_b = disc_outputs["x_b"][z_index]
    xmin = disc_outputs["xmin"][z_index]
    xmax = disc_outputs["xmax"][z_index] 
    x_arr = np.linspace(xmin, xmax, 200) # Generate array of x values

    Mstar_z = disc_outputs["Mstar"][z_index].to('Msun') # Convert from g to Msun
    logMstar = np.log10(Mstar_z.value)

    # # Print Mh to find where code goes wrong
    # log10Mh = np.log10(disc_outputs["Mh"][-1].to('Msun').value) # Want Mh at z=0
    # print(f'Mh at z=0: {log10Mh}')
    
    # Establishing Zmin and ZCGM
    if logMstar <= 9: # For low-mass galaxies
        ZCGM = 0.05 # Sharda+2024 Sec. 2.2.3
    elif logMstar >= 10.5:
        ZCGM = 0.2 # Sharda+2024 Sec. 2.2-0.7
    else: # For intermediate-mass galaxies, interpolate value of ZCGM
        line_params = line_func(9, 0.05, 10.5, 0.2) # Getting slope and y-intercept of line
        ZCGM = line_params[0]*logMstar + line_params[1]

    Zmin = 0.01
    # Zmin = ZCGM

    # Defining common groups of terms
    Toomre_denom = A - 2*beta_z*(P + 2*beta_z)
    GMC_denom = A - (1+beta_z)*(1+P+beta_z)
    c1power = -0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower1 = 0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower2 = 0.5*(-np.sqrt(P**2 + 4*A) - P)  

    # Empty lists store maximum and minimum values of c1 and Zr0
    c1_min = []
    c1_max = []
    Zr0_min = []
    Zr0_max = []

    # print(f'xb:{x_b}, xmax:{xmax}')
    if xmin <= x_b and xmax <= x_b: # Entire galactic disc in Toomre regime
        regime = "Toomre"
        print("Entire galactic disc in Toomre regime")

        # Return S used to calculate Zr0
        S = S_Toomre

        # Calculating lower bound for c1
        c1_lowbound_Toomre = (Zmin - (S_Toomre*(xmax**(2*beta_z)) / Toomre_denom)) * xmax**c1power
        
        # Currently do not have a case for P = 0 - need to solve equation derived from boundary condition
        # if P == 0:
        #     print("No implemented P = 0 case for entire galactic disc in Toomre regime - no upper bound for c1")
        if logMstar > 10.5: # Calculating upper bound using c1 from Piyush's code and Zr0 from Sharda+2024
            print("Zr0 set by source and accretion - Massive galaxy")

            # Calculate c1
            c1_uppbound_Toomre = (A*S_Toomre*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                 (-2.*P - 4.*beta_z) - 4.*np.sqrt(4.*A + P**2)*S_Toomre*beta_z**2 + \
                                 A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                 (2.*A - 8.*beta_z**2 - 4.*beta_z*P) - \
                                 2.*np.sqrt(4.*A + P**2)*S_Toomre*beta_z*P + P*S_Toomre*(4.*beta_z**2 + 2.*beta_z*P))/\
                                 (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                                 (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*(A - 4.*beta_z**2 - 2.*beta_z*P))
            
            # Calculate Zr0
            Zr0_lowbound_Toomre = S_Toomre/A
            Zr0_uppbound_Toomre = S_Toomre/A

        else:
            if P == 0:
                print("No implemented P = 0 case for Zr0 set by diffusion and advection - Low-mass galaxy")
                # Use the expressions for low-mass case with P != 0 for now since I don't have P = 0 case
                # for Toomre regime

                # Calculate c1
                c1_uppbound_Toomre = (2.*A**2*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM - \
                                    4.*np.sqrt(4.*A + P**2)*S_Toomre*beta_z**2 + \
                                    P*S_Toomre*beta_z*(2.*np.sqrt(4.*A + P**2) - 4.*np.sqrt(4.*A + P**2)*\
                                    xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z) + 4.*beta_z)\
                                    + P**2*(S_Toomre*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                    (-2.*np.sqrt(4.*A + P**2) - 4.*beta_z) - 2.*S_Toomre*beta_z - \
                                    8.*np.sqrt(4.*A + P**2)*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**3*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-2.*S_Toomre*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P) + \
                                    A*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (2.*P**3*ZCGM + 2.*P**2*np.sqrt(4.*A + P**2)*ZCGM - \
                                    4.*S_Toomre*xmax**(2.*beta_z)*beta_z + \
                                    P*(-2.*S_Toomre*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P)))/\
                                    (A**2*(1.*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(-5. + 5.*xmax**(1.*np.sqrt(4.*A + P**2)))) + \
                                    A*(2.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(-2. + 2.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(20. - 20.*xmax**(1.*np.sqrt(4.*A + P**2)))*(1.*beta_z**2 + 0.5*beta_z*P) + \
                                    np.sqrt(4.*A + P**2)*(-4. - 4.*xmax**(1.*np.sqrt(4.*A + P**2)))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**2*(np.sqrt(4.*A + P**2)*\
                                    (-8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))) + P*(8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P))
                
                # Calculate Zr0
                Zr0_lowbound_Toomre = ((P**2)*S_Toomre + A*(2*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2) + S_Toomre) - \
                                    4*S_Toomre*(beta_z**2) + P*S_Toomre*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                Zr0_uppbound_Toomre = ((P**2)*S_Toomre + A*(2*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2) + S_Toomre) - \
                                    4*S_Toomre*(beta_z**2) + P*S_Toomre*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                
            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculate c1
                c1_uppbound_Toomre = (2.*A**2*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM - \
                                    4.*np.sqrt(4.*A + P**2)*S_Toomre*beta_z**2 + \
                                    P*S_Toomre*beta_z*(2.*np.sqrt(4.*A + P**2) - 4.*np.sqrt(4.*A + P**2)*\
                                    xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z) + 4.*beta_z)\
                                    + P**2*(S_Toomre*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                    (-2.*np.sqrt(4.*A + P**2) - 4.*beta_z) - 2.*S_Toomre*beta_z - \
                                    8.*np.sqrt(4.*A + P**2)*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**3*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-2.*S_Toomre*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P) + \
                                    A*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (2.*P**3*ZCGM + 2.*P**2*np.sqrt(4.*A + P**2)*ZCGM - \
                                    4.*S_Toomre*xmax**(2.*beta_z)*beta_z + \
                                    P*(-2.*S_Toomre*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P)))/\
                                    (A**2*(1.*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(-5. + 5.*xmax**(1.*np.sqrt(4.*A + P**2)))) + \
                                    A*(2.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(-2. + 2.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(20. - 20.*xmax**(1.*np.sqrt(4.*A + P**2)))*(1.*beta_z**2 + 0.5*beta_z*P) + \
                                    np.sqrt(4.*A + P**2)*(-4. - 4.*xmax**(1.*np.sqrt(4.*A + P**2)))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**2*(np.sqrt(4.*A + P**2)*\
                                    (-8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))) + P*(8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P))
                
                # Calculate Zr0
                Zr0_lowbound_Toomre = ((P**2)*S_Toomre + A*(2*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2) + S_Toomre) - \
                                    4*S_Toomre*(beta_z**2) + P*S_Toomre*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                Zr0_uppbound_Toomre = ((P**2)*S_Toomre + A*(2*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2) + S_Toomre) - \
                                    4*S_Toomre*(beta_z**2) + P*S_Toomre*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                    
        if c1_uppbound_Toomre < c1_lowbound_Toomre: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for Toomre - decrease Zmin")
        
        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_Toomre)
        c1_max.append(c1_uppbound_Toomre)
        Zr0_max.append(Zr0_uppbound_Toomre)
        Zr0_min.append(Zr0_lowbound_Toomre)

        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_min = (S_Toomre*(x_arr**(2*beta_z)) / Toomre_denom) + c1_lowbound_Toomre*(x_arr**solpower1) + \
                            (Zr0_lowbound_Toomre - (S_Toomre/Toomre_denom) - c1_lowbound_Toomre) * (x_arr**solpower2) # Sharda+2024 Eqn 29
        normZ_profile_max = (S_Toomre*(x_arr**(2*beta_z)) / Toomre_denom) + c1_uppbound_Toomre*(x_arr**solpower1) + \
                            (Zr0_uppbound_Toomre - (S_Toomre/Toomre_denom) - c1_uppbound_Toomre) * (x_arr**solpower2) # Sharda+2024 Eqn 29

        # Finding teqbm
        dnormZdx_min = (2*beta_z*S_Toomre*(x_arr**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_lowbound_Toomre*(x_arr**(solpower1 - 1)) + \
                        solpower2*(Zr0_lowbound_Toomre - S_Toomre/Toomre_denom - c1_lowbound_Toomre)*(x_arr**(solpower2 - 1))
        dnormZdx_max = (2*beta_z*S_Toomre*(x_arr**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_uppbound_Toomre*(x_arr**(solpower1 - 1)) + \
                        solpower2*(Zr0_uppbound_Toomre - S_Toomre/Toomre_denom - c1_uppbound_Toomre)*(x_arr**(solpower2 - 1))
        
        adv_term_min = np.abs(P/x_arr + dnormZdx_min)
        adv_term_max = np.abs(P/x_arr + dnormZdx_max)
        
        d2normZdx2_min = (2*beta_z*(2*beta_z - 1)*S_Toomre*(x_arr**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_lowbound_Toomre*(x_arr**(solpower1 - 2)) + \
                         solpower2*(solpower2 - 1)*(Zr0_lowbound_Toomre - S_Toomre/Toomre_denom - c1_lowbound_Toomre)*(x_arr**(solpower2 - 2))
        d2normZdx2_max = (2*beta_z*(2*beta_z - 1)*S_Toomre*(x_arr**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_uppbound_Toomre*(x_arr**(solpower1 - 2)) + \
                         solpower2*(solpower2 - 1)*(Zr0_uppbound_Toomre - S_Toomre/Toomre_denom - c1_uppbound_Toomre)*(x_arr**(solpower2 - 2))
        
        diffusion_term_min = np.abs(dnormZdx_min/x_arr + d2normZdx2_min)
        diffusion_term_max = np.abs(dnormZdx_max/x_arr + d2normZdx2_max)
       
        sstar = x_arr**(2*(beta_z - 1))
        source_term = S_Toomre * sstar

        cstar = 1 / (x_arr**2)
        acc_term_min = normZ_profile_min*A*cstar
        acc_term_max = normZ_profile_max*A*cstar
        
        sg = (x_arr**beta_z) / x_arr
        teqbm_denom_min = normZ_profile_min*T*sg / Omega0 
        teqbm_denom_max = normZ_profile_max*T*sg / Omega0 

        teqbm_min = ((adv_term_min + diffusion_term_min + source_term + acc_term_min)/teqbm_denom_min)**(-1)
        teqbm_max = ((adv_term_max + diffusion_term_max + source_term + acc_term_max)/teqbm_denom_max)**(-1)

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1)

    elif xmin > x_b and xmax > x_b: # Entire galactic disc in GMC regime
        regime = "GMC"
        print("Entire galactic disc in GMC regime")
        
        # Return S used to calculate Zr0
        S = S_GMC

        # Calculating lower of c1
        c1_lowbound_GMC = (Zmin - (S_GMC*(xmax**(1+beta_z)) / GMC_denom)) * xmax**c1power

        # Using c1 upper bound values from Piyush's code
        # Mstar check first
        if logMstar > 10.5:
            print("Zr0 set by source and accretion - Massive galaxy")
            
            # Calculating c1
            c1_uppbound_GMC = (-1.*np.sqrt(4.*A + P**2)*S_GMC - 1.*P*np.sqrt(4.*A + P**2)*S_GMC + \
                            A*S_GMC*xmax**(1. + 0.5*P + 0.5*np.sqrt(4.*A + P**2) + beta_z)*\
                            (-2. - 2.*P - 2.*beta_z) - 2.*np.sqrt(4.*A + P**2)*S_GMC*beta_z - \
                            1.*P*np.sqrt(4.*A + P**2)*S_GMC*beta_z - 1.*np.sqrt(4.*A + P**2)*S_GMC*beta_z**2 + \
                            P*S_GMC*(1. + 1.*beta_z)*(1. + P + 1.*beta_z) + \
                            A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                            (2.*A + P*(-2. - 2.*beta_z) - 2.0000000000000004*(1. + 1.*beta_z)**2))/\
                            (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                            (-1. + 1.*A - 1.*P - 2.*beta_z - 1.*P*beta_z - 1.*beta_z**2))
            
            # Accounting for nan values for c1 max
            if np.isnan(c1_uppbound_GMC):
                g = np.sqrt(4.0*A + P*P)
                logx = np.log(xmax)

                # Choose sign so that the exponent in exp() is always <= 0 (prevents overflow)
                s = 1.0 if logx >= 0.0 else -1.0

                # Common, always-safe factors
                invT = np.exp(-s*g*logx)  # = xmax**(-s*g) ∈ (0, 1] in floating-point; may underflow to 0, which is fine

                # These are xmax**(E - s*g) evaluated in log space
                # E1 = 1 + beta_z + 0.5*(P + g)
                # E2 = 0.5*(P + g)
                x_pow_E1_minus_sg = np.exp((1.0 + beta_z + 0.5*(P + g) - s*g) * logx)
                x_pow_E2_minus_sg = np.exp((0.5*(P + g) - s*g) * logx)

                # Scaled numerator (already multiplied by xmax**(-s*g))
                const_part = (
                    -g*S_GMC
                    - P*g*S_GMC
                    - 2.0*g*S_GMC*beta_z
                    - P*g*S_GMC*beta_z
                    - g*S_GMC*beta_z**2
                    + P*S_GMC*(1.0 + beta_z)*(1.0 + P + beta_z)
                )

                num_scaled = (
                    invT*const_part
                    + A*S_GMC*x_pow_E1_minus_sg*(-2.0 - 2.0*P - 2.0*beta_z)
                    + A*P*x_pow_E2_minus_sg*ZCGM*(2.0*A + P*(-2.0 - 2.0*beta_z) - 2.0*(1.0 + beta_z)**2)
                )

                # Scaled denominator (also multiplied by xmax**(-s*g); the x**(±g) terms collapse to 1 ± invT)
                den_scaled = (
                    A*(P*(-invT + 1.0) + g*(invT + 1.0))
                    * (-1.0 + A - 1.0*P - 2.0*beta_z - 1.0*P*beta_z - 1.0*beta_z**2)
                )

                c1_uppbound_GMC = num_scaled/den_scaled

            # Calculating Zr0
            Zr0_lowbound_GMC = S_GMC/A
            Zr0_uppbound_GMC = S_GMC/A

        else:
            if Fsigma_z == 0:
                print("Zr0 set by diffusion and source - No transport")
                # Calculating c1

                c1_uppbound_GMC = ((2.*A + P*(1.*P + 1.*np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                (-1.*P*ZCGM + (0.5*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (4.*A + P**2 + P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - \
                                4.*(1. + beta_z)**2))/((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + (0.25*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*(4.*A + P**2 + \
                                P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - 4.*(1. + beta_z)**2))/\
                                ((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) - \
                                (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*P*S_GMC*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*S_GMC*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z))))/\
                                (A*P*(3. - 3.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*A*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))

                # Calculate Zr0
                Zr0_lowbound_GMC = (0.5 * (S_GMC*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                Zr0_uppbound_GMC = (0.5 * (S_GMC*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     
            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculating c1                    
                c1_uppbound_GMC = ((A + P*(P + np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-1.*P*ZCGM - (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                    (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                    xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S_GMC*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*S_GMC*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + \
                                    (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2))))/\
                                    (A*P*(2.5 - 2.5*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 0.5*A*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 1.*P**2*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
                
                # Include code that accounts for overflow error in xmax**(..P..) for really high P
                if np.isnan(c1_uppbound_GMC) or c1_uppbound_GMC == 0.0:
                    # print(f'This condition is triggered')
                    g = np.sqrt(4.0*A + P**2) # = sqrt(4A + P^2)
                    logx = np.log(xmax)

                    den1 = A - (1.0 + beta_z)*(1.0 + P + beta_z)
                    den2 = (A + P*(P + g)) * (A + P*(-1.0 - beta_z) - (1.0 + beta_z)**2)

                    # Stable powers (only small or negative exponents)
                    invT = np.exp(-g*logx)                    # = xmax**(-g) ~ 0 (safe)
                    x_pow_E_minus_g = np.exp((0.5*(P + g) - g)*logx)  # = xmax**(0.5*P - 0.5*g) ~ O(1)
                    x_pow_negE = np.exp(-0.5*(P + g)*logx)    # = xmax**(-0.5P - 0.5g) ~ 0 (safe)
                    x_pow_1p_beta = np.exp((1.0 + beta_z)*logx)       # = xmax**(1+beta) ~ moderate

                    # Bracket term (no huge positive powers remain)
                    B = (
                        -P*ZCGM
                        - (P*S_GMC*x_pow_negE)/den1
                        - (0.5*(-P - g)*S_GMC*x_pow_negE)/den1
                        + (P*S_GMC*x_pow_1p_beta)/den1
                        + (S_GMC*x_pow_1p_beta*(1.0 + beta_z))/den1
                        + (P*S_GMC*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                        + (0.5*(-P - g)*S_GMC*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                    )

                    # Numerator and denominator divided by xmax**g (so no overflow)
                    num_scaled = (A + P*(P + g)) * x_pow_E_minus_g * B # Numerator after beign multipled across by xmax**g
                    den_scaled = ( # Numerator after being multiplied by xmax**(-g)
                        A*P*(2.5*invT - 2.5)
                        + P**3*(invT - 1.0)
                        - 0.5*A*g*(invT + 1.0)
                        - P**2*g*(invT + 1.0)
                    ) 
                    
                    c1_uppbound_GMC = num_scaled/den_scaled

                # if c1_uppbound_GMC < c1_lowbound_GMC: # Recalculate c1 min after decreasing Zmin
                #     while c1_uppbound_GMC < c1_lowbound_GMC:
                #         Zmin = Zmin - 0.001 # Zmin = 0.01 originally
                #         c1_lowbound_GMC = (Zmin - (S_GMC*(xmax**(1+beta_z)) / GMC_denom)) * xmax**c1power 
                #         if c1_lowbound_GMC < c1_uppbound_GMC:
                #             break

                 # Calculating Zr0
                Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S_GMC*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S_GMC*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
        
        if c1_uppbound_GMC < c1_lowbound_GMC: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for GMC - decrease Zmin")

        # print(f'c1 max: {c1_uppbound_GMC}, c1 min:{c1_lowbound_GMC} for logMhz0 = {np.log10(disc_outputs['Mh'][-1].to('Msun').value)}, P = {P}, A={A}, S_GMC={S_GMC}, xmax={xmax}, beta_z={beta_z}, ZCGM={ZCGM}, at z = {z_given}')

        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_GMC)
        c1_max.append(c1_uppbound_GMC)
        Zr0_max.append(Zr0_uppbound_GMC)
        Zr0_min.append(Zr0_lowbound_GMC)

        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_min = (S_GMC*(x_arr**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_arr**solpower1) + \
                            (Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*(x_arr**solpower2) # Sharda+2024 Eqn 30
        normZ_profile_max = (S_GMC*(x_arr**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_arr**solpower1) + \
                            (Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*(x_arr**solpower2) # Sharda+2024 Eqn 30

        # Finding teqbm
        dnormZdx_min = ((S_GMC*(1+beta_z)*x_arr**(beta_z)) / GMC_denom) + solpower1*c1_lowbound_GMC*x_arr**(solpower1 - 1) + \
                       solpower2*(Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*x_arr**(solpower2 - 1)
        dnormZdx_max = ((S_GMC*(1+beta_z)*x_arr**(beta_z)) / GMC_denom) + solpower1*c1_uppbound_GMC*x_arr**(solpower1 - 1) + \
                       solpower2*(Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*x_arr**(solpower2 - 1)
        
        adv_term_min = np.abs((P/x_arr) * dnormZdx_min)
        adv_term_max = np.abs((P/x_arr) * dnormZdx_max) 
        
        d2normZdx2_min = ((S_GMC*beta_z*(1+beta_z)*x_arr**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_lowbound_GMC*x_arr**(solpower1 - 2) + \
                         solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*x_arr**(solpower2 - 2)
        d2normZdx2_max = ((S_GMC*beta_z*(1+beta_z)*x_arr**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_uppbound_GMC*x_arr**(solpower1 - 2) + \
                         solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*x_arr**(solpower2 - 2)
        
        diffusion_term_min = np.abs((dnormZdx_min/x_arr + d2normZdx2_min))
        diffusion_term_max = np.abs((dnormZdx_max/x_arr + d2normZdx2_max))
        
        sstar = x_arr**(beta_z-1)
        source_term = S_GMC * sstar

        cstar = 1 / (x_arr**2)
        acc_term_min = normZ_profile_min*A*cstar
        acc_term_max = normZ_profile_max*A*cstar
        
        sg = (x_arr**beta_z) / x_arr
        teqbm_denom_min = normZ_profile_min*T*sg / Omega0 
        teqbm_denom_max = normZ_profile_max*T*sg / Omega0 

        teqbm_min = ((adv_term_min + diffusion_term_min + source_term + acc_term_min)/teqbm_denom_min)**(-1)
        teqbm_max = ((adv_term_max + diffusion_term_max + source_term + acc_term_max)/teqbm_denom_max)**(-1)

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1)

    elif xmin <= x_b and xmax > x_b: # Inner disc in Toomre and outer disc in GMC
        regime = "Toomre and GMC" 
        print("Inner disc is Toomre, outer disc is GMC")

        # Calculating GMC part first which is the outer part of the disc
        # Calculating lower bound of c1
        c1_lowbound_GMC = (Zmin - (S_GMC*(xmax**(1+beta_z)) / GMC_denom)) * xmax**c1power
        
        # Using c1 upper bound values from Piyush's code
        # Mstar check first
        if logMstar > 10.5:
            print("Zr0 set by source and accretion - Massive galaxy")

            # Calculating c1
            c1_uppbound_GMC = (-1.*np.sqrt(4.*A + P**2)*S_GMC - 1.*P*np.sqrt(4.*A + P**2)*S_GMC + \
                            A*S_GMC*xmax**(1. + 0.5*P + 0.5*np.sqrt(4.*A + P**2) + beta_z)*\
                            (-2. - 2.*P - 2.*beta_z) - 2.*np.sqrt(4.*A + P**2)*S_GMC*beta_z - \
                            1.*P*np.sqrt(4.*A + P**2)*S_GMC*beta_z - 1.*np.sqrt(4.*A + P**2)*S_GMC*beta_z**2 + \
                            P*S_GMC*(1. + 1.*beta_z)*(1. + P + 1.*beta_z) + \
                            A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                            (2.*A + P*(-2. - 2.*beta_z) - 2.0000000000000004*(1. + 1.*beta_z)**2))/\
                            (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                            (-1. + 1.*A - 1.*P - 2.*beta_z - 1.*P*beta_z - 1.*beta_z**2))

            if np.isnan(c1_uppbound_GMC):
                g = np.sqrt(4.0*A + P*P)
                logx = np.log(xmax)

                # Choose sign so that the exponent in exp() is always <= 0 (prevents overflow)
                s = 1.0 if logx >= 0.0 else -1.0

                # Common, always-safe factors
                invT = np.exp(-s*g*logx)  # = xmax**(-s*g) ∈ (0, 1] in floating-point; may underflow to 0, which is fine

                # These are xmax**(E - s*g) evaluated in log space
                # E1 = 1 + beta_z + 0.5*(P + g)
                # E2 = 0.5*(P + g)
                x_pow_E1_minus_sg = np.exp((1.0 + beta_z + 0.5*(P + g) - s*g) * logx)
                x_pow_E2_minus_sg = np.exp((0.5*(P + g) - s*g) * logx)

                # Scaled numerator (already multiplied by xmax**(-s*g))
                const_part = (
                    -g*S_GMC
                    - P*g*S_GMC
                    - 2.0*g*S_GMC*beta_z
                    - P*g*S_GMC*beta_z
                    - g*S_GMC*beta_z**2
                    + P*S_GMC*(1.0 + beta_z)*(1.0 + P + beta_z)
                )

                num_scaled = (
                    invT*const_part
                    + A*S_GMC*x_pow_E1_minus_sg*(-2.0 - 2.0*P - 2.0*beta_z)
                    + A*P*x_pow_E2_minus_sg*ZCGM*(2.0*A + P*(-2.0 - 2.0*beta_z) - 2.0*(1.0 + beta_z)**2)
                )

                # Scaled denominator (also multiplied by xmax**(-s*g); the x**(±g) terms collapse to 1 ± invT)
                den_scaled = (
                    A*(P*(-invT + 1.0) + g*(invT + 1.0))
                    * (-1.0 + A - 1.0*P - 2.0*beta_z - 1.0*P*beta_z - 1.0*beta_z**2)
                )

                c1_uppbound_GMC = num_scaled/den_scaled
            
            # Calculating Zr0
            # # Using S_GMC
            # Zr0_lowbound_GMC = S_GMC/A
            # Zr0_uppbound_GMC = S_GMC/A
            # Using S_Toomre
            Zr0_lowbound_GMC = S_Toomre/A # Using S_Toomre gives reasonable Zr0 for massive galaxies
                                          # Using S_GMC returns low Zr0 and high metallicity gradients
            Zr0_uppbound_GMC = S_Toomre/A

            # Return S used to calculate Zr0
            S = S_Toomre

        else:
            if Fsigma_z == 0:
                print("Zr0 set by diffusion and source - No transport")
                # Calculating c1
                c1_uppbound_GMC = ((2.*A + P*(1.*P + 1.*np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                (-1.*P*ZCGM + (0.5*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (4.*A + P**2 + P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - \
                                4.*(1. + beta_z)**2))/((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + (0.25*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*(4.*A + P**2 + \
                                P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - 4.*(1. + beta_z)**2))/\
                                ((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) - \
                                (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*P*S_GMC*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*S_GMC*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z))))/\
                                (A*P*(3. - 3.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*A*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
                
                # Calculate Zr0
                # Using S_GMC - S_GMC is used because when calculating MZGR I get negative Zr0 values when using S_Toomre
                Zr0_lowbound_GMC = (0.5 * (S_GMC*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                   4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                   ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                Zr0_uppbound_GMC = (0.5 * (S_GMC*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                   4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                   ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     
                # # Using S_Toomre
                # Zr0_lowbound_GMC = (0.5 * (S_Toomre*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                #                     4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                #                     ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                # Zr0_uppbound_GMC = (0.5 * (S_Toomre*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                #                     4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                #                     ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     

                # Return S used to calculate Zr0
                S = S_GMC

            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculating c1
                c1_uppbound_GMC = ((A + P*(P + np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-1.*P*ZCGM - (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                    (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*\
                                    xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S_GMC*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*S_GMC*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + \
                                    (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S_GMC*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2))))/\
                                    (A*P*(2.5 - 2.5*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 0.5*A*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 1.*P**2*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))

                # Account for getting nan values for c1 max
                if np.isnan(c1_uppbound_GMC):
                    g = np.sqrt(4.0*A + P**2) # = sqrt(4A + P^2)
                    logx = np.log(xmax)

                    den1 = A - (1.0 + beta_z)*(1.0 + P + beta_z)
                    den2 = (A + P*(P + g)) * (A + P*(-1.0 - beta_z) - (1.0 + beta_z)**2)

                    # Stable powers (only small or negative exponents)
                    invT = np.exp(-g*logx)                    # = xmax**(-g) ~ 0 (safe)
                    x_pow_E_minus_g = np.exp((0.5*(P + g) - g)*logx)  # = xmax**(0.5*P - 0.5*g) ~ O(1)
                    x_pow_negE = np.exp(-0.5*(P + g)*logx)    # = xmax**(-0.5P - 0.5g) ~ 0 (safe)
                    x_pow_1p_beta = np.exp((1.0 + beta_z)*logx)       # = xmax**(1+beta) ~ moderate

                    # Bracket term (no huge positive powers remain)
                    B = (
                        -P*ZCGM
                        - (P*S_GMC*x_pow_negE)/den1
                        - (0.5*(-P - g)*S_GMC*x_pow_negE)/den1
                        + (P*S_GMC*x_pow_1p_beta)/den1
                        + (S_GMC*x_pow_1p_beta*(1.0 + beta_z))/den1
                        + (P*S_GMC*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                        + (0.5*(-P - g)*S_GMC*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                    )

                    # Numerator and denominator divided by xmax**g (so no overflow)
                    num_scaled = (A + P*(P + g)) * x_pow_E_minus_g * B # Numerator after beign multipled across by xmax**g
                    den_scaled = ( # Numerator after being multiplied by xmax**(-g)
                        A*P*(2.5*invT - 2.5)
                        + P**3*(invT - 1.0)
                        - 0.5*A*g*(invT + 1.0)
                        - P**2*g*(invT + 1.0)
                    ) 
                    
                    c1_uppbound_GMC = num_scaled/den_scaled

                # Calculating Zr0
                # Using S_GMC - S_GMC is used because when calculating MZGR I get negative Zr0 values when using S_Toomre
                Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S_GMC*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S_GMC*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                
                # # Using S_Toomre
                # Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                #                     S_Toomre*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                #                     ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                # Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                #                     S_Toomre*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                #                     ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C

                # Return S used to calculate Zr0
                S = S_GMC

        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_GMC)
        c1_max.append(c1_uppbound_GMC)
        Zr0_max.append(Zr0_uppbound_GMC)
        Zr0_min.append(Zr0_lowbound_GMC)

        # Finding metallicity profiles corresponding to minimum and maximum of c1_GMC for x_b < x < xmax
        normZ_profile_GMC_min = (S_GMC*(x_arr[x_arr > x_b]**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_arr[x_arr > x_b]**solpower1) + \
                                (Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*(x_arr[x_arr > x_b]**solpower2) # Sharda+2024 Eqn 30
        normZ_profile_GMC_max = (S_GMC*(x_arr[x_arr > x_b]**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_arr[x_arr > x_b]**solpower1) + \
                                (Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*(x_arr[x_arr > x_b]**solpower2) # Sharda+2024 Eqn 30

        # Finding teqbm
        dnormZdx_GMC_min = ((S_GMC*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z)) / GMC_denom) + solpower1*c1_lowbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 1) + \
                           solpower2*(Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 1)
        dnormZdx_GMC_max = ((S_GMC*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z)) / GMC_denom) + solpower1*c1_uppbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 1) + \
                           solpower2*(Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 1)
        
        adv_term_GMC_min = np.abs((P/x_arr[x_arr > x_b]) * dnormZdx_GMC_min)
        adv_term_GMC_max = np.abs((P/x_arr[x_arr > x_b]) * dnormZdx_GMC_max) 
        
        d2normZdx2_GMC_min = ((S_GMC*beta_z*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_lowbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 2) + \
                             solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 2)
        d2normZdx2_GMC_max = ((S_GMC*beta_z*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_uppbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 2) + \
                             solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 2)
        
        diffusion_term_GMC_min = np.abs((dnormZdx_GMC_min/x_arr[x_arr > x_b] + d2normZdx2_GMC_min))
        diffusion_term_GMC_max = np.abs((dnormZdx_GMC_max/x_arr[x_arr > x_b] + d2normZdx2_GMC_max))
        
        sstar = x_arr[x_arr > x_b]**(beta_z-1)
        source_term_GMC = S_GMC * sstar

        cstar = 1 / x_arr[x_arr > x_b]**2
        acc_term_GMC_min = normZ_profile_GMC_min*A*cstar
        acc_term_GMC_max = normZ_profile_GMC_max*A*cstar
        
        sg = (x_arr[x_arr > x_b]**beta_z) / x_arr[x_arr > x_b]
        teqbm_denom_GMC_min = normZ_profile_GMC_min*T*sg 
        teqbm_denom_GMC_max = normZ_profile_GMC_max*T*sg 

        teqbm_GMC_min = (Omega0*(adv_term_GMC_min + diffusion_term_GMC_min + source_term_GMC + acc_term_GMC_min)/teqbm_denom_GMC_min)**(-1)
        teqbm_GMC_max = (Omega0*(adv_term_GMC_max + diffusion_term_GMC_max + source_term_GMC + acc_term_GMC_max)/teqbm_denom_GMC_max)**(-1)

        # Finding value of normZ at x = x_b from GMC part
        normZ_GMC_xb_min = (S_GMC*(x_b**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_b**solpower1) + \
                           (Zr0_lowbound_GMC - S_GMC/GMC_denom - c1_lowbound_GMC)*(x_b**solpower2) # normZ at x = x_b for c1_GMC_min   
        normZ_GMC_xb_max = (S_GMC*(x_b**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_b**solpower1) + \
                           (Zr0_uppbound_GMC - S_GMC/GMC_denom - c1_uppbound_GMC)*(x_b**solpower2) # normZ at x = x_b

        # Set Zr0_Toomre to be the same as Zr0_GMC
        Zr0_lowbound_Toomre = Zr0_lowbound_GMC
        Zr0_uppbound_Toomre = Zr0_uppbound_GMC  

        # # At x_b, normZ from Toomre and GMC part must be the same - use this to calculate c1 for Toomre part
        c1_Toomre_fromGMC_min = (normZ_GMC_xb_min - (S_Toomre*(x_b**(2*beta_z)) / Toomre_denom) - (Zr0_lowbound_Toomre - S_Toomre/Toomre_denom)*(x_b**solpower2)) \
                                / (x_b**solpower1 - x_b**solpower2)
        c1_Toomre_fromGMC_max = (normZ_GMC_xb_max - (S_Toomre*(x_b**(2*beta_z)) / Toomre_denom) - (Zr0_uppbound_Toomre - S_Toomre/Toomre_denom)*(x_b**solpower2)) \
                                / (x_b**solpower1 - x_b**solpower2)

        # Ensure that normZ > normZmin at x_b
        c1_lowbound_xb_Toomre = (Zmin - (S_Toomre*(x_b**(2*beta_z)) / Toomre_denom)) * x_b**c1power # Lower bound for c1 in Toomre part at x_b
        if c1_lowbound_xb_Toomre > c1_Toomre_fromGMC_max or  c1_lowbound_xb_Toomre > c1_Toomre_fromGMC_min:
            warnings.warn("New c1 from Toomre part is invalid", UserWarning)
            print(f'Toomre c1 min at xb: {c1_lowbound_xb_Toomre}')
            print(f'Toomre c1 min from GMC: {c1_Toomre_fromGMC_min}')
            print(f'Toomre c1 max from GMC: {c1_Toomre_fromGMC_max}')
            print(f'Error occured for z={z_arr[z_index]}, logMstar={logMstar}, logMh={np.log10(disc_outputs['Mh'][z_index].to('Msun').value)}')
            
        c1_min.append(c1_Toomre_fromGMC_min)
        c1_max.append(c1_Toomre_fromGMC_max)
        Zr0_min.append(Zr0_lowbound_Toomre)
        Zr0_max.append(Zr0_uppbound_Toomre)

        # Solve normZ in Toomre part
        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_Toomre_min = (S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z)) / Toomre_denom) + c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**solpower1) + \
                                   (Zr0_lowbound_Toomre - (S_Toomre/Toomre_denom) - c1_Toomre_fromGMC_min) * (x_arr[x_arr <= x_b]**solpower2) # Sharda+2024 Eqn 29
        normZ_profile_Toomre_max = (S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z)) / Toomre_denom) + c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**solpower1) + \
                                   (Zr0_uppbound_Toomre - (S_Toomre/Toomre_denom) - c1_Toomre_fromGMC_max) * (x_arr[x_arr <= x_b]**solpower2) # Sharda+2024 Eqn 29

        # Finding teqbm
        dnormZdx_Toomre_min = (2*beta_z*S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**(solpower1 - 1)) + \
                              solpower2*(Zr0_lowbound_Toomre - S_Toomre/Toomre_denom - c1_Toomre_fromGMC_min)*(x_arr[x_arr <= x_b]**(solpower2 - 1))
        dnormZdx_Toomre_max = (2*beta_z*S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**(solpower1 - 1)) + \
                              solpower2*(Zr0_uppbound_Toomre - S_Toomre/Toomre_denom - c1_Toomre_fromGMC_max)*(x_arr[x_arr <= x_b]**(solpower2 - 1))
        
        adv_term_Toomre_min = np.abs(P/x_arr[x_arr <= x_b] + dnormZdx_Toomre_min)
        adv_term_Toomre_max = np.abs(P/x_arr[x_arr <= x_b] + dnormZdx_Toomre_max)
        

        d2normZdx2_Toomre_min = (2*beta_z*(2*beta_z - 1)*S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**(solpower1 - 2)) + \
                                solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S_Toomre/Toomre_denom - c1_Toomre_fromGMC_min)*(x_arr[x_arr <= x_b]**(solpower2 - 2))
        d2normZdx2_Toomre_max = (2*beta_z*(2*beta_z - 1)*S_Toomre*(x_arr[x_arr <= x_b]**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**(solpower1 - 2)) + \
                                solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S_Toomre/Toomre_denom - c1_Toomre_fromGMC_max)*(x_arr[x_arr <= x_b]**(solpower2 - 2))
        
        diffusion_term_Toomre_min = np.abs(dnormZdx_Toomre_min/x_arr[x_arr <= x_b] + d2normZdx2_Toomre_min)
        diffusion_term_Toomre_max = np.abs(dnormZdx_Toomre_max/x_arr[x_arr <= x_b] + d2normZdx2_Toomre_max)


        sstar = x_arr[x_arr <= x_b]**(2*(beta_z - 1))
        source_term_Toomre = S_Toomre * sstar

        cstar = 1 / (x_arr[x_arr <= x_b]**2)
        acc_term_Toomre_min = normZ_profile_Toomre_min*A*cstar
        acc_term_Toomre_max = normZ_profile_Toomre_max*A*cstar
        
        sg = (x_arr[x_arr <= x_b]**beta_z) / x_arr[x_arr <= x_b]
        teqbm_denom_Toomre_min = normZ_profile_Toomre_min*T*sg 
        teqbm_denom_Toomre_max = normZ_profile_Toomre_max*T*sg

        teqbm_Toomre_min = (Omega0*(adv_term_Toomre_min + diffusion_term_Toomre_min + source_term_Toomre + acc_term_Toomre_min)/teqbm_denom_Toomre_min)**(-1)
        teqbm_Toomre_max = (Omega0*(adv_term_Toomre_max + diffusion_term_Toomre_max + source_term_Toomre + acc_term_Toomre_max)/teqbm_denom_Toomre_max)**(-1)
        
        # Combine parts from Toomre and GMC denom
        adv_term_min = np.array(list(adv_term_Toomre_min) + list(adv_term_GMC_min))
        adv_term_max = np.array(list(adv_term_Toomre_max) + list(adv_term_GMC_max))

        diffusion_term_min = np.array(list(diffusion_term_Toomre_min) + list(diffusion_term_GMC_min))
        diffusion_term_max = np.array(list(diffusion_term_Toomre_max) + list(diffusion_term_GMC_max))

        source_term = np.array(list(source_term_Toomre) + list(source_term_GMC))

        acc_term_min = np.array(list(acc_term_Toomre_min) + list(acc_term_GMC_min))
        acc_term_max = np.array(list(acc_term_Toomre_max) + list(acc_term_GMC_max))

        teqbm_min = np.array(list(teqbm_Toomre_min.value) + list(teqbm_GMC_min.value)) * u.s
        teqbm_max = np.array(list(teqbm_Toomre_max.value) + list(teqbm_GMC_max.value)) * u.s

        normZ_profile_min = np.array(list(normZ_profile_Toomre_min) + list(normZ_profile_GMC_min))
        normZ_profile_max = np.array(list(normZ_profile_Toomre_max) + list(normZ_profile_GMC_max))

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting

    return {"regime": regime, "x": x_arr, "x_b": x_b, "metgrad_min": metgrad_min, "metgrad_max": metgrad_max, "log10Zr0_min": log10Zr0_min,
            "log10Zr0_max":log10Zr0_max, "normZ_profile_min": normZ_profile_min, "normZ_profile_max": normZ_profile_max, "teqbm_min": teqbm_min, 
            "teqbm_max": teqbm_max, "adv_term_min": adv_term_min,"adv_term_max": adv_term_max, "diffusion_term_min": diffusion_term_min, 
            "diffusion_term_max": diffusion_term_max, "source_term": source_term, "acc_term_min": acc_term_min, "acc_term_max": acc_term_max, 
            "c1_min": c1_min, "c1_max": c1_max, "Zr0_min": Zr0_min, "Zr0_max": Zr0_max, "T":T, "P": P, "A": A, 'S':S, "S_Toomre": S_Toomre, "S_GMC": S_GMC, 
            "logMstar": logMstar, "beta": beta_z, "ZCGM": ZCGM, "Zmin": Zmin, "vphi": vphi_z, "R": R_z, "Omega0": Omega0, "disc_outputs": disc_outputs
            }

def generate_curve(regime, c1_Toomre, c1_GMC, Zr0, P, S_Toomre, S_GMC, A, T, x, xmax, x_b, beta, Omega0):
    """
    Function to calculate metallicity profile and teqbm for given regime, c1 bounds, and disk parameters:

    Parameters:
        regime: string
            'Toomre', 'GMC', 'Toomre and GMC'
        c1_Toomre: float
            Constant c1 for Toomre regime
        c1_GMC: float
            Constant c1 for GMC regime
        Zr0: float
            Constant Zr0
        P: float
            Dimensionless ratio P
        S_Toomre: float
            Dimensionless ratio S_Toomre
        S_GMC: float
            Dimensionless ratio S_GMC
        A: float
            Dimensionless ratio A
        T: float
            Dimensionless ratio T
        x: array
            Dimensionless disk radius
        xmax: float
            Dimensionless outer disk radius
        x_b: float
            Transition radius between Toomre and GMC regime
        beta: float
            Rotation curve index
        Omega0: float
            Rotational frequency at r0, innermost disk radius, in 1/s

    Returns:
        normZ: array
            Normalized metallicity profile
        teqbm: array
            Equilibration time in s
        metgrad: float
            Metallicity gradient for log10normZ vs. x
    """ 
    solpower1 = 0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower2 = 0.5*(-np.sqrt(P**2 + 4*A) - P)

    if regime == 'Toomre':
        Toomre_denom = (A - 2*beta*(P+2*beta))

        # Calculating normZ
        normZ = (S_Toomre*(x**(2*beta)) / Toomre_denom) + c1_Toomre*(x**solpower1) + \
                            (Zr0 - (S_Toomre/Toomre_denom) - c1_Toomre) * (x**solpower2) # Sharda+2024 Eqn 29
    
        # Finding teqbm
        dnormZdx = (2*beta*S_Toomre*(x**(2*beta - 1)) / Toomre_denom) + solpower1*c1_Toomre*(x**(solpower1 - 1)) + \
                        solpower2*(Zr0 - S_Toomre/Toomre_denom - c1_Toomre)*(x**(solpower2 - 1))
        
        adv_term = np.abs(P/x + dnormZdx)
        
        # k = x_arr / (x_arr**beta_z)
        d2normZdx2 = (2*beta*(2*beta - 1)*S_Toomre*(x**(2*beta - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_Toomre*(x**(solpower1 - 2)) + \
                         solpower2*(solpower2 - 1)*(Zr0 - S_Toomre/Toomre_denom - c1_Toomre)*(x**(solpower2 - 2))
        
        diffusion_term = np.abs(dnormZdx/x + d2normZdx2)
        
        sstar = x**(2*(beta - 1))
        source_term = S_Toomre * sstar

        cstar = 1 / (x**2)
        acc_term = normZ*A*cstar
        
        sg = (x**beta) / x
        teqbm_denom = normZ*T*sg / Omega0 

        teqbm = ((adv_term + diffusion_term + source_term + acc_term)/teqbm_denom)**(-1)

        # Calculating metallicity gradient using polyfit
        metgrad, log10Zr0 = np.polyfit(x[10:-10], np.log10(normZ[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting

    elif regime == 'GMC':
        GMC_denom = (A - (1+beta)*(1+P+beta))

        # Calculate normZ
        normZ = (S_GMC*(x**(1+beta)) / GMC_denom) + c1_GMC*(x**solpower1) + \
                            (Zr0 - S_GMC/GMC_denom - c1_GMC)*(x**solpower2) # Sharda+2024 Eqn 30

        # Finding teqbm
        dnormZdx= ((S_GMC*(1+beta)*x**(beta)) / GMC_denom) + solpower1*c1_GMC*x**(solpower1 - 1) + \
                    solpower2*(Zr0 - S_GMC/GMC_denom - c1_GMC)*x**(solpower2 - 1)

        
        adv_term = np.abs((P/x) * dnormZdx)
        
        # k = x_arr / (x_arr**beta_z)
        d2normZdx2 = ((S_GMC*beta*(1+beta)*x**(beta - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_GMC*x**(solpower1 - 2) + \
                       solpower2*(solpower2 - 1)*(Zr0 - S_GMC/GMC_denom - c1_GMC)*x**(solpower2 - 2)
        
        diffusion_term = np.abs((dnormZdx/x + d2normZdx2))
        
        sstar = x**(beta-1)
        source_term = S_GMC * sstar

        cstar = 1 / (x**2)
        acc_term = normZ*A*cstar
        
        sg = (x**beta) / x
        teqbm_denom = normZ*T*sg / Omega0 

        teqbm = ((adv_term + diffusion_term + source_term + acc_term)/teqbm_denom)**(-1)

        # Calculating metallicity gradient using polyfit
        metgrad, log10Zr0 = np.polyfit(x[10:-10], np.log10(normZ[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting

    elif regime == 'Toomre and GMC':
        Toomre_denom = (A - 2*beta*(P+2*beta))
        GMC_denom = (A - (1+beta)*(1+P+beta))
        
        # Calculating normZ from Toomre and GMC part using
        normZ_Toomre = (S_Toomre*x[x <= x_b]**(2*beta) / Toomre_denom) + c1_Toomre*x[x <= x_b]**solpower1 + \
                        (Zr0 - S_Toomre/Toomre_denom - c1_Toomre)*x[x <= x_b]**solpower2
        normZ_GMC = (S_GMC*x[x > x_b]**(1+beta) / GMC_denom) + c1_GMC*x[x > x_b]**solpower1 + (Zr0 - S_GMC/GMC_denom - c1_GMC)*x[x > x_b]**solpower2
        
        # print(f'normZ Toomre:{normZ_Toomre}\n normZ GMC: {normZ_GMC}')

        # Calculating teqbm
        dnormZdx_Toomre = (2*beta*S_Toomre*(x[x <= x_b]**(2*beta - 1)) / Toomre_denom) + solpower1*c1_Toomre*(x[x <= x_b]**(solpower1 - 1)) + \
                          solpower2*(Zr0 - S_Toomre/Toomre_denom - c1_Toomre)*(x[x <= x_b]**(solpower2 - 1))
        dnormZdx_GMC = ((S_GMC*(1+beta)*x[x > x_b]**(beta)) / GMC_denom) + solpower1*c1_GMC*x[x > x_b]**(solpower1 - 1) + \
                        solpower2*(Zr0 - S_GMC/GMC_denom - c1_GMC)*x[x > x_b]**(solpower2 - 1)
        
        adv_term_Toomre = np.abs(P/x[x <= x_b] + dnormZdx_Toomre)
        adv_term_GMC = np.abs(P/x[x > x_b] * dnormZdx_GMC)

        d2normZdx2_Toomre = (2*beta*(2*beta - 1)*S_Toomre*(x[x <= x_b]**(2*beta - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_Toomre*(x[x <= x_b]**(solpower1 - 2)) + \
                            solpower2*(solpower2 - 1)*(Zr0 - S_Toomre/Toomre_denom - c1_Toomre)*(x[x <= x_b]**(solpower2 - 2))
        d2normZdx2_GMC = ((S_GMC*beta*(1+beta)*x[x > x_b]**(beta - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_GMC*x[x > x_b]**(solpower1 - 2) + \
                         solpower2*(solpower2 - 1)*(Zr0 - S_GMC/GMC_denom - c1_GMC)*x[x > x_b]**(solpower2 - 2)
        
        diffusion_term_Toomre = np.abs(dnormZdx_Toomre/x[x <= x_b] + d2normZdx2_Toomre)
        diffusion_term_GMC = np.abs((dnormZdx_GMC/x[x > x_b] + d2normZdx2_GMC))    

        sstar_Toomre = x[x <= x_b]**(2*(beta - 1))
        source_term_Toomre = np.abs(S_Toomre * sstar_Toomre)
        sstar_GMC = x[x > x_b]**(beta-1)
        source_term_GMC = np.abs(S_GMC * sstar_GMC)
        
        cstar_Toomre = 1 / (x[x <= x_b]**2)
        acc_term_Toomre = np.abs(normZ_Toomre*A*cstar_Toomre)
        cstar_GMC = 1 / (x[x > x_b]**2)
        acc_term_GMC = np.abs(normZ_GMC*A*cstar_GMC)

        sg_Toomre = (x[x <= x_b]**beta) / x[x <= x_b]
        teqbm_denom_Toomre = normZ_Toomre*T*sg_Toomre/Omega0 
        
        sg_GMC = (x[x > x_b]**beta) / x[x > x_b]
        teqbm_denom_GMC = normZ_GMC*T*sg_GMC / Omega0 

        teqbm_Toomre = ((adv_term_Toomre + diffusion_term_Toomre + source_term_Toomre + acc_term_Toomre)/teqbm_denom_Toomre)**(-1)
        teqbm_GMC = ((adv_term_GMC + diffusion_term_GMC + source_term_GMC + acc_term_GMC)/teqbm_denom_GMC)**(-1)

        # Combining two interpolated normZ and teqbm values from Toomre and GMC part
        normZ = np.array(list(normZ_Toomre) + list(normZ_GMC))
        teqbm = np.array(list(teqbm_Toomre.value) + list(teqbm_GMC.value)) * u.s
        adv_term = np.array(list(adv_term_Toomre) + list(adv_term_GMC))
        diffusion_term = np.array(list(diffusion_term_Toomre) + list(diffusion_term_GMC))
        source_term = np.array(list(source_term_Toomre) + list(source_term_GMC))
        acc_term = np.array(list(acc_term_Toomre) + list(acc_term_GMC))

        # Calculating metgrad
        metgrad, log10Zr0 = np.polyfit(x[10:-10], np.log10(normZ[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting

    return normZ, teqbm, metgrad

def normZ_func_varyPSA(disk_code_outputs, z_given, P, S, A):
    """
    Function to calculate metallicity gradients while varying P, S, A

    Parameters:
        z_given: Float - Redshift to calculate disc parameters
        disc_outputs: Dictionary - Outputs from disc_code_outputs
        P: Float - Value of P to use
        S: Float - Value of S to use
        A: Float - Value of A to use

    Returns:
        normZ: Float - Metallicity at radius, x, normalized to solar metallicity
    """
    # Calculate parameters from galactic disc model
    disc_outputs = disk_code_outputs
    
    # Check log10Mh0, Mstar, Mdotacc, etaw
    logMhz_final = np.log10(disc_outputs['Mh'][-1].to('Msun').value)
    logMstarz_final = np.log10(disc_outputs['Mstar'][-1].to('Msun').value)
    z_init = disc_outputs['z'][0]
    z_final = disc_outputs['z'][-1]
    Mdotacc = disc_outputs['Mdotacc_option']
    etaw = disc_outputs['etaw_option']
    # print(f'log10Mhz_final = {logMhz_final} ({logMstarz_final}) integrated from {z_init} to {z_final} with Mdotacc = {Mdotacc} and etaw = {etaw}')

    # Extract important parameters and their value at given z
    z_arr = disc_outputs["z"]
    z_index = find_nearest(z_arr, z_given)

    # print(f'Calculated z={z_arr[z_index]} for given z={z_given}')

    y = 0.028 # Yield factor, i.e., how much ISM is enriched with metals by SNII - Sharda+2024 Eqn. 26 
    solarZ = 0.0134 # Solar metallicity
    T = disc_outputs['T'][z_index]

    beta_z = disc_outputs["beta"][z_index]
    Fsigma_z = disc_outputs["Fsigma"][z_index]

    vphi_z = disc_outputs["vphi"][z_index] # In cm/s
    R_z = disc_outputs["R"][z_index] # In cm
    r0 = disc_outputs["r0"][z_index] # In cm
    Omega0 = vphi_z/r0 * ((r0/R_z)**beta_z) # Angular frequency at r0 in s^-1
    x_b = disc_outputs["x_b"][z_index]
    xmin = disc_outputs["xmin"][z_index]
    xmax = disc_outputs["xmax"][z_index] 
    x_arr = np.linspace(xmin, xmax, 200) # Generate array of x values

    Mstar_z = disc_outputs["Mstar"][z_index].to('Msun') # Convert from g to Msun
    logMstar = np.log10(Mstar_z.value)

    # # Print Mh to find where code goes wrong
    # log10Mh = np.log10(disc_outputs["Mh"][-1].to('Msun').value) # Want Mh at z=0
    # print(f'Mh at z=0: {log10Mh}')
    
    # Establishing Zmin and ZCGM
    if logMstar <= 9: # For low-mass galaxies
        ZCGM = 0.05 # Sharda+2024 Sec. 2.2.3
    elif logMstar >= 10.5:
        ZCGM = 0.2 # Sharda+2024 Sec. 2.2-0.7
    else: # For intermediate-mass galaxies, interpolate value of ZCGM
        line_params = line_func(9, 0.05, 10.5, 0.2) # Getting slope and y-intercept of line
        ZCGM = line_params[0]*logMstar + line_params[1]

    Zmin = 0.01
    # Zmin = ZCGM

    # Defining common groups of terms
    Toomre_denom = A - 2*beta_z*(P + 2*beta_z)
    GMC_denom = A - (1+beta_z)*(1+P+beta_z)
    c1power = -0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower1 = 0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower2 = 0.5*(-np.sqrt(P**2 + 4*A) - P)  

    # Empty lists store maximum and minimum values of c1 and Zr0
    c1_min = []
    c1_max = []
    Zr0_min = []
    Zr0_max = []
 
    if xmin <= x_b and xmax <= x_b: # Entire galactic disc in Toomre regime
        regime = "Toomre"
        print("Entire galactic disc in Toomre regime")

        # Calculating lower bound for c1
        c1_lowbound_Toomre = (Zmin - (S*(xmax**(2*beta_z)) / Toomre_denom)) * xmax**c1power
        
        # Currently do not have a case for P = 0 - need to solve equation derived from boundary condition
        # if P == 0:
        #     print("No implemented P = 0 case for entire galactic disc in Toomre regime - no upper bound for c1")
        if logMstar > 10.5: # Calculating upper bound using c1 from Piyush's code and Zr0 from Sharda+2024
            print("Zr0 set by source and accretion - Massive galaxy")

            # Calculate c1
            c1_uppbound_Toomre = (A*S*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                 (-2.*P - 4.*beta_z) - 4.*np.sqrt(4.*A + P**2)*S*beta_z**2 + \
                                 A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                 (2.*A - 8.*beta_z**2 - 4.*beta_z*P) - \
                                 2.*np.sqrt(4.*A + P**2)*S*beta_z*P + P*S*(4.*beta_z**2 + 2.*beta_z*P))/\
                                 (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                                 (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*(A - 4.*beta_z**2 - 2.*beta_z*P))
            
            # Calculate Zr0
            Zr0_lowbound_Toomre = S/A
            Zr0_uppbound_Toomre = S/A

        else:
            if P == 0:
                print("No implemented P = 0 case for Zr0 set by diffusion and advection - Low-mass galaxy")
                # Use the expressions for low-mass case with P != 0 for now since I don't have P = 0 case
                # for Toomre regime

                # Calculate c1
                c1_uppbound_Toomre = (2.*A**2*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM - \
                                    4.*np.sqrt(4.*A + P**2)*S*beta_z**2 + \
                                    P*S*beta_z*(2.*np.sqrt(4.*A + P**2) - 4.*np.sqrt(4.*A + P**2)*\
                                    xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z) + 4.*beta_z)\
                                    + P**2*(S*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                    (-2.*np.sqrt(4.*A + P**2) - 4.*beta_z) - 2.*S*beta_z - \
                                    8.*np.sqrt(4.*A + P**2)*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**3*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-2.*S*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P) + \
                                    A*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (2.*P**3*ZCGM + 2.*P**2*np.sqrt(4.*A + P**2)*ZCGM - \
                                    4.*S*xmax**(2.*beta_z)*beta_z + \
                                    P*(-2.*S*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P)))/\
                                    (A**2*(1.*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(-5. + 5.*xmax**(1.*np.sqrt(4.*A + P**2)))) + \
                                    A*(2.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(-2. + 2.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(20. - 20.*xmax**(1.*np.sqrt(4.*A + P**2)))*(1.*beta_z**2 + 0.5*beta_z*P) + \
                                    np.sqrt(4.*A + P**2)*(-4. - 4.*xmax**(1.*np.sqrt(4.*A + P**2)))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**2*(np.sqrt(4.*A + P**2)*\
                                    (-8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))) + P*(8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P))
                
                # Calculate Zr0
                Zr0_lowbound_Toomre = ((P**2)*S + A*(2*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2) + S) - \
                                    4*S*(beta_z**2) + P*S*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                Zr0_uppbound_Toomre = ((P**2)*S + A*(2*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2) + S) - \
                                    4*S*(beta_z**2) + P*S*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                
            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculate c1
                c1_uppbound_Toomre = (2.*A**2*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM - \
                                    4.*np.sqrt(4.*A + P**2)*S*beta_z**2 + \
                                    P*S*beta_z*(2.*np.sqrt(4.*A + P**2) - 4.*np.sqrt(4.*A + P**2)*\
                                    xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z) + 4.*beta_z)\
                                    + P**2*(S*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta_z)*\
                                    (-2.*np.sqrt(4.*A + P**2) - 4.*beta_z) - 2.*S*beta_z - \
                                    8.*np.sqrt(4.*A + P**2)*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**3*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-2.*S*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P) + \
                                    A*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (2.*P**3*ZCGM + 2.*P**2*np.sqrt(4.*A + P**2)*ZCGM - \
                                    4.*S*xmax**(2.*beta_z)*beta_z + \
                                    P*(-2.*S*xmax**(2.*beta_z) - 8.*ZCGM*beta_z**2 - 4.*ZCGM*beta_z*P)))/\
                                    (A**2*(1.*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(-5. + 5.*xmax**(1.*np.sqrt(4.*A + P**2)))) + \
                                    A*(2.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(-2. + 2.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P*(20. - 20.*xmax**(1.*np.sqrt(4.*A + P**2)))*(1.*beta_z**2 + 0.5*beta_z*P) + \
                                    np.sqrt(4.*A + P**2)*(-4. - 4.*xmax**(1.*np.sqrt(4.*A + P**2)))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P)) + P**2*(np.sqrt(4.*A + P**2)*\
                                    (-8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))) + P*(8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                                    (1.*beta_z**2 + 0.5*beta_z*P))
                
                # Calculate Zr0
                Zr0_lowbound_Toomre = ((P**2)*S + A*(2*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2) + S) - \
                                    4*S*(beta_z**2) + P*S*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                Zr0_uppbound_Toomre = ((P**2)*S + A*(2*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2) + S) - \
                                    4*S*(beta_z**2) + P*S*(np.sqrt(4*A + P**2) + 2*beta_z) - \
                                    4*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta_z**2) + P*beta_z)) / \
                                    ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta_z**2) - 2*P*beta_z))
                    
        if c1_uppbound_Toomre < c1_lowbound_Toomre: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for Toomre - decrease Zmin")
        
        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_Toomre)
        c1_max.append(c1_uppbound_Toomre)
        Zr0_max.append(Zr0_uppbound_Toomre)
        Zr0_min.append(Zr0_lowbound_Toomre)

        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_min = (S*(x_arr**(2*beta_z)) / Toomre_denom) + c1_lowbound_Toomre*(x_arr**solpower1) + \
                            (Zr0_lowbound_Toomre - (S/Toomre_denom) - c1_lowbound_Toomre) * (x_arr**solpower2) # Sharda+2024 Eqn 29
        normZ_profile_max = (S*(x_arr**(2*beta_z)) / Toomre_denom) + c1_uppbound_Toomre*(x_arr**solpower1) + \
                            (Zr0_uppbound_Toomre - (S/Toomre_denom) - c1_uppbound_Toomre) * (x_arr**solpower2) # Sharda+2024 Eqn 29

        # Finding teqbm
        dnormZdx_min = (2*beta_z*S*(x_arr**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_lowbound_Toomre*(x_arr**(solpower1 - 1)) + \
                        solpower2*(Zr0_lowbound_Toomre - S/Toomre_denom - c1_lowbound_Toomre)*(x_arr**(solpower2 - 1))
        dnormZdx_max = (2*beta_z*S*(x_arr**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_uppbound_Toomre*(x_arr**(solpower1 - 1)) + \
                        solpower2*(Zr0_uppbound_Toomre - S/Toomre_denom - c1_uppbound_Toomre)*(x_arr**(solpower2 - 1))
        
        adv_term_min = np.abs(P/x_arr + dnormZdx_min)
        adv_term_max = np.abs(P/x_arr + dnormZdx_max)
        
        d2normZdx2_min = (2*beta_z*(2*beta_z - 1)*S*(x_arr**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_lowbound_Toomre*(x_arr**(solpower1 - 2)) + \
                         solpower2*(solpower2 - 1)*(Zr0_lowbound_Toomre - S/Toomre_denom - c1_lowbound_Toomre)*(x_arr**(solpower2 - 2))
        d2normZdx2_max = (2*beta_z*(2*beta_z - 1)*S*(x_arr**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_uppbound_Toomre*(x_arr**(solpower1 - 2)) + \
                         solpower2*(solpower2 - 1)*(Zr0_uppbound_Toomre - S/Toomre_denom - c1_uppbound_Toomre)*(x_arr**(solpower2 - 2))
        
        diffusion_term_min = np.abs(dnormZdx_min/x_arr + d2normZdx2_min)
        diffusion_term_max = np.abs(dnormZdx_max/x_arr + d2normZdx2_max)
       
        sstar = x_arr**(2*(beta_z - 1))
        source_term = S * sstar

        cstar = 1 / (x_arr**2)
        acc_term_min = normZ_profile_min*A*cstar
        acc_term_max = normZ_profile_max*A*cstar
        
        sg = (x_arr**beta_z) / x_arr
        teqbm_denom_min = normZ_profile_min*T*sg / Omega0 
        teqbm_denom_max = normZ_profile_max*T*sg / Omega0 

        teqbm_min = ((adv_term_min + diffusion_term_min + source_term + acc_term_min)/teqbm_denom_min)**(-1)
        teqbm_max = ((adv_term_max + diffusion_term_max + source_term + acc_term_max)/teqbm_denom_max)**(-1)

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1)

    elif xmin > x_b and xmax > x_b: # Entire galactic disc in GMC regime
        regime = "GMC"
        print("Entire galactic disc in GMC regime")
        
        # Calculating lower of c1
        c1_lowbound_GMC = (Zmin - (S*(xmax**(1+beta_z)) / GMC_denom)) * xmax**c1power

        # Using c1 upper bound values from Piyush's code
        # Mstar check first
        if logMstar > 10.5:
            print("Zr0 set by source and accretion - Massive galaxy")
            
            # Calculating c1
            c1_uppbound_GMC = (-1.*np.sqrt(4.*A + P**2)*S - 1.*P*np.sqrt(4.*A + P**2)*S + \
                            A*S*xmax**(1. + 0.5*P + 0.5*np.sqrt(4.*A + P**2) + beta_z)*\
                            (-2. - 2.*P - 2.*beta_z) - 2.*np.sqrt(4.*A + P**2)*S*beta_z - \
                            1.*P*np.sqrt(4.*A + P**2)*S*beta_z - 1.*np.sqrt(4.*A + P**2)*S*beta_z**2 + \
                            P*S*(1. + 1.*beta_z)*(1. + P + 1.*beta_z) + \
                            A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                            (2.*A + P*(-2. - 2.*beta_z) - 2.0000000000000004*(1. + 1.*beta_z)**2))/\
                            (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                            (-1. + 1.*A - 1.*P - 2.*beta_z - 1.*P*beta_z - 1.*beta_z**2))
            
            # Accounting for nan values for c1 max
            if np.isnan(c1_uppbound_GMC):
                g = np.sqrt(4.0*A + P*P)
                logx = np.log(xmax)

                # Choose sign so that the exponent in exp() is always <= 0 (prevents overflow)
                s = 1.0 if logx >= 0.0 else -1.0

                # Common, always-safe factors
                invT = np.exp(-s*g*logx)  # = xmax**(-s*g) ∈ (0, 1] in floating-point; may underflow to 0, which is fine

                # These are xmax**(E - s*g) evaluated in log space
                # E1 = 1 + beta_z + 0.5*(P + g)
                # E2 = 0.5*(P + g)
                x_pow_E1_minus_sg = np.exp((1.0 + beta_z + 0.5*(P + g) - s*g) * logx)
                x_pow_E2_minus_sg = np.exp((0.5*(P + g) - s*g) * logx)

                # Scaled numerator (already multiplied by xmax**(-s*g))
                const_part = (
                    -g*S
                    - P*g*S
                    - 2.0*g*S*beta_z
                    - P*g*S*beta_z
                    - g*S*beta_z**2
                    + P*S*(1.0 + beta_z)*(1.0 + P + beta_z)
                )

                num_scaled = (
                    invT*const_part
                    + A*S*x_pow_E1_minus_sg*(-2.0 - 2.0*P - 2.0*beta_z)
                    + A*P*x_pow_E2_minus_sg*ZCGM*(2.0*A + P*(-2.0 - 2.0*beta_z) - 2.0*(1.0 + beta_z)**2)
                )

                # Scaled denominator (also multiplied by xmax**(-s*g); the x**(±g) terms collapse to 1 ± invT)
                den_scaled = (
                    A*(P*(-invT + 1.0) + g*(invT + 1.0))
                    * (-1.0 + A - 1.0*P - 2.0*beta_z - 1.0*P*beta_z - 1.0*beta_z**2)
                )

                c1_uppbound_GMC = num_scaled/den_scaled

            # Calculating Zr0
            Zr0_lowbound_GMC = S/A
            Zr0_uppbound_GMC = S/A

        else:
            if Fsigma_z == 0:
                print("Zr0 set by diffusion and source - No transport")
                # Calculating c1

                c1_uppbound_GMC = ((2.*A + P*(1.*P + 1.*np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                (-1.*P*ZCGM + (0.5*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (4.*A + P**2 + P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - \
                                4.*(1. + beta_z)**2))/((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + (0.25*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*(4.*A + P**2 + \
                                P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - 4.*(1. + beta_z)**2))/\
                                ((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) - \
                                (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*P*S*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*S*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z))))/\
                                (A*P*(3. - 3.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*A*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))

                # Calculate Zr0
                Zr0_lowbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                Zr0_uppbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     
            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculating c1                    
                c1_uppbound_GMC = ((A + P*(P + np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-1.*P*ZCGM - (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                    (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                                    xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*S*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + \
                                    (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2))))/\
                                    (A*P*(2.5 - 2.5*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 0.5*A*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 1.*P**2*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
                
                # Include code that accounts for overflow error in xmax**(..P..) for really high P
                if np.isnan(c1_uppbound_GMC) or c1_uppbound_GMC == 0.0:
                    # print(f'This condition is triggered')
                    g = np.sqrt(4.0*A + P**2) # = sqrt(4A + P^2)
                    logx = np.log(xmax)

                    den1 = A - (1.0 + beta_z)*(1.0 + P + beta_z)
                    den2 = (A + P*(P + g)) * (A + P*(-1.0 - beta_z) - (1.0 + beta_z)**2)

                    # Stable powers (only small or negative exponents)
                    invT = np.exp(-g*logx)                    # = xmax**(-g) ~ 0 (safe)
                    x_pow_E_minus_g = np.exp((0.5*(P + g) - g)*logx)  # = xmax**(0.5*P - 0.5*g) ~ O(1)
                    x_pow_negE = np.exp(-0.5*(P + g)*logx)    # = xmax**(-0.5P - 0.5g) ~ 0 (safe)
                    x_pow_1p_beta = np.exp((1.0 + beta_z)*logx)       # = xmax**(1+beta) ~ moderate

                    # Bracket term (no huge positive powers remain)
                    B = (
                        -P*ZCGM
                        - (P*S*x_pow_negE)/den1
                        - (0.5*(-P - g)*S*x_pow_negE)/den1
                        + (P*S*x_pow_1p_beta)/den1
                        + (S*x_pow_1p_beta*(1.0 + beta_z))/den1
                        + (P*S*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                        + (0.5*(-P - g)*S*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                    )

                    # Numerator and denominator divided by xmax**g (so no overflow)
                    num_scaled = (A + P*(P + g)) * x_pow_E_minus_g * B # Numerator after beign multipled across by xmax**g
                    den_scaled = ( # Numerator after being multiplied by xmax**(-g)
                        A*P*(2.5*invT - 2.5)
                        + P**3*(invT - 1.0)
                        - 0.5*A*g*(invT + 1.0)
                        - P**2*g*(invT + 1.0)
                    ) 
                    
                    c1_uppbound_GMC = num_scaled/den_scaled

                # if c1_uppbound_GMC < c1_lowbound_GMC: # Recalculate c1 min after decreasing Zmin
                #     while c1_uppbound_GMC < c1_lowbound_GMC:
                #         Zmin = Zmin - 0.001 # Zmin = 0.01 originally
                #         c1_lowbound_GMC = (Zmin - (S*(xmax**(1+beta_z)) / GMC_denom)) * xmax**c1power 
                #         if c1_lowbound_GMC < c1_uppbound_GMC:
                #             break

                 # Calculating Zr0
                Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
        
        if c1_uppbound_GMC < c1_lowbound_GMC: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for GMC - decrease Zmin")

        # print(f'c1 max: {c1_uppbound_GMC}, c1 min:{c1_lowbound_GMC} for logMhz0 = {np.log10(disc_outputs['Mh'][-1].to('Msun').value)}, P = {P}, A={A}, S={S}, xmax={xmax}, beta_z={beta_z}, ZCGM={ZCGM}, at z = {z_given}')

        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_GMC)
        c1_max.append(c1_uppbound_GMC)
        Zr0_max.append(Zr0_uppbound_GMC)
        Zr0_min.append(Zr0_lowbound_GMC)

        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_min = (S*(x_arr**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_arr**solpower1) + \
                            (Zr0_lowbound_GMC - S/GMC_denom - c1_lowbound_GMC)*(x_arr**solpower2) # Sharda+2024 Eqn 30
        normZ_profile_max = (S*(x_arr**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_arr**solpower1) + \
                            (Zr0_uppbound_GMC - S/GMC_denom - c1_uppbound_GMC)*(x_arr**solpower2) # Sharda+2024 Eqn 30

        # Finding teqbm
        dnormZdx_min = ((S*(1+beta_z)*x_arr**(beta_z)) / GMC_denom) + solpower1*c1_lowbound_GMC*x_arr**(solpower1 - 1) + \
                       solpower2*(Zr0_lowbound_GMC - S/GMC_denom - c1_lowbound_GMC)*x_arr**(solpower2 - 1)
        dnormZdx_max = ((S*(1+beta_z)*x_arr**(beta_z)) / GMC_denom) + solpower1*c1_uppbound_GMC*x_arr**(solpower1 - 1) + \
                       solpower2*(Zr0_uppbound_GMC - S/GMC_denom - c1_uppbound_GMC)*x_arr**(solpower2 - 1)
        
        adv_term_min = np.abs((P/x_arr) * dnormZdx_min)
        adv_term_max = np.abs((P/x_arr) * dnormZdx_max) 
        
        d2normZdx2_min = ((S*beta_z*(1+beta_z)*x_arr**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_lowbound_GMC*x_arr**(solpower1 - 2) + \
                         solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S/GMC_denom - c1_lowbound_GMC)*x_arr**(solpower2 - 2)
        d2normZdx2_max = ((S*beta_z*(1+beta_z)*x_arr**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_uppbound_GMC*x_arr**(solpower1 - 2) + \
                         solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S/GMC_denom - c1_uppbound_GMC)*x_arr**(solpower2 - 2)
        
        diffusion_term_min = np.abs((dnormZdx_min/x_arr + d2normZdx2_min))
        diffusion_term_max = np.abs((dnormZdx_max/x_arr + d2normZdx2_max))
        
        sstar = x_arr**(beta_z-1)
        source_term = S * sstar

        cstar = 1 / (x_arr**2)
        acc_term_min = normZ_profile_min*A*cstar
        acc_term_max = normZ_profile_max*A*cstar
        
        sg = (x_arr**beta_z) / x_arr
        teqbm_denom_min = normZ_profile_min*T*sg / Omega0 
        teqbm_denom_max = normZ_profile_max*T*sg / Omega0 

        teqbm_min = ((adv_term_min + diffusion_term_min + source_term + acc_term_min)/teqbm_denom_min)**(-1)
        teqbm_max = ((adv_term_max + diffusion_term_max + source_term + acc_term_max)/teqbm_denom_max)**(-1)

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1)

    elif xmin <= x_b and xmax > x_b: # Inner disc in Toomre and outer disc in GMC
        regime = "Toomre and GMC" 
        print("Inner disc is Toomre, outer disc is GMC")

        # Calculating GMC part first which is the outer part of the disc
        # Calculating lower bound of c1
        c1_lowbound_GMC = (Zmin - (S*(xmax**(1+beta_z)) / GMC_denom)) * xmax**c1power
        
        # Using c1 upper bound values from Piyush's code
        # Mstar check first
        if logMstar > 10.5:
            print("Zr0 set by source and accretion - Massive galaxy")

            # Calculating c1
            c1_uppbound_GMC = (-1.*np.sqrt(4.*A + P**2)*S - 1.*P*np.sqrt(4.*A + P**2)*S + \
                            A*S*xmax**(1. + 0.5*P + 0.5*np.sqrt(4.*A + P**2) + beta_z)*\
                            (-2. - 2.*P - 2.*beta_z) - 2.*np.sqrt(4.*A + P**2)*S*beta_z - \
                            1.*P*np.sqrt(4.*A + P**2)*S*beta_z - 1.*np.sqrt(4.*A + P**2)*S*beta_z**2 + \
                            P*S*(1. + 1.*beta_z)*(1. + P + 1.*beta_z) + \
                            A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                            (2.*A + P*(-2. - 2.*beta_z) - 2.0000000000000004*(1. + 1.*beta_z)**2))/\
                            (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                            (-1. + 1.*A - 1.*P - 2.*beta_z - 1.*P*beta_z - 1.*beta_z**2))

            if np.isnan(c1_uppbound_GMC):
                g = np.sqrt(4.0*A + P*P)
                logx = np.log(xmax)

                # Choose sign so that the exponent in exp() is always <= 0 (prevents overflow)
                s = 1.0 if logx >= 0.0 else -1.0

                # Common, always-safe factors
                invT = np.exp(-s*g*logx)  # = xmax**(-s*g) ∈ (0, 1] in floating-point; may underflow to 0, which is fine

                # These are xmax**(E - s*g) evaluated in log space
                # E1 = 1 + beta_z + 0.5*(P + g)
                # E2 = 0.5*(P + g)
                x_pow_E1_minus_sg = np.exp((1.0 + beta_z + 0.5*(P + g) - s*g) * logx)
                x_pow_E2_minus_sg = np.exp((0.5*(P + g) - s*g) * logx)

                # Scaled numerator (already multiplied by xmax**(-s*g))
                const_part = (
                    -g*S
                    - P*g*S
                    - 2.0*g*S*beta_z
                    - P*g*S*beta_z
                    - g*S*beta_z**2
                    + P*S*(1.0 + beta_z)*(1.0 + P + beta_z)
                )

                num_scaled = (
                    invT*const_part
                    + A*S*x_pow_E1_minus_sg*(-2.0 - 2.0*P - 2.0*beta_z)
                    + A*P*x_pow_E2_minus_sg*ZCGM*(2.0*A + P*(-2.0 - 2.0*beta_z) - 2.0*(1.0 + beta_z)**2)
                )

                # Scaled denominator (also multiplied by xmax**(-s*g); the x**(±g) terms collapse to 1 ± invT)
                den_scaled = (
                    A*(P*(-invT + 1.0) + g*(invT + 1.0))
                    * (-1.0 + A - 1.0*P - 2.0*beta_z - 1.0*P*beta_z - 1.0*beta_z**2)
                )

                c1_uppbound_GMC = num_scaled/den_scaled
            
            # Calculating Zr0
            # # Using S
            # Zr0_lowbound_GMC = S/A
            # Zr0_uppbound_GMC = S/A
            # Using S
            Zr0_lowbound_GMC = S/A # Using S gives reasonable Zr0 for massive galaxies
                                          # Using S returns low Zr0 and high metallicity gradients
            Zr0_uppbound_GMC = S/A

        else:
            if Fsigma_z == 0:
                print("Zr0 set by diffusion and source - No transport")
                # Calculating c1
                c1_uppbound_GMC = ((2.*A + P*(1.*P + 1.*np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                (-1.*P*ZCGM + (0.5*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (4.*A + P**2 + P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - \
                                4.*(1. + beta_z)**2))/((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + (0.25*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*(4.*A + P**2 + \
                                P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta_z) - 4.*(1. + beta_z)**2))/\
                                ((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                                (A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) - \
                                (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                                xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*P*S*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                (1.*S*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z))))/\
                                (A*P*(3. - 3.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*A*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                                1.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
                
                # Calculate Zr0
                # Using S - S is used because when calculating MZGR I get negative Zr0 values when using S
                Zr0_lowbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                   4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                   ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                Zr0_uppbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                                   4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                                   ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     
                # # Using S
                # Zr0_lowbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                #                     4*(1+beta_z)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                #                     ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4
                # Zr0_uppbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta_z) - \
                #                     4*(1+beta_z)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2))) / \
                #                     ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn. C4     

            else:
                print("Zr0 set by diffusion and advection - Low-mass galaxy")

                # Calculating c1
                c1_uppbound_GMC = ((A + P*(P + np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                    (-1.*P*ZCGM - (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                                    (A - 1.*(1. + beta_z)*(1. + P + beta_z)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                                    xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S*xmax**(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*S*xmax**(1. + beta_z)*(1. + beta_z))/(A - 1.*(1. + beta_z)*(1. + P + beta_z)) + \
                                    (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2)) + \
                                    (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                    (A + P**2 - 1.*(1. + beta_z)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta_z)))/\
                                    ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta_z) - 1.*(1. + beta_z)**2))))/\
                                    (A*P*(2.5 - 2.5*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                    P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 0.5*A*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 1.*P**2*np.sqrt(4.*A + P**2)*\
                                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))

                # Account for getting nan values for c1 max
                if np.isnan(c1_uppbound_GMC):
                    g = np.sqrt(4.0*A + P**2) # = sqrt(4A + P^2)
                    logx = np.log(xmax)

                    den1 = A - (1.0 + beta_z)*(1.0 + P + beta_z)
                    den2 = (A + P*(P + g)) * (A + P*(-1.0 - beta_z) - (1.0 + beta_z)**2)

                    # Stable powers (only small or negative exponents)
                    invT = np.exp(-g*logx)                    # = xmax**(-g) ~ 0 (safe)
                    x_pow_E_minus_g = np.exp((0.5*(P + g) - g)*logx)  # = xmax**(0.5*P - 0.5*g) ~ O(1)
                    x_pow_negE = np.exp(-0.5*(P + g)*logx)    # = xmax**(-0.5P - 0.5g) ~ 0 (safe)
                    x_pow_1p_beta = np.exp((1.0 + beta_z)*logx)       # = xmax**(1+beta) ~ moderate

                    # Bracket term (no huge positive powers remain)
                    B = (
                        -P*ZCGM
                        - (P*S*x_pow_negE)/den1
                        - (0.5*(-P - g)*S*x_pow_negE)/den1
                        + (P*S*x_pow_1p_beta)/den1
                        + (S*x_pow_1p_beta*(1.0 + beta_z))/den1
                        + (P*S*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                        + (0.5*(-P - g)*S*x_pow_negE*(A + P**2 - (1.0 + beta_z)**2 + P*(1.0 + g + beta_z)))/den2
                    )

                    # Numerator and denominator divided by xmax**g (so no overflow)
                    num_scaled = (A + P*(P + g)) * x_pow_E_minus_g * B # Numerator after beign multipled across by xmax**g
                    den_scaled = ( # Numerator after being multiplied by xmax**(-g)
                        A*P*(2.5*invT - 2.5)
                        + P**3*(invT - 1.0)
                        - 0.5*A*g*(invT + 1.0)
                        - P**2*g*(invT + 1.0)
                    ) 
                    
                    c1_uppbound_GMC = num_scaled/den_scaled

                # Calculating Zr0
                # Using S - S is used because when calculating MZGR I get negative Zr0 values when using S
                Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                                S*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                
                # # Using S
                # Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                #                     S*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                #                     ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C3
                # Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta_z) - 2*(1+beta_z)**2) + \
                #                     S*(A + P**2 - (1+beta_z)**2 + P*(1 + np.sqrt(4*A + P**2) + beta_z))) / \
                #                     ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta_z) - (1+beta_z)**2)) # Sharda+2024 Eqn C

        # Saving minimum and maximum values to plot family of curves
        c1_min.append(c1_lowbound_GMC)
        c1_max.append(c1_uppbound_GMC)
        Zr0_max.append(Zr0_uppbound_GMC)
        Zr0_min.append(Zr0_lowbound_GMC)

        # Finding metallicity profiles corresponding to minimum and maximum of c1_GMC for x_b < x < xmax
        normZ_profile_GMC_min = (S*(x_arr[x_arr > x_b]**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_arr[x_arr > x_b]**solpower1) + \
                                (Zr0_lowbound_GMC - S/GMC_denom - c1_lowbound_GMC)*(x_arr[x_arr > x_b]**solpower2) # Sharda+2024 Eqn 30
        normZ_profile_GMC_max = (S*(x_arr[x_arr > x_b]**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_arr[x_arr > x_b]**solpower1) + \
                                (Zr0_uppbound_GMC - S/GMC_denom - c1_uppbound_GMC)*(x_arr[x_arr > x_b]**solpower2) # Sharda+2024 Eqn 30

        # Finding teqbm
        dnormZdx_GMC_min = ((S*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z)) / GMC_denom) + solpower1*c1_lowbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 1) + \
                           solpower2*(Zr0_lowbound_GMC - S/GMC_denom - c1_lowbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 1)
        dnormZdx_GMC_max = ((S*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z)) / GMC_denom) + solpower1*c1_uppbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 1) + \
                           solpower2*(Zr0_uppbound_GMC - S/GMC_denom - c1_uppbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 1)
        
        adv_term_GMC_min = np.abs((P/x_arr[x_arr > x_b]) * dnormZdx_GMC_min)
        adv_term_GMC_max = np.abs((P/x_arr[x_arr > x_b]) * dnormZdx_GMC_max) 
        
        d2normZdx2_GMC_min = ((S*beta_z*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_lowbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 2) + \
                             solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S/GMC_denom - c1_lowbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 2)
        d2normZdx2_GMC_max = ((S*beta_z*(1+beta_z)*x_arr[x_arr > x_b]**(beta_z - 1)) / GMC_denom) + solpower1*(solpower1 - 1)*c1_uppbound_GMC*x_arr[x_arr > x_b]**(solpower1 - 2) + \
                             solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S/GMC_denom - c1_uppbound_GMC)*x_arr[x_arr > x_b]**(solpower2 - 2)
        
        diffusion_term_GMC_min = np.abs((dnormZdx_GMC_min/x_arr[x_arr > x_b] + d2normZdx2_GMC_min))
        diffusion_term_GMC_max = np.abs((dnormZdx_GMC_max/x_arr[x_arr > x_b] + d2normZdx2_GMC_max))
        
        sstar = x_arr[x_arr > x_b]**(beta_z-1)
        source_term_GMC = S * sstar

        cstar = 1 / x_arr[x_arr > x_b]**2
        acc_term_GMC_min = normZ_profile_GMC_min*A*cstar
        acc_term_GMC_max = normZ_profile_GMC_max*A*cstar
    
        sg = (x_arr[x_arr > x_b]**beta_z) / x_arr[x_arr > x_b]
        # print(f'len sg:{len(sg)}, len normZ_profile_GMC_min:{len(normZ_profile_GMC_min)}, len normZ_profile_GMC_max:{len(normZ_profile_GMC_min)}')
        teqbm_denom_GMC_min = normZ_profile_GMC_min*T*sg 
        teqbm_denom_GMC_max = normZ_profile_GMC_max*T*sg 

        teqbm_GMC_min = (Omega0*(adv_term_GMC_min + diffusion_term_GMC_min + source_term_GMC + acc_term_GMC_min)/teqbm_denom_GMC_min)**(-1)
        teqbm_GMC_max = (Omega0*(adv_term_GMC_max + diffusion_term_GMC_max + source_term_GMC + acc_term_GMC_max)/teqbm_denom_GMC_max)**(-1)

        # Finding value of normZ at x = x_b from GMC part
        normZ_GMC_xb_min = (S*(x_b**(1+beta_z)) / GMC_denom) + c1_lowbound_GMC*(x_b**solpower1) + \
                           (Zr0_lowbound_GMC - S/GMC_denom - c1_lowbound_GMC)*(x_b**solpower2) # normZ at x = x_b for c1_GMC_min   
        normZ_GMC_xb_max = (S*(x_b**(1+beta_z)) / GMC_denom) + c1_uppbound_GMC*(x_b**solpower1) + \
                           (Zr0_uppbound_GMC - S/GMC_denom - c1_uppbound_GMC)*(x_b**solpower2) # normZ at x = x_b

        # Set Zr0_Toomre to be the same as Zr0_GMC
        Zr0_lowbound_Toomre = Zr0_lowbound_GMC
        Zr0_uppbound_Toomre = Zr0_uppbound_GMC  

        # # At x_b, normZ from Toomre and GMC part must be the same - use this to calculate c1 for Toomre part
        c1_Toomre_fromGMC_min = (normZ_GMC_xb_min - (S*(x_b**(2*beta_z)) / Toomre_denom) - (Zr0_lowbound_Toomre - S/Toomre_denom)*(x_b**solpower2)) \
                                / (x_b**solpower1 - x_b**solpower2)
        c1_Toomre_fromGMC_max = (normZ_GMC_xb_max - (S*(x_b**(2*beta_z)) / Toomre_denom) - (Zr0_uppbound_Toomre - S/Toomre_denom)*(x_b**solpower2)) \
                                / (x_b**solpower1 - x_b**solpower2)

        # Ensure that normZ > normZmin at x_b
        c1_lowbound_xb_Toomre = (Zmin - (S*(x_b**(2*beta_z)) / Toomre_denom)) * x_b**c1power # Lower bound for c1 in Toomre part at x_b
        if c1_lowbound_xb_Toomre > c1_Toomre_fromGMC_max or  c1_lowbound_xb_Toomre > c1_Toomre_fromGMC_min:
            warnings.warn("New c1 from Toomre part is invalid", UserWarning)
            print(f'Toomre c1 min at xb: {c1_lowbound_xb_Toomre}')
            print(f'Toomre c1 min from GMC: {c1_Toomre_fromGMC_min}')
            print(f'Toomre c1 max from GMC: {c1_Toomre_fromGMC_max}')
            print(f'Error occured for z={z_arr[z_index]}, logMstar={logMstar}, logMh={np.log10(disc_outputs['Mh'][z_index].to('Msun').value)}')
            
        c1_min.append(c1_Toomre_fromGMC_min)
        c1_max.append(c1_Toomre_fromGMC_max)
        Zr0_min.append(Zr0_lowbound_Toomre)
        Zr0_max.append(Zr0_uppbound_Toomre)

        # Solve normZ in Toomre part
        # Finding metallicity profiles corresponding to minimum and maximum of c1
        normZ_profile_Toomre_min = (S*(x_arr[x_arr <= x_b]**(2*beta_z)) / Toomre_denom) + c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**solpower1) + \
                                   (Zr0_lowbound_Toomre - (S/Toomre_denom) - c1_Toomre_fromGMC_min) * (x_arr[x_arr <= x_b]**solpower2) # Sharda+2024 Eqn 29
        normZ_profile_Toomre_max = (S*(x_arr[x_arr <= x_b]**(2*beta_z)) / Toomre_denom) + c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**solpower1) + \
                                   (Zr0_uppbound_Toomre - (S/Toomre_denom) - c1_Toomre_fromGMC_max) * (x_arr[x_arr <= x_b]**solpower2) # Sharda+2024 Eqn 29

        # Finding teqbm
        dnormZdx_Toomre_min = (2*beta_z*S*(x_arr[x_arr <= x_b]**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**(solpower1 - 1)) + \
                              solpower2*(Zr0_lowbound_Toomre - S/Toomre_denom - c1_Toomre_fromGMC_min)*(x_arr[x_arr <= x_b]**(solpower2 - 1))
        dnormZdx_Toomre_max = (2*beta_z*S*(x_arr[x_arr <= x_b]**(2*beta_z - 1)) / Toomre_denom) + solpower1*c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**(solpower1 - 1)) + \
                              solpower2*(Zr0_uppbound_Toomre - S/Toomre_denom - c1_Toomre_fromGMC_max)*(x_arr[x_arr <= x_b]**(solpower2 - 1))
        
        adv_term_Toomre_min = np.abs(P/x_arr[x_arr <= x_b] + dnormZdx_Toomre_min)
        adv_term_Toomre_max = np.abs(P/x_arr[x_arr <= x_b] + dnormZdx_Toomre_max)
        

        d2normZdx2_Toomre_min = (2*beta_z*(2*beta_z - 1)*S*(x_arr[x_arr <= x_b]**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_Toomre_fromGMC_min*(x_arr[x_arr <= x_b]**(solpower1 - 2)) + \
                                solpower2*(solpower2 - 1)*(Zr0_lowbound_GMC - S/Toomre_denom - c1_Toomre_fromGMC_min)*(x_arr[x_arr <= x_b]**(solpower2 - 2))
        d2normZdx2_Toomre_max = (2*beta_z*(2*beta_z - 1)*S*(x_arr[x_arr <= x_b]**(2*beta_z - 2)) / Toomre_denom) + solpower1*(solpower1 - 1)*c1_Toomre_fromGMC_max*(x_arr[x_arr <= x_b]**(solpower1 - 2)) + \
                                solpower2*(solpower2 - 1)*(Zr0_uppbound_GMC - S/Toomre_denom - c1_Toomre_fromGMC_max)*(x_arr[x_arr <= x_b]**(solpower2 - 2))
        
        diffusion_term_Toomre_min = np.abs(dnormZdx_Toomre_min/x_arr[x_arr <= x_b] + d2normZdx2_Toomre_min)
        diffusion_term_Toomre_max = np.abs(dnormZdx_Toomre_max/x_arr[x_arr <= x_b] + d2normZdx2_Toomre_max)


        sstar = x_arr[x_arr <= x_b]**(2*(beta_z - 1))
        source_term_Toomre = S * sstar

        cstar = 1 / (x_arr[x_arr <= x_b]**2)
        acc_term_Toomre_min = normZ_profile_Toomre_min*A*cstar
        acc_term_Toomre_max = normZ_profile_Toomre_max*A*cstar
        
        sg = (x_arr[x_arr <= x_b]**beta_z) / x_arr[x_arr <= x_b]
        teqbm_denom_Toomre_min = normZ_profile_Toomre_min*T*sg 
        teqbm_denom_Toomre_max = normZ_profile_Toomre_max*T*sg

        teqbm_Toomre_min = (Omega0*(adv_term_Toomre_min + diffusion_term_Toomre_min + source_term_Toomre + acc_term_Toomre_min)/teqbm_denom_Toomre_min)**(-1)
        teqbm_Toomre_max = (Omega0*(adv_term_Toomre_max + diffusion_term_Toomre_max + source_term_Toomre + acc_term_Toomre_max)/teqbm_denom_Toomre_max)**(-1)
        
        # Combine parts from Toomre and GMC denom
        adv_term_min = np.array(list(adv_term_Toomre_min) + list(adv_term_GMC_min))
        adv_term_max = np.array(list(adv_term_Toomre_max) + list(adv_term_GMC_max))

        diffusion_term_min = np.array(list(diffusion_term_Toomre_min) + list(diffusion_term_GMC_min))
        diffusion_term_max = np.array(list(diffusion_term_Toomre_max) + list(diffusion_term_GMC_max))

        source_term = np.array(list(source_term_Toomre) + list(source_term_GMC))

        acc_term_min = np.array(list(acc_term_Toomre_min) + list(acc_term_GMC_min))
        acc_term_max = np.array(list(acc_term_Toomre_max) + list(acc_term_GMC_max))

        teqbm_min = np.array(list(teqbm_Toomre_min.value) + list(teqbm_GMC_min.value)) * u.s
        teqbm_max = np.array(list(teqbm_Toomre_max.value) + list(teqbm_GMC_max.value)) * u.s

        normZ_profile_min = np.array(list(normZ_profile_Toomre_min) + list(normZ_profile_GMC_min))
        normZ_profile_max = np.array(list(normZ_profile_Toomre_max) + list(normZ_profile_GMC_max))

        # Calculating metallicity gradient using polyfit
        metgrad_min, log10Zr0_min = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_min[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting
        metgrad_max, log10Zr0_max = np.polyfit(x_arr[10:-10], np.log10(normZ_profile_max[10:-10]), deg = 1) # Remove the first and last 10 points for the gradient fitting

    return {"regime": regime, "x": x_arr, "x_b": x_b, "metgrad_min": metgrad_min, "metgrad_max": metgrad_max, "log10Zr0_min": log10Zr0_min,
            "log10Zr0_max":log10Zr0_max, "normZ_profile_min": normZ_profile_min, "normZ_profile_max": normZ_profile_max, "teqbm_min": teqbm_min, 
            "teqbm_max": teqbm_max, "adv_term_min": adv_term_min,"adv_term_max": adv_term_max, "diffusion_term_min": diffusion_term_min, 
            "diffusion_term_max": diffusion_term_max, "source_term": source_term, "acc_term_min": acc_term_min, "acc_term_max": acc_term_max, 
            "c1_min": c1_min, "c1_max": c1_max, "Zr0_min": Zr0_min, "Zr0_max": Zr0_max, "T":T, "P": P, "A": A, 'S':S, "logMstar": logMstar, "beta": beta_z, 
            "ZCGM": ZCGM, "Zmin": Zmin, "vphi": vphi_z, "R": R_z, "Omega0": Omega0, "disc_outputs": disc_outputs
            }