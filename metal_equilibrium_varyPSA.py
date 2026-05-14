import numpy as np

# Importing astropy
from astropy import units as u
from astropy.constants import G
from astropy.cosmology import FlatLambdaCDM # Import cosmology
cosmo = FlatLambdaCDM(H0=71, Om0=0.27)

# Importing scipy
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d

# Import warnings
import warnings

# Importing other modules 
import gas_accretion as acc

# Varying c as a function of Mh using Fig. 16 Zhao2009
Mh_arr = np.loadtxt("Mh_vs_c.txt", usecols = 0)
c_arr = np.loadtxt("Mh_vs_c.txt", usecols = 1)
c_Mh = interp1d(Mh_arr, c_arr) 

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

# Now include the ability to fix and vary the following parameters: P, S, A, xmax (or R), beta, ZCGM

def normZ_varyPSA_func(z_given, disc_outputs, P, S, A, R, beta, ZCGM):
    """
    Calculates metallicity at given galactic radius and equilibration time to see if equilibrium model applies

    Parameters:
        z_given: Float - Redshift to calculate disc parameters
        disc_outputs: Dictionary - Outputs from galactic disc code
        P: Float - Value of P to use
        S: Float - Value of S to use
        A: Float - Value of A to use
        R: Float - Value of R to use
        beta: Float - Value of beta to use
        ZCGM: Float - Value of ZCGM to use

    Returns:
        normZ: Float - Metallicity at radius, x, normalized to solar metallicity
    """
    # Extract important parameters from galactic disc model and their value at given z
    z_arr = disc_outputs["z"]
    z_index = np.where(np.isclose(z_arr, z_given))[0][0]

    Mdotacc_arr = disc_outputs['Mdotacc']
    Mdotacc_z = disc_outputs['Mdotacc'][z_index]

    MdotSF_z = disc_outputs['MdotSF'][z_index]

    Fsigma_arr = disc_outputs['Fsigma'] 
    Fsigma_z = disc_outputs['Fsigma'][z_index]

    sigmag_arr = disc_outputs["sigmag"]
    sigmag_z = disc_outputs['sigmag'][z_index]

    MstaR = disc_outputs['Mstar'][z_index]

    fgQ_arr = disc_outputs['fgQ'] 
    fgQ_z = disc_outputs['fgQ'][z_index]

    vphi_arr = disc_outputs["vphi"]
    vphi_z = disc_outputs['vphi'][z_index]

    phint_arr = disc_outputs["phint"]
    phint_z = disc_outputs['phint'][z_index]

    phiQ_arr = disc_outputs["phiQ"] # Note that this is just an array of phiQ_init = 2 since we do not recalculate phiQ
    phiQ_z = disc_outputs['phiQ'][z_index]

    Q_arr = disc_outputs["Q"] # Note that this is just an array of Q_init = 1 since we do not recalculate phiQ
    Qval_z = disc_outputs['Q'][z_index]

    fsf_arr = disc_outputs["fsf"]
    fsf_z = disc_outputs['fsf'][z_index]

    # Defining constants
    eta = 1.5 # Defines dissipation of turbulence over one disc scale height - Sharda+2024 Sec. 2.1.3
    Qmin = 1 # Sharda+2024 Sec. 2 
    epsff = 0.015 # Table 1 Sharda+2024
    phimp = 1.4 # Table 1 Sharda+2024
    tSFmax = (2 * u.Gyr).cgs
    y = 0.028 # Yield factor, i.e., how much ISM is enriched with metals by SNII - Sharda+2024 Eqn. 26 
    solarZ = 0.0134 # Solar metallicity
    G_const = G.cgs
    logMstar = np.log10(MstaR.value)

    # Calculating tdep, i.e., depletion timescale
    tdep = (MstaR / MdotSF_z).cgs # In s

    # Making array of x values   
    r0 = (1 * u.kpc).cgs # Sharda+2024 Eqn. 18
    xmin = 1
    xmax = R/r0 # Disc normalised by r0 - Sharda+2024 Sec. 2.3

    x_arr = np.linspace(xmin, xmax, 100)
    
    # Calculating critical radius where Toomre regime = GMC regime
    x_b = (4*np.sqrt(2*(1+beta))*fgQ_z*epsff*vphi_z*tSFmax / (np.pi*Qval_z*np.sqrt(3*fgQ_z*phimp)) * ((r0/R)**beta) * 1/r0)**(1 / (1-beta))

    # Calculating dimensionless quantities - do not depend on radius 
    T = (3*np.sqrt(2*(1+beta))*phiQ_z*fgQ_z/Qval_z) * (vphi_z/sigmag_z * (r0/R)**beta)**2 # Sharda+2024 Eqn. 22
        
    # P = (6*eta*phiQ_arr**2 * phint_arr**1.5 * fgQ_arr**2 / (Qmin**2)) * (1+beta_arr / (1-beta_arr)) * Fsigma_arr # Calculate history of P to plot vs. redshift/stellar mass

    # S_arr = (24*phiQ_arr*(fgQ_arr**2)*epsff*fsf_arr / (np.pi*(Q_arr**2)*np.sqrt(3*fgQ_arr*phimp))) * (phiy*y/solarZ) * (1+beta_arr) * (vphi_arr/sigmag_arr * (r0/R)**beta_arr)**2 # Sharda+2024 Eqn. 24
    # S_arr = (3*np.sqrt(2*(1+beta_arr))*fsf_arr*fgQ_arr*phiQ_arr / (Q_arr*tSFmax)) * (phiy*y/solarZ) * (r0**(beta_arr+1) / (R**beta_arr)) * (vphi_arr / (sigmag_arr**2)) 
    # S = (24*phiQ_z*(fgQ_z**2)*epsff*fsf_z / (np.pi*(Qmin**2)*np.sqrt(3*fgQ_z*phimp))) * (phiy*y/solarZ) * (1+beta) * (vphi_z/sigmag_z * (r0/R)**beta)**2 # Sharda+2024 Eqn. 24
    # S = (3*np.sqrt(2*(1+beta))*fsf_z*fgQ_z*phiQ_z / (Qmin*tSFmax)) * (phiy*y/solarZ) * (r0**(beta+1) / (R**beta)) * (vphi_z / (sigmag_z**2)) 
    
    # A_arr = 3*G_const*phiQ_arr*Mdotacc_arr / (2 * sigmag_arr**3 * np.log(xmax))  
    # A = 3*G_const*phiQ_z*Mdotacc_z / (2 * sigmag_z**3 * np.log(xmax)) 

    # Establishing Zmin and ZCGM
    Zmin = 0.01
    
    # # ZCGM_func = interp1d([9, 10.5], [0.05, 0.2], kind = "linear")
    # if logMstar <= 9: # For low-mass galaxies
    #     ZCGM = 0.05 # Sharda+2024 Sec. 2.2.3
    # elif logMstar >= 10.5:
    #     ZCGM = 0.2 # Sharda+2024 Sec. 2.2-0.7
    # else: # For intermediate-mass galaxies, interpolate value of ZCGM
    #     line_params = line_func(9, 0.05, 10.5, 0.2) # Getting slope and y-intercept of line
    #     ZCGM = line_params[0]*logMstar + line_params[1]

    # Defining common groups of terms
    Toomre_denom = A - 2*beta*(P + 2*beta)
    GMC_denom = A - (1+beta)*(1+P+beta)
    c1power = -0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower1 = 0.5*(np.sqrt(P**2 + 4*A) - P)
    solpower2 = 0.5*(- np.sqrt(P**2 + 4*A) - P)  

    if xmin <= x_b and xmax <= x_b: # Entire galactic disc in Toomre regime
        regime = "Toomre"
        print("Entire galactic disc in Toomre regime")

        # Calculating lower bound for c1
        c1_lowbound_Toomre = (Zmin - (S*(xmax**(2*beta)) / Toomre_denom)) * xmax**c1power
        
        # Currently do not have a case for P = 0 - need to solve equation derived from boundary condition
        if P == 0:
            print("No implemented P = 0 case for entire galactic disc in Toomre regime - no upper bound for c1")
        elif logMstar >= 10.5: # Calculating upper bound using c1 from Piyush's code and Zr0 from Sharda+2024
            print("Zr0 set by source and accretion - Massive galaxy")

            # Calculate c1
            c1_uppbound_Toomre = (A*S*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta)*\
                    (-2.*P - 4.*beta) - 4.*np.sqrt(4.*A + P**2)*S*beta**2 + \
                    A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                    (2.*A - 8.*beta**2 - 4.*beta*P) - \
                    2.*np.sqrt(4.*A + P**2)*S*beta*P + P*S*(4.*beta**2 + 2.*beta*P))/\
                    (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                    (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*(A - 4.*beta**2 - 2.*beta*P))
            
            # Calculate Zr0
            Zr0_lowbound_Tomore = S/A
            Zr0_uppbound_Toomre = S/A
        else:
            print("Zr0 set by diffusion and advection - Low-mass galaxy")
            # Calculate c1
            c1_uppbound_Toomre = (2.*A**2*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM - \
                                4.*np.sqrt(4.*A + P**2)*S*beta**2 + \
                                P*S*beta*(2.*np.sqrt(4.*A + P**2) - 4.*np.sqrt(4.*A + P**2)*\
                                xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta) + 4.*beta)\
                                + P**2*(S*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2) + 2.*beta)*\
                                (-2.*np.sqrt(4.*A + P**2) - 4.*beta) - 2.*S*beta - \
                                8.*np.sqrt(4.*A + P**2)*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                                (1.*beta**2 + 0.5*beta*P)) + P**3*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                (-2.*S*xmax**(2.*beta) - 8.*ZCGM*beta**2 - 4.*ZCGM*beta*P) + \
                                A*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                                (2.*P**3*ZCGM + 2.*P**2*np.sqrt(4.*A + P**2)*ZCGM - \
                                4.*S*xmax**(2.*beta)*beta + \
                                P*(-2.*S*xmax**(2.*beta) - 8.*ZCGM*beta**2 - 4.*ZCGM*beta*P)))/\
                                (A**2*(1.*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P*(-5. + 5.*xmax**(1.*np.sqrt(4.*A + P**2)))) + \
                                A*(2.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P**3*(-2. + 2.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                                P*(20. - 20.*xmax**(1.*np.sqrt(4.*A + P**2)))*(1.*beta**2 + 0.5*beta*P) + \
                                np.sqrt(4.*A + P**2)*(-4. - 4.*xmax**(1.*np.sqrt(4.*A + P**2)))*\
                                (1.*beta**2 + 0.5*beta*P)) + P**2*(np.sqrt(4.*A + P**2)*\
                                (-8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))) + P*(8. - 8.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                                (1.*beta**2 + 0.5*beta*P))
            
            # Calculate Zr0
            Zr0_lowbound_Toomre = ((P**2)*S + A*(2*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2) + S) - \
                                4*S*(beta**2) + P*S*(np.sqrt(4*A + P**2) + 2*beta) - \
                                4*c1_lowbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta**2) + P*beta)) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta**2) - 2*P*beta))
            Zr0_uppbound_Toomre = ((P**2)*S + A*(2*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2) + S) - \
                                4*S*(beta**2) + P*S*(np.sqrt(4*A + P**2) + 2*beta) - \
                                4*c1_uppbound_Toomre*P*np.sqrt(4*A + P**2)*(2*(beta**2) + P*beta)) / \
                                ((A + P*(P + np.sqrt(4*A + P**2)))*(A - 4*(beta**2) - 2*P*beta))
                    
        if c1_uppbound_Toomre < c1_lowbound_Toomre: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for Toomre - decrease Zmin")
        
        c1_arr = np.linspace(c1_lowbound_Toomre, c1_uppbound_Toomre, 10) # Create array of values for c1
        Zr0_arr = np.linspace(Zr0_lowbound_Toomre, Zr0_uppbound_Toomre, 10) # Create array of values for Zr0

        # Empty lists/dictionaries to save data
        normZ_dict = {} # Each key corresponds to different c1
        dnormZdx_dict = {}
        d2normZdx2_dict = {}
        teqbm_dict = {}
        Omega0_arr = []
        k_dict = {}
        sg_dict = {}
        cstar_dict = {}
        sstar_dict = {}
        adv_term_dict = {}
        diffus_term_dict = {}
        source_term_dict = {}
        acc_term_dict = {}
        metgrad_polyfit_arr = []
        log10Zr0_polyfit_arr = []
        
        for i, (c1, Zr0) in enumerate(zip(c1_arr, Zr0_arr)):
            # Calculating normZ 
            normZ = (S*(x_arr**(2*beta)) / Toomre_denom) + c1*(x_arr**solpower1) + \
                    (Zr0 - (S/Toomre_denom) - c1) * (x_arr**solpower2) # Sharda+2024 Eqn 29
            normZ_dict[i] = normZ # Saving normZ
            
            # Calculating teqbm
            Omega0 = vphi_z/r0 * ((r0/R)**beta) # In s^-1
            k = x_arr / (x_arr**beta)
            sg = (x_arr**beta) / x_arr
            cstar = 1 / (x_arr**2)
            sstar = x_arr**(2*(beta - 1))
            dnormZdx = (2*beta*S*(x_arr**(2*beta - 1)) / Toomre_denom) + solpower1*c1*(x_arr**(solpower1 - 1)) + \
                    solpower2*(Zr0 - S/Toomre_denom - c1)*(x_arr**(solpower2 - 1))
            d2normZdx2 = (2*beta*(2*beta - 1)*S*(x_arr**(2*beta - 2)) / Toomre_denom) + \
                        solpower1*(solpower1 - 1)*c1*(x_arr**(solpower1 - 2)) + \
                        solpower2*(solpower2 - 1)*(Zr0 - S/Toomre_denom - c1)*(x_arr**(solpower2 - 2))

            adv_term = np.abs((P * dnormZdx) / x_arr)
            diffus_term = np.abs((dnormZdx + x_arr*d2normZdx2) / x_arr)
            source_term = np.abs(S*sstar)
            acc_term = np.abs(normZ*A*cstar)
            teqbm_denom = normZ*sg*T

            teqbm = ((Omega0*(adv_term + diffus_term + source_term + acc_term) / teqbm_denom)**(-1)).cgs # In s

            Omega0_arr.append(Omega0)
            k_dict[i] = k
            sg_dict[i] = sg
            cstar_dict[i] = cstar
            sstar_dict[i] = sstar
            dnormZdx_dict[i] = dnormZdx
            d2normZdx2_dict[i] = d2normZdx2
            adv_term_dict[i] = adv_term
            diffus_term_dict[i] = diffus_term
            source_term_dict[i] = source_term
            acc_term_dict[i] = acc_term
            teqbm_dict[i] = teqbm

            # Calculating metallicity gradient using polyfit
            metgrad, log10Zr0 = np.polyfit(x_arr, np.log10(normZ.value), deg = 1)
            metgrad_polyfit_arr.append(metgrad)
            log10Zr0_polyfit_arr.append(log10Zr0)

    elif xmin > x_b and xmax > x_b: # Entire galactic disc in GMC regime
        regime = "GMC"
        print("Entire galactic disc in GMC regime")
        
        # Calculating lower of c1
        c1_lowbound_GMC = (Zmin - (S*(xmax**(1+beta)) / GMC_denom)) * xmax**c1power

        # Using c1 upper bound values from Piyush's code
        if Fsigma_z == 0:
            print("Zr0 set by diffusion and source - No transport")

            # Calculating c1
            c1_uppbound_GMC = ((2.*A + P*(1.*P + 1.*np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                            (-1.*P*ZCGM + (0.5*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                            (4.*A + P**2 + P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta) - \
                            4.*(1. + beta)**2))/((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                            (A + P*(-1. - 1.*beta) - 1.*(1. + beta)**2)) + (0.25*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                            xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*(4.*A + P**2 + \
                            P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta) - 4.*(1. + beta)**2))/\
                            ((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                            (A + P*(-1. - 1.*beta) - 1.*(1. + beta)**2)) - \
                            (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                            (A - 1.*(1. + beta)*(1. + P + beta)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                            xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*P*S*xmax**(1. + beta))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*S*xmax**(1. + beta)*(1. + beta))/(A - 1.*(1. + beta)*(1. + P + beta))))/\
                            (A*P*(3. - 3.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                            P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                            1.*A*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                            1.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
            
            # Calculate Zr0
            Zr0_lowbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta) - \
                            4*(1+beta)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta) - 2*(1+beta)**2))) / \
                            ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta) - (1+beta)**2)) # Sharda+2024 Eqn. C4
            Zr0_uppbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta) - \
                            4*(1+beta)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta) - 2*(1+beta)**2))) / \
                            ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta) - (1+beta)**2)) # Sharda+2024 Eqn. C4
            
        elif logMstar >= 10.5:
            print("Zr0 set by source and accretion - Massive galaxy")
            
            # Calculating c1
            c1_uppbound_GMC = (-1.*np.sqrt(4.*A + P**2)*S - 1.*P*np.sqrt(4.*A + P**2)*S + \
                            A*S*xmax**(1. + 0.5*P + 0.5*np.sqrt(4.*A + P**2) + beta)*\
                            (-2. - 2.*P - 2.*beta) - 2.*np.sqrt(4.*A + P**2)*S*beta - \
                            1.*P*np.sqrt(4.*A + P**2)*S*beta - 1.*np.sqrt(4.*A + P**2)*S*beta**2 + \
                            P*S*(1. + 1.*beta)*(1. + P + 1.*beta) + \
                            A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                            (2.*A + P*(-2. - 2.*beta) - 2.0000000000000004*(1. + 1.*beta)**2))/\
                            (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                            (-1. + 1.*A - 1.*P - 2.*beta - 1.*P*beta - 1.*beta**2))
            
            # Calculating Zr0
            Zr0_lowbound_GMC = S/A
            Zr0_uppbound_GMC = S/A

        else:
            print("Zr0 set by diffusion and advection - Low-mass galaxy")

            # Calculating c1
            c1_uppbound_GMC = ((A + P*(P + np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                            (-1.*P*ZCGM - (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                            (A - 1.*(1. + beta)*(1. + P + beta)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                            xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*P*S*xmax**(1. + beta))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*S*xmax**(1. + beta)*(1. + beta))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (A + P**2 - 1.*(1. + beta)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta)))/\
                            ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta) - 1.*(1. + beta)**2)) + \
                            (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (A + P**2 - 1.*(1. + beta)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta)))/\
                            ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta) - 1.*(1. + beta)**2))))/\
                            (A*P*(2.5 - 2.5*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                            P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 0.5*A*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 1.*P**2*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
            
            # Calculating Zr0
            Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta) - 2*(1+beta)**2) + \
                            S*(A + P**2 - (1+beta)**2 + P*(1 + np.sqrt(4*A + P**2) + beta))) / \
                            ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta) - (1+beta)**2)) # Sharda+2024 Eqn C3
            Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta) - 2*(1+beta)**2) + \
                            S*(A + P**2 - (1+beta)**2 + P*(1 + np.sqrt(4*A + P**2) + beta))) / \
                            ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta) - (1+beta)**2)) # Sharda+2024 Eqn C3
        
        if c1_uppbound_GMC < c1_lowbound_GMC: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for GMC - decrease Zmin")

        c1_arr = np.linspace(c1_lowbound_GMC, c1_uppbound_GMC, 10) # Create array of values for c1
        Zr0_arr = np.linspace(Zr0_lowbound_GMC, Zr0_uppbound_GMC, 10) # Create array of values for Zr0

        # Empty lists/dictionaries to save data
        normZ_dict = {} # Each key corresponds to different c1
        dnormZdx_dict = {}
        d2normZdx2_dict = {}
        teqbm_dict = {}
        Omega0_arr = []
        k_dict = {}
        sg_dict = {}
        cstar_dict = {}
        sstar_dict = {}
        adv_term_dict = {}
        diffus_term_dict = {}
        source_term_dict = {}
        acc_term_dict = {}
        metgrad_polyfit_arr = []
        log10Zr0_polyfit_arr = []
                    
        Omega0 = vphi_z/r0 * ((r0/R)**beta) # In s^-1

        for i, (c1, Zr0) in enumerate(zip(c1_arr, Zr0_arr)):
            # Calculating normZ 
            normZ = (S*(x_arr**(1+beta)) / GMC_denom) + c1*(x_arr**solpower1) + \
                    (Zr0 - S/GMC_denom - c1)*(x_arr**solpower2) # Sharda+2024 Eqn 30
            normZ_dict[i] = normZ # Saving normZ
            
            # Calculating teqbm
            k = x_arr / (x_arr**beta)
            sg = (x_arr**beta) / x_arr
            cstar = 1 / (x_arr**2)
            sstar = x_arr**(beta-1)
            dnormZdx = (S*(1+beta)*x_arr**(beta) / GMC_denom) + solpower1*c1*x_arr**(solpower1 - 1) + \
                    solpower2*(Zr0 - S/GMC_denom - c1)*x_arr**(solpower2 - 1)
            d2normZdx2 = (S*beta*(1+beta)*x_arr**(beta - 1) / GMC_denom) + solpower1*(solpower1 - 1)*c1*x_arr**(solpower1 - 2) + \
                        solpower2*(solpower2 - 1)*(Zr0 - S/GMC_denom - c1)*x_arr**(solpower2 - 2)

            adv_term = np.abs((P * dnormZdx) / x_arr)
            diffus_term = np.abs((dnormZdx + x_arr*d2normZdx2) / x_arr)
            source_term = np.abs(S*sstar)
            acc_term = np.abs(normZ*A*cstar)
            teqbm_denom = normZ*sg*T

            teqbm = ((Omega0*(adv_term + diffus_term + source_term + acc_term) / teqbm_denom)**(-1)).cgs # In s

            k_dict[i] = k
            sg_dict[i] = sg
            cstar_dict[i] = cstar
            sstar_dict[i] = sstar
            dnormZdx_dict[i] = dnormZdx
            d2normZdx2_dict[i] = d2normZdx2
            adv_term_dict[i] = adv_term
            diffus_term_dict[i] = diffus_term
            source_term_dict[i] = source_term
            acc_term_dict[i] = acc_term
            teqbm_dict[i] = teqbm

            # Calculating metallicity gradient using polyfit
            metgrad, log10Zr0 = np.polyfit(x_arr, np.log10(normZ.value), deg = 1)
            metgrad_polyfit_arr.append(metgrad)
            log10Zr0_polyfit_arr.append(log10Zr0)

    elif xmin <= x_b and xmax > x_b: # Inner disc in Toomre and outer disc in GMC
        regime = "Toomre and GMC" 
        print("Inner disc is Toomre, outer disc is GMC")

        # Calculating GMC part first which is the outer part of the disc
        # Calculating lower bound of c1
        c1_lowbound_GMC = (Zmin - (S*(xmax**(1+beta)) / GMC_denom)) * xmax**c1power
        
        # Using c1 upper bound values from Piyush's code
        if Fsigma_z == 0:
            print("Zr0 set by diffusion and source - No transport")

            # Calculating c1
            c1_uppbound_GMC = ((2.*A + P*(1.*P + 1.*np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                            (-1.*P*ZCGM + (0.5*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                            (4.*A + P**2 + P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta) - \
                            4.*(1. + beta)**2))/((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                            (A + P*(-1. - 1.*beta) - 1.*(1. + beta)**2)) + (0.25*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                            xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*(4.*A + P**2 + \
                            P*(-2. + np.sqrt(4.*A + P**2) - 2.*beta) - 4.*(1. + beta)**2))/\
                            ((A + 0.5*P*(P + np.sqrt(4.*A + P**2)))*\
                            (A + P*(-1. - 1.*beta) - 1.*(1. + beta)**2)) - \
                            (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                            (A - 1.*(1. + beta)*(1. + P + beta)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                            xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*P*S*xmax**(1. + beta))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*S*xmax**(1. + beta)*(1. + beta))/(A - 1.*(1. + beta)*(1. + P + beta))))/\
                            (A*P*(3. - 3.*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                            P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                            1.*A*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - \
                            1.*P**2*np.sqrt(4.*A + P**2)*(1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
            
            # Calculate Zr0
            Zr0_lowbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta) - \
                            4*(1+beta)**2) + c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta) - 2*(1+beta)**2))) / \
                            ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta) - (1+beta)**2)) # Sharda+2024 Eqn. C4
            Zr0_uppbound_GMC = (0.5 * (S*(4*A + P**2 + P*(-2 + np.sqrt(4*A + P**2) - 2*beta) - \
                            4*(1+beta)**2) + c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta) - 2*(1+beta)**2))) / \
                            ((A + 0.5*P*(P + np.sqrt(4*A + P**2))) * (A + P*(-1-beta) - (1+beta)**2)) # Sharda+2024 Eqn. C4
            
        elif logMstar >= 10.5:
            print("Zr0 set by source and accretion - Massive galaxy")
            
            # Calculating c1
            c1_uppbound_GMC = (-1.*np.sqrt(4.*A + P**2)*S - 1.*P*np.sqrt(4.*A + P**2)*S + \
                            A*S*xmax**(1. + 0.5*P + 0.5*np.sqrt(4.*A + P**2) + beta)*\
                            (-2. - 2.*P - 2.*beta) - 2.*np.sqrt(4.*A + P**2)*S*beta - \
                            1.*P*np.sqrt(4.*A + P**2)*S*beta - 1.*np.sqrt(4.*A + P**2)*S*beta**2 + \
                            P*S*(1. + 1.*beta)*(1. + P + 1.*beta) + \
                            A*P*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*ZCGM*\
                            (2.*A + P*(-2. - 2.*beta) - 2.0000000000000004*(1. + 1.*beta)**2))/\
                            (A*(P*(-1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) + 1.*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))*\
                            (-1. + 1.*A - 1.*P - 2.*beta - 1.*P*beta - 1.*beta**2))
            
            # Calculating Zr0
            Zr0_lowbound_GMC = S/A
            Zr0_uppbound_GMC = S/A

        else:
            print("Zr0 set by diffusion and advection - Low-mass galaxy")

            # Calculating c1
            c1_uppbound_GMC = ((A + P*(P + np.sqrt(4.*A + P**2)))*xmax**(0.5*P + 0.5*np.sqrt(4.*A + P**2))*\
                            (-1.*P*ZCGM - (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/\
                            (A - 1.*(1. + beta)*(1. + P + beta)) - (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*\
                            xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2)))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*P*S*xmax**(1. + beta))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*S*xmax**(1. + beta)*(1. + beta))/(A - 1.*(1. + beta)*(1. + P + beta)) + \
                            (1.*P*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (A + P**2 - 1.*(1. + beta)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta)))/\
                            ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta) - 1.*(1. + beta)**2)) + \
                            (0.5*(-1.*P - 1.*np.sqrt(4.*A + P**2))*S*xmax**(-0.5*P - 0.5*np.sqrt(4.*A + P**2))*\
                                (A + P**2 - 1.*(1. + beta)**2 + P*(1. + np.sqrt(4.*A + P**2) + beta)))/\
                            ((A + P*(P + np.sqrt(4.*A + P**2)))*(A + P*(-1. - 1.*beta) - 1.*(1. + beta)**2))))/\
                            (A*P*(2.5 - 2.5*xmax**(1.*np.sqrt(4.*A + P**2))) + \
                            P**3*(1. - 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 0.5*A*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))) - 1.*P**2*np.sqrt(4.*A + P**2)*\
                            (1. + 1.*xmax**(1.*np.sqrt(4.*A + P**2))))
            
            # Calculating Zr0
            Zr0_lowbound_GMC = (c1_lowbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta) - 2*(1+beta)**2) + \
                            S*(A + P**2 - (1+beta)**2 + P*(1 + np.sqrt(4*A + P**2) + beta))) / \
                            ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta) - (1+beta)**2)) # Sharda+2024 Eqn C3
            Zr0_uppbound_GMC = (c1_uppbound_GMC*P*np.sqrt(4*A + P**2)*(2*A + P*(-2 - 2*beta) - 2*(1+beta)**2) + \
                            S*(A + P**2 - (1+beta)**2 + P*(1 + np.sqrt(4*A + P**2) + beta))) / \
                            ((A + P*(P + np.sqrt(4*A + P**2)))*(A + P*(-1-beta) - (1+beta)**2)) # Sharda+2024 Eqn C3
        
        if c1_uppbound_GMC < c1_lowbound_GMC: # In the case lower bound is greater than upper bound
            warnings.warn("Invalid c1 range for GMC - decrease Zmin")

        c1_arr_GMC = np.linspace(c1_lowbound_GMC, c1_uppbound_GMC, 10)
        Zr0_arr = np.linspace(Zr0_lowbound_GMC, Zr0_uppbound_GMC, 10) # Create array of values for Zr0

        Omega0 = vphi_z/r0 * ((r0/R)**beta) # In s^-1

        # Dictionaries to store solutions for GMC part
        normZ_dict_GMC = {} # Each key corresponds to different c1
        dnormZdx_dict_GMC = {}
        d2normZdx2_dict_GMC = {}
        teqbm_dict_GMC = {}
        k_dict_GMC = {}
        sg_dict_GMC = {}
        cstar_dict_GMC = {}
        sstar_dict_GMC = {}
        adv_term_dict_GMC = {}
        diffus_term_dict_GMC = {}
        source_term_dict_GMC = {}
        acc_term_dict_GMC = {}

        c1_arr_Toomre = []

        for i, (c1, Zr0) in enumerate(zip(c1_arr_GMC, Zr0_arr)):
            # Calculating normZ for GMC part
            normZ_GMC = (S*(x_arr[x_arr > x_b]**(1+beta)) / GMC_denom) + c1*(x_arr[x_arr > x_b]**solpower1) + \
                    (Zr0 - S/GMC_denom - c1)*(x_arr[x_arr > x_b]**solpower2) # Sharda+2024 Eqn 30
            normZ_dict_GMC[i] = normZ_GMC # Saving normZ
            
            # Calculating teqbm
            k_GMC = x_arr[x_arr > x_b] / (x_arr[x_arr > x_b]**beta)
            sg_GMC = (x_arr[x_arr > x_b]**beta) / x_arr[x_arr > x_b]
            cstar_GMC = 1 / (x_arr[x_arr > x_b]**2)
            sstar_GMC = x_arr[x_arr > x_b]**(beta-1)
            dnormZdx_GMC = (S*(1+beta)*x_arr[x_arr > x_b]**(beta) / GMC_denom) + \
                        solpower1*c1*x_arr[x_arr > x_b]**(solpower1 - 1) + \
                        solpower2*(Zr0 - S/GMC_denom - c1)*x_arr[x_arr > x_b]**(solpower2 - 1)
            d2normZdx2_GMC = (S*beta*(1+beta)*x_arr[x_arr > x_b]**(beta - 1) / GMC_denom) + \
                            solpower1*(solpower1 - 1)*c1*x_arr[x_arr > x_b]**(solpower1 - 2) + \
                            solpower2*(solpower2 - 1)*(Zr0 - S/GMC_denom - c1)*x_arr[x_arr > x_b]**(solpower2 - 2)

            adv_term_GMC = np.abs((P * dnormZdx_GMC) / x_arr[x_arr > x_b])
            diffus_term_GMC = np.abs((dnormZdx_GMC + x_arr[x_arr > x_b]*d2normZdx2_GMC) / x_arr[x_arr > x_b])
            source_term_GMC = np.abs(S*sstar_GMC)
            acc_term_GMC = np.abs(normZ_GMC*A*cstar_GMC)
            teqbm_denom_GMC = normZ_GMC*sg_GMC*T

            teqbm_GMC = ((Omega0*(adv_term_GMC + diffus_term_GMC + source_term_GMC + acc_term_GMC) / teqbm_denom_GMC)**(-1)).cgs # In s

            # Saving GMC part
            k_dict_GMC[i] = k_GMC
            sg_dict_GMC[i] = sg_GMC
            cstar_dict_GMC[i] = cstar_GMC
            sstar_dict_GMC[i] = sstar_GMC
            dnormZdx_dict_GMC[i] = dnormZdx_GMC
            d2normZdx2_dict_GMC[i] = d2normZdx2_GMC
            adv_term_dict_GMC[i] = adv_term_GMC
            diffus_term_dict_GMC[i] = diffus_term_GMC
            source_term_dict_GMC[i] = source_term_GMC
            acc_term_dict_GMC[i] = acc_term_GMC
            teqbm_dict_GMC[i] = teqbm_GMC

            # Constraining c1 values for Toomre part using solution from GMC part
            # Calculating normZ at x_b to constrain c1 for Toomre part and using Zr0 from GMC
            normZ_xb_GMC = (S*(x_b**(1+beta)) / GMC_denom) + c1*(x_b**solpower1) + \
                        (Zr0 - S/GMC_denom - c1)*(x_b**solpower2) # normZ at x = x_b                      

            # At x_b, normZ from Toomre and GMC part must be the same - use this to calculate c1 for Toomre part
            c1_Toomre_fromGMC = (normZ_xb_GMC - (S*(x_b**(2*beta)) / Toomre_denom) - (Zr0 - S/Toomre_denom)*(x_b**solpower2)) \
                                / (x_b**solpower1 - x_b**solpower2)

            # Check if new c1 for Toomre part is less than lower bound from Zmin
            c1_lowbound_xb_Toomre = (Zmin - (S*(x_b**(2*beta)) / Toomre_denom)) * x_b**c1power # Lower bound for c1 in Toomre part at x_b
            if c1_Toomre_fromGMC < c1_lowbound_xb_Toomre:
                warnings.warn("New c1 from Toomre part is invalid")

            c1_arr_Toomre.append(c1_Toomre_fromGMC) # Append c1 for Toomre part

        # With c1 for Toomre part from GMC part, calculate normZ for xmin <= x_break <= x
        # Dictionaries to store solutions for GMC part
        normZ_dict_Toomre = {} # Each key corresponds to different c1
        dnormZdx_dict_Toomre = {}
        d2normZdx2_dict_Toomre = {}
        teqbm_dict_Toomre = {}
        k_dict_Toomre = {}
        sg_dict_Toomre = {}
        cstar_dict_Toomre = {}
        sstar_dict_Toomre = {}
        adv_term_dict_Toomre = {}
        diffus_term_dict_Toomre = {}
        source_term_dict_Toomre = {}
        acc_term_dict_Toomre = {}
        
        for i, (c1, Zr0) in enumerate(zip(c1_arr_Toomre, Zr0_arr)):
            # Calculating normZ
            normZ_Toomre = (S*(x_arr[x_arr <= x_b]**(2*beta)) / Toomre_denom) + \
                        c1*(x_arr[x_arr <= x_b]**solpower1) + \
                        (Zr0 - (S/Toomre_denom) - c1) * (x_arr[x_arr <= x_b]**solpower2) # Sharda+2024 Eqn 29
            normZ_dict_Toomre[i] = normZ_Toomre # Saving normZ

            # Calculating teqbm
            k_Toomre = x_arr[x_arr <= x_b] / (x_arr[x_arr <= x_b]**beta)
            sg_Toomre = (x_arr[x_arr <= x_b]**beta) / x_arr[x_arr <= x_b]
            cstar_Toomre = 1 / (x_arr[x_arr <= x_b]**2)
            sstar_Toomre = x_arr[x_arr <= x_b]**(2*(beta - 1))
            dnormZdx_Toomre = (2*beta*S*(x_arr[x_arr <= x_b]**(2*beta - 1)) / Toomre_denom) + \
                            solpower1*c1*(x_arr[x_arr <= x_b]**(solpower1 - 1)) + \
                            solpower2*(Zr0 - S/Toomre_denom - c1)*(x_arr[x_arr <= x_b]**(solpower2 - 1))
            d2normZdx2_Toomre = (2*beta*(2*beta - 1)*S*(x_arr[x_arr <= x_b]**(2*beta - 2)) / Toomre_denom) + \
                                solpower1*(solpower1 - 1)*c1*(x_arr[x_arr <= x_b]**(solpower1 - 2)) + \
                                solpower2*(solpower2 - 1)*(Zr0 - S/Toomre_denom - c1)*(x_arr[x_arr <= x_b]**(solpower2 - 2))

            adv_term_Toomre = np.abs((P * dnormZdx_Toomre) / x_arr[x_arr <= x_b])
            diffus_term_Toomre = np.abs((dnormZdx_Toomre + x_arr[x_arr <= x_b]*d2normZdx2_Toomre) / x_arr[x_arr <= x_b])
            source_term_Toomre = np.abs(S*sstar_Toomre)
            acc_term_Toomre = np.abs(normZ_Toomre*A*cstar_Toomre)
            teqbm_denom_Toomre = normZ_Toomre*sg_Toomre*T

            teqbm_Toomre = ((Omega0*(adv_term_Toomre + diffus_term_Toomre + source_term_Toomre + acc_term_Toomre) / teqbm_denom_Toomre)**(-1)).cgs # In s

            # Saving only Toomre part
            k_dict_Toomre[i] = k_Toomre
            sg_dict_Toomre[i] = sg_Toomre
            cstar_dict_Toomre[i] = cstar_Toomre
            sstar_dict_Toomre[i] = sstar_Toomre
            dnormZdx_dict_Toomre[i] = dnormZdx_Toomre
            d2normZdx2_dict_Toomre[i] = d2normZdx2_Toomre
            adv_term_dict_Toomre[i] = adv_term_Toomre
            diffus_term_dict_Toomre[i] = diffus_term_Toomre
            source_term_dict_Toomre[i] = source_term_Toomre
            acc_term_dict_Toomre[i] = acc_term_Toomre
            teqbm_dict_Toomre[i] = teqbm_Toomre

        # Combining Toomre and GMC solutions together
        # Dictionaries to store solutions for both Toomre and GMC part together
        normZ_dict = {}
        dnormZdx_dict = {}
        d2normZdx2_dict = {}
        teqbm_dict = {}
        k_dict = {}
        sg_dict = {}
        cstar_dict = {}
        sstar_dict = {}
        adv_term_dict = {}
        diffus_term_dict = {}
        source_term_dict = {}
        acc_term_dict = {}

        metgrad_polyfit_arr = []
        log10Zr0_polyfit_arr = []

        for i in range(len(c1_arr_GMC)):
            normZ_arr = np.array(list(normZ_dict_Toomre[i]) + list(normZ_dict_GMC[i]))
            teqbm_arr = np.array(list(teqbm_dict_Toomre[i].value) + list(teqbm_dict_GMC[i].value)) * u.s
            dnormZdx_arr = np.array(list(dnormZdx_dict_Toomre[i]) + list(dnormZdx_dict_GMC[i]))
            d2normZdx2_arr = np.array(list(d2normZdx2_dict_Toomre[i]) + list(d2normZdx2_dict_GMC[i]))
            k_arr = np.array(list(k_dict_Toomre[i]) + list(k_dict_GMC[i]))
            sg_arr = np.array(list(sg_dict_Toomre[i]) + list(sg_dict_GMC[i]))
            cstar_arr = np.array(list(cstar_dict_Toomre[i]) + list(cstar_dict_GMC[i]))
            sstar_arr = np.array(list(sstar_dict_Toomre[i]) + list(sstar_dict_GMC[i]))
            adv_term_arr = np.array(list(adv_term_dict_Toomre[i]) + list(adv_term_dict_GMC[i]))
            diffus_term_arr = np.array(list(diffus_term_dict_Toomre[i]) + list(diffus_term_dict_GMC[i]))
            source_term_arr = np.array(list(source_term_dict_Toomre[i]) + list(source_term_dict_GMC[i]))
            acc_term_arr = np.array(list(acc_term_dict_Toomre[i]) + list(acc_term_dict_GMC[i]))

            normZ_dict[i] = normZ_arr
            teqbm_dict[i] = teqbm_arr
            dnormZdx_dict[i] = dnormZdx_arr
            d2normZdx2_dict[i] = d2normZdx2_arr
            k_dict[i] = k_arr
            sg_dict[i] = sg_arr
            cstar_dict[i] = cstar_arr
            sstar_dict[i] = sstar_arr
            adv_term_dict[i] = adv_term_arr
            diffus_term_dict[i] = diffus_term_arr
            source_term_dict[i] = source_term_arr
            acc_term_dict[i] = acc_term_arr

            # Calculating metallicity gradient using polyfit
            metgrad, log10Zr0 = np.polyfit(x_arr, np.log10(normZ_arr), deg = 1)
            metgrad_polyfit_arr.append(metgrad)
            log10Zr0_polyfit_arr.append(log10Zr0)

    if regime == "Toomre" or regime == "GMC":
        return {'regime': regime, 'normZ_dict': normZ_dict, 'metgrad_polyfit': metgrad_polyfit_arr, 'log10Zr0_polyfit': log10Zr0_polyfit_arr, 'c1_arr': c1_arr, 'Zr0_arr': Zr0_arr, 
                'x': x_arr, 'x_b': x_b, 'logMstar': logMstar, 'k_dict': k_dict, 'sg_dict': sg_dict, 'cstar_dict': cstar_dict, 'sstar_dict': sstar_dict, 'dnormZdx_dict': dnormZdx_dict, 
                'd2normZdx2_dict': d2normZdx2_dict, 'teqbm_dict': teqbm_dict, 'T': T, 'P': P, 'S': S, 'A': A, 'Omega0_arr': Omega0_arr, 'adv_term': adv_term_dict, 'diffus_term': diffus_term_dict,
                'source_term': source_term_dict, 'acc_term': acc_term_dict, "tdep": tdep, "disc_outputs": disc_outputs, "z_index": z_index
                }
    
    elif regime == "Toomre and GMC":
        return {'regime': regime, 'normZ_dict': normZ_dict, 'normZ_dict_Toomre': normZ_dict_Toomre, 'normZ_dict_GMC': normZ_dict_GMC, 'metgrad_polyfit': metgrad_polyfit_arr, 'log10Zr0_polyfit': log10Zr0_polyfit_arr, 
                'c1_arr_Toomre': c1_arr_Toomre, 'c1_arr_GMC': c1_arr_GMC, 'Zr0_arr': Zr0_arr, 'x': x_arr, 'x_b': x_b, 'logMstar': logMstar, 'k_dict': k_dict, 'k_dict_Toomre': k_dict_Toomre, 'k_dict_GMC': k_dict_GMC, 
                'sg_dict': sg_dict, 'sg_dict_Toomre': sg_dict_Toomre, 'sg_dict_GMC': sg_dict_GMC, 'cstar_dict': cstar_dict, 'cstar_dict_Toomre': cstar_dict_Toomre, 'cstar_dict_GMC': cstar_dict_GMC, 
                'sstar_dict': sstar_dict, 'sstar_dict_Toomre': sstar_dict_Toomre, 'sstar_dict_GMC': sstar_dict_GMC, 'dnormZdx_dict': dnormZdx_dict, 'dnormZdx_dict_Toomre': dnormZdx_dict_Toomre, 
                'dnormZdx_dict_GMC': dnormZdx_dict_GMC, 'd2normZdx2_dict': d2normZdx2_dict, 'd2normZdx2_dict_Toomre': d2normZdx2_dict_Toomre, 'd2normZdx2_dict_GMC': d2normZdx2_dict_GMC, 'teqbm_dict': teqbm_dict, 
                'teqbm_dict_Toomre': teqbm_dict_Toomre, 'teqbm_dict_GMC': teqbm_dict_GMC, 'T': T, 'P': P, 'S': S, 'A': A, 'Omega0': Omega0, 'adv_term': adv_term_dict, 'diffus_term': diffus_term_dict, 
                'source_term': source_term_dict, 'acc_term': acc_term_dict, "tdep": tdep, "disc_outputs": disc_outputs, "z_index": z_index
                }