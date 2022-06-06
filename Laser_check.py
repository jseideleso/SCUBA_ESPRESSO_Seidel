##############################
#Script to check an ESPRESSO file for laser contamination, works for both 1UT and 4UT mode
#author: jseidel@eso.org
#date: June 2022
##############################

#python packages to import
import numpy as np
#these can also be substituted if not provided within SCUBA
import matplotlib.pyplot as plt #only for plotting
from astropy import constants as const #I only need c, can be hardcoded if astropy not available
import astropy.units as u #optional
import astropy.io.fits as fits #if the data is already read in differently in SCUBA this is also optional

#-----------------------------
#modular parts of the code
def estimate_bg_flux_peak(wave,flux,wave_mask, wave_full, line_centre, poly_deg = 1):
    """
    function to calculate the background flux in a sodium range and the peak flux
    INPUT:
        wave: numpy array containing the wavelength in Angstrom
        flux: numpy array containing the flux extracted from an S1D
        wave_mask: numpy array of indices delimiting the modeling region
        wave_full: numpy array of indices delimiting the full sodium region
        line_centre: float, line centre wavelength in Angstrom
        poly_deg: degree of the polynomial to fit, currently linear
    OUTPUT:
        bg_flux: numpy array containting the background flux in the modelled wavelength region
        peak_flux: float, peak flux in the modelled wavelength region
    """
    
    param = np.polyfit(wave[wave_mask],flux[wave_mask],deg=poly_deg)
    fp = np.poly1d(param)
    #background flux at line centre
    bg_flux = fp(line_centre)
    #peak, taking into account background flux shape
    peak_arg = np.argmax(flux[wave_full]-fp(wave[wave_full]))
    peak_flux = flux[wave_full][peak_arg]
    return bg_flux, peak_flux
    
def create_wave_regions(wave, laser_wave, laser_window):
   """
   function to create masks for wavelength regions centered around a central wavelength
   INPUT:
       wave: numpy array containing the wavelength grid in Angstrom
       laser_wave: float, central wavelength in Angstrom
       laser_window: float, specified wavelength distance to the flux peak
    OUTPUT:
       wave_range: array of mask indices
   """
   wave_range = np.all((wave>laser_wave-laser_window,wave<laser_wave+laser_window), axis=0)
   return wave_range
       
def create_masked_wave_regions(wave, wave_laser, wave_window, wave_exclusion):
    """
    function to create more complicated masking regions excluding the sodium lines
    INPUT:
        wave: numpy array containing the wavelength grid in Angstrom
        wave_laser: float, central wavelength in Angstrom
        wave_window: float, offset from central wavelength to be included
        wave_exclusion: float, offset from central wavelength to be excluded
    OUTPUT:
        masked_wave: numpy array containing masked indices
    """
    part1_range = np.all((wave<wave_laser-wave_window, wave>wave_laser-wave_exclusion), axis=0)
    part2_range = np.all((wave>wave_laser+wave_window, wave<wave_laser+wave_exclusion), axis=0)
    masked_wave = np.any((part1_range,part2_range), axis=0)
    return masked_wave
    
### reversing the barycentric correction
def rev_berv(wave, berv):
    """
    function to reverse the BERV correction done automatically by ESPRESSO, we need everything in the
    observer's rest frame
    -uses speed of light from astropy package, this can also be hardcoded if needed
    INPUT:
        wave: numpy array containing the wavelength grid in Angstrom
        berv: float, BERV velocity in km/s
    OUTPUT:
        wave (numpy array) in the observer's rest frame
    """
    vdc = berv/const.c.to('km/s')
    return wave - wave * vdc
            


def read_from_file(path):
    """
    function that reads in all needed values from an ESPRESSO S1D fits file. This works currently for all science S1D files, both for 1UT and 4UT mode
    This function will most likely have to be adapted to SCUBA needs
    INPUT:
        path: string, location of the S1D file
    OUTPUT:
        flux: numpy array containing the S1D flux
        wave: numpy array containing the corresponding wavelength grid in Angstrom
    """

    hdul = fits.open(path)
    try:
        data = hdul[1].data
        berv = hdul[0].header['HIERARCH ESO QC BERV']*(u.km/u.s)
        wave = rev_berv(data.field(1),berv)
        flux = data.field(2)
        hdul.close()
        del hdul
    #clean up in case things went wrong while the file was open
    except:
        hdul.close()
        del hdul
    
    return flux, wave
    
    
#main function to be integrated into SCUBA
def check_laser_contamination(path):
    """
    MAIN function, takes path as argument currently, this can be adapted to directly feed wavelength, flux and berv from an S1D file if SCUBA already reads it anyway. Just get rid of read_from_file() module.
    INPUT:
        path: string, pointing to ESPRESSO S1D science file
    OUTPUT:
        contaminated: boolean, indicating if the spectrum was laser contaminated (TRUE == contamination)
    """

    #some wavelength values that remain constant throughout the script
    #the limits are taken from the document provided by Matias "ESPRESSO_laser_contamination_report_v2"
    CONST_LASER_SODIUM = 5891.5912   # Wavelength of the laser sodium line in Angstrom
    CONST_NON_LASER_SODIUM = 5897.558147 # Wavelength of the "non-laser" sodium line in Angstrom
    CONST_HW_RANGE = 0.05 # half-width of the wavelength range in which we check for the sodium lines
    CONST_SODIUM_REGION = 0.3 #exclusion region around the sodium lines in Angstrom

    #gets flux and wave from file, corrects wave for BERV
    flux, wave = read_from_file(path)
    
    #creates the full wavelength range around the sodium lines, both laser and non laser
    wave_laser = create_wave_regions(wave, CONST_LASER_SODIUM, CONST_HW_RANGE)
    wave_non_laser = create_wave_regions(wave, CONST_NON_LASER_SODIUM, CONST_HW_RANGE)
    
    ### Create masks for the surrounding of the sodium lines but excluding the lines, will be used for the background flux estimation
    wave_mask_laser = create_masked_wave_regions(wave,CONST_LASER_SODIUM,CONST_HW_RANGE,CONST_SODIUM_REGION)
    wave_mask_non_laser = create_masked_wave_regions(wave,CONST_NON_LASER_SODIUM,CONST_HW_RANGE,CONST_SODIUM_REGION)
 
    #Estimate the background flux using a first-order polynomial at the position of the lines
    #and the peak flux of the laser sodium lines, taking background flux shape into account
    flux_bg_laser, peak_laser = estimate_bg_flux_peak(wave,flux,wave_mask_laser, wave_laser, CONST_LASER_SODIUM, 1)
    flux_bg_non_laser, peak_non_laser = estimate_bg_flux_peak(wave,flux,wave_mask_non_laser, wave_non_laser, CONST_NON_LASER_SODIUM, 1)
    
    #Compute the ration between the peak flux of the laser sodium line and the background.
    #This checks if the sodium line is present in the spectrum and is therefore reasonamble to compute the ration of the sodium lines (peak_ratio)
    laser_ratio = peak_laser/flux_bg_laser

    # Compute the ratio between peak fluxes of sodium lines, the background flux is taken into account
    peak_ratio = (peak_laser-flux_bg_laser)/(peak_non_laser-flux_bg_non_laser)
    
    #Is there a sodium line and if so, is there contamination
    #If laser_ratio is small, indicating there are no sodium lines, the peak_ratio value is meaningless, as it is a ratio of two numbers close to zero which are defined by noise and/or fitting inaccuracy
    if laser_ratio > 1.2 and peak_ratio > 5.0:
        contaminated = True
    else:
        contaminated = False
    return contaminated



##TESTING
if __name__ == "__main__":
    #this is the input needed from SCUBA, path to S1D ESPRESSO file
    # alterantively if header and flux + wave is read in elsewhere, no need for function read_from_file()
    data_path = '/Volumes/SEIDEL ESO SHARE/Data/WASP-121/4UT/2018-12-01/r.ESPRE.2018-12-01T05_31_39.106_S1D_A.fits'


    print(check_laser_contamination(data_path))


