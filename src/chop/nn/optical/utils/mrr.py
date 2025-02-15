"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-07-18 00:03:04
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-07-18 00:03:05
"""

import numpy as np


__all__ = [
    "MORRConfig_20um_MQ",
    "MRRConfig_5um_HQ",
    "MRRConfig_5um_MQ",
    "MRRConfig_5um_LQ",
    "MORRConfig_10um_MQ",
]


class MORRConfig_20um_MQ:
    attenuation_factor = 0.8578
    coupling_factor = 0.8985
    radius = 20000  # nm
    group_index = 2.35316094
    effective_index = 2.35
    resonance_wavelength = 1554.252  # nm
    bandwidth = 0.67908  # nm
    quality_factor = 2288.7644639


class MRRConfig_5um_HQ:
    attenuation_factor = 0.987
    coupling_factor = 0.99
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 0.2278  # nm
    quality_factor = 6754.780509


class MRRConfig_5um_MQ:
    attenuation_factor = 0.925
    coupling_factor = 0.93
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 1.5068  # nm
    quality_factor = 1021.1965755


class MRRConfig_5um_LQ:
    attenuation_factor = 0.845
    coupling_factor = 0.85
    radius = 5000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 2.522  # nm
    quality_factor = 610.1265


class MORRConfig_10um_MQ:
    attenuation_factor = 0.8578
    coupling_factor = 0.8985
    radius = 10000  # nm
    group_index = 2.35316094
    effective_index = 2.4
    resonance_wavelength = 1538.739  # nm
    bandwidth = 1.6702  # nm
    quality_factor = 1213.047
