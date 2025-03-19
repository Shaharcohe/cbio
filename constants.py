from typing import Literal
trisomy_type_21 = 21
trisomy_type_13 = 13
trisomy_type_18 = 18
trisomy_types = [trisomy_type_21, trisomy_type_13, trisomy_type_18]

chr21_frac = 0.0147
chr18_frac = 0.025
chr13_frac = 0.035
genome_size = 3.2 * 1e9
read_length = 167

# https://pmc.ncbi.nlm.nih.gov/articles/PMC7163439/ - read length maternal / fetal
read_length_maternal = 167 # https://www.sciencedirect.com/science/article/pii/S1040842821002432
read_length_fetal = 143
read_length_std = 15

trisomy_prior = {
    trisomy_type_21: 1 / 700, # https://www.health.state.mn.us/diseases/cy/downsyndrome.html
    trisomy_type_13: 1 / 10000, # https://www.health.state.mn.us/diseases/cy/trisomy13.html#:~:text=Prevalence,and%20sperm%20in%20healthy%20parents.
    trisomy_type_18: 1 / 2000, # https://www.health.state.mn.us/diseases/cy/trisomy18.html#:~:text=Prevalence,and%20sperm%20in%20healthy%20parents.
}

ff_mean = {
    4: 0.01,
    5: 0.012,
    6: 0.02,
    7: 0.04,
    8: 0.06,
    9: 0.079,
    10: 0.082,
    11: 0.0805,
    12: 0.08,
    13: 0.09,
    14: 0.081,
    15: 0.08,
    16: 0.081,
    17: 0.082,
    18: 0.082,
    19: 0.081,
    20: 0.09,
    21: 0.098,
    22: 0.10,
    23: 0.10,
    24: 0.119,
    25: 0.122,
    26: 0.122,
    27: 0.14,
    28: 0.15,
    29: 0.16,
    30: 0.18,
    31: 0.18,
    32: 0.187,
    33: 0.21,
    34: 0.198,
    35: 0.208,
    36: 0.208,
    37: 0.207,
    38: 0.298,
    39: 0.202,
}

ff_std = {
    4: 0.01852,
    5: 0.01111,
    6: 0.01704,
    7: 0.02889,
    8: 0.03037,
    9: 0.03852,
    10: 0.03704,
    11: 0.03704,
    12: 0.04296,
    13: 0.04296,
    14: 0.03037,
    15: 0.03519,
    16: 0.03556,
    17: 0.03074,
    18: 0.03556,
    19: 0.03481,
    20: 0.04296,
    21: 0.04074,
    22: 0.03852,
    23: 0.04519,
    24: 0.05185,
    25: 0.04444,
    26: 0.04519,
    27: 0.05111,
    28: 0.04444,
    29: 0.05333,
    30: 0.05556,
    31: 0.06667,
    32: 0.0644,
    33: 0.08074,
    34: 0.08074,
    35: 0.08889,
    36: 0.05629,
    37: 0.08444,
    38: 0.1037,
    39: 0.04074,
}
feature_age = 'gestational_age'
feature_x = 'total_chromosome_reads'
feature_n = 'total_reads'
feature_ff = 'fetal_fraction'
feature_trisomy_label = 'trisomy_label'
feature_trisomy_type = 'trisomy_type'
feature_coverage_depth = 'coverage_depth'



def get_c(trisomy: int):
    if trisomy == trisomy_type_21:
        c = chr21_frac
    elif trisomy == trisomy_type_18:
        c = chr18_frac
    elif trisomy == trisomy_type_13:
        c = chr13_frac
    else:
        raise ValueError('invalid trisomy parameter.')
    return c

number_of_participants = 500
coverage_depths = [(1/30),(1/15), 0.1, 0.2, 1, 5, 10, 15, 30]
pregnancy_weeks = list(range(5, 16))

plot_coverage_depths = [(1/30),(1/15), 0.1, 1, 10, 15, 30]