from scipy.stats import chi2
from typing import Literal, Callable, Optional
import numpy as np
import scipy.stats as stats
import math
import constants as const
from scipy.integrate import quad



class BayesianModel:
    def __init__(self,
                 mode: Literal['bayesian', 'regular'],
                 significance_level: float = 0.05,
                 incorporate_prior: bool = False,
                 trisomy_prior: Optional[float] = None):
        self._mode = mode
        self._incorporate_prior = incorporate_prior
        if incorporate_prior and (trisomy_prior is None or not (1 > trisomy_prior > 0)):
            raise ValueError('trisomy_prior must be an integer between 0 and 1')
        self._trisomy_prior = trisomy_prior
        self._significance_level = significance_level


    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: Literal['bayesian', 'regular']):
        if mode not in ['bayesian', 'regular']:
            raise ValueError(f"invalid mode: {mode}")
        self._mode = mode


    def likelihood_ratio(self, x: int, n: int, gestational_age: int, c: float):
        if not (5 <= gestational_age <= 38):
            print("Gestational age must be between 5 and 38")
        if self.mode == "bayesian":
            return self._bayesian_likelihood(x, n, gestational_age, c)
        elif self.mode == 'regular':
            return self._regular_likelihood(x, n, gestational_age, c)
        else:
            raise ValueError("invalid likelihood ratio mode.")

    def _ff_probability(self,
                        x: int,
                        ff: float,
                        n: int,
                        gestational_age: int,
                        c: float):
        ff_prob = stats.norm.logpdf(ff, const.ff_mean[gestational_age], const.ff_std[gestational_age])
        try:
            return math.exp(x * math.log(1 + ff / 2) + (n - x) * math.log(1 - c * (1 + ff / 2))
                            + ff_prob - (n-x) * math.log(1 - c))
        except OverflowError:
            return float('inf')


    def _bayesian_likelihood(self, x, n, gestational_age, c):
        p = lambda ff: self._ff_probability(x, ff, n, gestational_age, c)
        integral, _ = quad(p, 0, 1)  # Integrate from 0 to 1 (valid fetal fraction range)
        return integral

    def _x_log_likelihood(self, x: int,
                          ff: float,
                          n: int,
                          c: float,
                          trisomy: bool):
        if trisomy:
            chr_prob = c * (1 + ff / 2)
        else:
            chr_prob = c
        mean = n * chr_prob
        std_dev = np.sqrt(n * chr_prob * (1 - chr_prob))
        return stats.norm.logpdf(x, mean, std_dev)

    def _regular_likelihood(self, x, n, gestational_age, c, log = False):
        ff = const.ff_mean[gestational_age]
        scale = 1 + (ff / 2)

        try:
            ll = x * math.log(scale) * (n-x) * math.log((1 - c * scale) / (1 - c))
            if log:
                return ll
            else:
                return math.exp(ll)
        except OverflowError:
            return float('inf')

    def predict(self, row) -> int:
        x = row[const.feature_x]
        n = row[const.feature_n]
        gestational_age = row[const.feature_age]
        trisomy = row[const.feature_trisomy_type]
        if self.mode == "bayesian":
            D = -2 * math.log(self._bayesian_likelihood(x, n, gestational_age, const.get_c(trisomy)))
        elif self.mode == 'regular':
            D = -2 * self._regular_likelihood(x, n, gestational_age, const.get_c(trisomy), True)
        return 1 if chi2.sf(D, df=1) < self._significance_level else 0

    def score(self, row) -> int:
        x = row[const.feature_x]
        n = row[const.feature_n]
        gestational_age = row[const.feature_age]
        trisomy = row[const.feature_trisomy_type]
        return self.likelihood_ratio(x, n, gestational_age, const.get_c(trisomy))
