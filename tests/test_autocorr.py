import numpy as np

from invrg.autocorr import autocorrelation, calc_error


def test_autocorrelation_is_normalized():
    x = np.random.default_rng(0).normal(size=4096)
    acf = autocorrelation(x)
    assert acf[0] == 1.0
    assert len(acf) == 2048


def test_integrated_autocorrelation_time_of_ar1_process():
    phi = 0.9
    rng = np.random.default_rng(0)
    noise = rng.normal(size=2**18)
    x = np.empty_like(noise)
    x[0] = 0.0
    for i in range(1, len(x)):
        x[i] = phi * x[i - 1] + noise[i]
    _, _, tau_int = calc_error(x)
    assert abs(tau_int - (1 + phi) / (2 * (1 - phi))) < 2.0  # exact value 9.5
