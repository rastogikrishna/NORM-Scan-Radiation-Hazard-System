# formula_engine.py


# Dose Rate Calculation
def calculate_dose_rate(ra, th, k):

    dose_rate = (
        (0.043 * ra) +
        (0.666 * th) +
        (0.047 * k)
    )

    return round(dose_rate, 3)


# Radium Equivalent Activity
def calculate_raeq(ra, th, k):

    raeq = (
        ra +
        (1.43 * th) +
        (0.077 * k)
    )

    return round(raeq, 3)


# External Hazard Index
def calculate_hex(ra, th, k):

    return round(
        (ra / 370) +
        (th / 259) +
        (k / 4810),
        3
    )


# Internal Hazard Index
def calculate_hin(ra, th, k):

    return round(
        (ra / 185) +
        (th / 259) +
        (k / 4810),
        3
    )


# Annual Effective Dose
def calculate_aed(dose_rate):

    din = 1.4 * dose_rate

    aed = (
        ((dose_rate * 0.2) + (din * 0.8))
        * 8760
        * 0.7e-6
    )

    return round(aed, 3)


# Excess Lifetime Cancer Risk
def calculate_elcr(aed):

    elcr = aed * 70 * 0.05

    return round(elcr, 3)