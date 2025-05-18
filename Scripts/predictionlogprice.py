import numpy as np

def predire_log_price(logement, coefficients, variables):

    if len(coefficients) != len(variables):
        raise ValueError("Le nombre de coefficients doit correspondre au nombre de variables.")

    valeur = 0
    for coef, var in zip(coefficients, variables):
        valeur += coef * logement.get(var, 0)  # .get(var, 0) évite les KeyError

    return valeur
