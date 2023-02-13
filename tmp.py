import logging
import os

def hypotenuse(a, b):
    """Compute the hypotenuse"""
    return (a**2 + b**2)**0.5

kwargs = {'a':3, 'b':4, 'c':hypotenuse(3, 4)}
logging.basicConfig(level=logging.INFO, filename='sample.log')
logging.info("Hypotenuse of {a}, {b} is {c}".format(**kwargs))
