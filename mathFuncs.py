#!/usr/bin/python3
import numpy as np

def vectorDot(v1, v2):
    """Return dot product of vectors v1 and v2

    Different from np.dot, np.inner in that if the input is an array of arrays, returns the right array."""

    return np.sum(np.multiply(v1,v2)) if isinstance(v1, list) or v1.ndim < 2 else np.sum(np.multiply(v1,v2), axis=1)

def cosTheta(x,y,z):
    """ Returns the cos(theta) of the cartezian vector (x,y,z) in radians"""
    return z / np.sqrt(x**2 + y**2 + z**2)

def angleBetween(v1, v2):
    """ Returns the angle in radians between vectors v1 and v2 """

    v1norms = np.linalg.norm(v1) if isinstance(v1, list) or v1.ndim < 2 else np.linalg.norm(v1, axis = 1)
    v2norms = np.linalg.norm(v2) if isinstance(v2, list) or v2.ndim < 2 else np.linalg.norm(v2, axis = 1)

    return np.arccos(vectorDot(v1, v2) / v1norms / v2norms)

def lorentzDot(v1, v2):
    """ Perform 4 vector dot product
    Assume time like value at end"""

    multiplied = np.multiply(v1,v2)

    if isinstance(v1, list) or v1.ndim < 2:
        return multiplied[3] - np.sum(multiplied[:3])

    return multiplied[:, 3] - multiplied[:, 0] - multiplied[:, 1] - multiplied[:, 2]

def mass2(lorentzVector):
    """Get mass of lorentz vector squared"""
    return lorentzDot(lorentzVector, lorentzVector)

def mass(lorentzVector):
    """Get mass of lorentz vecot"""
    return np.sqrt(mass2(lorentzVector))

def cosHelicity1(grandParent, parent, daughter):
    """ Calculate cosine helicity of the daughter.
    grandParent, parent, daughter are the 4 vectors of the particles in any frame
    Taken from BAD522 v6, page 120, eq. 141
    grandparent, parent and daughter are P, Q, D in that equation
    """
    return (lorentzDot(grandParent,daughter)*mass2(parent) - lorentzDot(grandParent,parent)*lorentzDot(parent,daughter))/np.sqrt((lorentzDot(grandParent,parent)**2 - mass2(parent)*mass2(grandParent))*(lorentzDot(parent,daughter)**2 - mass2(parent)*mass2(daughter)))

def effError(nom, denom):
    """Calculate efficiency error using binomial distribution
    eff = nom / denom

    Does not represent errors close to 0 or 1!!!
    Can be extended to include edge cases with CDF note 5894"""

    nom, denom = np.asarray(nom, dtype=float), np.asarray(denom)
    eff = nom/denom

    return np.sqrt(eff*(1-eff)/denom)
