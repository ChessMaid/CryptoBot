# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 22:50:29 2021

@author: Cedric
"""


def calc(n, lev, funding, quot, res):
    monthly_funding = (1 - funding)**(30*24/8)
    monthly_quot    = (1 + lev * (quot - 1))**(30*24*60*60/res)
    
    monthly = monthly_funding * monthly_quot
    
    return (monthly**(n+1) - 1)/(monthly - 1), monthly**n

if __name__ == "__main__":
    n       = 6
    lev     = 3
    funding = 0.0003
    quot    = 1.00015
    res     = 900
    
    facs = calc(n, lev, funding, quot, res)
    
    income  = 350
    in_bank = 12000
    
    print(facs)
    print(income * facs[0], in_bank * facs[1])
    