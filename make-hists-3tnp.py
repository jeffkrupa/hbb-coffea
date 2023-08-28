#!/usr/bin/python  

import os, sys
import subprocess
import json
import uproot3
import awkward as ak
import numpy as np
from coffea import processor, util, hist
import pickle

n2ddtthr = 0

# Main method
def main():

    if len(sys.argv) < 2:
        print("Enter year")
        return

    elif len(sys.argv) > 3:
        print("Incorrect number of arguments")
        return

    year = sys.argv[1]

    # Check if pickle exists     
    picklename = year+'/templates-tnp.pkl'
    if not os.path.isfile(picklename):
        print("You need to create the pickle")
        return

    # Read the histogram from the pickle file
    tnp = pickle.load(open(picklename,'rb')).integrate('region','tnp')

    if os.path.isfile(year+'/wtemplates_n2cvb.root'):
        os.remove(year+'/wtemplates_n2cvb.root')
    fout1 = uproot3.create(year+'/wtemplates_n2cvb.root')

    # data first
    p = "muondata"
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).sum('genflavor').sum('ddb1')
    hfail = tnp.integrate('n2ddt',int_range=slice(0,1)).integrate('process',p).sum('genflavor').sum('ddb1')

    fout1["data_obs_pass_nominal"] = hist.export1d(hpass)
    fout1["data_obs_fail_nominal"] = hist.export1d(hfail)

    # samples included
    p = ["ttbar","singlet","QCD","Wjets","Zjets"]

    # matched
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).integrate('genflavor',int_range=slice(1,4)).sum('ddb1')
    hfail = tnp.integrate('n2ddt',int_range=slice(0,1)).integrate('process',p).integrate('genflavor',int_range=slice(1,4)).sum('ddb1')
    fout1["catp2_pass_nominal"] = hist.export1d(hpass)
    fout1["catp2_fail_nominal"] = hist.export1d(hfail)

    # unmatched
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).integrate('genflavor',int_range=slice(0,1)).sum('ddb1')
    hfail = tnp.integrate('n2ddt',int_range=slice(0,1)).integrate('process',p).integrate('genflavor',int_range=slice(0,1)).sum('ddb1')
    fout1["catp1_pass_nominal"] = hist.export1d(hpass)
    fout1["catp1_fail_nominal"] = hist.export1d(hfail)

    if os.path.isfile(year+'/wtemplates_cvl.root'):
        os.remove(year+'/wtemplates_cvl.root')
    fout2 = uproot3.create(year+'/wtemplates_cvl.root')

    # data first                                                                                          
    p = "muondata"
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).sum('genflavor').integrate('ddb1',int_range=slice(0.64,1))
    hfail = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).sum('genflavor').integrate('ddb1',int_range=slice(0,0.64))

    fout2["data_obs_pass_nominal"] = hist.export1d(hpass)
    fout2["data_obs_fail_nominal"] = hist.export1d(hfail)

    # samples included                                                                                              
    p = ["ttbar","singlet","QCD","Wjets","Zjets"]

    # matched                                                                                                                     
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).integrate('genflavor',int_range=slice(1,4)).integrate('ddb1',int_range=slice(0.64,1))
    hfail = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).integrate('genflavor',int_range=slice(1,4)).integrate('ddb1',int_range=slice(0,0.64))
    fout2["catp2_pass_nominal"] = hist.export1d(hpass)
    fout2["catp2_fail_nominal"] = hist.export1d(hfail)

    # unmatched                                                                                                                 
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).integrate('genflavor',int_range=slice(0,1)).integrate('ddb1',int_range=slice(0.64,1))
    hfail = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).integrate('genflavor',int_range=slice(0,1)).integrate('ddb1',int_range=slice(0,0.64))
    fout2["catp1_pass_nominal"] = hist.export1d(hpass)
    fout2["catp1_fail_nominal"] = hist.export1d(hfail)

    return

if __name__ == "__main__":
    main()
