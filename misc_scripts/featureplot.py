#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Script for generating plots of BDT variable histograms. Executed as:

ldmx python3 FeaturePlot.py [name of output file] (no extension needed)

"""

import sys
tmpargv = sys.argv
sys.argv = []
import getopt
import numpy as np
import ROOT
from ROOT import gROOT, TFile, TTree, TChain, gDirectory, TLine, gStyle, TCanvas, TLegend, TH1F, TLatex, TF1, TEfficiency, TColor
sys.argv = tmpargv

options, remainder = getopt.gnu_getopt(sys.argv[1:], 'hl:')

# Parse the command line arguments
for opt, arg in options:
    if opt=='-l':
        label = str(arg)
    if opt=='-h':
        print_usage()
        sys.exit(0)
            
def openPDF(outfile,canvas):
	canvas.Print(outfile+".pdf[")

def closePDF(outfile,canvas):
	canvas.Print(outfile+".pdf]")

gStyle.SetOptStat(0)

outdir = "/sfs/qumulo/qhome/tgh7hx/ldmx/visiblesnoteplots/"

# path to files containing histograms of features (formatted as bkg, 50MeV
# signal, 5MeV signal)
infile1 = TFile("/sfs/qumulo/qhome/tgh7hx/ldmx/visiblesnoteplots/PN.root")
infile2 = TFile("/sfs/qumulo/qhome/tgh7hx/ldmx/visiblesnoteplots/mAp_050.root")
infile3 = TFile("/sfs/qumulo/qhome/tgh7hx/ldmx/visiblesnoteplots/mAp_005.root")

histoname = ["Etot", "Eupstream", "Edownstream", "hits", "downstreamhits", "isohits", "isoE", "xmean", "ymean", "xstd", "ystd", "zstd", "downstreamrmean_gammaproj", "downstreamhits_within1", "downstreamhits_within2", "downstreamhits_within3", "downstreamE_within1", "downstreamE_within2", "downstreamE_within3", "layershit", "xmean_unweighted", "ymean_unweighted", "xstd_unweighted", "ystd_unweighted", "zstd_unweighted"]

axistitle = ["Total energy [MeV]", "Upstream energy [MeV]", "Downstream energy [MeV]", "Number of readout hits", "Downstream hits", "Isolated hits", "Isolated enrgy [MeV]", "Mean x [mm]", "Mean y [mm]", "Std. dev. x [mm]", "Std. dev. y [mm]", "Std. dev. z [mm]", "Mean distance of downstream hits from photon trajectory line [mm]", "Downstream hits within 1 cell width of photon trajectory", "Downstream hits within 2 cell widths of photon trajectory", "Downstream hits within 3 cell widths of photon trajectory", "Downstream energy within 1 cell width of photon trajectory [MeV]", "Downstream energy within 2 cell widths of photon trajectory [MeV]", "Downstream energy within 3 cell widths of photon trajectory [MeV]", "Number of layers hit", "Mean x (unweighted) [mm]", "Mean y (unweighted) [mm]", "Std. dev. x (unweighted) [mm]", "Std. dev. y (unweighted) [mm]", "Std. dev. z (unweighted) [mm]"]

axisrange = [[2400,5000], [0,2000], [2400,5000], [0,130], [0,100], [0,60], [0,4000], [-300,300], [-300,300], [0,180], [0,180], [0,300], [0,350], [0,70], [0,70], [0,70], [0,4000], [0,4000], [0,4000], [0,40], [-300,300], [-300,300], [0,180], [0,180], [0,300]]

for i in range(len(histoname)):
    print("Now plotting feature", histoname[i])
    #create canvas
    c = TCanvas("c","c",800,600)

    # name of histogram in the root file
    h1 = infile1.Get("myana/myana_"+histoname[i])
    h2 = infile2.Get("myana/myana_"+histoname[i])
    h3 = infile3.Get("myana/myana_"+histoname[i])

    # normalize the histograms
    h1.Scale(1./h1.Integral())
    h2.Scale(1./h2.Integral())
    h3.Scale(1./h3.Integral())

    # create error bars
    h1.Sumw2()
    h2.Sumw2()
    h3.Sumw2()

    openPDF(outdir+histoname[i],c)

    h1.SetLineColor(634)
    h2.SetLineColor(797)
    h3.SetLineColor(862)

    # parameters to change for each variable plotted
    h1.GetXaxis().SetRangeUser(axisrange[i][0], axisrange[i][1])
    h1.GetYaxis().SetRangeUser(0.00001, 1)

    h1.Draw("histo e1")
    h2.Draw("same hist e1")
    h3.Draw("same hist e1")

    h1.SetTitle(" ")
    h1.GetXaxis().SetTitle(axistitle[i])
    h1.GetYaxis().SetTitle("A.U.")

    # legend that maybe needs to move for each variable plotted
    legend = TLegend(.63,.68,.97,.89)#.63,.66,.97,.87)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.035)
    legend.AddEntry(h1,"Photonuclear","LP")
    legend.AddEntry(h2,"m_{A'} = 50MeV", "LP")
    legend.AddEntry(h3,"m_{A'} = 5MeV", "LP")
    legend.Draw()

    # making the plots look fancy and official
    l = TLatex()
    l.SetTextFont(72)
    xtext=0.15
    ytext=0.91
    l.DrawLatexNDC(xtext,ytext,"LDMX")
    l.SetTextFont(52)
    l.DrawLatexNDC(xtext+0.1,ytext,"Simulation")

    c.SetLogy(1)
    c.Print(outdir+histoname[i]+".pdf")

    closePDF(outdir+histoname[i],c)

    #delete objects to prevent memory leaks
    del c
    del h1
    del h2
    del h3
