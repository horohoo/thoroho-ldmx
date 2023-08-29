#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:59:09 2020

@author: matthewsolt
"""

import sys
tmpargv = sys.argv
sys.argv = []
import getopt
import numpy as np
import ROOT
from ROOT import gROOT, TFile, TTree, TChain, gDirectory, TLine, gStyle, TCanvas, TLegend, TH1F, TLatex, TF1, TEfficiency
sys.argv = tmpargv

#List arguments
def print_usage():
    print ("\nUsage: {0} <output file base name> <input file>".format(sys.argv[0]))
    print ('\t-l: plot label')
    print ('\t-h: this help message')
    print

label = ""

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
c = TCanvas("c","c",800,600)

outfile = remainder[0]
outfileroot = TFile(remainder[0]+".root","RECREATE")

infile1 = TFile("/sfs/qumulo/qhome/tgh7hx/ldmx/analysis/july23/signaleffs/50MeV_passtrackerveto.root")
c1 = infile1.Get("c1_n3")
h1 = c1.GetPrimitive("passTrackerVeto")

infile2 = TFile("/sfs/qumulo/qhome/tgh7hx/ldmx/analysis/july23/signaleffs/5MeV_passtrackerveto.root")
c2 = infile2.Get("c1_n3")
h2 = c2.GetPrimitive("passTrackerVeto")


c.cd()
openPDF(outfile, c)
h1.SetLineColor(797)
h1.SetLineWidth(2)
h2.SetLineColor(862)
h2.SetLineWidth(2)
h1.GetXaxis().SetRangeUser(0, 900)
h1.GetYaxis().SetRangeUser(0.0, 1.0)

h1.Draw("histo e1")
h2.Draw("same hist e1")

h1.SetTitle(" ")
h1.GetXaxis().SetTitle("Decay distance downstream of target [mm]")
h1.GetYaxis().SetTitle("Efficiency")

legend = TLegend()#.10,.72,.25,.83)
legend.SetBorderSize(0)
legend.SetFillColor(0)
legend.SetFillStyle(0)
legend.SetTextFont(42)
legend.SetTextSize(0.035)
legend.AddEntry(h1,"m_{A'} = 50MeV","LP")
legend.AddEntry(h2,"m_{A'} = 5MeV", "LP")
legend.Draw()

l = TLatex()
l.SetTextFont(72)
xtext=0.15
ytext=0.91
l.DrawLatexNDC(xtext,ytext,"LDMX")
l.SetTextFont(52)
l.DrawLatexNDC(xtext+0.1,ytext,"Simulation")

c.Print(outfile+".pdf")

closePDF(outfile,c)
