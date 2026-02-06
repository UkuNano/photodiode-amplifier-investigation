#!/usr/bin/env python

import sys
import numpy as np
from sklearn.linear_model import LinearRegression

##### DEFINE RESITORS #####

resistorsVA = [
   200E3,   # bit 0
   470E3,   # bit 1
   390E3,   # bit 2
   300E3,   # bit 3
    51E3,   # bit 4
    75E3,   # bit 5
   100E3,   # bit 6
    33E3    # bit 7
]

resistorsTIA = [
   1500E3,  # bit 8
  10000E3,  # bit 9
   5100E3,  # bit 10
   3300E3,  # bit 11
    470E3,  # bit 12
   1000E3,  # bit 13
    680E3,  # bit 14
    300E3   # bit 15
]

def gainTIA(state):
    if (state >> 8) & 0xFF == 0: return 0

    gain = 0

    for i in range(8):
        gain += 1 / resistorsTIA[i] * ((state >> (i + 8)) & 1)

    return 1 / gain

def gainVA(state):
    if state & 0xFF == 0: return 0

    gain = 0

    for i in range(8):
        gain += 1 / resistorsVA[i] * ((state >> i) & 1)

    return (1 / gain + 100) / 100

v_gainTIA = np.vectorize(gainTIA)
v_gainVA = np.vectorize(gainVA)

gainTIA_list = v_gainTIA(np.arange(1, 256) << 8)
gainVA_list = v_gainVA(np.arange(1, 256))

##### READ THE DATA #####

filecount = len(sys.argv) - 1

fileindices = [None] * filecount
voltageMatrices = [None] * filecount
slopeLists = [None] * filecount

for i in range(filecount):
    slopeLists[i] = np.empty(255)

for i in range(filecount):
    print(F"Reading data... ({i + 1}/{filecount})  ", end='\r')

    filename = sys.argv[i + 1]
    fileindices[i] = filename[(filename.rindex('.') + 1):]
    datafile = open("data/" + filename, 'r')

    data = {"state": [], "raw_voltage": []}

    for line in datafile.readlines():
        if len(line.strip()) == 0: continue
        if line[0] == "#": continue

        # Leave out the middle comma
        data["state"].append(int(line.split()[0].strip()[:-1], 16))
        data["raw_voltage"].append(int(line.split()[1].strip(), 16))

    datafile.close()

    print(F"Analysing data... ({i + 1}/{filecount})", end='\r')

    # First index corresponds to gainTIA_list index
    # Second to gainVA_list index
    voltageMatrix = np.empty((255, 255))

    for j in range(len(data["state"])):
        state = data["state"][j]
        
        # Skip invalid data
        if (state >> 8) & 0xFF == 0 or state & 0xFF == 0: continue
        
        voltage = data["raw_voltage"][j] * 62.5E-6
        voltageMatrix[((state >> 8) & 0xFF) - 1, (state & 0xFF) - 1] = voltage

    voltageMatrices[i] = voltageMatrix

    for j in range(255):
        voltageList = voltageMatrix[np.full(255, j), np.arange(255)]
        mask = voltageList < 2.0479

        xlist = gainVA_list[mask]
        ylist = voltageList[mask]

        model = LinearRegression()
        model.fit(np.reshape(xlist, (-1, 1)),
                  np.reshape(ylist, (-1, 1)))

        slopeLists[i][j] = model.coef_[0][0]

        

##### FINAL ANALYSIS #####

print("Analysing data...                      ", end='\r')

THE_INFLEXION_DATASET = 12
THE_SPECIAL_ONE = 16

compoundSlopeList = slopeLists[0]
for i in range(1, filecount):
    compoundSlopeList = np.concatenate((compoundSlopeList, slopeLists[i]))


datasetmask = np.full(slopeLists[0].size, fileindices[0], dtype=int)
for i in range(1, filecount):
    datasetmask = np.concatenate((datasetmask, np.full(slopeLists[i].size, int(fileindices[i]), dtype=int)))

maskA = datasetmask < THE_INFLEXION_DATASET
maskB = np.logical_and(datasetmask > THE_INFLEXION_DATASET, datasetmask != THE_SPECIAL_ONE)

HISTOGRAM_XMIN = 0.003598
HISTOGRAM_XMAX = 0.003662
BINWIDTH = 0.0000004
BINCOUNT = int((HISTOGRAM_XMAX - HISTOGRAM_XMIN) / BINWIDTH)

histogramA = np.empty(BINCOUNT, dtype=float)
histogramB = np.empty(BINCOUNT, dtype=float)

for i in range(BINCOUNT):
    binmask = compoundSlopeList[maskA] > HISTOGRAM_XMIN + BINWIDTH * i
    binmask = np.logical_and(binmask, compoundSlopeList[maskA] < HISTOGRAM_XMIN + BINWIDTH * (i + 1))
    histogramA[i] = compoundSlopeList[maskA][binmask].size

    binmask = compoundSlopeList[maskB] > HISTOGRAM_XMIN + BINWIDTH * i
    binmask = np.logical_and(binmask, compoundSlopeList[maskB] < HISTOGRAM_XMIN + BINWIDTH * (i + 1))
    histogramB[i] = compoundSlopeList[maskB][binmask].size


histogramA = histogramA / np.sum(histogramA)
histogramB = histogramB / np.sum(histogramB)


gaussianMaskA = np.logical_and(compoundSlopeList[maskA] > HISTOGRAM_XMIN, compoundSlopeList[maskA] < HISTOGRAM_XMAX)
gaussianMaskB = np.logical_and(compoundSlopeList[maskB] > HISTOGRAM_XMIN, compoundSlopeList[maskB] < HISTOGRAM_XMAX)

stddevGuessA = np.std(compoundSlopeList[maskA][gaussianMaskA])
stddevGuessB = np.std(compoundSlopeList[maskB][gaussianMaskB])
avgGuessA = np.average(compoundSlopeList[maskA][gaussianMaskA])
avgGuessB = np.average(compoundSlopeList[maskB][gaussianMaskB])


##### PERFORM GRADIENT DESCENT ON THE GAUSSIAN DISTRIBUTIONS #####

def gaussianDistribution(x, avg, stddev):
    return 1 / np.sqrt(2 * np.pi * stddev**2) * np.exp(-(x - avg)**2/(2 * stddev**2))

def calculateErrorSquared(histogram, avg, stddev):
    error = 0

    for i in range(BINCOUNT):
        x = HISTOGRAM_XMIN + BINWIDTH * i
        thisError = gaussianDistribution(x, avg, stddev) * BINWIDTH - histogram[i]
        error += thisError * thisError

    return error

def gaussianGradientDescent(histogram, avgGuess, stddevGuess):
    avg = avgGuess
    SD = stddevGuess

    stepAVG = 1
    epsilonAVG = 1E-10
    stepSD = 1
    epsilonSD = 1E-10
    treshold = 1E-15

    iteration = 0

    while True:
        iteration += 1

        if iteration > 1000:
            break

        thisError = calculateErrorSquared(histogram, avg, SD)
        newErrorAVG = calculateErrorSquared(histogram, avg + epsilonAVG, SD)
        newErrorSD = calculateErrorSquared(histogram, avg, SD + epsilonSD)

        derivativeAVG = (newErrorAVG - thisError)
        derivativeSD = (newErrorSD - thisError)
        
        avg = avg - stepAVG * derivativeAVG
        SD = SD - stepSD * derivativeSD

        #print(F"\r{iteration}: dE/d(AVG) = {derivativeAVG:.4g} dE/d(SD) = {derivativeSD:.4g}   ", end='\r')

        if abs(derivativeAVG) < treshold and abs(derivativeSD) < treshold:
            break

    SD = abs(SD)

    print("                                                                                ", end='\r')
    print(F"GUESS:\t\tµ = {(avgGuess * 1000):.3f} mV\t\tσ = {(stddevGuess * 1000):.6f} mV")
    print(F"RESULT:\t\tµ = {(avg * 1000):.3f} mV\t\tσ = {(SD * 1000):.6f} mV")
    print(F"ΔE = {(thisError - calculateErrorSquared(histogram, avgGuess, stddevGuess)):.4g} with {iteration} iterations")
    print()

    return (avg, SD)

(avgA, stddevA) = gaussianGradientDescent(histogramA, avgGuessA, stddevGuessA)
(avgB, stddevB) = gaussianGradientDescent(histogramB, avgGuessB, stddevGuessB)


##### PLOT THE DATA #####

print("Building the LaTeX file...          ", end='\r')

latex = """
\\documentclass[tikz, border=5mm]{standalone}

\\usepackage{siunitx}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}

\\definecolor{princetonorange}{RGB}{255, 157, 52}
\\definecolor{wheelorange}{RGB}{245, 126, 0}
\\definecolor{crayolablue}{RGB}{55, 119, 255}
\\definecolor{risdblue}{RGB}{10, 88, 255}
\\definecolor{vermilion}{RGB}{255, 54, 54}
\\definecolor{applegreen}{RGB}{153, 184, 0}
\\definecolor{mexicanpink}{RGB}{220, 0, 115}
\\definecolor{mauve}{RGB}{242, 190, 252}
\\definecolor{verdigris}{RGB}{5, 168, 170}

\\begin{document}
\\begin{tikzpicture}
    \\begin{axis}[
            width=0.9\\linewidth,
            x filter/.expression={x*1000},
            y filter/.expression={y*100},
            xmin=3.600, xmax=3.660,
            ymin=0.0, ymax=10,
            restrict y to domain=0:10,
            xlabel=Slope of output voltage vs. voltage amplifier gain / \\si{\\milli\\volt},
            ylabel=Percentage of occurence,
            xtick={3.600,3.610,...,3.660},
            ytick={2,4,...,10},
            legend style={at={(0.98, 0.98)},anchor=north east},
            xticklabel style={
                /pgf/number format/.cd,
                fixed,
                fixed zerofill,
                precision=3,
                /tikz/.cd
            },
            yticklabel=\\pgfmathprintnumber{\\tick}\\,\\%%,
            yticklabel style={
                /pgf/number format/.cd,
                fixed,
                fixed zerofill,
                precision=0,
                /tikz/.cd
            },
            scaled x ticks=false,
            scaled y ticks=false,
            clip marker paths=true,
            area style
        ]

        \\addplot+[
            color=crayolablue,
            fill opacity=0.5,
            ybar interval,
            mark=no
        ]
        coordinates{%s};

        \\addplot [
            domain=%.6f:%.6f, 
            samples=200, 
            color=risdblue,
            dashed,
            line width=1.4pt,
            forget plot
        ]
        {%.9f/sqrt(2 * %.9f * %.9f^2) * exp(-(x - %.9f)^2/(2 * %.9f^2))};

        \\addplot [
            domain=%.6f:%.6f, 
            samples=200, 
            color=risdblue,
            line width=1.2pt,
            forget plot
        ]
        {%.9f/sqrt(2 * %.9f * %.9f^2) * exp(-(x - %.9f)^2/(2 * %.9f^2))};



        \\addplot+[
            color=princetonorange,
            fill opacity=0.5,
            ybar interval,
            mark=no
        ]
        coordinates{%s};

        \\addplot [
            domain=%.6f:%.6f,
            samples=200, 
            color=wheelorange,
            line width=1.2pt,
            forget plot
        ]
        {%.9f/sqrt(2 * %.9f * %.9f^2) * exp(-(x - %.9f)^2/(2 * %.9f^2))};

        \\addplot [
            domain=%.6f:%.6f,
            samples=200, 
            color=wheelorange,
            dashed,
            line width=1.4pt,
            forget plot
        ]
        {%.9f/sqrt(2 * %.9f * %.9f^2) * exp(-(x - %.9f)^2/(2 * %.9f^2))};



        \\legend{{%s}, {%s}}
    \\end{axis}
\\end{tikzpicture}
\\end{document}
"""


coordinatesA = ""
coordinatesB = ""
for i in range(BINCOUNT):
    xpos = HISTOGRAM_XMIN + BINWIDTH * i
    coordinatesA += F"({xpos:.9f},{histogramA[i]:.6f})"
    coordinatesB += F"({xpos:.9f},{histogramB[i]:.6f})"


legendA = ""
legendB = ""
for i in range(filecount):
    if int(fileindices[i]) == THE_SPECIAL_ONE:
        continue

    if int(fileindices[i]) < THE_INFLEXION_DATASET:
        legendA += fileindices[i] + ",\\,"
    else:
        legendB += fileindices[i] + ",\\,"

latex = latex % (
    coordinatesA,
    HISTOGRAM_XMIN, HISTOGRAM_XMAX,
    BINWIDTH, np.pi, stddevGuessA, avgGuessA, stddevGuessA,
    HISTOGRAM_XMIN, HISTOGRAM_XMAX,
    BINWIDTH, np.pi, stddevA, avgA, stddevA,

    coordinatesB,
    HISTOGRAM_XMIN, HISTOGRAM_XMAX,
    BINWIDTH, np.pi, stddevB, avgB, stddevB,
    HISTOGRAM_XMIN, HISTOGRAM_XMAX,
    BINWIDTH, np.pi, stddevGuessB, avgGuessB, stddevGuessB,

    legendA[:-3], legendB[:-3]
)

latexfilename = "latex/slopedistribution-set-4."
for i in range(filecount):
    latexfilename += fileindices[i] + "."
latexfilename += "tex"

latexfile = open(latexfilename, "w")
latexfile.write(latex)
latexfile.close()

sampleSizeA = compoundSlopeList[maskA].size
sampleSizeB = compoundSlopeList[maskB].size

print(F"A: µ = {(avgA * 1000):.6f} mV    σ = {(stddevA * 1000):.6f} mV")
print(F"   The average is {(avgA * 1000):.3f} mV ± {(2 * stddevA / np.sqrt(sampleSizeA) * 1E6):.2f} µV")
print(F"   The slope is ({(avgA * 1000):.3f} ± {(3 * stddevA * 1000):.3f}) mV")
print(F"B: µ = {(avgB * 1000):.6f} mV    σ = {(stddevB * 1000):.6f} mV")
print(F"   The average is {(avgB * 1000):.3f} mV ± {(2 * stddevB / np.sqrt(sampleSizeB) * 1E6):.2f} µV")
print(F"   The slope is ({(avgB * 1000):.3f} ± {(3 * stddevB * 1000):.3f}) mV")

print(sampleSizeA, sampleSizeB)
