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
    datafile = open("data/" + filename, 'rb')

    data = {"state": [], "raw_voltage": []}

    corruptedCount = 0

    for rawline in datafile.readlines():
        corrupted = False

        for rawbyte in rawline:
            if rawbyte >= 127:
                corrupted = True
                corruptedCount += 1
                break

        if corrupted: continue

        line = rawline[:-1].decode('ascii')

        if len(line.strip()) == 0: continue
        if line[0] == "#": continue

        # Leave out the middle comma
        data["state"].append(int(line.split()[0].strip()[:-1], 16))
        data["raw_voltage"].append(int(line.split()[1].strip(), 16))

    datafile.close()

    if corruptedCount > 0:
        print(F"{filename}: {corruptedCount} corrupted data points out of {corruptedCount + len(data["raw_voltage"])}")
        print()

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

HISTOGRAM_XMIN = 130E-6
HISTOGRAM_XMAX = 170E-6
BINWIDTH = 1E-6

BINCOUNT = int((HISTOGRAM_XMAX - HISTOGRAM_XMIN) / BINWIDTH)

histograms = []
averages = []
stddevs = []

for i in range(filecount):
    histogram = np.empty(BINCOUNT, dtype=float)

    for j in range(BINCOUNT):
        binmask = slopeLists[i] > HISTOGRAM_XMIN + BINWIDTH * j
        binmask = np.logical_and(binmask, slopeLists[i] < HISTOGRAM_XMIN + BINWIDTH * (j + 1))
        histogram[j] = slopeLists[i][binmask].size

    mask = slopeLists[i] > HISTOGRAM_XMIN
    mask = np.logical_and(mask, slopeLists[i] < HISTOGRAM_XMAX)

    histogram = histogram / np.sum(histogram)
    histograms.append(histogram)
    averages.append(np.average(slopeLists[i][mask]))
    stddevs.append(np.std(slopeLists[i][mask]))



##### PLOT THE DATA #####

print("Building the LaTeX file...          ", end='\r')

latexStart = """
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
\\definecolor{pink}{RGB}{250, 30, 135}
\\definecolor{mauve}{RGB}{242, 190, 252}
\\definecolor{verdigris}{RGB}{5, 168, 170}

\\begin{document}
\\begin{tikzpicture}
    \\begin{axis}[
            width=0.9\\linewidth,
            x filter/.expression={x*1000},
            y filter/.expression={y*100},
            xmin=0.130, xmax=0.170,
            ymin=0.0, ymax=24,
            restrict y to domain=0:24,
            xlabel=Slope of output voltage vs. voltage amplifier gain / \\si{\\milli\\volt},
            ylabel=Percentage of occurence,
            xtick={0.130,0.135,...,0.170},
            ytick={2,4,...,24},
            legend style={at={(0.98, 0.98)},anchor=north east},
            xticklabel style={
                /pgf/number format/.cd,
                fixed,
                fixed zerofill,
                precision=3,
                /tikz/.cd
            },
            yticklabel=\\pgfmathprintnumber{\\tick}\\,\\%,
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
"""

latexPlot = """
        \\addplot+[
            color=%s,
            fill opacity=0.35,
            ybar interval,
            mark=no
        ]
        coordinates{%s};
"""

latexEnd = """
        \\legend{{%s}}
    \\end{axis}
\\end{tikzpicture}
\\end{document}
"""



coordinateArrays = []
for i in range(filecount):
    coordinates = ""

    for j in range(BINCOUNT):
        xpos = HISTOGRAM_XMIN + BINWIDTH * j
        coordinates += F"({xpos:.9f},{histograms[i][j]:.6f})"

    coordinateArrays.append(coordinates)

colors = [
    "princetonorange",
    "crayolablue",
    "pink"
]



latex = latexStart
for i in range(filecount):
    latex += latexPlot % (
        colors[i],
        coordinateArrays[i],
    )

latex += latexEnd % ",\\,".join(fileindices)



latexfilename = "latex/slopedistribution-set-vref."
for i in range(filecount):
    latexfilename += fileindices[i] + "."
latexfilename += "tex"

latexfile = open(latexfilename, "w")
latexfile.write(latex)
latexfile.close()


for i in range(filecount):
    print(F"{sys.argv[i + 1]}:                     ")
    print(F"    µ = {(averages[i] * 1000):.6f} mV    σ = {(stddevs[i] * 1000):.6f} mV")
    print(F"    ({(averages[i] * 1000):.3f} ± {(3 * stddevs[i] * 1000):.3f}) mV")
    print()
