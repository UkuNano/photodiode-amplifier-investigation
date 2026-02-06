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
interceptLists = [None] * filecount
rmseLists = [None] * filecount

for i in range(filecount):
    slopeLists[i] = np.empty(255)
    interceptLists[i] = np.empty(255)
    rmseLists[i] = np.empty(255)

for i in range(filecount):
    print(F"Reading data... ({i + 1}/{filecount})  ", end='\r')

    filename = sys.argv[i + 1]
    fileindices[i] = filename[(filename.index('.') + 1):]
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
        
        # Too few datapoints, let's discard this
        if xlist.size < 10:
            rmseLists[i][j] = 100
            continue

        model = LinearRegression()
        model.fit(np.reshape(xlist, (-1, 1)),
                  np.reshape(ylist, (-1, 1)))

        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        
        rmse = 0
        
        for k in range(xlist.size):
            predicted = slope * xlist[k] + intercept
            error = ylist[k] - predicted
            rmse += error * error
        
        rmse = np.sqrt(rmse / xlist.size)
        
        rmseLists[i][j] = rmse
        slopeLists[i][j] = slope
        interceptLists[i][j] = intercept

        

##### FINAL ANALYSIS #####

print("Analysing data...                      ", end='\r')

compoundSlopeList = slopeLists[0]
for i in range(1, filecount):
    compoundSlopeList = np.concatenate((compoundSlopeList, slopeLists[i]))

compoundInterceptList = interceptLists[0]
for i in range(1, filecount):
    compoundInterceptList = np.concatenate((compoundInterceptList, interceptLists[i]))

compoundRmseList = rmseLists[0]
for i in range(1, filecount):
    compoundRmseList = np.concatenate((compoundRmseList, rmseLists[i]))

compoundGainList = np.concatenate([gainTIA_list] * filecount)

cleanMask = compoundRmseList < 0.001 #0.0007
outlierMask = np.logical_not(cleanMask)

cleanCount = compoundGainList[cleanMask].size
outlierCount = compoundGainList[outlierMask].size


xmin = 0.00357
xmax = 0.00371
binwidth = 0.000001
bincount = int((xmax - xmin) / binwidth)
cleanHistogram = np.empty(bincount, dtype=int)
outlierHistogram = np.empty(bincount, dtype=int)

for i in range(bincount):
    binmask = compoundSlopeList[cleanMask] > xmin + binwidth * i
    binmask = np.logical_and(binmask, compoundSlopeList[cleanMask] < xmin + binwidth * (i + 1))
    cleanHistogram[i] = compoundSlopeList[cleanMask][binmask].size

for i in range(bincount):
    binmask = compoundSlopeList[outlierMask] > xmin + binwidth * i
    binmask = np.logical_and(binmask, compoundSlopeList[outlierMask] < xmin + binwidth * (i + 1))
    outlierHistogram[i] = compoundSlopeList[outlierMask][binmask].size


stddev = np.std(compoundSlopeList)
avg = np.average(compoundSlopeList)

##### PLOT THE DATA #####

print("Building the LaTeX file...          ", end='\r')

latex = """
\\documentclass[11pt, a4paper]{article}

\\usepackage[T1]{fontenc}
\\usepackage{lscape}
\\usepackage{pdflscape}
\\usepackage{siunitx}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\pgfplotsset{set layers} %% Activate layer system

%% Define layer order
\\pgfplotsset{
    /pgfplots/layers/mylayers/.define layer set={
        axis background,axis grid,main,axis ticks,axis lines,axis tick labels,
        axis descriptions,axis foreground
    }{/pgfplots/layers/standard},
}

\\definecolor{malachite}{RGB}{4, 231, 98}
\\definecolor{selectiveyellow}{RGB}{245, 183, 0}
\\definecolor{mexicanpink}{RGB}{220, 0, 115}
\\definecolor{licorice}{RGB}{38, 28, 21}
\\definecolor{verdigris}{RGB}{5, 168, 170}
\\definecolor{ylnmnblue}{RGB}{74, 88, 153}

\\begin{document}
\\pagenumbering{gobble}

\\begin{landscape}
\\hspace{-30mm} \\begin{tikzpicture}[baseline=0pt]
    \\begin{semilogxaxis}[
            width=0.6\\linewidth,
            xmin=100000, xmax=11000000,
            ymin=%.3f, ymax=%.3f,
            restrict y to domain=%.7f:%.7f,
            xlabel=Transimpedance amplifier stage gain / $\\frac{\\si{\\volt}}{\\si{\\ampere}}$,
            ylabel=Slope of output voltage vs. voltage amplifier gain / \\si{\\milli\\volt},
            grid=both,
            ytick={%.5f,%.5f,...,%.5f},
            legend style={at={(0.98, 0.98)},anchor=north east},
            log ticks with fixed point,
            xticklabel style={
                /pgf/number format/.cd,
                1000 sep={\,}, %% Thin space as thousands separator
                fixed,
                /tikz/.cd,
                rotate=45,
                anchor=north east
            },
            scaled x ticks=false,
            yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3,/tikz/.cd},
            clip marker paths=true
        ]

        \\addplot+[
            color=verdigris,
            only marks,
            mark size=0.6pt,
            mark=*,
            mark options={fill=verdigris}
        ]
        coordinates{%s};

        %%\\addplot+[mark=none, color=ylnmnblue, line width=1.5pt] coordinates{(100000, %.8f) (11000000, %.8f)};

        \\addplot+[
            color=mexicanpink,
            only marks,
            mark size=1.5pt,
            mark=triangle*,
            mark options={fill=mexicanpink}
        ]
        coordinates{%s};
    \\end{semilogxaxis}
\\end{tikzpicture} \\hspace{-15.5pt} \\begin{tikzpicture}[baseline=0pt]
    \\begin{axis}[
            width=0.6\\linewidth,
            ymin=%.3f, ymax=%.3f,
            xmin=0, xmax=300,
            restrict x to domain=0:300,
            %%xmajorgrids=true,
            grid style={/pgfplots/on layer=axis background},
            set layers=mylayers,
            ymajorticks=false,
            yticklabel=\\empty,
            axis x line*=top,
            xlabel=Count,
            xtick={0,50,...,300},
            xticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=0,/tikz/.cd},
            legend style={at={(0.98, 0.98)},anchor=north east},
            area style,
            execute at begin axis={
                \draw[line width=1.0pt] (rel axis cs:0,0) -- (rel axis cs:1,0);
            }
        ]

        \\addplot+[
            color=verdigris,
            fill opacity=0.4,
            xbar interval,
            mark=no
        ]
        coordinates{%s};

        \\addplot+[
            color=mexicanpink,
            fill opacity=0.4,
            xbar interval,
            mark=no
        ]
        coordinates{%s};

        %%\\addplot+[mark=none, color=ylnmnblue, line width=1.5pt] coordinates{(0, %.8f) (305, %.8f)};

        \\legend{{RMSE < \\SI{1.0}{\\milli\\volt} (%i)}, {RMSE > \\SI{1.0}{\\milli\\volt} (%i)}} %%, $\mu = \SI{%.3f}{\\milli\\volt}$}
    \\end{axis}
\\end{tikzpicture}
\\end{landscape}
\\end{document}
"""

cleanCoordinates = ""
for i in range(cleanCount):
    cleanCoordinates += F"({compoundGainList[cleanMask][i]:.1f},{(compoundSlopeList[cleanMask][i] * 1000):.6f})"

outlierCoordinates = ""
for i in range(outlierCount):
    outlierCoordinates += F"({compoundGainList[outlierMask][i]:.1f},{(compoundSlopeList[outlierMask][i] * 1000):.6f})"

cleanHistogramCoordinates = ""
outlierHistogramCoordinates = ""
for i in range(bincount):
    xpos = xmin + binwidth * i
    # Swapping the coordinates here is intentional
    cleanHistogramCoordinates += F"({cleanHistogram[i]},{(xpos * 1000):.3f})"
    outlierHistogramCoordinates += F"({outlierHistogram[i]},{(xpos * 1000):.3f})"

ymin = 3.58
ymax = 3.705

ytickStep = np.floor((ymax - ymin) / 5 / 0.0025) * 0.0025
ytickStart = np.floor(ymin / ytickStep) * ytickStep
ytickEnd = np.ceil(ymax / ytickStep) * ytickStep

latex = latex % (
    ymin, ymax,
    ymin, ymax,
    ytickStart, ytickStart + ytickStep, ytickEnd,
    cleanCoordinates, avg * 1000, avg * 1000, outlierCoordinates,

    ymin, ymax,
    cleanHistogramCoordinates, outlierHistogramCoordinates, avg * 1000, avg * 1000,
    cleanCount, outlierCount, avg * 1000 # Legend
)

latexfilename = "latex/slope-rmse-log-distribution."
for i in range(filecount):
    latexfilename += fileindices[i] + "."
latexfilename += "tex"

latexfile = open(latexfilename, "w")
latexfile.write(latex)
latexfile.close()

print(F"Average slope is {(avg * 1000):.3f} mV with a standard deviation of {(stddev * 1E6):.3f} µV")
print(F"There are {cleanCount} clean datapoints and {outlierCount} noisy datapoints")
