#!/usr/bin/env python

import sys
import random
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

##### READ THE DATA #####

filecount = len(sys.argv) - 1
fileindices = [None] * filecount
voltageMatrices = [None] * filecount
slopeLists = [None] * filecount
interceptLists = [None] * filecount

for i in range(filecount):
    slopeLists[i] = np.empty(255)
    interceptLists[i] = np.empty(255)

for i in range(filecount):
    print(F"Reading data... ({i}/{filecount})  ", end='\r')

    filename = sys.argv[i + 1]
    fileindices[i] = filename[(filename.index('.') + 1):]
    datafile = open("data/" + filename, "rb")

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
        if line.strip()[0] == "#": continue

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

    gainTIA_list = v_gainTIA(np.arange(1, 256) << 8)
    gainVA_list = v_gainVA(np.arange(1, 256))

    for j in range(len(data["state"])):
        state = data["state"][j]
        voltage = data["raw_voltage"][j] * 62.5E-6
        voltageMatrix[((state >> 8) & 0xFF) - 1, (state & 0xFF) - 1] = voltage

    for j in range(255):
        voltageList = voltageMatrix[np.full(255, j), np.arange(255)]
        mask = voltageList < 2.0479

        model = LinearRegression()
        model.fit(np.reshape(gainVA_list[mask], (-1, 1)),
                  np.reshape(voltageList[mask], (-1, 1)))

        slopeLists[i][j] = model.coef_[0][0]
        interceptLists[i][j] = model.intercept_[0]

##### FINAL ANALYSIS #####

print("Analysing data...                      ", end='\r')

compoundSlopeList = slopeLists[0]
for i in range(1, filecount):
    compoundSlopeList = np.concatenate((compoundSlopeList, slopeLists[i]))

compoundInterceptList = interceptLists[0]
for i in range(1, filecount):
    compoundInterceptList = np.concatenate((compoundInterceptList, interceptLists[i]))

compoundGainList = np.concatenate([gainTIA_list] * filecount)

stddev = np.std(compoundSlopeList)
avg = np.average(compoundSlopeList)

print(F"Average slope is {(avg * 1000):.3f} mV with a standard deviation of {(stddev * 1E6):.3f} µV")
