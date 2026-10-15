import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(os.getcwd()).parent
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt

from python.fluorophores import FlStatic
from python.fluorophores import FlCollection
from python.psfs import PsfVectorial
from python.estimators import est_qLSQiter2D
from python.estimators import backgroundsubtractor
from python.simulators import Simulator

def Rolling_Average(Data,Window):
    if Window % 2 == 0: Window = Window + 1
    kernel = np.ones(Window)/Window
    
    DataPad = np.zeros(len(Data)+Window-1)
    DataPad[:int(Window/2)] = Data[:int(Window/2)]
    DataPad[int(Window/2):-int(Window/2)] = Data
    DataPad[-int(Window/2):] = Data[-int(Window/2):]

    av = np.correlate(DataPad,kernel,mode='valid')
    return av

psf_vec = PsfVectorial()
psf_vec.setpinhole(AU=0.8)

sim = Simulator()
laserpower = 10
numberOfLocalizations = 1000
L = 70
probecenter = False
estimatebackground = False
sim.definePattern("donut", psf_vec,
                  phasemask="vortex", 
                  makepattern="orbitscan", 
                  orbitpoints=6, 
                  probecenter=probecenter,
                  orbitL=L,
                  laserpower=laserpower,
                  pointdwelltime=0.1)
sim.defineComponent("estdonut", "estimator", est_qLSQiter2D, parameters=[L, probecenter], dim=(0,1))
sim.defineComponent("bg", "background", backgroundsubtractor, parameters=["background_estimated"])
seq = ["donut", "bg", "estdonut"]
fltestpos = [0, 0, 0]
fl1 = FlStatic()
fl1.pos = fltestpos
sim.fluorophores = fl1

# no background
sim.background_estimated = 0
fl1.pos = fltestpos
psf_vec.zerooffset = 0
out = sim.runSequence(seq, maxlocs=numberOfLocalizations)
print("no background")
summ1 = sim.summarize_results(out)

BasePhot = summ1.photrate

fl2 = FlStatic()

biases = []
photrates = []
Xs = range(1,501,1)

for x in Xs:
  
  fl2.pos = [x, 0, 000]

  fc = FlCollection()
  fc.add([fl1, fl2])
  psf_vec.zerooffset = 0
  sim.fluorophores = fc  # make a simulator and attach fluorophore
  sim.background_estimated = 0  # we don't know bg
  out = sim.runSequence(seq, maxlocs=1000)#numberOfLocalizations)

  print("nearby fluorophore:",x)
  summ = sim.summarize_results(out)

  biases.append(summ.bias[0])
  photrates.append(summ.photrate)

biases = np.asarray(biases)
photrates = np.asarray(photrates)/BasePhot

print(np.argmax(photrates))

peak = np.argmax(photrates)
lt2 = np.where(photrates<1.95)
lt13 = np.where(photrates<1.3)

#plt.plot(Xs[:peak],photrates[:peak])
#plt.plot(Xs[peak:],photrates[peak:])
plt.plot(Xs,photrates)
plt.ylabel('relative efo [x baseline]')
plt.xlabel('second fluorophore position [nm]')
plt.show()

#plt.plot(Xs[:peak],biases[:peak])
#plt.plot(Xs[peak:],biases[peak:])
plt.plot(Xs,biases)
plt.ylabel('measured position bias [nm]')
plt.xlabel('second fluorophore position [nm]')
plt.show()

LinFit = np.polyfit(photrates[lt13],biases[lt13],deg=1)
QuadFit = np.polyfit(photrates[peak:],biases[peak:],deg=2)

print(LinFit)

#plt.plot(photrates[:peak],biases[:peak])
#plt.plot(photrates[peak:],biases[peak:])
plt.plot(photrates,biases)
plt.plot(photrates[lt13],np.polyval(LinFit,photrates[lt13]))
plt.ylabel('relative efo [x baseline]')
plt.xlabel('measured position bias [nm]')
#plt.plot(photrates[peak:],np.polyval(QuadFit,photrates[peak:]))
plt.show()
