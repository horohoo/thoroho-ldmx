from LDMX.Framework import ldmxcfg

p=ldmxcfg.Process('validation')

from LDMX.DQM import dqm
ana = dqm.SampleValidation("validation")

# Define the order in which the analyzers will be executed.
p.sequence=[ana]

# input the file as an argument on the command line
import sys
p.inputFiles=[sys.argv[1]]

# Specify the output file
p.histogramFile=sys.argv[2]
