#!/bin/bash
echo "Train Deep Parameter Approximator"
python -c 'from IrrigationDetector_DoubleExp_v2 import main; main()'
echo "DPA Performance Evaluation"
python -c 'from NLSDoubleExpFit_v2 import main; main(50)'
echo "Train Irrigation Detector"
python -c 'from TrainHeatingClassifier import main; main()'
