## File Description:
  1. makeFEL.py -> Saves the EMMD estimated Free Energy Landscapes (FEL) in a <outfile.npy> file.
  2. GMModel.py -> Underlying class that fits Variational Bayesian Gaussian Mixture Model over the given data and generates the FEL.
  3. FELplot.py -> plots the FEL as a contour plot.

## How to run:
  ```python
  # make the FEL from Collective Variable ensemble
  python makeFEL.py inputFile.npy outputFile.npy
  
  # plot the FEL
  python FELplot.py outputFile.npy
  ```
