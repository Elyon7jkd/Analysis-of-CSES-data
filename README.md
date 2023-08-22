# Analysis-of-CSES-data
For code running you need to use Pyhton IDE such as Spider,Pycharm, Jupyter or Google Colab.
Install libraries:
-h5py
-numpy
-pandas
-scipy
-matplotlib

Dataset:In the section dataset there is the google drive link of some .h5 files samples from CSES satellite that must be read from code. Change every datapath in code with the folder where you save the dataset.

Code:There are .py and .ipynb files where notebooks are stored and pdf files with the code and relative notes for each line.

-'mutiplot spettri_zoom' provides the spectrograms in multiplot and the time focus on 5 min, 1 min, 30sec, 2sec,1sec.
-'spectrograms' is a program that reads all files, prepares and manipulates data and plot the spectrograms.
-'map', 'reader' and 'efd' are files read by the code 'heatmap', so you have to save it into the input folder.
-'heatmap' represents the amplitude into a geographic map.
-'statistic' calculates the amplitude for every orbits in single plots.
-'valMinMax' finds the max and min values that need to be selected when you build plot function in order to make comparison between orbits.
-'AVG amplitude total' produces a global graph with all the amplitude of every orbits for different frequencies.

Plot:In the plot section there are some output images and a drive link where find all the pdf of plots from code.
