# Cell_detection
Script to detect cells in microfluidics content and calculate the extracellular area for pH correction.

Networks used: MobileNets, Inception v2

script description:
0 - extracts the roi of the cell for further analysis.
1 - divides empty droplets and droplets containing cells.
2 - divides empty droplets and droplets containing cells using both the networks.
3 - evaluates the network accuracy and recall, using the testset
4 - calculates the extracellular area
5 - complete script to divide the cells, extract the roi of the cell and calculate the extracellular area 

