# Cell_detection
Script to detect cells in microfluidics content and calculate the extracellular area for pH correction.

Networks used: MobileNets, Inception v2

Files description:<br>
0 - extracts the roi of the cell for further analysis<br>
1 - divides empty droplets and droplets containing cells<br>
2 - divides empty droplets and droplets containing cells using both the networks<br>
3 - evaluates the network accuracy and recall, using the testset<br>
4 - calculates the extracellular area<br>
5 - complete script to divide the cells, extract the roi of the cell and calculate the extracellular area <br>

