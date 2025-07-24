Matlab can be configured to work with the cluster by importing the torque.settings configuration file. 
Open the graphical matlab at the top of the screen click Parallel > Manage Cluster Profiles > Import. 
Then navigate to /share/qsub_examples/matlab and select torque.settings.

Once the configuration file has been imported Matlab will handle starting processes on the compute nodes. 
Worker processes can be started by running "parpool('torque',12)". 

You may use up to 80 worker licenses at a time if they are available.  
You can run the "matlablicense" command on the Linux command line to check the current available licenses. 
There are 500 worker licenses shared between all users on Acropolis. 

The matlab local configuration "parpool('local')" will open 12 workers on the local machine by default. 
Please do not run parallel Matlab jobs directly on the Acropolis head node. This work should be submitted to the nodes.
The local configuration does not count against the 500 worker licenses available with the torque configuration. 
You may use the matlabparallelbatch.sh script to submit a 12 worker distributed batch job using the local confuration.  
You may submit multiple 12 worker instances of Matlab at a time. 
The matlabparallelbatcharray.sh script can submit multiple jobs with the local profile at a time. 
Users are permitted to use up to 200 cores at a time.
