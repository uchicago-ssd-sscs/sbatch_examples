% set the parallel profile as local
pc = parcluster('local')
% set the temp file location to MATLABWORKDIR variable set in submit script
pc.JobStorageLocation = strcat(getenv('MATLABWORKDIR'))
% open a pool of 12 worker processes
parpool(pc,12)

% run a simple parfor loop
a = ones(1000,1000)

disp('begin running parfor')
parfor i = 1:10000
  b=ones./a
end

disp('done running parfor')

% close the workers and exit
poolobj = gcp('nocreate');
delete(poolobj);
exit
