% Set path to the python in the pytracking conda environment
python_path = '/home/zj/anaconda3/envs/pytracking/bin/python';

% Set path to pytracking
pytracking_path = '/home/zj/tracking/metricNet/pytracking-master/pytracking';

% Set path to trax installation. Check
% https://trax.readthedocs.io/en/latest/tutorial_compiling.html for
% compilation information
trax_path = '/home/zj/tracking/vot-toolkit-master-new/native/trax';

tracker_name = 'atom';          % Name of the tracker to evaluate
runfile_name = 'atom_vot';    % Name of the parameter file to use
debug = 0;

%%
tracker_label = [tracker_name, '_', runfile_name];

% Generate python command
tracker_command = sprintf(['%s -c "import sys; sys.path.append(''%s'');', ...
                           'sys.path.append(''%s/support/python'');', ...
                           'import run_vot;', ...
                           'run_vot.run_vot(''%s'', ''%s'', debug=%d)"'],...
                           python_path, pytracking_path, trax_path, ...
                           tracker_name, runfile_name, debug);


tracker_interpreter = python_path;

tracker_linkpath = {[trax_path, '/build'],...
		[trax_path, '/build/support/client'],...
		[trax_path, '/build/support/opencv']};
