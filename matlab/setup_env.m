function setup_env(path_to_python, full_path_to_project)
    % 1) Locate yourself into the main project folder
    cd (full_path_to_project);
    % 2) Add the 'matlab' folder to MATLAB's paths
    path_to_matlab_folder = fullfile(full_path_to_project,'matlab');
    addpath(genpath(path_to_matlab_folder));
    % 3) Set Python environment in MATLAB
    if ispc
        % Windows requires that the Python process be external to MATLAB,
        % otherwise it generates the warning "Unrecognized command line option: -json."
        % which will prevent the tracker from starting.
        pyenv('ExecutionMode','OutOfProcess', 'Version',path_to_python);
    elseif isunix
        pyenv('Version',path_to_python);
    end
    pyenv('ExecutionMode','OutOfProcess', 'Version',path_to_python);
    % 4) Add the tracker folder to PYTHONPATH
    insert(py.sys.path,int32(0),fullfile(path_to_matlab_folder,'tracker'));
end