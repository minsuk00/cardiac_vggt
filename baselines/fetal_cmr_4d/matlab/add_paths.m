function add_paths( repoDir, myDir )
%ADD_PATHS  Put the MIITT-adapted fetal_cmr_4d functions on the MATLAB path.
%   myDir goes FIRST so the patched cardsync_intraslice.m shadows the repo copy.
addpath( myDir );                                  % miitt_preproc, patched cardsync_intraslice
addpath( fullfile( repoDir, 'cardsync' ) );        % estimate_heartrate_xf, calc_cardiac_timing, calc_freq, cardsync_interslice
addpath( fullfile( repoDir, '4drecon' ) );         % (preproc reference; we use miitt_preproc)
addpath( fullfile( repoDir, 'vis' ) );             % save_figs
addpath( genpath( fullfile( repoDir, 'lib', 'nifti' ) ) );        % load_untouch_nii
addpath( genpath( fullfile( repoDir, 'lib', 'general_tools' ) ) );
end
