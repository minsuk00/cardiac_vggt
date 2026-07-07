function miitt_interslice( reconDir, repoDir, myDir )
%MIITT_INTERSLICE  Inter-slice cardiac synchronisation (mirrors README step 10).
%   Reads the intra-slice results + per-slice cines; no Philips dependency.
add_paths( repoDir, myDir );

dataDir     = fullfile( reconDir, 'data' );
cardsyncDir = fullfile( reconDir, 'cardsync' );
cineDir     = fullfile( reconDir, 'slice_cine_vol' );
M = matfile( fullfile( cardsyncDir, 'results_cardsync_intraslice.mat' ) );

% target slice (optional)
tgtLoc = NaN;
tgtLocFile = fullfile( dataDir, 'tgt_slice_no.txt' );
if exist( tgtLocFile, 'file' )
    fid = fopen( tgtLocFile, 'r' ); tgtLoc = fscanf( fid, '%f' ); fclose( fid );
end

% excluded slices (zero-indexed in file)
excludeSlice = [];
excludeSliceFile = fullfile( dataDir, 'force_exclude_slice.txt' );
if exist( excludeSliceFile, 'file' )
    fid = fopen( excludeSliceFile, 'r' ); excludeSlice = fscanf( fid, '%f' ) + 1; fclose( fid );
end

S = cardsync_interslice( M.S, 'recondir', cineDir, 'resultsdir', cardsyncDir, ...
                         'tgtloc', tgtLoc, 'excludeloc', excludeSlice );  %#ok<NASGU>
disp( 'INTERSLICE DONE' );
end
