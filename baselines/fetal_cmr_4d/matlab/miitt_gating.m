function miitt_gating( reconDir, repoDir, myDir )
%MIITT_GATING  Run preproc + intra-slice self-gating on one MIITT recon dir.
%   Standalone validation of the self-gating stage (no SVRTK recon needed).
add_paths( repoDir, myDir );
S = miitt_preproc( reconDir );
save( fullfile( reconDir, 'data', 'results.mat' ), 'S', '-v7.3' );
fprintf( 'preproc ok: nLoc=%d frameDur=%.4f s\n', S(1).nLoc, S(1).frameDuration );

S = cardsync_intraslice( S, 'resultsDir', fullfile( reconDir, 'cardsync' ), 'verbose', false );

hr = 60 ./ cell2mat( [ S.tRR ] );
fprintf( 'HR per slice (bpm): ' ); fprintf( '%.0f ', hr ); fprintf( '\n' );
fprintf( 'median HR = %.1f bpm  (range %.0f-%.0f)\n', ...
    median( hr, 'omitnan' ), min(hr), max(hr) );
disp( 'GATING DONE' );
end
