function S = miitt_preproc( reconDir )
%MIITT_PREPROC  fetal_cmr_4d preproc replacement for a MIITT real-time volunteer.
%
%   Builds the stack struct S that preproc.m would normally produce, but WITHOUT
%   any Philips ReconFrame PARAM (*_rlt_parameters.mat) files -- MIITT ships
%   already-reconstructed images, so timing/geometry are taken from MIITT's known
%   acquisition constants + the NIfTI header instead. Only the fields used by the
%   magnitude 4D pipeline (cardsync_intraslice/interslice + the recon bash
%   scripts) are populated. Flow-only fields (rltRe/Im/slw/trn) are omitted.
%
%   Writes the same text files preproc.m does (tgt_stack_no, slice_thickness,
%   force_exclude_*), which the recon_*.bash scripts read.

dataDir = fullfile( reconDir, 'data' );
maskDir = fullfile( reconDir, 'mask' );

% MIITT real-time acquisition constant (temporal; spacing is read from the NIfTI
% header below so metadata rebuilds -- e.g. the 8->10 mm through-plane fix -- are
% picked up automatically rather than hard-coded).
FRAME_DURATION  = 0.025;   % s per frame (25 ms/frame golden-angle spiral RT)

rltFileList = dir( fullfile( dataDir, '*_rlt_ab.nii.gz' ) );
nStack = numel( rltFileList );
if nStack == 0
    error( 'miitt_preproc:noData', 'no *_rlt_ab.nii.gz in %s', dataDir );
end

S = struct([]);
for iStk = 1:nStack
    desc = strrep( rltFileList(iStk).name, '_rlt_ab.nii.gz', '' );
    S(iStk).desc          = desc;
    S(iStk).rltAbFile     = fullfile( dataDir, rltFileList(iStk).name );
    S(iStk).dcAbFile      = fullfile( dataDir, sprintf( '%s_dc_ab.nii.gz', desc ) );
    S(iStk).maskHeartFile = fullfile( maskDir, sprintf( '%s_mask_heart.nii.gz', desc ) );
    S(iStk).rltMatFile    = '';  % unused: cardsync_intraslice (MIITT) reads the NIfTI

    R = load_untouch_nii( S(iStk).rltAbFile );
    S(iStk).niiHdr = R.hdr;
    dims = R.hdr.dime.dim;      % [ndim d1 d2 d3 d4 ...]
    nLoc = double( dims(4) );   % 3rd image dim = slice-locations
    nDyn = double( dims(5) );   % 4th image dim = real-time frames

    % slice thickness = the true RF slab (SVR slice PSF), NOT the pitch. MIITT protocol
    % (J. Hamilton 2026-07-04): 8 mm slice thickness + 2 mm gap = pixdim(4) 10 mm pitch.
    % The 2 mm gaps are unsampled (filled by the SR spatial prior); use pitch - gap = 8 mm.
    SLICE_GAP_MM = 2.0;
    sliceThickness = double( R.hdr.dime.pixdim(4) ) - SLICE_GAP_MM;

    S(iStk).nLoc           = nLoc;
    S(iStk).sliceThickness = sliceThickness;
    S(iStk).frameDuration  = FRAME_DURATION;

    % slice-sequential acquisition: each slice is nDyn frames acquired back-to-back
    sliceDuration = nDyn * FRAME_DURATION;
    for iLoc = 1:nLoc
        sliceStart = (iLoc-1) * sliceDuration;
        S(iStk).tFrame{iLoc} = sliceStart + FRAME_DURATION * (0:(nDyn-1));
    end
end

% --- text files consumed by the recon_*.bash scripts ---
write_txt( fullfile( dataDir, 'tgt_stack_no.txt' ), '1' );
fid = fopen( fullfile( dataDir, 'slice_thickness.txt' ), 'w' );
fprintf( fid, '%g ', [S.sliceThickness] );
fclose( fid );
write_txt( fullfile( dataDir, 'force_exclude_stack.txt' ), '' );
write_txt( fullfile( dataDir, 'force_exclude_slice.txt' ), '' );
write_txt( fullfile( dataDir, 'force_exclude_frame.txt' ), '' );

end  % miitt_preproc


function write_txt( p, s )
fid = fopen( p, 'w' );
fprintf( fid, '%s', s );
fclose( fid );
end
