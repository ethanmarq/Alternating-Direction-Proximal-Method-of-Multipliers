function libsvm_to_mat(url, varargin)
%LIBSVM_TO_MAT  Download a LIBSVM-format dataset and convert it to .mat.
%
%   LIBSVM_TO_MAT(URL) downloads URL (which may point at a plain LIBSVM file
%   or a .bz2 / .gz archive of one), parses it, and writes a .mat file in
%   '/scratch/marque6/libsvm_data/' containing:
%
%       X            sparse  (n_samples x n_features)  -- feature matrix
%       Y            sparse  (n_samples x n_labels)    -- 0/1 label matrix
%                                                         (omitted if unlabeled)
%       label_names  cell array of original label ids, one per Y column
%
%   The output file name is derived from the URL basename (e.g.
%   rcv1_topics_train.svm.bz2  ->  rcv1_topics_train.mat). This drops
%   straight into load_spca.m / load_ssc.m via:
%
%       S = load(data_path);
%       X_data = S.X;
%
%   LIBSVM_TO_MAT(URL, 'OutDir', DIR) writes to DIR instead of the default.
%   LIBSVM_TO_MAT(URL, 'Name',   NAME) uses NAME.mat as the output filename.
%   LIBSVM_TO_MAT(URL, 'Overwrite', true) replaces an existing .mat file
%                                          (default: skip if it already exists).
%
%   Examples
%   --------
%       % Default location (/scratch/marque6/libsvm_data/):
%       libsvm_to_mat(['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/' ...
%                      'datasets/multilabel/rcv1_topics_train.svm.bz2']);
%
%       % Custom directory:
%       libsvm_to_mat(url, 'OutDir', '/path/to/dir');
%
%       % Custom filename:
%       libsvm_to_mat(url, 'Name', 'rcv1_train');
%
%   Requirements: curl and bunzip2 (or gunzip for .gz inputs) on PATH.

% ------------------------------- parse args -----------------------------
DEFAULT_OUTDIR = '/scratch/marque6/libsvm_data/';

p = inputParser;
addRequired(p,  'url',        @(s) ischar(s) || (isstring(s) && isscalar(s)));
addParameter(p, 'OutDir',     DEFAULT_OUTDIR, @(s) ischar(s) || isstring(s));
addParameter(p, 'Name',       '',             @(s) ischar(s) || isstring(s));
addParameter(p, 'Overwrite',  false,          @(x) islogical(x) && isscalar(x));
parse(p, url, varargin{:});

url       = char(p.Results.url);
out_dir   = char(p.Results.OutDir);
out_name  = char(p.Results.Name);
overwrite = p.Results.Overwrite;

% Derive output filename from URL if not supplied:
%   rcv1_topics_train.svm.bz2  ->  rcv1_topics_train
if isempty(out_name)
    [~, base, ext] = fileparts(url);
    if any(strcmpi(ext, {'.bz2', '.gz'}))
        [~, base, ~] = fileparts(base);    % strip the .svm (or whatever) too
    end
    out_name = base;
else
    [~, out_name, ~] = fileparts(out_name);   % strip user-supplied .mat
end

if ~exist(out_dir, 'dir')
    fprintf('Creating output directory: %s\n', out_dir);
    [ok, msg] = mkdir(out_dir);
    if ~ok
        error('libsvm_to_mat:mkdir', 'Could not create %s: %s', out_dir, msg);
    end
end

mat_path = fullfile(out_dir, [out_name, '.mat']);
if exist(mat_path, 'file') && ~overwrite
    fprintf('Output already exists, skipping: %s\n', mat_path);
    fprintf('  (pass ''Overwrite'', true to replace it.)\n');
    return;
end

% ------------------------- working directory ----------------------------
tmpdir  = tempname;
mkdir(tmpdir);
cleaner = onCleanup(@() rmdir(tmpdir, 's'));      %#ok<NASGU>

[~, fname, fext] = fileparts(url);
archive_path = fullfile(tmpdir, [fname, fext]);

% --------------------------- download (curl) ----------------------------
fprintf('[1/3] curl %s\n      -> %s\n', url, archive_path);
cmd = sprintf('curl -L --fail --retry 3 -sS -o "%s" "%s"', archive_path, url);
[status, msg] = system(cmd);
if status ~= 0
    error('libsvm_to_mat:download', ...
          'curl failed (exit %d): %s', status, strtrim(msg));
end

% ----------------------------- decompress -------------------------------
[~, ~, aext] = fileparts(archive_path);
if strcmpi(aext, '.bz2')
    fprintf('[2/3] bunzip2 %s\n', archive_path);
    [status, msg] = system(sprintf('bunzip2 -f "%s"', archive_path));
    if status ~= 0
        error('libsvm_to_mat:bunzip2', ...
              'bunzip2 failed (exit %d): %s', status, strtrim(msg));
    end
    text_path = archive_path(1:end-4);            % strip .bz2
elseif strcmpi(aext, '.gz')
    fprintf('[2/3] gunzip %s\n', archive_path);
    out = gunzip(archive_path, tmpdir);
    text_path = out{1};
else
    fprintf('[2/3] no decompression needed\n');
    text_path = archive_path;
end

% ------------------------------- parse ----------------------------------
fprintf('[3/3] parsing %s\n', text_path);
[X, Y, label_names] = local_parse_libsvm(text_path);                %#ok<ASGLU>

% ------------------------------- save -----------------------------------
fprintf('      writing %s\n', mat_path);
if isempty(Y)
    save(mat_path, 'X', '-v7');
else
    save(mat_path, 'X', 'Y', 'label_names', '-v7');
end

fprintf('Done.\n');
fprintf('  X: %d x %d sparse, nnz = %d\n', size(X,1), size(X,2), nnz(X));
if ~isempty(Y)
    fprintf('  Y: %d x %d sparse, nnz = %d (%d distinct labels)\n', ...
            size(Y,1), size(Y,2), nnz(Y), numel(label_names));
end
fprintf('  Load in MATLAB:  S = load(''%s''); X = S.X;\n', mat_path);

end


% ======================================================================= %
% Local: parse a LIBSVM (multi-label aware) text file into sparse X, Y    %
% ======================================================================= %
function [X, Y, label_names] = local_parse_libsvm(filepath)

    fid = fopen(filepath, 'r');
    if fid < 0
        error('libsvm_to_mat:fopen', 'Cannot open %s', filepath);
    end
    fid_cleaner = onCleanup(@() fclose(fid));                       %#ok<NASGU>

    % Pre-allocate in chunks; grow as needed. Trim at the end.
    CHUNK = 1e6;
    ii = zeros(CHUNK, 1);   jj = zeros(CHUNK, 1);   vv = zeros(CHUNK, 1);
    nnz_x = 0;

    lab_rows = zeros(CHUNK, 1);
    lab_strs = cell(CHUNK, 1);
    n_lab = 0;

    row      = 0;
    max_feat = 0;

    while true
        line = fgetl(fid);
        if ~ischar(line)               % EOF
            break;
        end
        line = strtrim(line);
        if isempty(line) || line(1) == '#'
            continue;
        end
        row = row + 1;

        tokens   = strsplit(line);     % splits on any whitespace, collapses runs
        startTok = 1;

        % First token is a label set iff it contains no ':' (features always do).
        if ~any(tokens{1} == ':')
            label_field = tokens{1};
            if ~isempty(label_field)
                labels = strsplit(label_field, ',');
                for k = 1:numel(labels)
                    lab = strtrim(labels{k});
                    if isempty(lab), continue; end
                    n_lab = n_lab + 1;
                    if n_lab > numel(lab_rows)
                        lab_rows(end + CHUNK) = 0;
                        lab_strs{end + CHUNK} = '';
                    end
                    lab_rows(n_lab) = row;
                    lab_strs{n_lab} = lab;
                end
            end
            startTok = 2;
        end

        for k = startTok:numel(tokens)
            tok   = tokens{k};
            colon = find(tok == ':', 1, 'first');
            if isempty(colon)
                error('libsvm_to_mat:format', ...
                      'Malformed token "%s" on line %d', tok, row);
            end
            f = sscanf(tok(1:colon-1), '%d');
            v = sscanf(tok(colon+1:end), '%g');
            if isempty(f) || isempty(v)
                error('libsvm_to_mat:format', ...
                      'Bad index/value "%s" on line %d', tok, row);
            end

            nnz_x = nnz_x + 1;
            if nnz_x > numel(ii)
                ii(end + CHUNK) = 0;
                jj(end + CHUNK) = 0;
                vv(end + CHUNK) = 0;
            end
            ii(nnz_x) = row;
            jj(nnz_x) = f;
            vv(nnz_x) = v;
            if f > max_feat
                max_feat = f;
            end
        end

        if mod(row, 5000) == 0
            fprintf('      ... %d rows parsed\n', row);
        end
    end

    n_samples  = row;
    n_features = max_feat;

    % Trim over-allocation.
    ii = ii(1:nnz_x);  jj = jj(1:nnz_x);  vv = vv(1:nnz_x);

    if n_samples == 0
        error('libsvm_to_mat:empty', 'File contained no parseable rows.');
    end
    X = sparse(ii, jj, vv, n_samples, n_features);

    % --- labels ---
    if n_lab == 0
        Y           = [];
        label_names = {};
        return;
    end

    lab_rows = lab_rows(1:n_lab);
    lab_strs = lab_strs(1:n_lab);

    unique_labels = unique(lab_strs);
    % Prefer numeric sort when all labels parse as numbers (matches LIBSVM
    % topic-id files like rcv1_topics_train, where labels are integers).
    nums = str2double(unique_labels);
    if all(~isnan(nums))
        [~, ord]      = sort(nums);
        unique_labels = unique_labels(ord);
    end

    label_to_col = containers.Map(unique_labels, ...
                                  num2cell(1:numel(unique_labels)));
    ycols = zeros(n_lab, 1);
    for k = 1:n_lab
        ycols(k) = label_to_col(lab_strs{k});
    end

    Y           = sparse(lab_rows, ycols, 1, n_samples, numel(unique_labels));
    label_names = unique_labels(:);
end
