function [Seg Cut Id Ncut] = NcutPartition(I, W, sNcut, sArea, cuts, id)
% NcutPartition - Partitioning
%
% Synopsis
%  [sub ids ncuts] = NcutPartition(I, W, sNcut, sArea, [id])
%
% Description
%  Partitioning. This function is called recursively.
%
% Inputs ([]s are optional)
%  (vector) I        N x 1 vector representing a segment to be partitioned.
%                    Each element has a node index of V (global segment).
%  (matrux) W        N x N matrix representing the computed similarity
%                    (weight) matrix.
%                    W(i,j) is similarity between node i and j.
%  (scalar) sNcut    The smallest Ncut value (threshold) to keep partitioning.
%  (scalar) sArea    The smallest size of area (threshold) to be accepted
%                    as a segment.
%  (ints)   cuts     cut indices
%  (string) [id]     A label of the segment (for debugg)
%
% Outputs ([]s are optional)
%  (cell)   Seg      A cell array of segments partitioned.
%                    Each cell is the each segment.
%  (cell)   Id       A cell array of strings representing labels of each segment.
%                    IDs are generated as children based on a parent id.
%  (cell)   Ncut     A cell array of scalars representing Ncut values
%                    of each segment.
%
% Requirements
%  NcutValue
%
% Authors
%  Naotoshi Seo <sonots(at)sonots.com>
%
% License
%  The program is free to use for non-commercial academic purposes,
%  but for course works, you must understand what is going inside to use.
%  The program can be used, modified, or re-distributed for any purposes
%  if you or one of your group understand codes (the one must come to
%  court if court cases occur.) Please contact the authors if you are
%  interested in using the program without meeting the above conditions.

% Changes
%  10/01/2006  First Edition
% Compute D
N = length(W);
d = sum(W, 2);
D = spdiags(d, 0, N, N); % diagonal matrix

% Step 2 and 3. Solve generalized eigensystem (D -W)*S = S*D*U (12).
% (13) is not necessary thanks to smart matlab.
% Get the 2 smallests ('sm')
warning off; % let me stop warning
[U,S] = eigs(D-W, D, 2, 'sm');
% 2nd smallest (1st smallest has all same value elements, and useless)
U2 = U(:, 2);
% T = abs(U2); m = max(T); T = T / m * 255; imshow(uint8(reshape(T, 15, 20)));

% Step 3. Refer 3.1 Example 3.
% (1). Bipartition the graph at 0 (hopefully, eigenvector can be
% splitted by + and -). % This did not work well.
%A = find(U2 > 0);
%B = find(U2 <= 0);
% (2). Bipartition the graph at median value.
%t = median(U2);
%A = find(U2 > t);
%B = find(U2 <= t);
% (3). Bipartition the graph at point that Ncut is minimized.
t = mean(U2);
t = fminsearch('NcutValue', t,  optimset('MaxFunEvals', 50000), U2, W, D);
A = find(U2 > t);
B = find(U2 <= t);

%find cut point
[c cut_index] = min(abs(U2-t));
% append cut to the list
cuts = [cuts; cut_index];

% Step 4. Decide if the current partition should be divided
% if either of partition is too small, stop recursion.
% if Ncut is larger than threshold, stop recursion.
ncut = NcutValue(t, U2, W, D);

%length(A)
%length(B)
%ncut
if (length(A) < sArea || length(B) < sArea) || ncut > sNcut
    Seg{1}   = I;
    Id{1}   = id; % for debugging
    Ncut{1} = ncut; % for duebuggin
    Cut{1} = cuts;
    return;
end

% Seg segments of A
[SegA CutA IdA NcutA] = NcutPartition(I(A), W(A, A), sNcut, sArea, cuts, [id '-A']);
% I(A): node index at V. A is index at the segment, I
% W(A, A); % weight matrix in segment A

% Seg segments of B
[SegB CutB IdB NcutB] = NcutPartition(I(B), W(B, B), sNcut, sArea, cuts, [id '-B']);

% concatenate cell arrays
Seg   = [SegA SegB];
Id   = [IdA IdB];
Ncut = [NcutA NcutB];
Cut = [CutA CutB];
end
