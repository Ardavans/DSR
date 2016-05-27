function W = NcutComputeW(I, SI, SX, r);
% NcutComputeW - Compute a similarity (weight) matrix
%
% Synopsis
%  W = NcutComputeW(I, SI, SX, r)
%
% Inputs ([]s are optional)
%  (matrix) I        nRow x nCol x c matrix representing image.
%                    c is 3 for color image, 1 for grayscale image.
%                    Let me define N = nRow x nCol.
%  (scalar) SI       Coefficient used to compute similarity (weight) matrix
%                    Read [1] for meanings.
%  (scalar) SX       Coefficient used to compute similarity (weight) matrix
%                    Read [1] for meanings.
%  (scalar) r        Coefficient used to compute similarity (weight) matrix
%                    Definition of neighborhood.
%
% Outputs ([]s are optional)
%  (matrux) W        N x N matrix representing the computed similarity 
%                    (weight) matrix.
%                    W(i,j) is similarity between node i and j.
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
[nRow, nCol, c] = size(I);
N = nRow * nCol;
W = sparse(N,N);

% Feature Vectors
if c == 3
    F = F3(I);
else
    F = F2(I);
end
F = reshape(F, N, 1, c); % col vector
% Spatial Location, e.g.,
% [1 1] [1 2] [1 2]
% [2 1] [2 2] [2 3]
X = cat(3, repmat((1:nRow)', 1, nCol), repmat((1:nCol), nRow, 1));
X = reshape(X, N, 1, 2); % col vector

% Future Work: Reduce computation to half. It can be done
% because W is symmetric mat
for ic=1:nCol
    for ir=1:nRow
        % matlab tricks for fast computation (Avoid 'for' loops as much as
        % possible, instead use repmat.)

        % This range satisfies |X(i) - X(j)| <= r (block distance)
        jc = (ic - floor(r)) : (ic + floor(r)); % vector
        jr = ((ir - floor(r)) :(ir + floor(r)))';
        jc = jc(jc >= 1 & jc <= nCol);
        jr = jr(jr >= 1 & jr <= nRow);
        jN = length(jc) * length(jr);

        % index at vertex. V(i)
        i = ir + (ic - 1) * nRow;
        j = repmat(jr, 1, length(jc)) + repmat((jc -1) * nRow, length(jr), 1);
        j = reshape(j, length(jc) * length(jr), 1); % a col vector

        % spatial location distance (disimilarity)
        XJ = X(j, 1, :);
        XI = repmat(X(i, 1, :), length(j), 1);
        DX = XI - XJ;
        DX = sum(DX .* DX, 3); % squared euclid distance
        %DX = sum(abs(DX), 3); % block distance
        % square (block) reagion may work better for skew lines than circle (euclid) reagion.

        % |X(i) - X(j)| <= r (already satisfied if block distance measurement)
        constraint = find(sqrt(DX) <= r);
        j = j(constraint);
        DX = DX(constraint);

        % feature vector disimilarity
        FJ = F(j, 1, :);
        FI = repmat(F(i, 1, :), length(j), 1);
        DF = FI - FJ;
        DF = sum(DF .* DF, 3); % squared euclid distance
        %DF = sum(abs(DF), 3); % block distance

        % Hint: W(i, j) is a col vector even if j is a matrix
        W(i, j) = exp(-DF / (SI*SI)) .* exp(-DX / (SX*SX)); % for squared distance
        %W(i, j) = exp(-DF / SI) .* exp(-DX / SX);
    end
end
end

% F1 - F4: Compute a feature vector F. See 4 EXPERIMENTS
%
%  F = F1(I) % for point sets
%  F = F2(I) % intensity
%  F = F3(I) % hsv, for color
%  F = F4(I) % DOOG
%
%  Input and output arguments ([]'s are optional):
%   I (scalar or vector). The image.
%   F (scalar or vector). The computed feature vector F
%
% Author : Naotoshi Seo
% Date   : Oct, 2006
% for point sets
function F = F1(I);
F = (I == 0);
end
function F = F2(I);
% intensity, for gray scale
F = I;
end
function F = F3(I);
% hsv, for color
F = I; % raw RGB
% Below hsv resulted in errors at eigs(). eigs returns erros so often.
%  F = rgb2hsv(double(I)); % V = [0, 255] with double, V = [0, 1] without double
%  % any fast way in matlab?
%  [nRow nCol c] = size(I);
%  for i=1:nRow
%      for j=1:nCol
%          HSV = reshape(F(i, j, :), 3, 1);
%          h = HSV(1); s = HSV(2); v = HSV(3);
%          F(i, j, :) = [v v*s*sin(h) v*s*cos(h)];
%      end
%  end
end
function F = F4(I);
% DOOG, for texture
% Future
end

function X = coord(i, nRow, nCol);
% After all, I did not use this
% coord: Convert vertex index into spatial coordinates
%
%  X = coordinate(n, nRow, nCol)
%
%  Input and output arguments ([]'s are optional):
%   n (scalar or col vector). The vertex index of graph
%   nRow (scalar). The # of rows in original image
%   nCol (scalar). The # of cols in original image
%   X (2 cols vector). The spatial (image) coordinates whose 1st col
%   expresses row, and 2nd col expressed col.
%
% Hint: reshape() connects into column way (up-to-down)
%
% Author : Naotoshi Seo
% Date   : Oct, 2006
i = i - 1; % let me start from 0 to make mod easy
row = mod(i, nRow);
col = floor(i / nRow);
% matlab index starts from 1
row = row + 1;
col = col + 1;
X = [row col];
end
