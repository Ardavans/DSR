function SegI = NcutImageSegment(I, SI, SX, r, sNcut, sArea)
% NcutImageSegment - Normalized Cuts and Image Segmentation [1]
%
% Synopsis
%  [SegI] = NcutImageSegment(I, SI, SX, r, sNcut, sArea)
%
% Description
%  Normalized Cuts and Image Segmentation [1]
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
%  (scalar) sNcut    The smallest Ncut value (threshold) to keep partitioning.
%  (scalar) sArea    The smallest size of area (threshold) to be accepted
%                    as a segment.
%
% Outputs ([]s are optional)
%  (cell)    SegI    cell array of segmented images of nRow x nCol x c.
%
% Requirements
%  NcutComputeW, NcutPartition, NcutValue
%
% References
%  [1] Jianbo Shi and Jitendra Malik, "Normalized Cuts and Image
%  Segmentation," IEEE Transactions on PAMI, Vol. 22, No. 8, Aug. 2000.
%  http://www.cs.berkeley.edu/~malik/papers/SM-ncut.pdf
%  [2] Graph Based Image Segmentation Tutorial
%  http://www.cis.upenn.edu/~jshi/GraphTutorial/
%  [3] MATLAB Normalized Cuts Segmentation Code
%  http://www.cis.upenn.edu/~jshi/software/
%  [4] D. Martin and C. Fowlkes and D. Tal and J. Malik, "A Database of
%  Human Segmented Natural Images and its Application to Evaluating
%  Segmentation Algorithms and Measuring Ecological Statistics",
%  Proc. 8th Int'l Conf. Computer Vision, vol. 2, pp. 416-423, July 2001.
%  http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
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
V = reshape(I, N, c); % connect up-to-down way. Vertices of Graph

% Step 1. Compute weight matrix W, and D
W = NcutComputeW(I, SI, SX, r);

% Step 5. recursively repartition
Seg = (1:N)'; % the first segment has whole nodes. [1 2 3 ... N]'
[Seg Id Ncut] = NcutPartition(Seg, W, sNcut, sArea, 'ROOT');

% Convert node ids into images
for i=1:length(Seg)
    subV = zeros(N, c); %ones(N, c) * 255;
    subV(Seg{i}, :) = V(Seg{i}, :);
    SegI{i} = uint8(reshape(subV, nRow, nCol, c));
    fprintf('%s. Ncut = %f\n', Id{i}, Ncut{i});
end
end