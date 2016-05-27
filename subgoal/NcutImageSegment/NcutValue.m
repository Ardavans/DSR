function ncut = NcutValue(t, U2, W, D);
% NcutValue - 2.1 Computing the Optimal Partition Ncut. eq (5)
%
% Synopsis
%  ncut = NcutValue(T, U2, D, W);
%
% Inputs ([]s are optional)
%  (scalar) t        splitting point (threshold)
%  (vector) U2       N x 1 vector representing the 2nd smallest
%                     eigenvector computed at step 2.
%  (matrix) W        N x N weight matrix
%  (matrix) D        N x N diagonal matrix
%
% Outputs ([]s are optional)
%  (scalar) ncut     The value calculated at the right term of eq (5).
%                    This is used to find minimum Ncut.
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
x = (U2 > t);
x = (2 * x) - 1; % convert [1 0 0 1 0]' to [1 -1 -1 1 -1]' to follow paper's way
d = diag(D);
k = sum(d(x > 0)) / sum(d);
b = k / (1 - k);
y = (1 + x) - b * (1 - x);
ncut = (y' * (D - W) * y) / ( y' * D * y );
end