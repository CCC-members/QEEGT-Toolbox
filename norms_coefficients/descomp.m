function X = descomp(minv, maxv, firstv, cdif)

[nf, nc] = size(cdif);
nc = nc+1;
X = zeros(nf,nc);

X(:,1)=firstv(:);

for k=1:nf
  X(k,2:nc) = int2real(cdif(k,:),minv(k),maxv(k));
end

for k=2:nc
  X(:, k) = X(:,k-1) + X(:, k);
end
