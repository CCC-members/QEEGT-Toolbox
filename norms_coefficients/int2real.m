function r = int2real(e, minr, maxr);

%Convierte el numero entero e en el intervalo [0:255] al numero real r en el intervalo [minr:maxr]

r = minr + e.*(maxr-minr)./255;