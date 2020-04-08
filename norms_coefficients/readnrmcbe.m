function [meancoef, stdcoef, state, pgcorrect, freqres, freqsHz]=readnrmcBE(nrmbdfname,age)

%%El fichero donde se almacenan las normas tiene la sgte estructura
%%array[1..100] of char // para cualquier cosa
%%state
%%pgcorrect
%%nder
%%nages
%%nfreqs
%%normage
%%freqsHz
%%array[1..nder,1..nages] of minmed:single;
%%array[1..nder,1..nages] of maxmed:single;
%%array[1..nder,1..nages] of minstd:single;
%%array[1..nder,1..nages] of maxstd:single;
%%array[1..nder,1..nages] of firstvmed:single;
%%array[1..nder,1..nages] of firstvstd:single;
%%array[1..nder,1..nfreq,1..nages] of cdifmed : byte;
%%array[1..nder,1..nfreq,1..nages] of cdifstd : byte;

%En este fichero se almacenan las normas para todas las frecuencias. En un fichero FICHREG vienen
%los coeficientes para una frecuencia. Hay que ubicarlos en el lugar que les corresponda.
%Lo primero es comprobar que el fichero exista. Si no existe se crea, si no, se chequea si es
%compatible con otros resultados almacenados anteriormente

nrmbdfname = setstr(nrmbdfname);

s = 'QEEGT Norms. Kernel Coefficients. Version 1.0';
if exist(nrmbdfname)
  fid = fopen(nrmbdfname,'r+');
  s1=fread(fid,100,'char*1');
  s1=setstr(s1);
  if strcmp(s1(1:length(s))',s) == 0
    error('Invalid File Type');
  end
  v=fread(fid,5,'integer*2');
  state=v(1);
  pgcorrect=v(2);
  nvar=v(3);
  nages=v(4);
  nfreqs=v(5);
  %Leer las edades y las frecuencias
  freqres=fread(fid,1,'real*4'); %%resolucion en frecuencias de las normas
  ages=fread(fid,nages,'real*4'); %%rango de edades para los cuales se almacenan los coeficientes
  freqsHz=fread(fid,nfreqs,'real*4'); %%rango de frecuencias para el cual se almacenan los coefs.
  ptrbase = ftell(fid);
  age = log(age); %la regresion se hace contra al Log de la edad y las edades se almacenan en Log
  [mn,ageindex]=min(abs(ages-age));
  
  %Posicionarme en la edad corresp. para leer los minimos por deriv para la media
  ptr = ptrbase + (ageindex-1)*nvar*4;
  fseek(fid,ptr,'bof');
  minmed=fread(fid,nvar,'real*4'); %%para minmed
  %Posicionarme en la edad corresp. para leer los maximos por deriv para la media
  ptr = ptrbase + nages*nvar*4 + (ageindex-1)*nvar*4;
  fseek(fid,ptr,'bof');
  maxmed=fread(fid,nvar,'real*4'); %%para maxmed
  %Posicionarme en la edad corresp. para leer los minimos por deriv para la varianza
  ptr = ptrbase + 2*nages*nvar*4 + (ageindex-1)*nvar*4;
  fseek(fid,ptr,'bof');
  minstd=fread(fid,nvar,'real*4'); %%para minstd
  %Posicionarme en la edad corresp. para leer los maximos por deriv para la varianza
  ptr = ptrbase + 3*nages*nvar*4 + (ageindex-1)*nvar*4;
  fseek(fid,ptr,'bof');
  maxstd=fread(fid,nvar,'real*4'); %%para maxstd
  %Posicionarme en la edad corresp. para leer los primeros valores por deriv para la media
  ptr = ptrbase + 4*nages*nvar*4 + (ageindex-1)*nvar*4;
  fseek(fid,ptr,'bof');
  firstvmed=fread(fid,nvar,'real*4'); %%para firstvmed
  %Posicionarme en la edad corresp. para leer los primeros valores por deriv para la varianza
  ptr = ptrbase + 5*nages*nvar*4 + (ageindex-1)*nvar*4;
  fseek(fid,ptr,'bof');
  firstvstd=fread(fid,nvar,'real*4'); %%para firstvstd
  %Posicionarme en la edad corresp. para leer el resto de los valores por deriv para la media y la varianza
  ptr = ptrbase + 6*nages*nvar*4 + 2*(ageindex-1)*nvar*(nfreqs-1)*1; %1 byte
  fseek(fid,ptr,'bof');
  cdifmed=fread(fid,[nvar, nfreqs-1],'uint8');
  cdifstd=fread(fid,[nvar, nfreqs-1],'uint8');
  fclose(fid);
  meancoef = descomp(minmed, maxmed, firstvmed, cdifmed);
  stdcoef = descomp(minstd, maxstd, firstvstd, cdifstd);
  
else
  error('File does not exist');
end
