function [mncoefPA, stdcoefPA, mncoefPR, stdcoefPR, mncoefFM, stdcoefFM, state, pgcorrect, freqres, nrm_band_index]=readnrmcBA(nrmbdfname,age)

%%El fichero donde se almacenaran las normas tiene la sgte estructura
%%array[1..100] of char // para cualquier cosa
%%state
%%pgcorrect
%%nder
%%nages
%%nbands
%%ncols_nrm_band_index
%%nrm_band_index
%%freqres
%%normage
%%array[1..nder,1..nbands,1..nages] of regcoefPA : single;
%%array[1..nder,1..nbands,1..nages] of stdcoefPA : single;
%%array[1..nder,1..nbands-1,1..nages] of regcoefPR : single;
%%array[1..nder,1..nbands-1,1..nages] of stdcoefPR : single;
%%array[1..nder,1..nbands,1..nages] of regcoefFM : single;
%%array[1..nder,1..nbands,1..nages] of stdcoefFM : single;

%En este fichero se almacenan las normas para todas las bandas del modelo Banda Ancha.

nrmbdfname = setstr(nrmbdfname);

s = 'QEEGT Norms. Broad Band Model. Kernel Coefficients. Version 1.0';

if exist(nrmbdfname)
  fid = fopen(nrmbdfname,'r+');
  s1=fread(fid,100,'char*1');
  s1=setstr(s1);
  if strcmp(s1(1:length(s))',s) == 0
    error('Invalid File Type');
  end
  v=fread(fid,6,'integer*2');
  state=v(1);
  pgcorrect=v(2);
  nvar=v(3);
  nages=v(4);
  nbandas=v(5);
  ncols_nrm_band_index=v(6);
  nrm_band_index=fread(fid,[nbandas, ncols_nrm_band_index],'integer*2');

  %Leer las edades y las frecuencias
  freqres=fread(fid,1,'real*4'); %%resolucion en frecuencias de las normas
  ages=fread(fid,nages,'real*4'); %%rango de edades para los cuales se almacenan los coeficientes
  ptrbase = ftell(fid);
  age = log(age); %la regresion se hace contra al Log de la edad y las edades se almacenan en Log
  [mn,ageindex]=min(abs(ages-age));
  
  %Posicionarme en la edad corresp. para leer los minimos por deriv para la media
  ptr = ptrbase + (ageindex-1)*( (((3*nbandas - 1)*nvar)*4) *2 );
  %El desplaz. corresponde a los coef de la media y la std de los PA, PR y FM de las edades anteriores
  fseek(fid,ptr,'bof');
  mncoefPA  = fread(fid,[nvar, nbandas],  'real*4');
  stdcoefPA = fread(fid,[nvar, nbandas],  'real*4');
  mncoefPR  = fread(fid,[nvar, nbandas-1],'real*4');
  stdcoefPR = fread(fid,[nvar, nbandas-1],'real*4');
  mncoefFM  = fread(fid,[nvar, nbandas],  'real*4');
  stdcoefFM = fread(fid,[nvar, nbandas],  'real*4');
  fclose(fid);
else
  error('File does not exist');
end
