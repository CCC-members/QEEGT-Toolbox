function qeegt(eeg_fname, state, lwin, fmin, freqres, fmax, wbands, brain, pg_apply, flags, output_folder)

% eeg_fname: File which contains the EEG data in PLG (Neuronic) format. Does mot include extension
% state:     EEG state to be selected for calculations
% lwin:      Length of epochs, in seconds. Example: 2.56 seg
% fmin:      Low cut frequency for analysis
% freqres:   Frequency resolution for analysis
% fmax:      High cut frequency for analysis
% wbands:    Broad Bands definition in string format. For example:
%                 wbands='1.56 3.51; 3.9 7.41; 7.8 12.48; 12.87 19.11; 1.56 11';
% brain:     1 indicates to restrict the inverse solution to only gray matter
% pg_apply:  1 indicates to apply correction by the General Scale Factor
%flags: Options for calculations in string format. For example:
%       flags='1 1 1 0 1 1 1 1 1 1 1 1';
%  flags(1)- 1 for calculating the Broad Band Spectral Model
%  flags(2)- 1 for calculating the Narrow Band Spectral Model
%  flags(3)- 1 for calculating the Spectra at the EEG sources
%  flags(4)- RESERVED, not used
%  flags(5)- 1 for calculating the Broad Band Spectral Model Z values
%  flags(6)- 1 for calculating the Narrow Band Spectral Model Z values
%  flags(7)- 1 for calculating the Sources Spectra Z values
%  flags(8)- 1 for calculating the correlations matrix bewteen all pairs of channels for each epoch
%  flags(9)- 1 for calculating the coherence matrix bewteen all pairs of channels for each frequency
% flags(10)- 1 for calculating the phase difference matrix bewteen all pairs of channels for each frequency
% flags(11)- 1 for calculating the frequency domain correlations bewteen all pairs of channels for each frequency and each epoch
% flags(12)- Store the XYZ components of the solutions at the sources
% output_folder: Folder name where to store the results. It should end with the folder separation character.
%            If empty, the results are saved in the same directory as the input data
%%%%%%%% NEW %%%%%%%%%%
% eeg_fname: can also be the name of an ASCII file (with a fixed structure)
% which contains the data of an EEG file. In that case, the file needs to
% have the extension ".txt" and mnust have the following structure:
% NAME=Jane Doe
% SEX=F
% AGE=20.94
% SAMPLING_FREQ=200
% EPOCH_SIZE=512
% NCHANNELS=19
% MONTAGE=
% Fp1-REF
% Fp2-REF
% F3_-REF
%and so on. The program expects NCHANNELS lines with the names
%AFTER THE CHANNELS NAMES THE EEG DATA
%where each lione is an instant of time and each column represents a channel.
%If the EEG contains 30 segments of 512 points each and 19 channels, then
%30*512 lines of 19 columns of numbers (either float or integer) are expected


SI_correction = 1;
pathnrm='';
pathbrain='';

try
    wbands = str2num(wbands);
catch
    error('Incorrect definition of the parameter "wbands"')
end

try
    flags = str2num(flags);
catch
    error('Incorrect definition of the parameter "flags"')
end

try
    if ~isnumeric(lwin), lwin = str2num(lwin); end
catch
    error('Incorrect definition of the parameter "lwin"')
end

try
    if ~isnumeric(fmin), fmin = str2num(fmin); end
catch
    error('Incorrect definition of the parameter "fmin"')
end

try
    if ~isnumeric(freqres), freqres = str2num(freqres); end
catch
    error('Incorrect definition of the parameter "freqres"')
end

try
    if ~isnumeric(fmax), fmax = str2num(fmax); end
catch
    error('Incorrect definition of the parameter "fmax"')
end

try
    if ~isnumeric(brain), brain = str2num(brain); end
catch
    error('Incorrect definition of the parameter "brain"')
end

try
    if ~isnumeric(pg_apply), pg_apply = str2num(pg_apply); end
catch
    error('Incorrect definition of the parameter "pg_apply"')
end

try
    if exist('output_folder', 'var')
        if ~isempty(output_folder) && (output_folder(end) ~= filesep)
            output_folder(end+1) = filesep;
        end
    else
        output_folder = '';
    end
catch
    output_folder = '';
end

if ~isempty(output_folder)
    if ~isdir(output_folder)
        try
            mkdir(output_folder);
        catch
            output_folder = '';
        end
    end
end

if isa(eeg_fname, 'char')
   [pp nn ee] = fileparts(eeg_fname);
   if strcmpi(ee, '.txt')
       [eeg, ok] = load_txt(eeg_fname);
       if ~ok
           error('Unknown TXT file format')
       else
           [pp nn ee] = fileparts(eeg_fname);
           if ~isempty(pp), pp = [pp filesep]; end
           if isempty(output_folder)
               eeg.file_res = [pp nn '.res'];
           else
               eeg.file_res = [output_folder nn '.res'];
           end
           eeg_fname = eeg;
           clear eeg
       end
   end       
end

if isa(eeg_fname, 'struct')
    try
        data = eeg_fname.data;
    catch
        error('eeg_fname should contain a field named "data", which is a matrix of #channels x #times');
    end
    
    try
        montage = eeg_fname.montage;
    catch
        error('eeg_fname should contain a field named "montage", with the name of channels, including the reference. Example: Fp1-REF. REF can also be "A12" for EAR LINKED, "Cz" foe Cz reference, etc');
    end
    
    try
        age = eeg_fname.age;
    catch
        error('eeg_fname should contain a field named "age", which the subject''s age');
    end
    
    try
        SAMPLING_FREQ = eeg_fname.SAMPLING_FREQ;
    catch
        error('eeg_fname should contain a field named "SAMPLING_FREQ", which the sampling frequency in Hz of the data');
    end
    
    try
        epoch_size = eeg_fname.epoch_size;
    catch
        error('eeg_fname should contain a field named "epoch_size", which the length of an epoch size. The number of instants of timne in data should epoch_size*#ofwindows');
    end
    
    try
        file_res = eeg_fname.file_res;
    catch
        error('eeg_fname should contain a field named "file_res", with the name of the file where to save the results, without Extension');
    end
    
elseif isa(eeg_fname, 'char')
    
    [pp nn ee] = fileparts(eeg_fname);
    if ~isempty(pp)
        eeg_fname = [pp filesep nn];
    else
        eeg_fname = nn;
    end
    
    [data, montage, age, SAMPLING_FREQ, epoch_size, wins, msg] = read_plgwindows(eeg_fname, state, lwin);

    [pp, nn, ee] = fileparts(eeg_fname);
    if ~isempty(pp), pp = [pp filesep]; end
    if ~isempty(output_folder)
        pp = output_folder;
    end
    file_res = [pp nn];

else
    error('eeg_fname variable is not of correct type');
end

% inf_info = read_plginf([eeg_fname '.inf']);

montage = setstr(montage);
sp = 1/SAMPLING_FREQ;
nit = round(lwin./sp);

%quedarse con ventanas que sean multiplos de nit
if length(epoch_size) == 1
    nepochs = size(data, 2) ./ epoch_size;
    epoch_size = epoch_size .* ones(nepochs, 1);
end
epoch_size = epoch_size(:);
starts = cumsum([1; epoch_size(1:end-1)]);
longs = nit.*floor(epoch_size ./ nit);
ends = starts+longs-1;
ind = [];
for h=1:length(starts)
    ind = union(ind, [starts(h):ends(h)]);
end

%seleccionar los electrodos a incluir
indices=[1:19];  %the first 19 channels of the 1020 system
montage = montage(indices, :);

data = data(indices,:);
nvt=floor(size(data,2)./nit);

% %garantizar montage AVR
for k=1:size(montage,1)
    montage(k,5:7)='AVR';
end

qeegt_all(data, age, state, montage, nit, sp, fmax, fmin, freqres, wbands, flags, pathnrm, pathbrain, pg_apply, SI_correction, brain, file_res);

end


function [eeg, ok] = load_txt(eeg_fname)
eeg = [];
ok = 0;
try
    fid = fopen(eeg_fname, 'r');
    nlines = 0;
    while ~feof(fid)
        txt = fgetl(fid); nlines = nlines+1;
        [desc, rest] = strtok(txt, '=');
        if isempty(rest) %no encontro el signo igual. O es un error o ya vienen los datos
            if ~all(isfield(eeg, {'age', 'montage', 'nchannels', 'SAMPLING_FREQ', 'epoch_size'}))
                return
            end
            nlines = nlines-1;
            fclose(fid);
            eeg.data = textread(eeg_fname, '%f', 'headerlines', nlines);
            nvt = length(eeg.data) ./ eeg.nchannels;
            eeg.data = reshape(eeg.data, eeg.nchannels, nvt);
            ok = 1;
        else
            rest(1) = '';
            rest = strtrim(rest);
            desc = lower(desc);
            switch lower(desc)
                case {'name', 'sex'}
                case 'sampling_freq'
                    rest = str2num(rest);
                    desc = 'SAMPLING_FREQ';
                case {'age', 'epoch_size', 'nchannels'}
                    rest = str2num(rest);
                case 'montage'
                    if ~isfield(eeg, 'nchannels')
                        fclose(fid);
                        return
                    end
                    mtg = cell(eeg.nchannels, 1);
                    for k=1:eeg.nchannels
                        mtg{k} = fgetl(fid); nlines = nlines+1;
                        mtg{k} = strrep(mtg{k}, '_', ' ');
                        mtg{k} = strtrim(mtg{k});
                    end
                    rest = char(mtg);
                otherwise
                    return
            end
            eeg.(desc) = rest;
        end
    end
    fclose(fid);
catch
    return
end
end


function qeegt_all(data, age, state, montage, nit, sp, fmax,...
    fmin, freqres, wbands, flags, pathnrm, pathbrain, pg_apply, SI_correction, brain, results_file)

%%%%%%Calcular
[fmin, freqres, fmax, FEG, flags, MCross, indfreq_in_Z, newmont, ZSpec, PA, PR, FM, ZPA, ZPR, ZFM, Sinv, ZSinv, lambda, CorrM, Coh, Phase, Jxyz,  Spt, indnocero] = ...
    qeegtker1_calc(data, age, state, montage, nit, sp, fmax, fmin, freqres, wbands, flags, pathnrm, pathbrain, pg_apply, SI_correction, brain);


newmont = montage;
if size(newmont,1) == 1, newmont = newmont'; end
% if size(newmont,1) ~= 118,
%     newmont = [newmont; zeros(118-size(newmont,1), size(newmont,2))];
% end
indfreq_in_Z = indfreq_in_Z';
lambda = lambda(:);
freqrange = [fmin:freqres:fmax];
mont = montage; mont(:,end-2:end) = repmat('AVR', size(mont,1),1);
ii = find(mont == '_'); mont(ii) = ' ';
ii = find(newmont == '_'); newmont(ii) = ' ';

if size(ZSpec,2) > length(indnocero)
    ZSpec = ZSpec(:, indnocero);
end
if size(ZPA,1) > length(indnocero)
    ZPA = ZPA(indnocero, :);
end
if size(ZPR,1) > length(indnocero)
    ZPR = ZPR(indnocero, :);
end
if size(ZFM,1) > length(indnocero)
    ZFM = ZFM(indnocero, :);
end

savemods(results_file, state, mont, freqrange, FEG, MCross, indfreq_in_Z, newmont, ZSpec, PA, PR, FM, ZPA, ZPR, ZFM, Sinv, ZSinv, lambda, SI_correction, CorrM, Coh, Phase, Jxyz);

% saveresults([results_file '_qeegt'], fmin, freqres, fmax, FEG, flags, MCross, indfreq_in_Z, newmont, ZSpec, PA, PR, FM, ZPA, ...
%     ZPR, ZFM, Sinv, ZSinv, lambda, SI_correction, CorrM, Coh, Phase, Jxyz);
end

% nd = size(data,1);
% dnames = s1020_dnames;
% frange = fmin:freqres:fmax;
% for k=1:nd
%     mysubplot(5,4,k, 0.03, 0.03, 0.03, 0.03, 0.05, 0.03);
%     plot(frange, ZSpec(1:min(48,size(ZSpec,1)),k), 'b',[fmin fmax], [1.96 1.96], 'r', [fmin fmax], [-1.96 -1.96], 'r');
%     title(['Z values ' dnames{k}]);
%  end

function [mncoefPA, stdcoefPA, mncoefPR, stdcoefPR, mncoefFM, stdcoefFM, state, pgcorrect, freqres, nrm_band_index]=readnrmcba(nrmbdfname,age)

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
end

function [PA, PR, FM, ZPA, ZPR, ZFM, flags] = banda_ancha(Spec, flags, wbands, fmin, freqres, fmax, nit, sp, montage, fn_eqreg_ba, age)
%Poner los limites de las bandas de frec. de banda ancha como multiplos de la resolucion en
%frecuencias real

wbands = freqres .* round(wbands ./ freqres);

freqrange = [fmin:freqres:fmax];

ndt = size(Spec,1);

%Si se va a normar BA compatibilizar las resoluc. en frecuencia

if flags(5) %Calcular las normas de banda ancha
    [PAMedia, PASigma, PRMedia, PRSigma, FMMedia, FMSigma, ba_state, ba_pgcorrect, ba_freqres, nrm_band_index]=readnrmcba(fn_eqreg_ba,age);
    if freqres - ba_freqres > 0.1,  %La resoluc. en frec. de los datos es muy grande. No se puede obtener
        %medidas de BA compatibles con las normas
        flags(5) = 0; %no se va a normar BA
        gap = 1;
    else
        gap = round(ba_freqres ./ freqres);
        
        nb=size(wbands,1);
        nbnorm=size(nrm_band_index,1);
        w_nrm_band_index = ba_freqres * nrm_band_index;
        posnorm=zeros(nb,1);
        %Chequear si las definiciones de las bandas coinciden con las bandas normadas
        for i=1:nb
            for j=1:nbnorm
                if (abs(wbands(i,1)-w_nrm_band_index(j,1))< 1.0e-1) & (abs(wbands(i,2)-w_nrm_band_index(j,2))<1.0e-1),
                    posnorm(i)=j;
                end
            end
        end
        
        %Si ninguna banda coincide con una de las normadas entonces no se pueden calcular las normas
        %para banda ancha
        if sum(posnorm) == 0
            flags(5) = 0;
        end
    end
else
    gap = 1;
end


[PA, PR, FM, btotal] = bbsp(Spec, freqrange, freqres, nit, sp, wbands, gap);

%Transformar PA por Log, PR por Logit, FM no se transforma
ind=find(PA > 0);
tPA = PA;
tPA(ind)=log(tPA(ind));
tPR = logit(PR);

nbandasba=size(wbands,1);
if flags(5),
    newmont=match_montage(montage');
    indnocero = find(newmont ~= 0);
    nderZ = size(newmont,1);
    
    ZPA=zeros(nderZ, nbandasba);
    ZFM=zeros(nderZ, nbandasba);
    ZPR=zeros(nderZ, nbandasba-1);
    for j=1:nbandasba,
        if (posnorm(j) > 0 & posnorm(j) <= nbandasba)
            ZPA(indnocero,j) = (tPA(newmont(indnocero),j) - PAMedia(indnocero,posnorm(j))) ./ PASigma(indnocero,posnorm(j));
            ZFM(indnocero,j) = (FM(newmont(indnocero),j) - FMMedia(indnocero,posnorm(j))) ./ FMSigma(indnocero,posnorm(j));
        end
    end
    
    %Para la Z del PR hay que tener en cuenta que si una de las bandas era la banda total, esta se le quito al PR
    nbandaspr = size(PR,2);
    if ~isempty(btotal)
        posnorm(btotal)=[];
    end
    for j=1:nbandaspr,
        if (posnorm(j) > 0 & posnorm(j) <= nbandaspr)
            ZPR(indnocero,j) = (tPR(newmont(indnocero),j) - PRMedia(indnocero,posnorm(j))) ./ PRSigma(indnocero,posnorm(j));
        end
    end
else
    ZPA = -1; ZPR = -1; ZFM = -1;
end

end
function [PA, PR, MF,btotal] = bbsp(Spect, freqrange, freqres, nit, sp, wbands, gap)

%SPECT es una matriz de nder x nfreq

%Matchear las frec. de wbands con las de freqrange para determinar los indices con los que se corresponden
nbandasba=size(wbands,1);
band_index = zeros(size(wbands));
for k=1:nbandasba
    for l=1:size(wbands,2)
        [mini,ii]=min(abs(wbands(k,l)-freqrange));
        if ~isempty(ii),
            band_index(k,l)=ii(1);
        else
            error('Invalid broad band frequency range')
        end
    end
end

nder = size(Spect,1);
nb=size(band_index,1);
bTotal = [min(band_index(:)) max(band_index(:))];

PA=zeros(nder,nb);
PR=zeros(nder,nb);
MF=zeros(nder,nb);

%save('c:\res0', freqrange, band_index, bTotal,Spect);
PTotal=sum(Spect(:,bTotal(1):gap:bTotal(2))')';

for k=1:nb,
    wb=band_index(k,1):gap:band_index(k,2);
    PA(:,k)=sum(Spect(:,wb)')';
    ind = find(PTotal); %Para protegerse de que haya algun PTotal que sea cero. Esto puede ocurrir si algun comemierda se pone a jugar
    %con los EEGs. Crealo o no
    PR(ind,k)=PA(ind,k)./PTotal(ind);
    wb_FM = freqrange(band_index(k,1))./freqres:gap:freqrange(band_index(k,2))./freqres;
    
    % save('c:\res1',wb_FM, freqrange, band_index, freqres, gap);
    wb_FM = round(wb_FM);
    %save('c:\res2',wb_FM);
    
    %%  MF(:,k)=sum((Spect(:,wb).*((freqres*gap)*wb(ones(ndt,1),:)))')' ./ (nit*sampper*PA(:,k));
    
    MF(:,k)=sum((Spect(:,wb).*((freqres*gap)*wb_FM(ones(nder,1),:)))')';
    ind = find(PA(:, k));  %Para protegerse de que haya algun PTotal que sea cero. Esto puede ocurrir si algun comemierda se pone a jugar
    %con los EEGs. Crealo o no
    MF(ind,k) = MF(ind,k)./ PA(ind,k);
end

indout = [];
for k=1:nb, %Ver si existe alguna banda total para quitarla del PR, porque siempre es 1
    if (band_index(k,1) == bTotal(1)) & (band_index(k,2) == bTotal(2))
        indout = [indout k];
    end
end
if ~isempty(indout)
    PR(:,indout) = [];
end
btotal = indout;

%Corregir al PA un factor de escala de 2.56 (supuestamente el tamao de las ventanas de analisis) para que sea igual al TW2
factor = nit*sp;
if factor > eps
    PA =PA ./ factor;
end
end

function Spect = calc_sp(ffteeg, nder)

%ffteeg son los coef. de Fourier calculados a partir de los segmentos de EEG por la funcion FFTCALC. Es una
%matriz de nder*nvt x nfreq

nfreq = size(ffteeg,2);
nvt = ceil(size(ffteeg,1) ./ nder);

Spect=zeros(nder,nfreq);
for i=1:nvt
    f=ffteeg((i-1)*nder+1:i*nder,:);
    Spect = Spect + abs(f).^2;
end

% Spect=Spect./nvt;
end

function ZSinv_j = calc_zsinv(Sinv_j, indfreq_in_Z, indnorma, cur_freq, SIMedia, SISigma);
if indfreq_in_Z(cur_freq) > 0
    ZSinv_j = (Sinv_j - SIMedia(:,indnorma(cur_freq))) ./ SISigma(:,indnorma(cur_freq));
else
    ZSinv_j = -1;
end
end

function Coh = coherence(MCross)
nder = size(MCross,1);
d = reshape(diag(MCross), nder, 1);
denom = sqrt(reshape(kron(d,d),nder,nder));
Coh = abs(MCross ./ denom);
Coh(1:nder+1:end) = 0; %the coherence in the diagonal is always 1 and it is not interesting

% Coh = MCross(:, var_number) ./ sqrt(MCross(var_number,var_number) * diag(MCross));
end

function Coh = coherence_allfreq(MCross)
[nd2,nf] = size(MCross);
nd = sqrt(nd2);
Coh = zeros(nd,nd,nf);
for k=1:nf
    MC = com2her(MCross(:,k));
    Coh(:,:,k)=coherence(MC);
end

end

function mat_H = com2her(Comp)
if size(Comp,2) > 1
    nd = size(Comp,1);
else
    nd=length(Comp);
    Comp = Comp(:);
end

nd=sqrt(nd);
if fix(nd)-nd~=0
    error('Matrix can not be square')
end
nf = size(Comp,2);
mat_H = zeros(nd, nd, nf);
for k=1:nf
    mat_R=reshape(Comp(:,k),nd,nd);
    tri=tril(mat_R,-1);
    mat_H(:,:,k)=triu(mat_R)+triu(mat_R,1)' +i.*(tri'-tri);
end

%#inbounds
%mbreal(Comp); mbreal(mat_R); mbreal(tri);

end

function nrm_fname = concat_nrm_fname(state, mont, pg_apply, model, brain, SI_correction, path_file)
%state          char (A, B, C...)
%mont:          1-Referencia Promedio,       2-Record,      3-Laplaciano
%pg_apply:      0-Do not apply PG,           1-Apply PG
%model:         1-Narrow Band,               2-Broad Band,  3-Electrical Tomography
%               4-ET LF,                     5-Operator for spectra reconstruction
%               6-Lambda value
%brain:         1-Just gray matter           2-Gray matter + Basal Ganglia
%SI_correction: 1-Factor Global              2-Factor por frecuencias       0-No correction
%path_file: directory where the file must be found

if mont == 1, %%Montaje AVR
    mnt = 'AVR';
elseif mont == 2, %%Montaje Record
    mnt = 'REC';
elseif mont == 3, %%Montaje Laplac
    mnt = 'LAP';
else
    mnt = '';
end

if pg_apply
    lpg = 'PG';
else
    lpg = 'RD';
end

if model == 1  %%Banda Estrecha
    modelo = 'NB';
    ext = '.NRM';
elseif model == 2  %%Banda Ancha
    modelo = 'BB';
    ext = '.NRM';
elseif model == 3  %%Tomografia
    modelo = 'ET';
    ext = '.NRM';
elseif model == 4  %%Operador de regularizacion
    modelo = 'LF';
    ext = '.mat';
elseif model == 5  %%Operador para reconstruir el espectro
    modelo = 'SR';
    ext = '.OPR';
elseif model == 6  %%Fichero con el lambda
    modelo = 'LA';
    ext = '.DAT';
end

path_file = path_file(:)';

if ~isempty(path_file)
    if path_file(length(path_file)) ~= '\'
        path_file = [path_file '\'];
    end
end

if model == 4 %%Caso del Lead Field, que no depende del estado ni del poder geometrico
    state = '_';
    lpg = '__';
end

nrm_fname = [path_file state '_' mnt '_' lpg '_' modelo '_'];

if model > 2 %%Tomografia, LF, operador para reconstruir el espectro o lambda
    if brain == 1  %%corteza solamente
        tbrain = 'GM';
    elseif brain == 2  %%corteza mas ganglios basales
        tbrain = 'BG';
    end
    nrm_fname = [nrm_fname tbrain];
end

if model == 3 %%Tomografia
    if SI_correction == 1
        correct = '_GF';
    elseif SI_correction == 2
        correct = '_FF';
    elseif SI_correction == 0
        correct = '_NC';
    else
        correct = '';
    end
    nrm_fname = [nrm_fname correct];
end


nrm_fname = [nrm_fname ext];
nrm_fname = setstr(nrm_fname);
end

function CorrM = corr_cal(eegdata, nit)
%nit es la long de un segmento
nvt=floor(size(eegdata,2)./nit); % numero de ventanas a procesar
nder = size(eegdata, 1);
eegdata = reshape(eegdata, nder, nit, nvt);
CorrM = zeros(nder, nder, nvt);

for k=1:nvt
    [tmp, CorrM(:,:,k)] = cov_corr(eegdata(:,:,k), eegdata(:,:,k));
end
end

function CorrF = corrfreq_cal(fftdata, nder)
%nder es numero de derivaciones
nvt=floor(size(fftdata,1)./nder); % numero de segmentos
nfreq = size(fftdata,2);
fftdata = reshape(fftdata, nder, nvt, nfreq);
CorrF = zeros(nder, nder, nvt+1, nfreq);

for cur_freq = 1:nfreq
    for k=1:nvt
        V = fftdata(:, k, cur_freq);
        M = (V*V');
        CorrF(:, :, k, cur_freq) = Fcorrel(M);
    end
end
CorrF(:,:,end,:) = mean(CorrF(:,:,1:end-1,:), 3);



end

function Corr = Fcorrel(MCross)
nder = size(MCross,1);
d = reshape(diag(MCross), nder, 1);
denom = sqrt(reshape(kron(d,d),nder,nder));
Corr = real(MCross) ./ denom;
Corr = triu(Corr);
end

function [S, R] = cov_corr(series1, series2);
%series1 is a matrix of #voxels1 x #times.
%series2 is a matrix of #voxels2 x #times.
%the function calculates the correlation matrix of all points in both
%matrices, producing a resultant matrix of #voxels2 x #voxels1.
%There are three resultant matrices:
%S(i,j): is the covariance of the i-th voxel in series2 Vs. the j-th voxel in series1
%R(i,j): is the correlation of the i-th voxel in series2 Vs. the j-th voxel in series1

nvoxels = size(series2,1);

series2=series2';

%Calculating covariances
T=size(series2,1);
H=eye(T,T) - (1/T)*ones(T,T);
S=(1./(T-1))*series1*H*series2;

series1=series1';

%This is not the whole covariance matrix of all voxels, but just the cov
%matrix of the selected voxels in vector voxels

if nargout > 1 %It means calculate the correlation matrix
    %Calculating variances
    Sigma1 = calc_var(series1);
    Sigma2 = calc_var(series2);
    
    nvoxels = length(Sigma1);
    nvar = length(Sigma2);
    tmp = sqrt(Sigma2);
    S_ii = repmat(tmp, nvoxels, 1);
    tmp = sqrt(Sigma1);
    S_jj = repmat(tmp',1,nvar);
    
    R = zeros(size(S));
    tmp =(S_ii.*S_jj);
    ind = find(tmp < eps);
    %as we are going to divide by tmp, if tmp is too small, then after division I will set the result to zero
    if ~isempty(ind), tmp(ind)=1; end %so the division does not creates problem
    %In this step sometime MEMORY OVERFLOW happens. So, I make a cycle and divide it in pieces of 500 columns
    totcols = size(tmp,2); ncols = 500;
    ndiv = ceil(totcols ./ ncols);
    for h=1:ndiv
        cols = [(h-1)*ncols+1:min([h*ncols totcols])];
        R(:, cols) = S(:, cols) ./ tmp(:, cols);
    end
    if ~isempty(ind), R(ind) = 0; end
    R = R';
end

S = S';

return

% %For Testing with a small example
% Sigma = cov(series2);
% sig = zeros(size(Sigma));
%
% n=size(Sigma,1);
% for k=1:n
%     for h=1:n
%         sig(k,h) = (sqrt(T) .* Sigma(k,h)) ./ (sqrt((Sigma(k,k) .* Sigma(h,h)) + (Sigma(k,h).^2)));
%     end
% end
%
% % tmp = reshape(S, nX*nY*nSlices, nvoxels);
% % tmp = tmp';
% max(max(abs(S - Sigma(:,voxels))))
%
% corr = corrcoef(series2);
% % tmp = reshape(R, nX*nY*nSlices, nvoxels);
% % tmp = tmp';
% max(max(abs(R - corr(:,voxels))))
%
% % tmp = reshape(S_Test, nX*nY*nSlices, nvoxels);
% % tmp = tmp';
% max(max(abs(S_Test - sig(:,voxels))))
%
%
end

function Sigma = calc_var(series);
n = size(series,2); nh = floor(n/2);
Sigma = zeros(1, n);
Sigma(1:nh) = std(series(:, 1:nh)).^2;
Sigma(nh+1:n) = std(series(:, nh+1:n)).^2;
end

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
end

function result = existsf(nfile);
fid = fopen( nfile, 'r' );
if fid == -1,
    result = 0;
else
    result = 1;
    fclose(fid);
end
%#realonly
%#inbounds

end

function MCross = fft2mcross(fftdata,ndt,avg_segs)

%fftdata es de (ndt*nvt,nfreq)
nfreq = size(fftdata, 2);

nvt = ceil(size(fftdata,1) ./ ndt);
if avg_segs
    MCross = zeros(ndt*ndt,nfreq);
else
    MCross = zeros(ndt*ndt,nfreq, nvt);
end

for cur_freq = 1:nfreq
    fftdata_freq = reshape(fftdata(:,cur_freq), ndt, nvt);
    if avg_segs
        M = (fftdata_freq*fftdata_freq');
%         M = (fftdata_freq*fftdata_freq')./nvt;
        M=her2com(M);
        MCross(:, cur_freq) = M(:);
    else
        for h=1:nvt
            M = (fftdata_freq(:,h)*fftdata_freq(:,h)');
            M=her2com(M);
            MCross(:, cur_freq, h) = M(:);
        end
    end
end
end

function [ffteeg, real_freqinterval] = fft_calc(eegdata, nt, sp)

%sp es el periodo de muestreo
%nt es la long de un segmento

real_freqres = 1 ./ (sp*nt);
real_fmin = real_freqres;
real_fmax = 1 ./ (2*sp);
real_freqinterval = [real_fmin, real_freqres, real_fmax];

nvt=floor(size(eegdata,2)./nt); % numero de ventanas a procesar
nder = size(eegdata, 1);

ffteeg = zeros(nder*nvt,round(nt./2));
A=zeros(nt,nder);

%hamm = hamming(nt);
%WARNING: VENTANAS DE HAMMING DISABLED POR COMPATIBILIDAD CON EL TW2
hamm = ones(nt,1);

for i=1:nvt
    A=eegdata(:,(i-1)*nt+1:i*nt)' .* hamm(:, ones(nder,1));  %Se multiplica por hamming
    f = fft(A)';
    f = f(:, 2:round(nt./2)+1); %quitar el primer coef porque es una linea de base y ademas, quedarse con
    %la mitad de los coeficientes por la frecuencia de Nyquist
    
    %%%%OJO: COMPATIBILIDAD CON EL TW2%%%%%%%
    f = f ./ sqrt(nt);
    %%%%OJO: COMPATIBILIDAD CON EL TW2%%%%%%%
    
    %  f(:,1) = f(:,2); %Esto es una cochinada aqui porque el primer coef. de la FFT sale mal
    ffteeg((i-1)*nder+1:i*nder,:) = f;
end

%%%%OJO: COMPATIBILIDAD CON EL TW2%%%%%%%
SamplingFreqHz = 1./sp;
TWfactor = (1000./(2 * pi * SamplingFreqHz * nvt * 100));
ffteeg = ffteeg * sqrt(TWfactor);
%%%%OJO: COMPATIBILIDAD CON EL TW2%%%%%%%
end

function [data, freqrange, nfreq, freqindex] = freq_manage(data, fmin, fmax, freqres, real_freqres, real_fmax);

%%%%%OJO: SI SE HACE ALGUN CAMBIO EN ESTE PROCEDIMIENTO, ACORDARSE QUE ESTE PROCED. SE USA TAMBIEN
%%%%%     EN CALCMCRO. VER COMO LO AFECTA

freqrange=[fmin:freqres:fmax];
freqrange = freqrange(:);
nfreq=length(freqrange);

%Buscar el indice de cada frecuencia en freqrange en el arreglo original de frecuencias
freqindex=zeros(nfreq,1);
real_freqrange = [real_freqres:real_freqres:real_fmax];
for k=1:nfreq
    ii=find(abs(freqrange(k)-real_freqrange) < 0.1);
    if ~isempty(ii),
        if length(ii) == 1
            freqindex(k)=ii;
        else
            [mm,ii]=min(abs(freqrange(k)-real_freqrange));
            freqindex(k)=ii;
        end
    else
        error('Invalid frequency range')
    end
end

%Quedarse con las frecuencias seleccionadas por parametro para los datos
data = data(:,freqindex);
end

function [data_freq] = get_data_freq(data, ndt, cur_freq, newmont)
%Aqui viene el calculo de la solucion inversa

%data es de (ndt*nvt,nfreq)
nvt = ceil(size(data,1) ./ ndt);
data_freq = reshape(data(:,cur_freq), ndt, nvt);

%Quedarse con las 19 deriv del 1020, y garantizar ponerlas en el orden usual del sistema 1020,
%que es el que se espera para el calculo de las SI
data_freq = data_freq(newmont(1:19),:);

return;

%%  data_freq = data_freq ./ sqrt(nvt); %esto creo que no hace falta hacerlo porque se hace en fftcalc
%para garantizar la compatibilidad con el TW2

%%Con motivos de prueba M = (data_freq*data_freq') debe ser igual a la matriz crossespectral a esa frecuencia


%%%Esto fue para probar que las matrices crossespectrales daban igual lo mismo si vienen del TW2, que si se calculan con calcmcro, que si se trabaja con la fft
%%%y no se llega a calcular la matriz cossespectral directamente.
%%%OJO: Recordar que para que la matriz crossespctral de igual que la del TW2, no se puede aplicar referencia promedio ni poder geometricoy que en la matriz
%%%cross del TW2 no viene la frecuencia 0.3906

%load c:\datos\sama\samamod
%MCross=transpone(MCross);  %%esto es para corregir el error del programa crossmod.dll

%M = (data_freq*data_freq');
%M=her2com(M);
%M = reshape(M, nd1, nd1);

%%load c:\datos\sama\sama.txt
%%eeg=sama'; clear sama
%%minfreq=0.3906; freqres=0.3906; maxfreq=19.11;
%%nt = 512;
%%[Spect,S, nvt] = calcmcro(eeg, minfreq, maxfreq, freqres, nt);
%%tmp2 = reshape(S(:,cur_freq), nd1, nd1);

%%  kk= M ./ tmp2;
%%  kk(1:5, 1:5)
%%  disp(max(max(abs(kk))))

%if cur_freq > 1
%    tmp1 = reshape(MCross(:,cur_freq-1), nd1, nd1);
%    kk= M ./ tmp1;
%    kk(1:5, 1:5)
%    disp(max(max(abs(kk))))
%
%    [tmp,factor]=pg(MCross);
%    tmp1 = reshape(tmp(:,cur_freq-1), nd1, nd1);
%    kk= M ./ tmp1;
%    kk(1:5, 1:5)
%    disp(max(max(abs(kk))))
%
%end
end

function [mont, montage] = get_montage(montage);
%%Devuelve:
%%  1 si el montaje es Referencia Promedio
%%  2 si el montaje es Record
%%  3 si el montaje es Laplaciano de 19 canales del S1020 estandar
%% -1 si el montaje es desconocido

montage = setstr(montage);

[m,n]=size(montage);
if (m==7) | (n==7)
    if n == 7
        montage = montage'; transp = 1;
    else
        transp = 0;
    end
    cmp = montage(4:7,1)';
    if strcmp('-AVR', cmp) > 0
        mont = 1;
    elseif strcmp('-A1 ', cmp) > 0
        mont = 2;
    elseif strcmp('-A2 ', cmp) > 0
        mont = 2;
    elseif strcmp('-A12', cmp) > 0
        mont = 2;
    elseif strcmp('-REF', cmp) > 0
        mont = 2;
    elseif strcmp('-L19', cmp) > 0
        mont = 3;
    else
        mont = -1;
    end
    
    montage = montage(1:3,:);
    if transp
        montage = montage';
    end
else
    mont = -1;
end
end

function mat_C = her2com(mat_H);

mat_C=triu(real(mat_H))-tril(imag(mat_H));
mat_C=mat_C(:);
end

function r = int2real(e, minr, maxr);

%Convierte el numero entero e en el intervalo [0:255] al numero real r en el intervalo [minr:maxr]

r = minr + e.*(maxr-minr)./255;
end

function [X, lambda, Glambda, reg_param, G] = invreg_c(U,s,V,Y,autom_lambda,lambda, leftbound,rightbound);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = invreg(U,s,V,Y,lambda);
% finds optimal regularization parameter lambda
%
%      lambda =   arg min G(l)= || P(l) Y ||^2 ./ Trace(n P(l))^2
%
% for using in Tikhonov regularization
%
%         min { || A x - Y ||^2 + lambda^2 ||L x ||^2 } (function tikh),
%
% where
%
%      P(l) = I-A inv(A' A+l^2 I) A'
%
% and
%
%      A = U diag(s) V' by compact SVD (csvd).
%
%
% Input arguments:
%
%   U          - left orthognal matrix from SVD of A
%   s          - column vector of singular values of A
%   V          - right orthognal matrix from SVD of A (if only the first r columns of V are passed
%              - then only the first r components of x are calculated)
%   Y          - data matrix
%   lambda     - regularization parameter when it is input (Then it is not calculated)
%
% Output arguments:
%
%   lambda  - optimal regularization parameter
%   X       - solution when Oper is input else it is the regularization operator.
%   Glambda - G(lambda) Note when this parameter is present a plot of
%             the GCV curve is created
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%# scalar autom_lambda n m p npoints smin_ratio No_Ok ratio tmp np Glambda minGi

[n,m]=size(Y);
p=length(s);
onesm=ones(1,m);
s2=s.^2;
UtY = U'*Y;

if nargin < 6
    lambda = -1;
end

if nargin < 7
    leftbound = -1;
end

if nargin < 8
    rightbound = [];
end

rho2_v = []; %esto es para engaar al compildor del Matcom y que lo cree como una matriz

if isempty(lambda) | (lambda == -1)
    % Set defaults.
    npoints = 100;                  % Number of points on the curve.
    smin_ratio = 16*eps;           % Smallest regularization parameter.
    % Initialization.
    
    No_Ok = 1;
    reg_param = zeros(npoints,1); G = reg_param;
    while No_Ok
        reg_param(npoints) = max([s(p),s(1)*smin_ratio,leftbound]);
        if isempty(rightbound)
            ratio = 1.2*(s(1)./reg_param(npoints)).^(1/(npoints-1));
            for i=npoints-1:-1:1, reg_param(i) = ratio*reg_param(i+1); end
        else
            reg_param = linspace(rightbound, reg_param(npoints), npoints)';
        end
        for i=1:npoints
            tmp = reg_param(i) * reg_param(i);
            f1 = tmp ./ (s2 + tmp);
            fb = (f1*onesm).*UtY;
            rho2_v = fb(:)'*fb(:);
            tmp = sum(f1);
            tmp = tmp * tmp;
            G(i) = rho2_v / tmp;
        end
        
        [mGlambda, mminGi] = min(G);
        lmin = minloc(G);
        np = length(lmin);
        if np == 1,
            minGi = lmin;
        else
            ind=find((lmin == npoints) | (lmin == 1));
            lmin(ind)=[];
            if isempty(lmin)
                if G(npoints) > G(1)
                    minGi = 1;
                else
                    minGi = npoints;
                end
            else
                [aa,bb]=min(G(lmin));
                minGi = lmin(bb);
            end
        end
        Glambda = real(G(minGi));
        lambda= reg_param(minGi);
        
        %if not autom_lambda
        %Mostrar ventana con Tres opciones:
        % - Aceptar
        % - Cambiar el limite inferior y recalcular
        % - Cambiar el lambda a mano
        % Si escogio Aceptar o Lambda a mano, poner No_Ok = 0;
        %end
        
        if 0
            %  figure
            loglog(reg_param,G,'-'), xlabel('lambda'), ylabel('G(lambda)')
            title('GCV function')
            HoldState = ishold; hold on;
            loglog([lambda,lambda],[Glambda/1000,Glambda],':')
            loglog(lambda,Glambda,'*');
            title(['GCV function, minimum at ',num2str(lambda)])
            if (~HoldState), hold off; end
            drawnow; pause
        end
        
        No_Ok = 0;
    end
    
    %  mbrealscalar(npoints);
    %  mbrealvector(reg_param); mbrealvector(G);
    %  mbreal(f1); mbrealvector(ratio);
else
    Glambda = -1;
end %end de u

% si se desea solucion inversa
X=(V*(((s./(s2+lambda.^2))* onesm).*UtY));

%return

%%%%ESTO ES UNA VERSION CON RANKRANGE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%umbral=1e-8;
%tolerance=s2(1)*umbral;
%indexes=find(s2>=tolerance);
%matrixrank=length(indexes);
%rankrange=1:matrixrank;
%UtY=UtY(rankrange,:);
%s2=s2(rankrange);
%X1=V(:,rankrange)*(diag(s(rankrange)./(s2+lambda.^2))*UtY);
%%%%ESTO ES UNA VERSION CON RANKRANGE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%mbrealvector(Glambda);
%mbreal(U); mbreal(s); mbreal(V);
%mbrealscalar(autom_lambda);

end

function [Mat1]=loadmat1(filename)

fid=fopen(filename,'r');
if fid == -1, error(['File ' filename ' does not exist']); end

%loading matrix 1
[x, c]=fread(fid, 2, 'real*4');
if c ~= 2, error( ['File ' filename ' is invalid']); end
Mat1=zeros(x(1), x(2));
[Mat1, c]=fread(fid, x', 'real*4');
if c ~= prod(x), error( ['File ' filename ' is invalid']); end

c=fclose(fid);

%#inbounds
%#realonly
%mbintscalar(c);

end

function y = logit(x)
y = zeros(size(x));
i1 = find(x < eps);
y(i1) = -36;
i2 = find(x > 0.9999999999999999);
y(i2) = 36;
ind = setdiff(1:numel(x),[i1(:); i2(:)]);
y(ind) = log(x(ind) ./ (1-x(ind)));

end

function [indnorma, indfreq_in_Z, nfreqs_norm] = match_freq_norm(freqrange,norm_freqsHz)
%ver de las frecuencias calculadas para cuales existe norma

nfreq=length(freqrange);
indnorma = zeros(nfreq,1);
indfreq_in_Z = zeros(nfreq,1); %Indica la columna de Z que se corresponde con la frecuencia k. Un valor
%cero indica que no hay columna en Z correspond. a esa frec., es decir,
%que no hay norma para esa frecuencia
current = 0;
for k=1:nfreq
    ind=find(abs(freqrange(k)-norm_freqsHz)<0.1);
    if ~isempty(ind),
        indnorma(k)=ind;
        current = current+1;
        indfreq_in_Z(k)=current;
    end
end
nfreqs_norm = length(find(indnorma > 0));
end

function [newmont, indnocero]=match_montage(montage)

%#inbounds
%#realonly

if size(montage,1) < size(montage,2)
    montage = montage';
end

montage = setstr(montage);
montage = mayus(montage);

nder_norms=118;
newmont=zeros(nder_norms,1);

%derivaciones que estan normadas. Derivaciones 56 y 57 no tienen norma
nderiv = cell(nder_norms,2);
for k=1:55
    nderiv{k,1} = num2str(k);
    nderiv{k,2} = '';
end
for k=58:120
    nderiv{k-2,1} = num2str(k);
    nderiv{k-2,2} = '';
end
nderiv{1,2} = 'FP1';
nderiv{2,2} = 'FP2';
nderiv{3,2} = 'F3';
nderiv{4,2} = 'F4';
nderiv{5,2} = 'C3';
nderiv{6,2} = 'C4';
nderiv{7,2} = 'P3';
nderiv{8,2} = 'P4';
nderiv{9,2} = 'O1';
nderiv{10,2} = 'O2';
nderiv{11,2} = 'F7';
nderiv{12,2} = 'F8';
nderiv{13,2} = 'T3';
nderiv{14,2} = 'T4';
nderiv{15,2} = 'T5';
nderiv{16,2} = 'T6';
nderiv{17,2} = 'FZ';
nderiv{18,2} = 'CZ';
nderiv{19,2} = 'PZ';

mm = cell(size(montage,1),1);
for k=1:size(montage,1)
    mm{k} = strtok(montage(k,:),'-');
    mm{k} = strtok(mm{k},'_');
    mm{k} = deblank(mm{k});
end

for h=1:size(nderiv,1)
    for k=1:size(montage,1)
        if strcmp(nderiv{h,1},mm{k}) || strcmp(nderiv{h,2},mm{k})
            newmont(h)=k; break;
        end
    end
end

indnocero = find(newmont ~= 0);
newmont = newmont(:);
indnocero = indnocero(:);
end

function y = mayus(x)

%#inbounds
%#realonly

y=zeros(size(x));
[nf,nc]=size(x);

st = abs(x);
alphabet = [abs('a'):abs('z')];
for k=1:nf,
    for l=1:nc,
        if ~isempty(find(st(k,l) == alphabet)),
            y(k,l) = x(k,l)-32;
        else,
            y(k,l)=x(k,l);
        end;
    end
end
y=setstr(y);
end

function lmin = minloc(x)

%#realonly
%#inbounds
%# scalar i l

if ~isreal(x), x = real(x); end

lmin = [];
if size(x,1) == 1, x = x'; end
if size(x,2) ~= 1, return; end

ll = size(x,1);  l = ll(1);
% if x(1) < x(2), lmin = 1; end
for i = 2:l-1
    if (x(i-1) > x(i)) & (x(i) < x(i+1)),
        lmin = [lmin; i];
    end
end
% if x(l-1) > x(l), lmin = [lmin; l]; end

if isempty(lmin), [kk, lmin]=min(x); return; end

ratio = zeros(length(lmin),1);
for k=1:length(lmin)
    ii = find((x(lmin(k)-1:-1:1) - x(lmin(k))) < 0);
    if ~isempty(ii), nantes = ii(1)-1; else nantes = lmin(k)-1; end
    
    ii = find((x(lmin(k)+1:end) - x(lmin(k))) < 0);
    if ~isempty(ii), ndesp = ii(1)-1; else ndesp = length(x) - lmin(k); end
    
    ratio(k) = min(nantes, ndesp);
end

ii = find(ratio < 3);
lmin(ii) = [];

if isempty(lmin), [kk, lmin]=min(x); return; end

end

function M = new2old_str(M);

nd = sqrt(size(M,1));
for k=1:size(M,2)
    temp = reshape(M(:,k), nd, nd);
    temp = triu(temp)' - tril(temp,-1)';
    M(:,k) = temp(:);
end
end

function Phase = phase_delay(MCross)
[nd2,nf] = size(MCross);
nd = sqrt(nd2);
Phase = zeros(nd,nd,nf);
for k=1:nf
    MC = com2her(MCross(:,k));
    Phase(:,:,k)=angle(MC);
end

end

function [fmin, freqres, fmax, FEG, flags, MCross, indfreq_in_Z, newmont, ZSpec, ...
    PA, PR, FM, ZPA, ZPR, ZFM, Sinv, ZSinv, lambda, CorrM, Coh, Phase, Jxyz, Spec, indnocero] = ...
    qeegtker1_calc(data, age, state, montage, nit, sp, fmax, fmin, freqres, wbands, ...
    flags, pathnrm, pathbrain, pg_apply, SI_correction, brain)

%      data - matriz de #deriv * #trials

%%state es un char (A, B, C...)

%% brain:
%%       1-Corteza
%%       2-Corteza + Ganglios Basales
%%pg_apply:
%%       0-Do not apply PG,
%%       1-Apply PG
%%pathbrain: directory where the file must be found

%flags
%  1- Calcular banda ancha
%  2- Calcular banda estrecha
%  3- Calcular soluciones inversas
%  4- RESERVED     %%%ANTERIORMENTE: Calcular espectros reconstruidos
%  5- Calcular normas de banda ancha
%  6- Calcular normas de banda estrecha
%  7- Calcular normas de soluciones inversas
%  8- Calcular coeficientes de correlacion entre derivaciones por segmentos
%  9- Calcular matriz de coherencias entre las derivaciones por frecuencias
% 10- Calcular matriz de diferencia de fases entre las derivaciones por frecuencias
% 11- Calcular matriz de correlaciones en la frecuencia para cada segmento
% 12- Salvar las XYZ de las fuentes en un fichero texto


%#inbounds
%#realonly

%%para matcom %# scalar nit state sp age fmin fmax freqres pg_apply fm ndt
%%para matcom %# scalar real_fmin, real_fmax, real_freqres factor nfreq ng

flags(4)=0; %%Desactivarla

montage = setstr(montage);
pathnrm = setstr(pathnrm);
pathbrain = setstr(pathbrain);

mont = get_montage(montage);
%mont:      1-Referencia Promedio,       2-Record,      3-Laplaciano


if mont == 1, %%Montaje AVR
    quitar_rp = 1;
    laplac = 0;
elseif mont == 2, %%Montaje Record
    quitar_rp = 0;
    laplac = 0;
elseif mont == 3, %%Montaje Laplac
    quitar_rp = 0;
    laplac = 1;
else  %%montaje desconocido
    quitar_rp = 0;
    laplac = 0;
end

fn_eqreg_be  = concat_nrm_fname(state, mont, pg_apply, 1, brain, SI_correction, pathnrm);
fn_eqreg_ba  = concat_nrm_fname(state, mont, pg_apply, 2, brain, SI_correction, pathnrm);
fn_eqreg_si  = concat_nrm_fname(state, mont, pg_apply, 3, brain, SI_correction, pathnrm);
fn_LF  = concat_nrm_fname(state, mont, pg_apply, 4, brain, SI_correction, pathbrain);
fn_lambda = concat_nrm_fname(state, mont, pg_apply, 6, brain, SI_correction, pathbrain);

if laplac  %Solo existe para la topografia. Se inhabilitan todos los de soluc. inversas
    flags(4) = 0; flags(7) = 0;
    fname = pathnrm(:)';
    fname = [fname filesep 'Le_surf1020.mat'];
    try
        load(fname, 'Le')
        %      if ~exist('Le'), error(['Laplacian matrix ' fname ' is not the proper type']); end
    catch
        error(['Laplacian matrix ' fname ' not found']);
    end
end

if ~existsf(fn_eqreg_be)
    flags(6)=0;
    msg0 = 'No hay normas de Banda Estrecha para el estado';
end

if ~existsf(fn_eqreg_ba)
    flags(5)=0;
    msg2 = 'No hay normas de Banda Ancha para el estado';
end

if ~existsf(fn_eqreg_si)
    flags(7)=0;
    msg1 = 'No hay normas de soluciones inversas para el estado';
end

if ~existsf(fn_LF)
    flags(3)=0; flags(4) = 0;
    msg1 = 'No hay Lead Field para soluciones inversas';
end

if ~quitar_rp %No hay normas sin referencia promedio. Por tanto, se deshabilitan las flags de las normas
    flags(3)=0; flags(4)=0; flags(7)=0;
end

if flags(3) | flags(4) | flags(7)
    lambda = loadmat1(fn_lambda);
else
    lambda = 0;
end

if (age < 4) | (age > 90) %Se inhabilitan las normas
    flags(5) = 0; flags(6) = 0; flags(7) = 0;
end

%Se van a calcular medidas Z. TEMPORALMENTE vamos a exigir que sea el mismo rango de frecuencias
%con el que se calcularon las normas. Esto es hasta que se arregle el problema de calcular el factor
%de escala para que no afecte los calculos en una resolucion de frecuencias diferente
if (flags(6) | flags(7))
    freqres = 0.3906; fmin = freqres*2; fmax = freqres * 49;
end

ndt = size(data,1);  %esto se puede cambiar por un codigo Pascal

[newmont, indnocero]=match_montage(montage);

ndernormas = min([ndt 19]);

if quitar_rp  %quitar la referencia promedio a los datos
    data = refprom(data, ndt, ndernormas);
end

if laplac == 1 %hacer transformacion Laplaciano. Por el momento solo se hace para el sistema 1020
    %%Chequear si es el sistema 1020
    %%Hacer transformacion Laplaciano
    try
        data = Le * data;
    catch
        error(['Laplacian matrix ' fname ' is not the expected one']);
    end
end

if flags(8)
    CorrM = corr_cal(data, nit);
else
    CorrM = -1;
end

%%Calcular las matrices cross-espectrales para salvarlas en un .MOD. Esto se hace unicamente por
%%compatibilidad con el TW2, ya que este calculo no se utiliza ni para calcular los espectros ni las
%%soluciones inversas. Asimismo, el factor de escala global, se calcula siempre para poder
%%guardarlo en el .MOD, pero no se le quita a la matriz cross que se devuelve para crear el .MOD

[data, real_fextremos]  = fft_calc(data, nit, sp);
MCross = fft2mcross(data, ndt, 1);

%calcular los coeficientes de Fourier
real_fmin = real_fextremos(1);
real_freqres = real_fextremos(2);
real_fmax = real_fextremos(3);

%Garantizar que fmin, fmax y freqres sean multiplos de resol. en frecuencias verdadera
if freqres < real_freqres, freqres = real_freqres; end
if fmax > real_fmax, fmax = real_fmax; end
if fmin < real_freqres, fmin = real_freqres; end

freqres = real_freqres .* (round(freqres ./ real_freqres));
fmin = real_freqres .* (round(fmin ./ real_freqres));
fmax = real_freqres .* (round(fmax ./ real_freqres));

% if (flags(6) | flags(7)) %Se van a calcular normas. Ver si tienen el mismo rango de frecuencias
%     if (abs(freqres - 0.3906) > 0.1) | ((abs(fmin - 0.3906) > 0.1) &  (abs(fmin - 0.3906*2) > 0.1)) | (abs(fmax - 0.3906*49) > 0.1)
%         flags(5)=0;  flags(6)=0;  flags(7)=0; %quitar el calculo de las normas
%     end
% end
%
[data, freqrange, nfreq, freqindex] = freq_manage(data, fmin, fmax, freqres, real_freqres, real_fmax);
nfreq = size(freqrange,1);
%en freq_manage DATA se transforma de una matriz de datos en el tiempo a una matriz de coeficientes
%de Fourier en la frec

%Restringir MCross al rango de frecuencias seleccionado
MCross = MCross(:, freqindex);

%Calcular el espectro
Spec = calc_sp(data, ndt);

if flags(9)
    Coh = coherence_allfreq(MCross);
else
    Coh = -1;
end

if flags(10)
    Phase = phase_delay(MCross);
else
    Phase = -1;
end

if flags(11)
    CorrFreq = corrfreq_cal(data, ndt);
else
    CorrFreq = -1;
end

MCross = new2old_str(MCross);  %% Esto es para almacenar MCross con el mismo convenio que seguia el TW2

[ii, nff] = min(abs(freqrange - 19.15)); %Para limitar el calculo del FEG al rango de 0.39 a 19.15 Hz
ind=find(Spec > eps);
lSpec = zeros(size(Spec));
lSpec(ind)=log(Spec(ind));
lSpec = lSpec(1:ndernormas,1:nff);
FEG = exp( sum(lSpec(:))./prod(size(lSpec)) );
if FEG < eps, FEG = 1.0; end %%esto es para protegerse del caso de que algun comemierda meta un EEG que de un factor
%%de escala muy chiquito o cero

if pg_apply, %Calcular el poder geometrico
    data = data ./ sqrt(FEG);
    Spec = Spec ./ FEG;
end

if flags(1) | flags(5) %Calcular banda ancha
    [PA, PR, FM, ZPA, ZPR, ZFM, flags] = banda_ancha(Spec, flags, wbands, fmin, freqres, fmax, nit, sp, montage, fn_eqreg_ba, age);
else
    PA = -1; PR = -1; FM = -1; ZPA = -1; ZPR = -1; ZFM = -1;
end

%Transformar el espectro a escala logaritmica
ind=find(Spec > eps);
Spec(ind)=log(Spec(ind));

if flags(6)  %Calcular normas de banda estrecha
    [ZSpec, flags, indnorma, indfreq_in_Z] = z_be(Spec, flags, fn_eqreg_be, age, freqrange, montage);
else
    indnorma = -1;  ZSpec = -1; indnorma = -1; indfreq_in_Z = -1; indnocero = -1;
end

SpecR = -1; %%En esta version esta deshabilitado

if flags(3) | flags(7)
    load(fn_LF, 'Ui', 'si', 'Vis', 'rind');  %%Carga la Ui, si, Vis
    ng = round(size(Vis,1) ./ 3);
    [flags, Sinv, ZSinv, SIMedia, SISigma, indnorma1, indfreq_in_Z1, indnocero] = tec_init(flags, montage, fn_eqreg_si, age, freqrange, ng);
    
    if ~flags(6)
        indnorma = indnorma1;
        indfreq_in_Z = indfreq_in_Z1;
    end
    
end

if length(lambda) < nfreq
    lambda = [lambda(:); lambda(length(lambda))*ones(nfreq-length(lambda),1)];
end

prueba = 1;


if flags(3) | flags(7)  %%Calcular soluciones inversas
    
    if length(flags) >= 12 && flags(12)
        Jxyz = zeros(size(Vis,1), nfreq);
    end
    
    for cur_freq=1:nfreq,
        [data_freq] = get_data_freq(data, ndt, cur_freq, newmont);
        Js = invreg_c(Ui, si, Vis, data_freq, 0, lambda(cur_freq));
        
        if length(flags) >= 12 && flags(12)
            Jxyz(:,cur_freq) = real(mean(Js,2));
        end
        
        if prueba == 1
            Js = abs(Js).^2;
            Js = mean(Js, 2); %aqui antes habia puesto sum
            Js = reshape(Js, 3, ng);
            Js = log(mean(Js));
            
        elseif prueba == 2
            %             Js = abs(Js).^2;
            %             Js = mean(Js, 2); %aqui antes habia puesto sum
            %             Js = reshape(Js, 3, ng);
            %             Js = log(mean(Js));
            %
            %             Oper = invreg_c(Ui, si, Vis, eye(ndernormas), 0, lambda(cur_freq));
            %             Oper = abs(Oper).^2;
            %             Oper = reshape(Oper, 3, ng);
            %             Oper = log(mean(Oper));
            %             umb = median(Oper) - mad(Oper)*2;
            %             ind = find(Oper < umb);
            %             Oper(ind) = umb;
            %             Js = Js - Oper;
            
        elseif prueba == 3
            Js = abs(Js).^2;
            Js = mean(Js, 2); %aqui antes habia puesto sum
            Js = reshape(Js, 3, ng);
            Js = log(mean(Js));
            
            Oper = invreg_c(Ui, si, Vis, eye(ndernormas), 0, lambda(cur_freq));
            Oper = abs(Oper).^2;
            Oper = reshape(Oper, 3, ng);
            Oper = log(mean(Oper));
            Js = Js - Oper;
            
        elseif prueba == 4
            Oper = invreg_c(Ui, si, Vis, eye(ndernormas), 0, lambda(cur_freq));
            J = zeros(ng,1);
            for kk=1:ng
                indd = (kk-1)*3+1:kk*3;
                gen = Oper(indd,:);
                gen = pinv(gen * gen');
                for ss =1:size(Js,2)
                    J(kk) = J(kk) + Js(indd,ss)' * gen * Js(indd,ss);
                end
            end
            Js = log(abs(J));
        end
        
        Js = Js(:);
        Sinv(:,cur_freq) = Js;
    end
    
    
    %%Estandarizar la Sinv de este caso
    switch SI_correction
        case 1
            factor = mean(Sinv(:));   %se calcula un factor global
            Sinv = Sinv - factor;       %%El factor se resta porque la SI esta en logaritmo
        case 2
            factor = mean(Sinv);      %se calcula un factor por frecuencias
            Sinv = Sinv - repmat(factor, ng, 1);       %%El factor se resta porque la SI esta en logaritmo
        otherwise
            factor = 1;
            %% disp('Sin estandarizar la J')
    end
    
    if length(flags) >= 12 && flags(12)
        Jxyz = Jxyz ./ factor;
    end
    
    if flags(7), %Calcular la ZSinv
        for cur_freq=1:nfreq,
            if indfreq_in_Z(cur_freq) > 0
                ZSinv_j = calc_zsinv(Sinv(:,cur_freq), indfreq_in_Z, indnorma, cur_freq, SIMedia, SISigma);
                ZSinv(:,indfreq_in_Z(cur_freq)) = ZSinv_j;
            end
        end
    end
    
else
    Sinv = -1; ZSinv = -1; SpecR = -1; Jxyz = [];
end

% Jt = Sinv(:);
% save Js.eng Jt -ascii
% Jt = ZSinv(:);
% save Jz.eng Jt -ascii
%
%%En QEEGT\KERNEL hay un programita que se llama checkingZSI que es para verificar los calculos de la ZSInv

% figure; plot(Sinv(10:100:end,:)'); title('J sujeto'); saveas(gcf, 'h:\temp\jsujeto.jpg');
% figure; plot(SIMedia(10:100:end,:)'); title('J Norma'); saveas(gcf, 'h:\temp\jnorma.jpg');
% figure; plot(SISigma(10:100:end,:)'); title('STD Norma'); saveas(gcf, 'h:\temp\stdnorma.jpg');
% figure; plot(ZSinv(10:100:end,:)'); title('ZJ sujeto'); saveas(gcf, 'h:\temp\zjsujeto.jpg');
%
% figure; plot(Spec'); title('Sp sujeto'); saveas(gcf, 'h:\temp\SPsujeto.jpg');

Spec = Spec';
SpecR = SpecR';
ZSpec = ZSpec';

%%save('c:\invsol', Sinv, ZSinv, SIMedia, SISigma);
%save "c:\\resqeegt" Spec ZSpec Sinv ZSinv
end

function [meancoef, stdcoef, state, pgcorrect, freqres, freqsHz]=readnrmcbe(nrmbdfname,age)

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
end

function data = refprom(data, ndt, ndernormas)

%Normalizar los datos por la referencia promedio
% H=eye(ndt)-ones(ndt,ndt)./ndt;
%Usar esta H modificada para corregir solo por las deriv normadas. Esto es
%necesario para no alterar las Zs
H=eye(ndt)-[ones(ndt,ndernormas) zeros(ndt,ndt-ndernormas)]/ndernormas;
data = H * data;
end

function saveresults(results_file, fmin, freqres, fmax, FEG, flags, MCross, indfreq_in_Z, newmont, ZSpec, PA, PR, FM, ZPA, ZPR, ZFM, Sinv, ZSinv, lambda, SI_correction, CorrM, Coh, Phase, Jxyz)


%%%%%%%%%%Salvar los resultados

%%  if ~exist('lambda')
%%      lambda = -1;
%%  end

%%  if ~exist('SI_correction')
%%      SI_correction = -1;
%%  end


results_file = setstr(results_file);

fid = fopen(results_file, 'w');
if fid == -1
    errort = 1; return;
end;

emptymat = -1;
FileVersion = 1.0;

fwrite(fid, FileVersion, 'real*4');
fwrite(fid, fmin, 'real*4');
fwrite(fid, freqres, 'real*4');
fwrite(fid, fmax, 'real*4');
fwrite(fid, FEG, 'real*4');


WriteMatrix(fid,flags'); %%un truco porque WriteMatrix transpone antes de grabar

%%Salvar las matrices cross
[nder, nfreq] = size(MCross);
nder = floor( sqrt(nder) );
fwrite(fid, nfreq, 'real*4');

for k = 1:nfreq
    mtemp = reshape(MCross(:,k), nder, nder);
    %%Cambiarle el signo a la triangular inferior de las matrices cross para que sea compatible con la forma de almacenamiento del TW2
    ind=find(tril(mtemp,-1));
    mtemp(ind) = -mtemp(ind);
    WriteMatrix(fid,mtemp);
end;

if flags(6) | flags(7) %%Se calculo Z de banda estrecha y/o soluciones inversas. Grabar los indices
    WriteMatrix(fid,indfreq_in_Z);
else
    WriteMatrix(fid,emptymat);
end

%%Grabar el orden que se uso para las derivaciones en caso de que se haya calculado Z
WriteMatrix(fid,newmont);

if flags(6)  %%Se calculo ZSpec
    WriteMatrix(fid, ZSpec);  %%ZSpec es una matriz de nfreq x nder y se salva por filas
else
    WriteMatrix(fid,emptymat);
end

if flags(1)  %%Se calculo Banda Ancha
    WriteMatrix(fid,PA');
    WriteMatrix(fid,PR');
    WriteMatrix(fid,FM');
else
    WriteMatrix(fid,emptymat);
    WriteMatrix(fid,emptymat);
    WriteMatrix(fid,emptymat);
end;

if flags(5)  %%Se calculo Z de Banda Ancha
    WriteMatrix(fid,ZPA');
    WriteMatrix(fid,ZPR');
    WriteMatrix(fid,ZFM');
else
    WriteMatrix(fid,emptymat);
    WriteMatrix(fid,emptymat);
    WriteMatrix(fid,emptymat);
end;

if flags(3) %%Se calculo soluciones inversas
    WriteMatrix(fid,Sinv');
else
    WriteMatrix(fid,emptymat);
end

if flags(7) %%Se calculo Z de soluciones inversas
    WriteMatrix(fid,ZSinv');
else
    WriteMatrix(fid,emptymat);
end

if lambda == -1
    WriteMatrix(fid,emptymat);
else
    WriteMatrix(fid,lambda);
end

if SI_correction == -1
    WriteMatrix(fid,emptymat);
else
    WriteMatrix(fid,SI_correction);
end

if CorrM == -1
    nseg = 1;
    fwrite(fid, nseg, 'real*4');
    WriteMatrix(fid,emptymat);
else
    %%Salvar las matrices de correlaciones por segmentos
    nseg = size(CorrM, 3);
    fwrite(fid, nseg, 'real*4');
    for k = 1:nseg
        mtemp = CorrM(:,:,k);
        WriteMatrix(fid,mtemp);
    end;
end

if Coh == -1
    nseg = 1;
    fwrite(fid, nseg, 'real*4');
    WriteMatrix(fid,emptymat);
else
    %%Salvar las matrices de coherencias por frecuencias
    nfreq = size(Coh, 3);
    fwrite(fid, nfreq, 'real*4');
    for k = 1:nfreq
        mtemp = Coh(:,:,k);
        WriteMatrix(fid,mtemp);
    end;
end

if Phase == -1
    nseg = 1;
    fwrite(fid, nseg, 'real*4');
    WriteMatrix(fid,emptymat);
else
    %%Salvar las matrices de delay de fase por frecuencias
    nfreq = size(Phase, 3);
    fwrite(fid, nfreq, 'real*4');
    for k = 1:nfreq
        mtemp = Phase(:,:,k);
        WriteMatrix(fid,mtemp);
    end;
end

if isempty(Jxyz)
    WriteMatrix(fid,emptymat);
else
    WriteMatrix(fid,Jxyz');
end

fclose(fid);
end

function WriteMatrix(fid, mat) %%La matriz se salva por filas, para comodidad de Aubert
[m,n]=size(mat);
mat = mat';
fwrite(fid, m, 'real*4');
fwrite(fid, n, 'real*4');
fwrite(fid, mat, 'real*4');

end

function [flags, Sinv, ZSinv, SIMedia, SISigma, indnorma, indfreq_in_Z, indnocero] = tec_init(flags, montage, fn_eqreg_si, age, freqrange, ng);


Sinv = -1; ZSinv = -1; SIMedia = -1; SISigma = -1; indnorma = -1; indfreq_in_Z = -1; indnocero = -1;

nfreq=length(freqrange);

fn_eqreg_si = setstr(fn_eqreg_si);

newmont=match_montage(montage');
indnocero = find(newmont ~= 0);

%Chequear el montaje para ver si se puede calcular la soluc. inversa
if length(find(newmont(1:19) ==0)) > 0, %Si no estan las primeras 19 deriv (1020) no se pueden calcular las soluciones inversas
    flags(3)=0; flags(4)=0; flags(7)=0;
    msg3 = 'Montaje no valido para el calculo de las soluciones inversas';
end
flags(4)=flags(4) & flags(3); %Para calcular espectro reconstruido hay que calcular SI

if flags(7)
    %Preparando para el calculo de la ZSinv de la solucion inversa
    [SIMedia, SISigma, si_state, si_pgcorrect, norm_freqres, norm_freqsHz]=readnrmcbe(fn_eqreg_si,age);
    %ver de las frecuencias calculadas para cuales existe norma
    [indnorma, indfreq_in_Z, nfreqs_norm] = match_freq_norm(freqrange,norm_freqsHz);
    %Si el rango de frecuencias no coincide no se puede calcular medidas Z
    if sum(indnorma) == 0,
        flags(7)=0;
        msg2 = 'El rango de frecuencias no corresponde con el rango de frecuencias de las normas';
    end
end

Sinv=zeros(ng,nfreq);

if flags(7)
    ZSinv=zeros(ng,nfreqs_norm);
end
end

function [ZSpec, flags, indnorma, indfreq_in_Z] = z_be(Spec, flags, fn_eqreg_be, age, freqrange, montage)

nfreq=length(freqrange);

[SpMedia, SpSigma, be_state, be_pgcorrect, norm_freqres, norm_freqsHz]=readnrmcbe(fn_eqreg_be,age);
%Chequeo de consistencia
%ver de las frecuencias calculadas para cuales existe norma
[indnorma, indfreq_in_Z, nfreqs_norm] = match_freq_norm(freqrange,norm_freqsHz);
%Si el rango de frecuencias no coincide no se puede calcular medidas Z
if sum(indnorma) == 0,
    flags(6)=0; flags(7)=0;
    msg2 = 'El rango de frecuencia no corresponde con el rango de frecuencias de las normas';
end

if flags(6),
    [newmont, indnocero]=match_montage(montage');
    nderZ = size(newmont,1);
    
    ZSpec = zeros( nderZ, nfreqs_norm );
    for j=1:nfreq,
        if indnorma(j) > 0
            ZSpec(indnocero,indfreq_in_Z(j)) = (Spec(newmont(indnocero),j) - SpMedia(indnocero,indnorma(j))) ./ SpSigma(indnocero,indnorma(j));
        end
    end
else
    ZSpec = -1;
end

nfreq = nfreq;
%%En QEEGT\KERNEL hay un programita que se llama checkingZSP que es para verificar los calculos de la ZSP
end


function [name, ext, pathd] = getfname( pathname )
%[name, ext, pathd] = getfname( pathname );

ind1 = find( pathname == '\' );
if ~isempty(ind1),
    pathd = pathname(1:ind1(length(ind1)));
    pathname = pathname(ind1(length(ind1))+1:length(pathname));
else
    pathd = '';
end

ind2 = find( pathname == '.' );
if ~isempty(ind2),
    ext = pathname(ind2(1)+1:length(pathname));
    pathname = pathname(1:ind2(1)-1);
else
    ext = '';
end
name = pathname;
end

function savemods(fname, state, montage, freqrange, FEG, MCross, indfreq_in_Z, newmont, ZS, PA, PR, FM, ZPA, ZPR, ZFM, Sinv, ZJ, lambda, SI_correction, CorrM, Coh, Phase, Jxyz)

[pp nn ee] = fileparts(fname);
nd = sqrt(size(MCross,1));
nf = size(MCross,2);

num = [];
if ~isequal(MCross, -1)
    crosstype = 'Cross Spectrum';
    fsave = [pp filesep nn '-cross-' state '-'];
    [fsave, num] = find_not_exist(fsave, num);
    Matrix = reshape(MCross,nd,nd,nf);
    writemodfrom0(fsave, Matrix, freqrange, FEG, montage, crosstype);
end

if ~isequal(ZS, -1)
    crosstype = 'Z Cross Spectrum';
    fsave = [pp filesep nn '-Zcross-' state '-'];
    [fsave, num] = find_not_exist(fsave, num);
    Matrix = ZS;
    writemodfrom0(fsave, Matrix, freqrange, FEG, montage, crosstype);
end

if ~isequal(PA, -1) && ~isequal(PR, -1) &&~isequal(FM, -1)
    crosstype = 'Broad Band';
    fsave = [pp filesep nn '-BBSP-' state '-'];
    [fsave, num] = find_not_exist(fsave, num);
    Matrix = zeros(1,nd,size(PA,2),3);
    Matrix(1,:,:,1) = PA;
    Matrix(1,:,1:size(PR,2),2) = PR;
    Matrix(1,:,1:size(FM,2),3) = FM;
    writemodfrom0(fsave, Matrix, freqrange, FEG, montage, crosstype);
end

if ~isequal(ZPA, -1) && ~isequal(ZPR, -1) &&~isequal(ZFM, -1)
    crosstype = 'Z Broad Band';
    fsave = [pp filesep nn '-ZBBSP-' state '-'];
    [fsave, num] = find_not_exist(fsave, num);
    Matrix = zeros(1,nd,size(ZPA,2),3);
    Matrix(1,:,:,1) = ZPA(1:nd,:);
    Matrix(1,:,1:size(ZPR,2),2) = ZPR(1:nd,:);
    Matrix(1,:,1:size(ZFM,2),3) = ZFM(1:nd,:);
    Matrix(1,:,:,4) = ZPA(1:nd,:);
    writemodfrom0(fsave, Matrix, freqrange, FEG, montage, crosstype);
end

if ~isequal(Sinv, -1)
    crosstype = 'ET Spectrum';
    fsave = [pp filesep nn '-ETC-' state '-'];
    [fsave, num] = find_not_exist(fsave, num);
    Matrix = Sinv;
    writemodfrom0(fsave, Matrix, freqrange, FEG, montage, crosstype);
end

if ~isequal(ZJ, -1)
    crosstype = 'Z ET Spectrum';
    fsave = [pp filesep nn '-ZETC-' state '-'];
    [fsave, num] = find_not_exist(fsave, num);
    Matrix = ZJ;
    writemodfrom0(fsave, Matrix, freqrange, FEG, montage, crosstype);
end

if ~isequal(Jxyz, -1) && ~isempty(Jxyz)
    crosstype = 'ETxyz Spectrum';
    fsave = [pp filesep nn '-ETCxyz-' state '-'];
    [fsave, num] = find_not_exist(fsave, num);
    Matrix = Jxyz;
    writemodfrom0(fsave, Matrix, freqrange, FEG, montage, crosstype);
end

if ~isequal(Coh, -1) && ~isempty(Coh)
    crosstype = 'Coherence';
    fsave = [pp filesep nn '-coh-' state '-'];
    [fsave, num] = find_not_exist(fsave, num);
    Matrix = reshape(Coh,nd,nd,nf);
    writemodfrom0(fsave, Matrix, freqrange, FEG, montage, crosstype);
end

% , CorrM, Coh, Phase, Jxyz
end




function [fsave, num] = find_not_exist(fsave, num)
if exist('num', 'var') && ~isempty(num)
    fsave = [fsave num2str(num) '.mod'];
else
    for k=1:1000
        fs = [fsave num2str(k) '.mod'];
        if ~exist(fs, 'file')
            fsave = fs; num = k; return
        end
    end
end
end

function  writemodfrom0(fName, Matrix, freqrange, FEG, montage, crosstype)
%writemodfrom0(fName, Matrix, freqrange, FEG, crosstype)


%Mod Structure
%integer*2    : ProtMask
%integer*1    : Byte con la long del comentario
%char*80      : Comentario
%6 integer*2  : 'MeasureSize', 'DurationSize', 'FirstSpaceSize', 'SecondSpaceSize', 'ReservedBytes', 'DataSize'

%MatrixSize = MeasureSize*DurationSize*FirstSpaceSize*SecondSpaceSize
%MatrixSize rel*DataSize : Data Matrix
%ReservedBytes : 0
%DataSize : 4

%integer*1    : Byte con la long del DurationUnit
%char*8       : DurationUnit
%

%Array de 8 HeaderList:
%  integer*4 : Ofs
%  integer*2 : Total
%end

% fseek(fid, HeaderLists.Ofs(1), 'bof') %should be the current position

fid = fopen(fName, 'w');
if fid == -1
    error(['Could not create file ' fName]);
end

ProtMask = 4657; %this is a mark for file models, the first three nibbles are a key protection 
%and the last is the current models version, this is a good means to protect 
%ourselves from mismatch versions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

count = fwrite(fid, ProtMask, 'integer*2');
if count ~= 1
    fclose(fid);
    error(['Error writing file ' fName]);
end

if ~iscell(montage)
    mont = cell(size(montage,1),1);
    for k=1:size(montage,1)
        mont{k} = deblank(montage(k,:));
    end
    montage = mont;
end

LabelD = cell(length(freqrange),1);
for k=1:length(freqrange)
    LabelD{k} = strcat(num2str(freqrange(k),'%5.3f'), ' Hz');
end
Total8 = 11;
Context.Name = {'Start Frequency:Single'; 'Frequency Resolution:Single'; 'Scale Factor:Single'; 'Scale Measures:Boolean[1]'; ...
    'Measure Label:String[20]'; 'Duration Label:String[20]'; 'Space1 Label:String[20]'; 'Space2 Label:String[20]'; 'Compact:Boolean'; ...
    'DurationFirst:Single'; 'DurationStep:Single'};
Context.Size = [4 4 4 1 21 21 21 21 1 4 4];
fres = mode(diff(freqrange));
Context.Data = {freqrange(1), fres, FEG, [], '', 'Frequency (Hz)', 'Derivation', 'Derivation', 1, freqrange(1), fres};

switch lower(crosstype)
    case 'cross spectrum'
        MComment = 'Cross Spectrum';
        Measure = 1;
        Duration2 = 51;
        LSp2 = montage;
        LabelM = {MComment};
        ListScale = {'0'};
        ListUnit = {'æVý/Hz'};
        ListTransform = {'æVý'};
        Context.Data{4} = Measure;
        Context.Data{6} = MComment;
    case 'z cross spectrum'
        MComment = 'Z Cross Spectrum';
        Measure = 0;
        Duration2 = 0;
        M = Matrix';
        Matrix = zeros(1,size(M,1), size(M,2), 2);
        Matrix(1,:,:,1) = M; 
        Matrix(1,:,:,2) = M;
        LSp2 = {};
        clear M
        LabelM = {MComment; MComment};
        ListScale = {'0'; '0'};
        ListUnit = {''; ''};
        ListTransform = {'0Z '; '0Z '};
        Context.Data{4} = Measure;
        Context.Data{6} = MComment;
    case 'broad band'
        MComment = 'Broad Band';
        Measure = 0;
        Duration2 = 0;
        LSp2 = {};
        clear M
        LabelM = {'Absolute Power'; 'Relative Power'; 'Mean Frequency'};
        ListScale = {'0'; '0'; '0'};
        ListUnit = {'æVý/Hz'; ''; 'Hz'};
        ListTransform = {''; ''; ''};
        LabelD = {'Delta'; 'Theta'; 'Alpha'; 'Beta'; 'Total'};
        Total8 = 8;
        Context.Name{4} = 'Scale Measures:Boolean[3]';
        Context.Name(9:end) = [];
        Context.Size(4) = 3;
        Context.Size(9:end) = [];
        Context.Data{4} = [1 0 0]';
        Context.Data{5} = 'Broad Band';
        Context.Data{6} = 'Frequency Band';
        Context.Data(9:end) = [];
    case 'z broad band'
        MComment = 'Z Broad Band';
        Measure = 0;
        Duration2 = 0;
        LSp2 = {};
        clear M
        LabelM = {'Z Absolute Power with GP'; 'Z Relative Power'; 'Z Mean Frequency'; 'Z Absolute Power with GP'};
        ListScale = {'0'; '0'; '0'; '0'};
        ListUnit = {''; ''; ''; ''};
        ListTransform = {''; ''; ''; ''};
        LabelD = {'Delta'; 'Theta'; 'Alpha'; 'Beta'; 'Total'};
        Total8 = 8;
        Context.Name{4} = 'Scale Measures:Boolean[3]';
        Context.Name(9:end) = [];
        Context.Size(4) = 3;
        Context.Size(9:end) = [];
        Context.Data{4} = [1 0 0]';
        Context.Data{5} = 'Broad Band';
        Context.Data{6} = 'Frequency Band';
    case {'et spectrum', 'etxyz spectrum'}
        if strcmpi(crosstype, 'et spectrum')
            MComment = 'Electrical Tomography';
        else
            MComment = 'XYZ Electrical Tomography';
        end
        Measure = [1; 0];
        Duration2 = 0;
        M = Matrix;
        Matrix = zeros(1,size(M,1), size(M,2), 2);
        Matrix(1,:,:,1) = M; 
        Matrix(1,:,:,2) = M;
        LSp2 = {};
        clear M
        if strcmpi(crosstype, 'et spectrum')
            LabelM = {'ET with GP'; 'ET with GP'};
        else
            LabelM = {'xyzET with GP'; 'xyzET with GP'};
        end
        ListScale = {'0'; '0'};
        ListUnit = {''; ''};
        ListTransform = {''; ''};
        Total8 = 8;
        Context.Data{4} = Measure;
        Context.Data{5} = MComment;
        Context.Data{6} = 'Frequency Band';
        Context.Data{7} = 'Source';
        Context.Data{8} = 'Source';
        Context.Data(9:end) = [];
        Context.Name{4} = 'Scale Measures:Boolean[2]';
        Context.Size(4) = 2;
        Context.Name(9:end) = [];
        Context.Data(9:end) = [];
        Context.Size(9:end) = [];
        montage = cell(size(Matrix,2),1);
        if strcmpi(crosstype, 'et spectrum')
            for k=1:size(Matrix,2)
                montage{k} = ['Source ' num2str(k)];
            end
        else
            xyz = 'xyz';
            p = 1;
            for k=1:size(Matrix,2)/3
                for hg=1:3
                    montage{p} = ['Source ' num2str(k) '-' xyz(hg)];
                    p = p+1;
                end
            end
        end
    case 'z et spectrum'
        MComment = 'Z Electrical Tomography';
        Measure = [1; 0; 0];
        Duration2 = 0;
        M = Matrix;
        Matrix = zeros(1,size(M,1), size(M,2), 2);
        Matrix(1,:,:,1) = M; 
        Matrix(1,:,:,2) = M;
        LSp2 = {};
        clear M
        LabelM = {'ZET with GP'; 'ZET with GP'};
        ListScale = {'0'; '0'};
        ListUnit = {''; ''};
        ListTransform = {''; ''};
        Total8 = 8;
        Context.Data{4} = Measure;
        Context.Data{5} = MComment;
        Context.Data{6} = 'Frequency Band';
        Context.Data{7} = 'Source';
        Context.Data{8} = 'Source';
        Context.Data(9:end) = [];
        Context.Name{4} = 'Scale Measures:Boolean[2]';
        Context.Size(4) = 3;
        Context.Name(9:end) = [];
        Context.Data(9:end) = [];
        Context.Size(9:end) = [];
        montage = cell(size(Matrix,2),1);
        for k=1:size(Matrix,2)
            montage{k} = ['Source ' num2str(k)];
        end
    case 'coherence'
        MComment = 'Coherence';
        Measure = 1;
        Duration2 = 0;
        LSp2 = montage;
        LabelM = {MComment};
        ListScale = {'0'};
        ListUnit = {'æVý/Hz'};
        ListTransform = {'æVý'};
        Context.Data{4} = Measure;
        Context.Data{6} = MComment;
    otherwise
        disp('unknown mod type'); return
end
    
%Write   MComment      : string[msComment];
msComment      = 80;
l = length(MComment);
count = fwrite(fid,l,'integer*1'); %escribir el byte de la long del string
MC = MComment(:);
if l < msComment
    MC = [MC; ' '*ones(msComment-l,1)];
end
count = fwrite(fid,MC,'char*1');


%%escribir los tamaños de las medidas
%6 integer*2  : 'MeasureSize', 'DurationSize', 'FirstSpaceSize', 'SecondSpaceSize', 'ReservedBytes', 'DataSize'
MI.MeasureSize = size(Matrix,4);
MI.DurationSize = size(Matrix,3);
MI.FirstSpaceSize = size(Matrix,2);
MI.SecondSpaceSize = size(Matrix,1);
MI.ReservedBytes = 0;
MI.DataSize = 4; %size de single
medidas = [MI.MeasureSize; MI.DurationSize; MI.FirstSpaceSize; MI.SecondSpaceSize; MI.ReservedBytes; MI.DataSize];
count = fwrite(fid,medidas,'integer*2');

MatrixSize = MI.MeasureSize*MI.DurationSize*MI.FirstSpaceSize*MI.SecondSpaceSize;  %MI.MeasureSize*MI.DurationSize*MI.FirstSpaceSize*MI.SecondSpaceSize;

Matrix = Matrix(:);
if MI.DataSize == 4  %gabar la matrix como single
    count=fwrite(fid,Matrix,'real*4');
else
    error('DataSize unexpected');
end

if MI.ReservedBytes ~= 0  %grabar los reservedbytes
    l = zeros(MI.ReservedBytes, 1);
    count = fwrite(fid,l,'char*1');
end;

%%Lectura de:   DurationUnit   : string[msDurationUnit];
msDurationUnit = 8;
DurationUnit = zeros(msDurationUnit,1);
DurationUnit(2) = Duration2;
l = length(DurationUnit);
count = fwrite(fid,l,'integer*1'); %grabar el byte de la long del string
count = fwrite(fid,DurationUnit,'char*1');



Labels.ListLabelM = LabelM;
Labels.ListLabelD = LabelD;

Labels.ListLabelSp1 = montage;
Labels.ListLabelSp2 = LSp2;
Labels.ListScale = ListScale;
Labels.ListUnit = ListUnit;
Labels.ListTransform = ListTransform;


%Lectura de:     HeaderLists   : packed array[1..MaxList] of TListInfo;
% TListInfo = packed record
%   Ofs   : Longint;
%   Total : Word;
% end;
%HeaderSize = SizeOf(TListInfo)*MaxList;

MaxList = 8;
HeaderLists.Total(1) = length(Labels.ListLabelM);
HeaderLists.Total(2) = length(Labels.ListLabelD);
HeaderLists.Total(3) = length(Labels.ListLabelSp1);
HeaderLists.Total(4) = length(Labels.ListLabelSp2);
HeaderLists.Total(5) = length(Labels.ListScale);
HeaderLists.Total(6) = length(Labels.ListUnit);
HeaderLists.Total(7) = length(Labels.ListTransform);
HeaderLists.Total(8) = Total8;

start = ftell(fid) + MaxList*(4 + 2);
HeaderLists.Ofs(1) = start;
HeaderLists.Ofs(2) = HeaderLists.Ofs(1) + sum(cellfun('length',Labels.ListLabelM))+length(Labels.ListLabelM);
HeaderLists.Ofs(3) = HeaderLists.Ofs(2) + sum(cellfun('length',Labels.ListLabelD))+length(Labels.ListLabelD);
HeaderLists.Ofs(4) = HeaderLists.Ofs(3) + sum(cellfun('length',Labels.ListLabelSp1))+length(Labels.ListLabelSp1);
HeaderLists.Ofs(5) = HeaderLists.Ofs(4) + sum(cellfun('length',Labels.ListLabelSp2))+length(Labels.ListLabelSp2);
HeaderLists.Ofs(6) = HeaderLists.Ofs(5) + sum(cellfun('length',Labels.ListScale))+length(Labels.ListScale);
HeaderLists.Ofs(7) = HeaderLists.Ofs(6) + sum(cellfun('length',Labels.ListUnit))+length(Labels.ListUnit);
HeaderLists.Ofs(8) = HeaderLists.Ofs(7) + sum(cellfun('length',Labels.ListScale))+length(Labels.ListScale);


for k=1:MaxList
    count=fwrite(fid,HeaderLists.Ofs(k),'integer*4');
    count=fwrite(fid,HeaderLists.Total(k),'integer*2');
end

status = fseek(fid, HeaderLists.Ofs(1), 'bof');

WriteList(fid, Labels.ListLabelM);
WriteList(fid, Labels.ListLabelD);
WriteList(fid, Labels.ListLabelSp1);
WriteList(fid, Labels.ListLabelSp2);
WriteList(fid, Labels.ListScale);
WriteList(fid, Labels.ListUnit);
WriteList(fid, Labels.ListTransform);

%% write information of the context (the last context)
%% TContextInfo = packed record
%%   Name    : string[msContext];
%%   Size    : Word;
%%   Ptrdata : Pointer;
%% end;

for k=1:HeaderLists.Total(MaxList)
    st=Context.Name{k};
    l=length(st);
    count = fwrite(fid,l,'integer*1');
    count = fwrite(fid,st,'char*1');
    l = Context.Size(k);
    count=fwrite(fid,l,'integer*2');
end

OfsDataContext = ftell(fid);  %No se para que lo usan, pero lo dejo por si acaso
for k=1:HeaderLists.Total(MaxList)
    st=char(Context.Name(k));
    if findstr('Single',st)
        count = fwrite(fid,Context.Data{k},'real*4');
    elseif findstr('String',st)
        l=length(Context.Data{k});
        count = fwrite(fid,l,'integer*1');
        if l < Context.Size(k)
            Context.Data{k} = [Context.Data{k}(:); zeros((Context.Size(k)-l-1),1)];
        end
        count = fwrite(fid,Context.Data{k},'char*1');
    elseif findstr('Boolean',st)
        count = fwrite(fid,Context.Data{k},'integer*1');
    else
       error(['Encontro un contexto desconocido: ' st]);
    end
end
fclose(fid);
end



function WriteList(fid, List)
Total = length(List);
for k=1:Total
    l = length(List{k});
    count=fwrite(fid,l,'integer*1'); %escribir el byte de la long del string
    st=List{k};
    count=fwrite(fid,st,'char*1');
end
end



function [data, epoch_size] = plg_readwinds(eeg_fname, wins)
%   function [data, MONTAGE, Age, SAMPLING_FREQ, epoch_size] = plg_read(eeg_fname, state);
%eeg_fname:   name of file to read (without extension(s))

%reading the Neuronic/Track Walker/www.cneuro.edu.cu EEG format

% Piotr J. Durka http://brain.fuw.edu.pl/~durka 2002.06.02

[pp nn ee] = fileparts(eeg_fname);
if isempty(ee)
else
    eeg_fname = strerep(eeg_fname, ee, '');
end
fname = [pp filesep nn '.xnf'];
if exist(fname, 'file')
    plg_info = plg_xmlread(fname);
    NUMBER_OF_CHANNELS =  plg_info.Record.Channels;
    if ~ischar(plg_info.Calibration.M)
        v = plg_info.Calibration.M;
    else
        v = strtovect( plg_info.Calibration.M);
    end
    v(1) = [];
    cdc = reshape(v, 2, NUMBER_OF_CHANNELS);
    
else
    info = read_plginf([eeg_fname '.inf']);
    NUMBER_OF_CHANNELS = info.NChannels;
    %%%%% reading *.CDC file %%%%%%%%%%%%%%%%
    %% calibration & DC offset %%%%%%%%%%%%%%
    cdc_filename=[eeg_fname '.CDC'];
    fi=fopen(cdc_filename, 'r');
    if fi==-1
        cdc_filename=[eeg_fname '.cdc'];
        fi=fopen(cdc_filename, 'r');
        if fi==-1
            error(sprintf('cannot open file  %s for reading', cdc_filename))
        end
    end
    cdc=fread(fi, [2, NUMBER_OF_CHANNELS], 'float32');
    fclose(fi);
    %%%%%% END reading *.CDC file %%%%%%%%%%%
end

NUM_OF_WINDOWS = length(wins.name);

%%%%% reading & calibrating data %%%%%%%%%%%%%%%%
data_filename=[eeg_fname '.PLG'];
datafile_handle=fopen(data_filename, 'r');
if datafile_handle==-1
    data_filename=[eeg_fname '.plg'];
    datafile_handle=fopen(data_filename, 'r');
    if datafile_handle==-1
        error(sprintf('cannot open file  %s for reading', data_filename));
    end
end

data = [];
epoch_size = [];

for w=1:NUM_OF_WINDOWS
    fseek(datafile_handle, (wins.start(w)-1)*NUMBER_OF_CHANNELS*2, 'bof');  %%El 2 es del sizeof de integer*2
    wdata=fread(datafile_handle, [NUMBER_OF_CHANNELS, wins.end(w)-wins.start(w)+1], 'integer*2');
    epoch_size = [epoch_size; size(wdata,2)];
    for c=1:size(wdata,1)
        wdata(c,:)=round((wdata(c,:).*cdc(1,c) - cdc(2,c)));
    end
    data = [data wdata];
end

fclose(datafile_handle);
%%%%%%% END reading & calibrating data %%%%%%%%%%%
end


function inf_info = read_plginf(inf_name)

inf_info = [];

if exist(inf_name, 'file')
    filename = inf_name;
else
    [pp nn ee] = fileparts(inf_name);
    if ~isempty(pp), pp = [pp filesep]; end
    inf_name = [pp nn];
    if exist([inf_name '.inf'], 'file')
        ext = '.inf';
    elseif exist([inf_name '.INF'], 'file')
        ext = '.INF';
    elseif exist([inf_name '.xnf'], 'file')
        ext = '.xnf';
    elseif exist([inf_name '.XNF'], 'file')
        ext = '.XNF';
    else
        return;
    end
    filename = [inf_name ext];
end
[pp nn ext] = fileparts(filename);

try
    if strcmpi(ext, '.xnf')
        plg_info = plg_xmlread(filename);
        if ~isempty(plg_info)
            inf_info.SamplingFreq = plg_info.Record.SamplingRate;
            inf_info.NSamples = plg_info.Record.Bursts;
            inf_info.NChannels = plg_info.Record.Channels;
            MONTAGE = plg_info.Record.Montage;
            MONTAGE = strread(MONTAGE, '%s', 'delimiter', ' ');
            MONTAGE = strrep(MONTAGE, '~', '_');
            inf_info.MONTAGE = MONTAGE;
            inf_info.Gains = strtovect(plg_info.Record.Gain);
            inf_info.LoCut = strtovect(plg_info.Record.LowCut);
            inf_info.HiCut = strtovect(plg_info.Record.HighCut);
        end
        return
    else
        fi=fopen(filename, 'r');
        if fi == -1
            return
        end
    end
    %%%%%%% reading some data from *.INF file (ASCII info) %%%%%
    
    tline=1;
    already_read = 0;
    while ~feof(fi) & tline~=-1
        if ~already_read, tline=fgets(fi); end
        already_read = 0;
        if tline==-1
            error('empty file???');
        else
            [is, tline]=strtok(tline);
            switch is
                case 'PLGMontage'                %electrode names
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);
                    if inf_info.NChannels ~= str2num(is)
                        error('different number of channels and electrode names or channels after the montage in file');
                    end
                    
                    inf_info.MONTAGE = cell(inf_info.NChannels,1);
                    for i=1:inf_info.NChannels
                        [is, tline]=strtok(tline);
                        if isempty(is)
                            tline=fgets(fi);
                            if strfind(tline, 'PLGSR(Hz)') == 1
                                already_read = 1; break;
                            end
                            [is, tline]=strtok(tline);
                        end
                        inf_info.MONTAGE{i}=is;
                    end
                case 'Gains'                %Ganancia de los electrodos
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);
                    if inf_info.NChannels ~= str2num(is)
                        error('different number of channels and electrode names or channels after the montage in file');
                    end
                    
                    inf_info.Gains = zeros(inf_info.NChannels,1);
                    for i=1:inf_info.NChannels
                        [is, tline]=strtok(tline);
                        if isempty(is)
                            tline=fgets(fi);
                            if strfind(tline, 'LCut(Hz)') == 1
                                already_read = 1; break;
                            end
                            [is, tline]=strtok(tline);
                        end
                        inf_info.Gains(i)=str2num(is);
                    end
                case 'LCut(Hz)'                %Low cut
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);
                    if inf_info.NChannels ~= str2num(is)
                        error('different number of channels and electrode names or channels after the montage in file');
                    end
                    
                    inf_info.LoCut = zeros(inf_info.NChannels,1);
                    for i=1:inf_info.NChannels
                        [is, tline]=strtok(tline);
                        if isempty(is)
                            tline=fgets(fi);
                            if strfind(tline, 'HCut(Hz)') == 1
                                already_read = 1; break;
                            end
                            [is, tline]=strtok(tline);
                        end
                        inf_info.LoCut(i)=str2num(is);
                    end
                case 'HCut(Hz)'                %High cut
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);
                    if inf_info.NChannels ~= str2num(is)
                        error('different number of channels and electrode names or channels after the montage in file');
                    end
                    
                    inf_info.HiCut = zeros(inf_info.NChannels,1);
                    for i=1:inf_info.NChannels
                        [is, tline]=strtok(tline);
                        if isempty(is)
                            tline=fgets(fi);
                            if strfind(tline, 'BytesDiskBuffer') == 1
                                already_read = 1; break;
                            end
                            [is, tline]=strtok(tline);
                        end
                        inf_info.HiCut(i)=str2num(is);
                    end
                case 'PLGNC'                      %number of channels
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);   %skip & check '1'
                    if ~strcmp(is, '1')
                        error(['why not 1?']);
                    end
                    [is, tline]=strtok(tline);
                    inf_info.NChannels = str2num(is);
                case 'PLGNS'                      %number of channels
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);   %skip & check '1'
                    if ~strcmp(is, '1')
                        error(['why not 1?']);
                    end
                    [is, tline]=strtok(tline);
                    inf_info.NSamples = str2num(is);
                case 'PLGSR(Hz)'                      %number of channels
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);   %skip & check '1'
                    if ~strcmp(is, '1')
                        error(['why not 1?']);
                    end
                    [is, tline]=strtok(tline);
                    inf_info.SamplingFreq = str2num(is);
                otherwise
                    %disp(['unused: ...' tline]);
            end
        end
    end
    fclose(fi);
end
end


function mrks = read_plgmrk(eeg_fname)

wins = [];
mrks = [];

%%%%%% reading marks from *.MRK file %%%%%%%%%%
if isempty(strfind(lower(eeg_fname), '.mrk'))
    eeg_fname = [eeg_fname '.mrk'];
end
fi=fopen(eeg_fname, 'r');
if fi==-1
    return;
end

NUM_OF_WINDOWS=0;
while ~feof(fi)
    c =fread(fi, 1, 'uint8');
    if feof(fi)
        break;
    end
    NUM_OF_WINDOWS=NUM_OF_WINDOWS+1;
    WIN_NAME(NUM_OF_WINDOWS) = c;
    WIN_START(NUM_OF_WINDOWS)=fread(fi, 1, 'integer*4');
end
fclose(fi);

if NUM_OF_WINDOWS > 0
    wins.name = WIN_NAME;
    wins.start = WIN_START;
end

mrks = wins;
end



function pat_info = read_plgpat(pat_name)

pat_info = [];
try
    [c1, c2, c3]=textread(pat_name,'%s = %d %[^\n]');
catch
    return
end
if ~exist('c1'), return; end

aa=ismember(c3,'$$ ');
ind=find(aa);
for k=1:length(ind)
    c3{ind(k)} = '';
end

c3 = char(c3);
p=find(c3 == 160); if ~isempty(p), c3(p) = ''; end
p=find(c3 == 130); if ~isempty(p), c3(p) = ''; end
p=find(c3 == 161); if ~isempty(p), c3(p) = ''; end
p=find(c3 == 162); if ~isempty(p), c3(p) = ''; end
p=find(c3 == 163); if ~isempty(p), c3(p) = ''; end
p=find(c3 == 164); if ~isempty(p), c3(p) = ''; end
p=find(c3 == 165); if ~isempty(p), c3(p) = ''; end
p=find(c3 == '_'); if ~isempty(p), c3(p) = ' '; end
p=find(c3 == 196); if ~isempty(p), c3(p) = '_'; end

ii=strmatch('Name', c1);
if isempty(ii), pat_info.Name = ''; else pat_info.Name = deblank(c3(ii,:)); end

ii=strmatch('Code', c1);
if isempty(ii), pat_info.Code = ''; else pat_info.Code = deblank(c3(ii,:)); end

ii=strmatch('Sex', c1);
if isempty(ii), pat_info.Sex = ''; else pat_info.Sex = deblank(c3(ii,:)); end
switch lower(pat_info.Sex)
    case {'male', 'masculino'}
        pat_info.Sex = 'M';
    case {'female', 'femenino'}
        pat_info.Sex = 'F';
    otherwise
        pat_info.Sex = ' ';
end

ii=strmatch('Race', c1);
if isempty(ii), pat_info.Race = ''; else pat_info.Race = deblank(c3(ii,:)); end

ii=strmatch('BirthDate', c1);
if isempty(ii), pat_info.BirthDate = ''; else pat_info.BirthDate = deblank(c3(ii,:)); end

ii=strmatch('Age', c1);
if isempty(ii), pat_info.Age = ''; else pat_info.Age = deblank(c3(ii,:)); end

ii=strmatch('GestAge', c1);
if isempty(ii), pat_info.GestAge = ''; else pat_info.GestAge = deblank(c3(ii,:)); end

ii=strmatch('RecordDate', c1);
if isempty(ii), pat_info.RecordDate = ''; else pat_info.RecordDate = deblank(c3(ii,:)); end

ii=strmatch('RecordTime', c1);
if isempty(ii), pat_info.RecordTime = ''; else pat_info.RecordTime = strtrim(c3(ii,:)); end

ii=strmatch('Technician', c1);
if isempty(ii), pat_info.Technician = ''; else pat_info.Technician = strtrim(c3(ii,:)); end

ii=strmatch('Status', c1);
if isempty(ii), pat_info.Status = ''; else pat_info.Status = strtrim(c3(ii,:)); end

ii=strmatch('RefPhysician', c1);
if isempty(ii), pat_info.Doctor = ''; else pat_info.Doctor = strtrim(c3(ii,:)); end


texto = [];
for k=1:size(c3,1)
    cc = strtrim(c3(k, :));
    s = upper(cc);
    if ~isempty(s)
        if length(strfind(s, 'NO PROYECTO')) > 0
            pat_info.excluido = 1;
            pat_info.causa_excluido = cc;
        elseif length(strfind(s, 'REPETIR')) > 0
            pat_info.repetir = 1;
            if isfield(pat_info, 'informe_eeg')
                pat_info.informe_eeg = [pat_info.informe_eeg ' ' cc];
            else
                pat_info.informe_eeg = cc;
            end
        elseif length(strfind(s, 'EEG NORMAL')) > 0
            pat_info.eeg_normal = 1;
            if isfield(pat_info, 'informe_eeg')
                pat_info.informe_eeg = [pat_info.informe_eeg ' ' cc];
            else
                pat_info.informe_eeg = cc;
            end
        elseif length(strfind(s, 'RUIDO')) > 0 | length(strfind(s, 'ARTEFACTO')) > 0
            pat_info.eeg_normal = 0;
            if isfield(pat_info, 'informe_eeg')
                pat_info.informe_eeg = [pat_info.informe_eeg ' ' cc];
            else
                pat_info.informe_eeg = cc;
            end
        elseif length(strfind(s, 'ELECTRODO')) > 0 | length(strfind(s, 'CANAL')) > 0
            if length(strfind(s, 'MALO'))
                cc = upper(cc);
                cc = strrep(cc, 'ELECTRODOS', ''); cc = strrep(cc, 'ELECTRODO', '');
                cc = strrep(cc, 'CANALES', ''); cc = strrep(cc, 'CANAL', '');
                cc = strrep(cc, 'MALOS', ''); cc = strrep(cc, 'MALO', '');
                cc = strrep(cc, ':', ' '); cc = strrep(cc, ',', ' ');
                cc = strrep(cc, '.', ' '); cc = strrep(cc, ';', ' ');
                v = str2num(cc);
                if isempty(v)
                    pat_info.canales_malos = cc;
                else
                    pat_info.canales_malos = v;
                end
            end
        end
    end
    texto{length(texto)+1} = [c1{k} '  =  ' c3(k,:)];
    %     st = strtrim(upper(c1{k}));
    %     if findstr(st, 'CLINICALDATA') | findstr(st, 'DIAGNOSIS') | findstr(st, 'MEDICATION')
    %         texto{length(texto)+1} = [c1{k} ' = '  c3(k,:)];
    %     end
    
end
pat_info.texto = texto;
end



function wins = read_plgwin(eeg_fname)

wins = [];

%%%%%% reading marked windows from *.WIN file %%%%%%%%%%
fi=fopen(eeg_fname, 'r');
if fi==-1
    filename=[eeg_fname '.win'];
    fi=fopen(filename, 'r');
    if fi==-1
        return;
    end
end
NUM_OF_WINDOWS=0;
while ~feof(fi)
    c =fread(fi, 1, 'uint8');
    if feof(fi)
        break;
    end
    NUM_OF_WINDOWS=NUM_OF_WINDOWS+1;
    WIN_NAME(NUM_OF_WINDOWS) = c;
    WIN_START(NUM_OF_WINDOWS)=fread(fi, 1, 'integer*4');
    WIN_END(NUM_OF_WINDOWS)  =fread(fi, 1, 'integer*4');
end
fclose(fi);

if NUM_OF_WINDOWS > 0
    wins.name = WIN_NAME;
    wins.start = WIN_START;
    wins.end = WIN_END;
end
end



function [data, MONTAGE, age, SAMPLING_FREQ, epoch_size, wins, msg] = read_plgwindows(plgname, state, lwin)
%plgname: nombre del PLG, con el path, sin el .PLG
%state: un caracter. ejemplo, 'A'
%lwin en seg, ejemplo: 0.005 (5 msec)

msg = ''; data = []; MONTAGE = []; age = []; SAMPLING_FREQ = []; epoch_size = []; wins = [];
[pp, nn, ee] = fileparts(plgname);
if ~isempty(pp), pp = [pp filesep]; end
try
    isPE = 0; mrk = []; npoints = [];
    [data, MONTAGE, age, SAMPLING_FREQ, epoch_size, wins, mrks] = plg_read([pp nn], state, isPE, mrk, npoints);
catch
    msg =['Error reading file ' pp filesep nn '. Please check if the file exists and there are windows marked'];
    data = []; MONTAGE = []; nt = []; sp = []; return
end
if isempty(data),
    msg = ['Error loading file ' pp filesep nn '. No windows found'];
    MONTAGE = []; nt = []; sp = []; return
end
sp = 1 ./ SAMPLING_FREQ;
nt = floor(lwin / sp);  %numero de ventanas de analisis
if nt > size(data,2)
    msg = ['Error loading file ' pp filesep nn '. Size of analysis windows too big'];
    data = []; MONTAGE = []; nt = []; sp = []; return
end
MONTAGE = char(MONTAGE);

quitar = [];
csum = cumsum([1; epoch_size]);
for k=1:length(epoch_size)
    if isempty(quitar)
        quitar = [csum(k)+nt*floor(epoch_size(k)/nt):csum(k+1)-1]';
    else
        quitar = [quitar; [csum(k)+nt*floor(epoch_size(k)/nt):csum(k+1)-1]'];
    end
end
data(:,quitar) = [];
end

function [data, MONTAGE, Age, SAMPLING_FREQ, epoch_size, wins, mrks] = plg_read(basename, state, isPE, mrk, npoints)

%   function [data, MONTAGE, Age, SAMPLING_FREQ, epoch_size] = plg_read(basename, state);
%basename:   name of file to read (without extension(s))

%reading the Neuronic/Track Walker/www.cneuro.edu.cu EEG format

% Piotr J. Durka http://brain.fuw.edu.pl/~durka 2002.06.02

if ~exist('PE','var'), isPE = 0; end
if ~exist('mrk','var'), mrk = []; end
if ~exist('npoints','var'), npoints = []; end

data = []; MONTAGE = []; Age = []; SAMPLING_FREQ = []; epoch_size = []; wins = []; mrks = [];

if nargin<1
    return;
end

if isPE && ~isempty(state)
    disp('If parameter isPE is TRUE then state has to be [] and mrk a character between 0 and 9'); return
end

if isPE && isempty(mrk)
    disp('If parameter isPE is TRUE then mrk has to be a character between 0 and 9. Assumning 0');
    mrk = 0;
end

if isPE && isempty(npoints)
    disp('Parameter isPE is TRUE and windows size not specified. Assumning 512');
    noints = 512;
end

if nargin<2
    state = 'A';
end

[pp nn ee] = fileparts(basename);
basename = strrep(basename, ee, '');

pat_info = read_plgpat([basename '.pat']);
if isfield(pat_info, 'Age')
    Age = str2double(pat_info.Age);
else
    pat_info = read_plgpat([basename '.PAT']);
    if isfield(pat_info, 'Age')
        Age = str2double(pat_info.Age);
    else
        Age = [];
    end
end

plgfname = [pp filesep nn '.xnf'];
if exist(plgfname, 'file')
    plg_info = plg_xmlread(plgfname);
    NUMBER_OF_CHANNELS =  plg_info.Record.Channels;
    MONTAGE = plg_info.Record.Montage;
    MONTAGE = strread(MONTAGE, '%s', 'delimiter', ' ');
    MONTAGE = strrep(MONTAGE, '~', '_');
    SAMPLING_FREQ = plg_info.Record.SamplingRate;
%     NUMBER_OF_SAMPLES_IN_FILE= plg_info.Record.Bursts;
    v = strtovect( plg_info.Calibration.M);
    v(1) = [];
    cdc = reshape(v, 2, NUMBER_OF_CHANNELS);
    w=plg_info.Window.A;
    if ~iscell(w)
         w = {w};
    end
    NUM_OF_WINDOWS = length(w);
    WIN_NAME = zeros(NUM_OF_WINDOWS,1);
    WIN_START = zeros(NUM_OF_WINDOWS,1);
    WIN_END = zeros(NUM_OF_WINDOWS,1);
    wnames = fields(plg_info.Title);
    wnames = str2double(strrep(wnames, 'A', ''));
    ind = find(wnames >= 'A'& wnames <= 'Z');
    wnames(ind) = char(wnames(ind));
    for k=1:NUM_OF_WINDOWS
        WIN_NAME(k) = w{k}(3);
        ii = find(ismember(wnames, WIN_NAME(k)));
        if ~isempty(ii)
            WIN_NAME(k) = wnames(ii);
        end
        WIN_START(k) = w{k}(1);
        WIN_END(k) =  WIN_START(k) + w{k}(3) - 1;
    end
    
else
%     %%%%%%% reading the age from *.PAT file (ASCII info) %%%%%
%     pat_filename=[basename '.PAT'];
%     fi=fopen(pat_filename, 'r');
%     if fi==-1
%         pat_filename=[basename '.pat'];
%         fi=fopen(pat_filename, 'r');
%         if fi==-1
%             error(sprintf('cannot open file  %s for reading', pat_filename))
%         end
%     end
%     tline=1;
%     while ~feof(fi) & tline~=-1
%         tline=fgets(fi);
%         if tline==-1
%             error('empty file???');
%         else
%             [is, tline]=strtok(tline);
%             switch is
%                 case 'Age'                      %Patient age
%                     [is, tline]=strtok(tline);   %skip & check '='
%                     if ~strcmp(is, '=')
%                         error(['missing ' is]);
%                     end
%                     [is, tline]=strtok(tline);   %skip & check '1'
%                     if ~strcmp(is, '1')
%                         error(['why not 1?']);
%                     end
%                     tline = strrep(tline, '_', ''); %to eliminate the _ characters, if present
%                     tline = deblank(tline); %removing tariling blanks
%                     Age = str2num(tline);
%                 otherwise
%                     %disp(['unused: ...' tline]);
%             end
%         end
%     end
%     fclose(fi);
%     %%%%%%%%%%%% END reading *.PAT file %%%%%%%%%%%%
%     
    %%%%%%% reading some data from *.INF file (ASCII info) %%%%%
    inf_filename=[basename '.INF'];
    fi=fopen(inf_filename, 'r');
    if fi==-1
        inf_filename=[basename '.inf'];
        fi=fopen(inf_filename, 'r');
        if fi==-1
            disp(sprintf('cannot open file  %s for reading', inf_filename)); return
        end
    end
    tline=1;
    while ~feof(fi) & tline~=-1
        tline=fgets(fi);
        if tline==-1
            error('empty file???');
        else
            [is, tline]=strtok(tline);
            switch is
                case 'PLGMontage'                %electrode names
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);
                    if NUMBER_OF_CHANNELS ~= str2num(is)
                        error('different number of channels and electrode names or channels after the montage in file');
                    end
                    
                    for i=1:NUMBER_OF_CHANNELS
                        [is, tline]=strtok(tline);
                        if isempty(is)
                            tline=fgets(fi);
                            [is, tline]=strtok(tline);
                        end
                        MONTAGE(i,:)=is;
                    end
                case 'PLGNC'                      %number of channels
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);   %skip & check '1'
                    if ~strcmp(is, '1')
                        error(['why not 1?']);
                    end
                    [is, tline]=strtok(tline);
                    NUMBER_OF_CHANNELS = str2num(is);
                case 'PLGNS'                      %number of channels
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);   %skip & check '1'
                    if ~strcmp(is, '1')
                        error(['why not 1?']);
                    end
                    [is, tline]=strtok(tline);
                    NUMBER_OF_SAMPLES_IN_FILE = str2num(is);
                case 'PLGSR(Hz)'                      %number of channels
                    [is, tline]=strtok(tline);   %skip & check '='
                    if ~strcmp(is, '=')
                        error(['missing ' is]);
                    end
                    [is, tline]=strtok(tline);   %skip & check '1'
                    if ~strcmp(is, '1')
                        error(['why not 1?']);
                    end
                    [is, tline]=strtok(tline);
                    SAMPLING_FREQ = str2num(is);
                otherwise
                    %disp(['unused: ...' tline]);
            end
        end
    end
    fclose(fi);
    %%%%%%%%%%%% END reading *.INF file %%%%%%%%%%%%
    
    
    %%%%% reading *.CDC file %%%%%%%%%%%%%%%%
    %% calibration & DC offset %%%%%%%%%%%%%%
    cdc_filename=[basename '.CDC'];
    fi=fopen(cdc_filename, 'r');
    if fi==-1
        cdc_filename=[basename '.cdc'];
        fi=fopen(cdc_filename, 'r');
        if fi==-1
            error(sprintf('cannot open file  %s for reading', cdc_filename))
        end
    end
    cdc=fread(fi, [2, NUMBER_OF_CHANNELS], 'float32');
    fclose(fi);
    %%%%%% END reading *.CDC file %%%%%%%%%%%
    
    if ~isPE
        %%%%%% reading marked windows from *.WIN file %%%%%%%%%%
        filename=[basename '.WIN'];
        fi=fopen(filename, 'r');
        if fi==-1
            filename=[basename '.win'];
            fi=fopen(filename, 'r');
            if fi==-1
                return;
                %         error(sprintf('cannot open file  %s for reading', filename))
            end
        end
        NUM_OF_WINDOWS=0;
        while ~feof(fi)
            c =fread(fi, 1, 'uint8');
            if feof(fi)
                break;
            end
            NUM_OF_WINDOWS=NUM_OF_WINDOWS+1;
            WIN_NAME(NUM_OF_WINDOWS) = c;
            WIN_START(NUM_OF_WINDOWS)=fread(fi, 1, 'integer*4');
            WIN_END(NUM_OF_WINDOWS)  =fread(fi, 1, 'integer*4');
        end
        fclose(fi);
        
        if NUM_OF_WINDOWS > 0
            wins.name = WIN_NAME;
            wins.start = WIN_START;
            wins.end = WIN_END;
        else
            wins = [];
        end
        
        %%for k=1:NUM_OF_WINDOWS
        %%    disp([WIN_NAME(k)  '  ' num2str(WIN_START(k)) '  ' num2str(WIN_END(k))  '  ' num2str(WIN_END(k) - WIN_START(k) + 1)]);
        %%end
        
        %% output to terminal -- comment out 3 following lines to turn off %%
        %for i=1:NUM_OF_WINDOWS
        %    disp(sprintf('%c(%03d) %d %d', WIN_NAME(i), WIN_NAME(i), WIN_START(i), WIN_END(i)));
        %end
        %%%%%% END  reading marked windows from *.WIN file %%%%%
    else %%reading the marks
        %%%%%% reading marked windows from *.mrk file %%%%%%%%%%
        filename=[basename '.MRK'];
        fi=fopen(filename, 'r');
        if fi==-1
            filename=[basename '.mrk'];
            fi=fopen(filename, 'r');
            if fi==-1
                return;
                %         error(sprintf('cannot open file  %s for reading', filename))
            end
        end
        NUM_OF_WINDOWS=0;
        while ~feof(fi)
            c =fread(fi, 1, 'char');
            if feof(fi)
                break;
            end
            start=fread(fi, 1, 'integer*4');
            if start - npoints > 1
                NUM_OF_WINDOWS=NUM_OF_WINDOWS+1;
                WIN_NAME(NUM_OF_WINDOWS) = c;
                WIN_START(NUM_OF_WINDOWS) = start - npoints + 1;
                WIN_END(NUM_OF_WINDOWS) = start;
            end
        end
        fclose(fi);
        
        if NUM_OF_WINDOWS > 0
            wins.name = WIN_NAME;
            wins.start = WIN_START;
            wins.end = WIN_END;
        else
            wins = [];
        end
        %%%%%% END  reading marks from *.MRK file %%%%%
    end
end    
    
%%%%% reading & calibrating data %%%%%%%%%%%%%%%%
data_filename=[basename '.PLG'];
datafile_handle=fopen(data_filename, 'r');
if datafile_handle==-1
    data_filename=[basename '.plg'];
    datafile_handle=fopen(data_filename, 'r');
    if datafile_handle==-1
        error(sprintf('cannot open file  %s for reading', data_filename));
    end
end

data = [];
epoch_size = [];
for w=1:NUM_OF_WINDOWS
    if WIN_NAME(w) == state
        fseek(datafile_handle, (WIN_START(w)-1)*NUMBER_OF_CHANNELS*2, 'bof');  %%El 2 es del sizeof de integer*2
        wdata=fread(datafile_handle, [NUMBER_OF_CHANNELS, WIN_END(w)-WIN_START(w)+1], 'integer*2');
        epoch_size = [epoch_size; WIN_END(w)-WIN_START(w)+1];
        for c=1:size(wdata,1)
            for p=1:size(wdata,2)
                wdata(c,p)=round((wdata(c,p).*cdc(1,c) - cdc(2,c)));
            end
        end
        data = [data wdata];
    end
end

fclose(datafile_handle);
%%%%%%% END reading & calibrating data %%%%%%%%%%%

if isPE
    mrks = wins; wins = [];
end

end
