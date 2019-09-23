% bd_basepathname = 'V:\MCH\';
% bd_basepathname = 'H:\Datos\MCH\';
bd_basepathname = 'D:\';

state = 'A';

eegs = dir([bd_basepathname 'EEGs\*.']);

eegs = {eegs.name};
if strcmp(eegs{1}, '.'), eegs(1) = []; end
if strcmp(eegs{1}, '..'), eegs(1) = []; end
montaje = cell(length(eegs),1);
% info_casos = cell(length(eegs),1);
for k=1:length(eegs)
    nuevo = eegs{k};
    disp(nuevo);
    plg_name = [bd_basepathname 'EEGs\' nuevo '\' nuevo '.inf'];
    inf_info = read_plginf(plg_name);
    montaje{k} = inf_info.MONTAGE;
    plg_name = [bd_basepathname 'EEGs\' nuevo '\' nuevo '.pat'];
%     pat_info = read_plgpat(plg_name);
%     info_casos{k,1} = nuevo;
%     info_casos{k,2} = pat_info.Sex;
%     info_casos{k,3} = pat_info.Age;
end

comunes = char(montaje{1});
comunes = cellstr(comunes(:,1:3));
montaje{1} = comunes;
for k=2:length(montaje)
    disp(k)
    c = char(montaje{k});
    c = cellstr(c(:,1:3));
    montaje{k} = c;
    comunes = intersect(comunes, c);
end

c = char(comunes);
ind = find(c == '_');
c(ind) = ' ';
c = str2num(c);
c = sort(c);
for k=1:size(c,1)
    d = num2str(c(k,:));
    comunes{k} = [d setstr(ones(1,3-length(d))*'_')];
end


%buscar los indices en cada caso
indices = zeros(length(comunes),length(montaje));
for k=1:length(montaje)
    c = char(montaje{k});
    [aa, indices(:,k)]=ismember(comunes, montaje{k});
end


return


%cargar los datos
longv = 512;
ind_d = 1;
for k=1:length(eegs)
    nuevo = eegs{k};
    disp(nuevo);
    plg_name = [bd_basepathname 'EEGs\' nuevo '\' nuevo];
    [data, MONTAGE, Age, SAMPLING_FREQ, epoch_size, wins] = plg_read(plg_name, state);
    if ~isempty(epoch_size)
        starts = cumsum([1; epoch_size(1:end-1)]);
        longs = longv.*floor(epoch_size ./ longv);
        ends = starts+longs-1;
        ind = [];
        for h=1:length(starts)
            ind = union(ind, [starts(h):ends(h)]);
        end
        ages(ind_d) = Age;
        sp(ind_d) = SAMPLING_FREQ;
        datos{ind_d} = data(indices(:,k),ind); ind_d = ind_d + 1;
    end
end

ind = cellfun('isempty', datos);
datos(ind) = [];

save datamch datos ages sp

