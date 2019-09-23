function [data, epoch_size] = plg_readwinds(basename, wins)
%   function [data, MONTAGE, Age, SAMPLING_FREQ, epoch_size] = plg_read(basename, state);
%basename:   name of file to read (without extension(s))

%reading the Neuronic/Track Walker/www.cneuro.edu.cu EEG format

% Piotr J. Durka http://brain.fuw.edu.pl/~durka 2002.06.02

[pp nn ee] = fileparts(basename);
if isempty(ee)
else
    basename = strerep(basename, ee, '');
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
    info = read_plginf([basename '.inf']);
    NUMBER_OF_CHANNELS = info.NChannels;
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
end

NUM_OF_WINDOWS = length(wins.name);

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
