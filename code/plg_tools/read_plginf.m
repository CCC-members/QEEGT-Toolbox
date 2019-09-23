function inf_info = read_plginf(inf_name)

inf_info = [];

[pp nn ee] = fileparts(inf_name);
if isempty(ee)
    filename = [inf_name '.inf'];
else
    filename = inf_name;
end

try
    %%%%%%% reading some data from *.INF file (ASCII info) %%%%%
    fi=fopen(filename, 'r');
    if (fi==-1) && isempty(ee)
        filename = strrep(inf_name, '.inf', '.xnf');
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
            return
        else
            return
        end
    elseif fi == -1
        return
    end
end
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
