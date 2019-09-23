function mrks = read_plgmrk(basename)

wins = [];
mrks = [];

%%%%%% reading marks from *.MRK file %%%%%%%%%%
if isempty(strfind(lower(basename), '.mrk'))
    basename = [basename '.mrk'];
end
fi=fopen(basename, 'r');
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
