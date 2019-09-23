function wins = read_plgwin(basename)

wins = [];

%%%%%% reading marked windows from *.WIN file %%%%%%%%%%
fi=fopen(basename, 'r');
if fi==-1
    filename=[basename '.win'];
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
