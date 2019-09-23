
call_with_plg = 0;  %Set to 1 if you have an EEG recording in the PLG Neuronic format; set to 0 if if it is text format

basename = ['example' filesep 'MC0000001'];
state = 'A';
lwin = 2.56; %length of analysis window in seconds
fmin = 0.390625; %Min freq
freqres = fmin; %frequency resolution
fmax=19.11; %max frequency
wbands='1.56 3.51; 3.9 7.41; 7.8 12.48; 12.87 19.11; 1.56 30';
flags='1 1 1 0 1 1 1 1 1 1 1 1';
%flags: Options for calculations
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

brain=1;
pg_apply=1;

if call_with_plg
    qeegt(basename, state, lwin, fmin, freqres, fmax, wbands, brain, pg_apply, flags)

else
    
%THIS IS AN EXAMPLE OF HOW TO CREATE THE TEXT FILE HAVE THE NECESSARY STRUCTURE

%From read_plgwindows the data comes already divided in windows of exact lwin points.
%This uses the PLG file that is provided as example. But it is just an example
    
% % [data, montage, age, SAMPLING_FREQ, epoch_size, wins, msg] = read_plgwindows(basename, state, lwin);
% % txt = cell(7+size(montage,1), 1);
% % txt{1,1} = 'NAME=Jane Doe';
% % txt{2,1} = 'SEX=F';
% % txt{3,1} = ['AGE=' num2str(age, '%.2f')];
% % txt{4,1} = ['SAMPLING_FREQ=' num2str(SAMPLING_FREQ, '%f')];
% % nit = round(lwin*SAMPLING_FREQ);
% % nvt = size(data,2) ./ nit;
% % epochs = nit;
% % txt{5,1} = ['EPOCH_SIZE=' num2str(epochs, '%d ')];
% % txt{6,1} = ['NCHANNELS=' num2str(size(data,1), '%d')];
% % txt{7,1} = 'MONTAGE=';
% % for k=1:size(montage,1)
% %     txt{7+k,1} = montage(k,:);
% % end
% % dlmwrite('H:\qeeg\qeegp\all_in_one\example\MC0000001_A.txt', char(txt), 'newline', 'pc', 'delimiter', '');
% % dlmwrite('H:\qeeg\qeegp\all_in_one\example\MC0000001_A.txt', data', 'newline', 'pc', 'delimiter', '\t', '-append');


    eegdata = 'example\MC0000001_A.txt';
    qeegt(eegdata, state, lwin, fmin, freqres, fmax, wbands, brain, pg_apply, flags)
end
