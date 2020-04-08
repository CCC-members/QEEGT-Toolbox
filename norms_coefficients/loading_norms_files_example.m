age = 20; 

%Reading a normative file for narrow band at the scalp
filename = 'A_AVR_PG_NB_.NRM';
[meancoef, stdcoef, state, pgcorrect, freqres, freqsHz]=readnrmcbe(filename, age);

%Reading a normative file for broad band at the scalp
filename = 'A_AVR_PG_BB_.NRM';
[mncoefPA, stdcoefPA, mncoefPR, stdcoefPR, mncoefFM, stdcoefFM, state, pgcorrect, freqres, nrm_band_index]=readnrmcba(filename,age);


%Reading a normative file for narrow band at the sources
filename = 'A_AVR_PG_ET_GM_GF.NRM';
[meancoef, stdcoef, state, pgcorrect, freqres, freqsHz]=readnrmcbe(filename,age);
