# QEEGT-Toolbox

qEEGt stands for Tomographic Quantitative Electroencephalography. This methodology was developed at the Cuban Neuroscience Center as part of the first wave of the Cuban Human Brain Mapping Project (CHBMP) and has been validated and used in different health systems for several decades (Bosch-Bayard et al., 2001; Hernandez-gonzalez et al., 2011). It produces age-corrected normative Statistical Parametric Maps of EEG log source spectra testing compliance to a normative database. Among other features are the calculation of EEG scalp spectra, and the estimation of their source spectra using VARETA source imaging. Crucially, this is completed by the evaluation of z spectra by means of the built-in age regression equations obtained from the CHBMP database (ages 5-87) to provide normative Statistical Parametric Mapping of EEG log source spectra.

This methodology has now been integrated into the the MNI Neuroinformatics Ecosystem as a docker in CBRAIN. Incorporation into the MNI ecosystem now provides CBRAIN registered users access to its full functionality. Openly releasing this software in the CBRAIN platform will facilitate the use of standardized qEEGt methods in different research and clinical settings. An updated precis of the methods is provided in the appendix of a paper which is currently under revision in Frontiers in Neuroinformatics. qEEGt/CBRAIN is the first installment of instruments developed by the C-C-C project.

We are the same time publicly releasing here the Matlab implementation of the qEEGt/CBRAIN facilities for users who want to run it outside the CBRAIN environment.

The toolbox structure is as follows:

1)	A directory named “code” that encapsulates all the Matlab code necessary for the toolbox.
2)	A directory named “leadfields” which contains 4 Leadfields that are used by the toolbox to solve the inverse problem by means of VARETA. These leadfields have been calculated for the Montreal Neurological Institute template. It implements two grids, one defined over the gray matter of the template (GM) and another one which is defined over the gray matter but also include subcortical structures as Thalamus, Insula, Caudate and others (BG). The GM grid contains 3244 voxels and the BG grid contains 3564 voxels. The files which contain the letters KL in the name contain the LeadField (Ke) and the Laplacian matrix (Le). The files containing the letters LF in the name, contain the USV decomposition of the Leadfield, which is internally used by the qEEGt toolbox. This directory has to be added to the Matlab path for the toolbox to work.
3)	A directory named “EEGnorms” contain the populational age dependent mean and standard deviation of the Cuban Normative database, as it is explained in Bosch-Bayard et al., 2001 as well as in the mentioned paper under revision. It includes the coefficients to calculate the z-scores for the log-spectra of the narrow and broad band models both at the scalp and at the sources. This directory has to be added to the Matlab path for the toolbox to work.
4)	The directory named “EEGexample” contains and EEG recording obtained from the Cuban EEG recording system Neuronic, both in its original proprietary format as well as in the text format that is accepted by the qEEGt toolbox. The Matlab functions to read those files are included in the directory “code”.

Use of the toolbox.

The main function of the toolbox is “qeegt.m”, included in the directory “code”, although there is also a qeegt.p, which is the pseudo .P code that was used for dockerization in CBRAIN. This is only function that needs to be called to obtain all the qEEGt calculations.
There is a file named “qeegt.hdr”, which contains the explanation and how to set all parameters. And there is a file named “main_call.m”, in the “code” directory, which explains how to call the “qeegt” function of the toolbox, both with the PLG file as well as the TXT file. The expected format for the TXT file is also explained in the “qeegt.hdr” and in the “main_call.m”.

At the end of the “main_call.m” file there is an example of how to create a TXT file with the expected structure, using the EEG recording provided in the directory “example”.

The qeegt function save all the calculations in files with extension .MOD. This is a proprietary general format created by Neuronic system to store their results. In the directory “plg_tools”, inside the directory “code”, the function “getdatamod.m” reads all the MOD files and returns all the information they contain.

According to the parameters set in the call to the qeegt.m function, there can be:

  a)	Raw cross-spectral matrices for the narrow band model at the scalp

  b)	z cross-spectral matrices for the narrow band model at the scalp. 

  c)	raw broad band matrices for absolute power, relative power and Median Frequency at the scalp.

  d)	z broad band matrices for absolute power, relative power and Median Frequency at the scalp.

  e)	Coherence matrices at the scalp for the narrow band model.

  f)	The energy of the Raw log spectra at the sources, either for the grid at the cortex or the grid including subcortical structures.

  g)	The lead components (XYZ) of the Raw log spectra at the sources, either for the grid at the cortex or the grid including subcortical structures.

  h)	z of the log spectra at the sources, either for the grid at the cortex or the grid including subcortical structures.


Independent Use of the Norms

The directory <norm_coeeficients> contain information on the nature and how to independently access (out of the qEEGT toolbox) the normative coefficients used in this toolbox. Those coefficients were obtained from the Cuban normal population and we are providing indepdent access to them for persons who may want to use them for their own projects. Information about how the Cuban Normative database from wehere the regressions coefficients were obtained is also provide in the file <Normative Data Information.docx>.
  
References

Bosch-Bayard, J., Valdés-Sosa, P., Virues-Alba, T., Aubert-Vázquez, E., John, E. R., Harmony, T., … Trujillo-Barreto, N. (2001). 3D statistical parametric mapping of EEG source spectra by means of variable resolution electromagnetic tomography (VARETA). Clinical EEG (Electroencephalography), 32(2), 47–61. https://doi.org/10.1177/155005940103200203

Hernandez-gonzalez, G., Bringas-vega, M. L., Galán-garcía, L., Bosch-bayard, J., Melie-garcia, Y. L. L., Valdes-urrutia, L., & Cobas-ruiz, M. (2011). P f o r o. Clinical Eeg And Neuroscience, 42(3).

