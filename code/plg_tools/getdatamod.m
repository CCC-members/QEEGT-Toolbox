%[Matrix, MComment, MI, DurationUnit,  HeaderLists, Context, Labels] = getdatamod(fName);
function [Matrix, MComment, MI, DurationUnit,  HeaderLists, Context, Labels] = getdatamod(fName)

%%fName = 'c:\datos\alar\ALAR___-CROSS-A-7.MOD';
%%fName = 'c:\datos\kaki\sup\1261181-ZETCBG-A-2.MOD';


ProtMask = 4657; %this is a mark for file models, the first three nibbles are a key protection 
%and the last is the current models version, this is a good means to protect 
%ourselves from mismatch versions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[pp nn ee]=fileparts(fName);
motfile = strcmp(lower(ee), '.mot');

fid = fopen(fName, 'r');
if fid == -1
    error(['Could not open file ' fName]);
end

[TmpProt, count] = fread(fid, 1, 'integer*2');
if count ~= 1
    fclose(fid);
    error(['Infalid file type. File ' fName]);
end


if TmpProt ~= ProtMask
    fclose(fid);
    error(['Infalid file type. File ' fName]);
end

%Lectura de:      MComment      : string[msComment];
msComment      = 80;
l = fread(fid,1,'integer*1'); %Leer el byte de la long del string
MComment = fread(fid,msComment,'char*1');
MComment = setstr(MComment(1:l))';

%%Lectura de los tamaños de las medidas
if motfile
    xx(1:4) = fread(fid,4,'integer*4');
    xx(5:6) = fread(fid,2,'integer*2');
else
    xx = fread(fid,6,'integer*2');
end
MI=struct('MeasureSize',xx(1),...
          'DurationSize',xx(2),...
          'FirstSpaceSize',xx(3),...
          'SecondSpaceSize',xx(4),...
          'ReservedBytes',xx(5),...
          'DataSize',xx(6));

MatrixSize = MI.MeasureSize*MI.DurationSize*MI.FirstSpaceSize*MI.SecondSpaceSize;

if MI.DataSize == 4
    Matrix =fread(fid,MatrixSize,'real*4');
else
    error('DataSize unexpected');
end

Matrix = reshape(Matrix,MI.SecondSpaceSize,MI.FirstSpaceSize,MI.DurationSize,MI.MeasureSize);

if MI.ReservedBytes ~= 0
    ReservedData = fread(fid,MI.ReservedBytes,'char*1');
else
    ReservedData = '';
end;

%%Lectura de:   DurationUnit   : string[msDurationUnit];
msDurationUnit = 8;
l = fread(fid,1,'integer*1'); %Leer el byte de la long del string
DurationUnit = fread(fid,msDurationUnit,'char*1');
%%DurationUnit = setstr(DurationUnit(1:l))';   %%%OJO%%%

%%Lectura de:     HeaderLists   : packed array[1..MaxList] of TListInfo;
%% TListInfo = packed record
%%   Ofs   : Longint;
%%   Total : Word;
%% end;
%%HeaderSize = SizeOf(TListInfo)*MaxList;
MaxList = 8;
HeaderLists = struct('Ofs',zeros(MaxList,1),  'Total', zeros(MaxList,1));
for k=1:MaxList
    HeaderLists.Ofs(k)=fread(fid,1,'integer*4');
    HeaderLists.Total(k)=fread(fid,1,'integer*2');
end

status = fseek(fid, HeaderLists.Ofs(1), 'bof') ;

Labels.ListLabelM      = ReadList(fid, HeaderLists.Total(1));
Labels.ListLabelD      = ReadList(fid, HeaderLists.Total(2));
Labels.ListLabelSp1  = ReadList(fid, HeaderLists.Total(3));
Labels.ListLabelSp2  = ReadList(fid, HeaderLists.Total(4));
Labels.ListScale         = ReadList(fid, HeaderLists.Total(5));
Labels.ListUnit            = ReadList(fid, HeaderLists.Total(6));
Labels.ListTransform = ReadList(fid, HeaderLists.Total(7));

%% Read information of the context (the last context)
%% TContextInfo = packed record
%%   Name    : string[msContext];
%%   Size    : Word;
%%   Ptrdata : Pointer;
%% end;

%Context = struct('Name', cell(1,1), 'Size', zeros(HeaderLists.Total(MaxList),1), 'Data', 0);
%Context.Name ={ 'a'};  %%Esto aqui esta de truco porque no se como declarar y asignar estructuras que contengan cells
for k=1:HeaderLists.Total(MaxList)
    l=fread(fid,1,'integer*1');
    st =fread(fid,l,'char*1');
    st=setstr(st)';
    Context.Name(k) ={st};
    Context.Size(k)=fread(fid,1,'integer*2');
end

OfsDataContext = ftell(fid);  %No se para que lo usan, pero lo dejo por si acaso
for k=1:HeaderLists.Total(MaxList)
    st=char(Context.Name(k));
    if findstr('Single',st)
       nitems = floor(Context.Size(k) ./ 4);
       st =fread(fid,nitems,'real*4');
    elseif findstr('String',st)
        nitems = floor(Context.Size(k) ./ 1);
        l=fread(fid,1,'integer*1');
        nitems=nitems-1;
        st =fread(fid,nitems,'char*1');
        st = char(st(1:min([l nitems]))');
    elseif findstr('Boolean',st)
        nitems = floor(Context.Size(k) ./ 1);
       st =fread(fid,nitems,'integer*1');
    else
    end
    Context.Data(k) = {st};
end

fclose(fid);



function List = ReadList(fid, Total);
List = cell(Total,1);
for k=1:Total
    l = fread(fid,1,'integer*1'); %Leer el byte de la long del string
    st=fread(fid,l,'char*1');
    st = setstr(st)';
    List(k)={st};
end
