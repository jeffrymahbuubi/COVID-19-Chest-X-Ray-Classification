%================================================================================================
% Function: Image Enhanchement
% Author: Aunuun Jeffry Mahbuubi / 馬杰睿
% Date: 2022/01/22
% NB: The code given will automatically enhance the image and save with the
% same name file of the image. 
%================================================================================================
%% Histogram Equalization

%Original Data Path
datapath = fullfile('C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\Computer Aided Engineering\Quiz, Mid-Term, Final-Project\Final Project\Deep Learning\train');
%Subfolder of Original File
folderori_C = fullfile(datapath,'COVID');
folderori_N = fullfile(datapath,'NORMAL');
folderori_P = fullfile(datapath,'PNEUMONIA');
%Images data inside Subfolder & To fetch the name of the data
filelist_C = dir(fullfile(folderori_C, '*.png')); filelist_C = natsortfiles({filelist_C.name}); 
filelist_N = dir(fullfile(folderori_N, '*.png')); filelist_N = natsortfiles({filelist_N.name});
filelist_P = dir(fullfile(folderori_P, '*.png')); filelist_P = natsortfiles({filelist_P.name});
%Output Path, make the output folder first. 
outputpath = fullfile('C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\Computer Aided Engineering\Quiz, Mid-Term, Final-Project\Final Project\Image Processingv2\histogrameq');
folderout_C = fullfile(outputpath,'COVID');
folderout_N = fullfile(outputpath,'NORMAL');
folderout_P = fullfile(outputpath,'PNEUMONIA');

%Process Image 
num_iterations = numel(filelist_C);
for fidx = 1:num_iterations
    outfilepattern_C = string(filelist_C(fidx));
    outfilepattern_N = string(filelist_N(fidx));
    outfilepattern_P = string(filelist_P(fidx));
    img_C = imread(fullfile(folderori_C, filelist_C{fidx}));
    img_N = imread(fullfile(folderori_N, filelist_N{fidx}));
    img_P = imread(fullfile(folderori_P, filelist_P{fidx}));
    histeq_C = histeq(img_C);
    histeq_N = histeq(img_N);
    histeq_P = histeq(img_P);
    imwrite(histeq_C, fullfile(folderout_C, sprintf(outfilepattern_C, fidx)));
    imwrite(histeq_N, fullfile(folderout_N, sprintf(outfilepattern_N, fidx)));
    imwrite(histeq_P, fullfile(folderout_P, sprintf(outfilepattern_P , fidx)));
end


%% Gamma Correction

%Original Data Path
datapath = fullfile('C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\Computer Aided Engineering\Quiz, Mid-Term, Final-Project\Final Project\Deep Learning\train');
%Subfolder of Original File
folderori_C = fullfile(datapath,'COVID');
folderori_N = fullfile(datapath,'NORMAL');
folderori_P = fullfile(datapath,'PNEUMONIA');
%Images data inside Subfolder & To fetch the name of the data
filelist_C = dir(fullfile(folderori_C, '*.png')); filelist_C = natsortfiles({filelist_C.name});
filelist_N = dir(fullfile(folderori_N, '*.png')); filelist_N = natsortfiles({filelist_N.name});
filelist_P = dir(fullfile(folderori_P, '*.png')); filelist_P = natsortfiles({filelist_P.name});
%Output Path, make the output folder first. 
outputpath = fullfile('C:\Users\LENOVO X1E\Desktop\Course Energy Engineering Year 1\Semester Ganjil\5\Computer Aided Engineering\Quiz, Mid-Term, Final-Project\Final Project\Image Processingv2\gammacorrection');
folderout_C = fullfile(outputpath,'COVID');
folderout_N = fullfile(outputpath,'NORMAL');
folderout_P = fullfile(outputpath,'PNEUMONIA');

%Process Image 
num_iterations = numel(filelist_C);
for fidx = 1:num_iterations
    outfilepattern_C = string(filelist_C(fidx));
    outfilepattern_N = string(filelist_N(fidx));
    outfilepattern_P = string(filelist_P(fidx));
    img_C = imread(fullfile(folderori_C, filelist_C{fidx}));
    img_N = imread(fullfile(folderori_N, filelist_N{fidx}));
    img_P = imread(fullfile(folderori_P, filelist_P{fidx}));
    gammacor_C = lin2rgb(img_C);
    gammacor_N = lin2rgb(img_N);
    gammacor_P = lin2rgb(img_P);
    imwrite(gammacor_C, fullfile(folderout_C, sprintf(outfilepattern_C, fidx)));
    imwrite(gammacor_N, fullfile(folderout_N, sprintf(outfilepattern_N, fidx)));
    imwrite(gammacor_P , fullfile(folderout_P, sprintf(outfilepattern_P , fidx)));
end

%% 



