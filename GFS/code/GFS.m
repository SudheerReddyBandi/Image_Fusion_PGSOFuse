
% Please cite this article as: Bavirisetti DP, Kollu V, Gang X, Dhuli R. Fusion of MRI and CT images using guided image filter and image statistics. 
% Int. J. Imaging Syst. Technol. 2017;27:227–237. https://doi.org/10.1002/ima.22228
close all;
clear all;
clc;
  
 % Parameters           
r=25; 
eps=2.1;
cov_wsize=5;

is_overlap = 1;

%% input pair images
for j=1:13
index = j;
path1 = ['E:\my research\guide change\phd data\fusion\data\proposed work\images\optical\',num2str(index),'.png'];
path2 = ['E:\my research\guide change\phd data\fusion\data\proposed work\images\sar\',num2str(index),'.png'];
                if is_overlap == 1
                    path_temp = './fused/';
                    if exist(path_temp,'dir')==0
                        mkdir(path_temp);
                    end
                    fuse_path = [path_temp, 'fused',num2str(index),'.png'];
                end
                if exist(fuse_path,'file')~=0
                    continue;
                end
fig_origin1 = imread(path1);
fig_origin2 = imread(path2);

%I1 = im2double(fig_origin1);
%I2 = im2double(fig_origin2);

 I1=imresize(fig_origin1,[128 128]);
 I2=imresize(fig_origin2,[128 128]);
 I(:,:,1)=I1;
 I(:,:,2)=I2;
 
tic
% Base and detail layers seperation
A1= guidedfilter(double(I1), double(I2), r, eps);
B1=uint8(A1);
D1=double(I1)-A1;
C1=uint8(D1);
A2=guidedfilter(double(I2), double(I1), r, eps);
B2=uint8(A2);
D2=double(I2)-A2;
C2=uint8(D2);

 D(:,:,1)=D1;
 D(:,:,2)=D2;
 % Fusion rule
 xfused=GFS_fusion_rule(I,D,cov_wsize);
  toc
 FF=uint8(xfused);
 imwrite(FF,fuse_path,'png');
end
% Display of images
%figure, imshow(I1,[]);
%figure, imshow(I2,[]);
%figure, imshow(FF,[]);
 
