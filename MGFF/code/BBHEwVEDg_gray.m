function F_BBHEwVEDg_gray = BBHEwVEDg_gray( F )

% clear all;
% close all;
% clc;
%x0=imread('eswar.jpg');
x0=F;
%x0=imread('11.bmp');
% x0=imread('12.bmp');
%y0=rgb2ycbcr(x0);
luma=x0;%(:,:,1);
[m,n]=size(luma);
% cb=y0(:,:,2);
% cr=y0(:,:,3);
k=mean(mean(luma));
r=round(k);
l=length(luma(luma<=r));
         listindex=find(luma<=r);
 k1=reshape(luma(listindex),l,1);%obtained below fist mean values put it into an arry
 k2=mean(k1);%find mean of mean in first half
 k3=round(k2);%round to nearest integer to find pdf and cdf till first quarter
  sum=0;
 l1=length(luma(luma<=k3));%find all  values in luma falls below fist mean
 listindex1=find(luma<=k3);%all  values in luma falls below fist mean given in their locations
 k4=reshape(luma(listindex1),l1,1);%arrange all fist mean values in a vector
     xpdf=hist(k4,[0:k3]);%pdf from 0:r
      xpdf=xpdf/l1;%normalized pdf to get nk/n,l=sum of xpdf,total no of pixels form 0 to r.
%       plot(xpdf);
%       xlabel('gray levels up to mean');
%       ylabel('pdf up to mean');
%       title('histogram for half an image up to mean');
      sk=xpdf*triu(ones(k3+1));
%       figure(2);
%       plot(sk);
%       xlabel('gray levels upto 1st mean');
%       ylabel('cdf upto mean');
%       title('cdf for half  of an image up to ist mean');
      alpha=0.6;
      for l2=0:k3
              list1=find(k4==l2);%find value in an vector i.e converted from matrix
              list(list1)=alpha*sk(l2+1)*(k3+1)+(1-alpha)*(k3+1);
              %list(list1)=sk(l2+1)*(k3+1);%map dont disturb to get bhe as
              %it is 13/3/2011
              ert(l2+1)=alpha*sk(l2+1)*(k3+1)+(1-alpha)*(k3+1);
      end
          p=zeros(m,n);             
      p(listindex1)= list;
%      figure(3);
%      imshow(p);
%      xlabel('gray levels up to first mean');
%       ylabel('luma component equilized image up to first mean');
%      title('processed luma image up to first mean');
     k=mean(mean(luma));
r=round(k);
l=length(luma(luma<=r));
        listindex=find(luma<=r);
 k1=reshape(luma(listindex),l,1);%obtained below fist mean values put it into an arry
 k2=mean(k1);%find mean of mean in first half
 k3=round(k2);%round to nearest integer to find pdf and cdf till first quarter
%   sum=0;
   b=k3;
%  l2=length(luma(luma>k3));%find all  values in luma falls below fist mean
  listindex2=find((luma>k3)&(luma<=r));%all  values in luma falls below fist mean given in their locations
  l2=length(listindex2);
  k5=reshape(luma(listindex2),l2,1);%arrange all fist mean values in a vector
      x2pdf=hist(k5,[k3+1:r]);%pdf from 0:r
       x2pdf=x2pdf/l2;%normalized pdf to get nk/n,l=sum of xpdf,total no of pixels form 0 to r.
%        figure(4);
%        plot(x2pdf);
%        xlabel('gray levels 2nd mean');
%        ylabel('pdf of 2nd mean');
%        title('histogram for 2nd mean');
       sk2=x2pdf*triu(ones(r-k3));
%        figure(5);
%        plot(sk2);
%        xlabel('gray levels of 2nd mean');
%        ylabel('cdf upto mean');
%        title('cdf for half  of an image of 2nd mean');
       k2u=1;
       for l3=k3+1:r
               list2=find(k5==l3);%find value in an vector i.e converted from matrix
               list1(list2)=alpha*sk2(k2u)*(r-k3)+(1-alpha)*(k3+1);
               %list1(list2)=(k3+1)+(sk2(k2u))*(r-k3);%map dont disturb to
               %get BHE 13/3/2011
               ert(l3)=alpha*sk2(k2u)*(r-k3)+(1-alpha)*(k3+1);
               k2u=k2u+1;
       end
           p1=zeros(m,n);             
       p1(listindex2)= list1;
%       figure(6);
%       imshow(p1);
%      xlabel('gray levels up to first 2nd mean');
%       ylabel('luma component equilized image 2nd mean');
%      title('processed luma image up to 2nd mean');
 %lupper30=length(luma(luma>r);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 lupper30=length(luma(luma>r));
%for i=0:r
    %if(luma(luma<=r))
        listindexupper3=find(luma>r);
   % end
%end
 k1upper=reshape(luma(listindexupper3),lupper30,1);
 mean3=mean(k1upper);
  r3=round(mean3);
 %length30=length((luma<=r3)&(luma>r));
  listindexupper30=find((luma>r)&(luma<=r3));
  length30=length(listindexupper30);
  k30upper=reshape(luma(listindexupper30),length30,1);
  
  %length30=length((luma<=r3)&(luma>r));
%  sum=0;
      xpdfupper30=hist(k30upper,[r+1:r3]);%pdf from r+1:r3
      xpdfupper30=xpdfupper30/length30;%normalized pdf to get nk/n,l=sum of xpdf,total no of pixels form r+1 to 255.
%       figure(7);
%      plot(xpdfupper30);
%      xlabel('gray levels   3rd mean');
%      ylabel('pdf of 3rd mean');
%      title('histogram for upper half an image  3rd mean');
      skupper30=xpdfupper30*triu(ones(r3-r));
%       figure(8);
%       plot(skupper30);
%       xlabel('gray levels after mean');
%       ylabel('cdf after mean');
%       title('cdf for upper half  of an image after mean');
      k3u=1;
      for k3upper=(r+1):r3
              list1upper30=find(k30upper==k3upper);%find value in an vector i.e converted from matrix
              listnew(list1upper30)=(r+1)+skupper30(k3u)*(r3-r);%map
              ert(k3upper)=(r+1)+skupper30(k3u)*(r3-r);
         k3u=k3u+1;
     end
     
          p2=zeros(m,n);
             
     p2(listindexupper30)= listnew;
%     figure(9);
%      imshow(p2);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     listindexupper40=find(luma>r3);
     lupper40=length(listindexupper40);
   % end
%end
 k1upper40=reshape(luma(listindexupper40),lupper40,1);
 %sum=0;
 %for i=0:r
     xpdfupper40=hist(k1upper40,[r3+1:255]);%pdf from r+1:255
     xpdfupper40=xpdfupper40/lupper40;%normalized pdf to get nk/n,l=sum of xpdf,total no of pixels form r+1 to 255.
%      figure(10);
%      plot(xpdfupper40);
%      xlabel('gray levels after 4mean');
%      ylabel('pdf after 4mean');
%      title('histogram for upper half an image after 4mean');
     skupper40=xpdfupper40*triu(ones(255-r3));
%      figure(11);
%      plot(skupper40);
%      xlabel('gray levels after 4mean');
%      ylabel('cdf after 4mean');
%      title('cdf for upper half  of an image after 4mean');
     k4u=1;
     for k4upper=(r3+1):255
         %if(xpdfupper(k2upper)>r)
             list1upper40=find(k1upper40==k4upper);%find value in an vector i.e converted from matrix
             %for k2u=1:58
             listnew4(list1upper40)=(r3+1+skupper40(k4u)*(255-r3));%map
         %end
         ert(k4upper)=(r3+1+skupper40(k4u)*(255-r3));
         k4u=k4u+1;
     end
     
%      for i=0:l-1
          p3=zeros(m,n);
%          if (p(listindex))
%              p(:)=list;
%          end
%      end
             
     p3(listindexupper40)= listnew4;
%     figure(12);
%     imshow(p3);
%     xlabel('gray levels after 4mean');
%      ylabel('luma component equilized image after 4mean');
%      title('processed luma image after 4mean');
     om=p+p1+p2+p3;
     F_BBHEwVEDg_gray=om;
     %ommmmm=p1+p;
%     figure(13);
%  imshow(uint8(om));
    % colormap('gray');
%       xlabel('gray level');
%      ylabel('combined lower and upper half luma component equilized image');
%      title('combined luma image');
%     figure(8);
    %image(om);
%     for j1=0:255
%         count=0;
%         for i1=0:m*n-1
%             if om(i1+1)==j1
%                 count=count+1;
%             end
%         end
%             prob(j1+1)=count/m*n;
%     end
%     figure(16);
%     plot(prob);
%     
%    for j2=0:255
%         count1=0;
%         for i2=0:m*n-1
%             if luma(i2+1)==j2
%                 count1=count1+1;
%             end
%         end
%             prob2(j2+1)=count1/m*n;
%     end
%     figure(17);
%     plot(prob2); 
%     xlabel('gray levels after mean');
%      ylabel('luma component equilized image after mean');
%      title('processed luma image after mean');
%     ommmmm=p1+p;
%     figure(7);
%      colormap('gray');
%       xlabel('gray level');
%      ylabel('combined lower and upper half luma component equilized image');
%      title('combined luma image');
%     image(ommmmm);
%      cat1=cat(3,om,cb,cr);
%     figure(14);
%     imshow(cat1);
% %     xlabel('gray level(ycbcr)');
% %      ylabel('combined lower and upper half luma,cromablue,croma red component equilized image');
% %      title('luma croma b and r color processed image');
%     catconversion=ycbcr2rgb(cat1);
%     figure(15);
%     imshow(catconversion);
%     xlabel('gray level(rgb)');
%      ylabel('combined lower and upper half  RGB component equilized image');
%      title('converted from ycbcr2rgb color(RGB) processed image');

