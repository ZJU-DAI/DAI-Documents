

im2 = imread('flowers.tiff');     
im1 = imread('flowers-tampered.tiff');
     
% 获取定位篡改区域的图像以及统计特征
[map, stat1, stat2] = CFAloc(im1,im2);

data1 = log(stat1(:));
data2 = log(stat2(:));

n_bins1 = round(sqrt(length(data1)));
n_bins2 = round(sqrt(length(data2)));   


subplot(3,2,1), imshow(im2), title('Not tampered image');
subplot(3,2,2), imshow(im1), title('Manipulated image');
subplot(3,2,5), imagesc(map),colormap('gray'), title('Probability map ');
subplot(3,2,4), hist(data1, n_bins1), title('Histogram of the proposed feature');
subplot(3,2,3), hist(data2, n_bins2), title('Histogram of the original image');    
