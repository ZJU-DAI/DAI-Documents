
function [map, stat1,stat2] = CFAloc(im1,im2)

% Pattern of CFA on green channel
Bayer = [0, 1;
         1, 0];
     
% 取绿色通道
im1 = im1(:,:,2);
im2 = im2(:,:,2);
% 获取图像的宽度和高度
dim1 = size(im1);
dim2 = size(im2);

% 插值
pre1 = prediction(im1);
pre2 = prediction(im2);
% 采集像素和内插像素的局部变化
var_map1 = getVarianceMap(pre1, Bayer, dim1);
var_map2 = getVarianceMap(pre2, Bayer, dim2);

% 提取特征
stat1 = getFeature(var_map1, Bayer);
stat2 = getFeature(var_map2, Bayer);
% GMM模型参数估计
[mu, sigma] = MoGEstimationZM(stat1);

% 对数似然图
loglikelihood_map = loglikelihood(stat1, mu, sigma);

% 对似然图进行中值滤波
map = medfilt2(loglikelihood_map,[5 5],'symmetric');

return
