

function [var_map,var_map_interpolated] = getVarianceMap(im,Bayer,dim)

% 以Bayer为模版扩大纬度, pattern 和图像大小相同 pattern的作用是 将采样信号和插值信号分开
pattern = kron(ones(dim(1)/2,dim(2)/2), Bayer);

window = gaussian_window();

% 归一化
mc = sum(sum(window));% 高斯核的系数
window_mean = window./mc;% 归一化
vc = 1 - (sum(sum((window.^2)))); %无偏比例因子

%  variance of acquired pixels
acquired = im.*(pattern);%把采样信号抽离出来
mean_map_acquired = imfilter(acquired,window_mean,'replicate').*pattern;%高斯滤波后抽离采样区域信号 结果为预测误差局部加权均值
sqmean_map_acquired = imfilter(acquired.^2,window_mean,'replicate').*pattern;%高斯滤波抽离采样区域信号
var_map_acquired =  (sqmean_map_acquired - (mean_map_acquired.^2))/vc;   % 局部加权方差

%  variance of interpolated pixels
interpolated = im.*(1-pattern);
mean_map_interpolated = imfilter(interpolated,window_mean,'replicate').*(1-pattern); %高斯滤波
sqmean_map_interpolated = imfilter(interpolated.^2,window_mean,'replicate').*(1-pattern); %高斯滤波
var_map_interpolated = (sqmean_map_interpolated - (mean_map_interpolated.^2))/vc;% 局部加权方差

var_map = var_map_acquired + var_map_interpolated;

return