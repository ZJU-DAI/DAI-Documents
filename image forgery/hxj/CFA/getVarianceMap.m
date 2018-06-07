

function [var_map] = getVarianceMap(im,Bayer,dim)

% 以Bayer为模版扩大纬度, pattern 和图像大小相同
pattern = kron(ones(dim(1)/2,dim(2)/2), Bayer);

% 1值区域被处理, 分别处理采样信号和插值信号
mask = [1, 0, 1, 0, 1, 0, 1;
        0, 1, 0, 1, 0, 1, 0;
        1, 0, 1, 0, 1, 0, 1;
        0, 1, 0, 1, 0, 1, 0;
        1, 0, 1, 0, 1, 0, 1;
        0, 1, 0, 1, 0, 1, 0;
        1, 0, 1, 0, 1, 0, 1];

% gaussian window fo mean and variance
window = gaussian_window().*mask;
mc = sum(sum(window));
window_mean = window./mc;

%  variance of acquired pixels
vc = 1 - (sum(sum((window.^2))));
acquired = im.*(pattern);
mean_map_acquired = imfilter(acquired,window_mean,'replicate').*pattern;
sqmean_map_acquired = imfilter(acquired.^2,window_mean,'replicate').*pattern;
var_map_acquired =  (sqmean_map_acquired - (mean_map_acquired.^2))/vc;

%  variance of interpolated pixels
interpolated = im.*(1-pattern);
mean_map_interpolated = imfilter(interpolated,window_mean,'replicate').*(1-pattern);
sqmean_map_interpolated = imfilter(interpolated.^2,window_mean,'replicate').*(1-pattern);
var_map_interpolated = (sqmean_map_interpolated - (mean_map_interpolated.^2))/vc;

var_map = var_map_acquired + var_map_interpolated;

return