
function [map,stat1] = CFAloc(im1)

% Pattern of CFA on green channel
Bayer = [0, 1;
         1, 0];
     
% 取绿色通道
im1 = im1(:,:,2);

% 获取图像的宽度和高度
dim1 = size(im1);

% 插值
pre1 = prediction(im1);

% 采集像素和内插像素的局部变化
[var_map1,] = getVarianceMap(pre1, Bayer, dim1);

% 提取特征
stat1 = getFeature(var_map1, Bayer);

% GMM模型参数估计
[mu, sigma] = MoGEstimationZM(stat1);

% 对数似然图 对统计的特征进行分类
loglikelihood_map = loglikelihood(stat1, mu, sigma);

% 对似然图进行中值滤波
func = @(x) sum(x(:));
log_L_cum = blkproc(loglikelihood_map,[3 3],func);
map = medfilt2(log_L_cum,[3 3]);

return
