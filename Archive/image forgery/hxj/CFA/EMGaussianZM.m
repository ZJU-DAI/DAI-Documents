
function [mu1,v1,v2] = EMGaussianZM(x)
%
% estimate Gaussian mixture parameters from data x with EM algorithm
% assume x distributed as alpha * N(0,v1) + (1 - alpha) * N(mu2, v2)
% 绝对误差限
tol = 1e-3;
% 最大迭代次数
max_iter = 500;
% 初始化像素参数
alpha = 0.5;
mu1 = mean(x);
v2 = var(x);
v1 = v2/10;
alpha_old = 1;
k = 1;
while abs(alpha - alpha_old) > tol && k < max_iter
    alpha_old = alpha;
    k = k + 1;
    % expectation E步 根据当前的参数计算后验概率
    f1 = alpha * exp(-x.^2/2/v1)/sqrt(v1);
    f2 = (1 - alpha) * exp(-(x - mu1).^2/2/v2)/sqrt(v2);
    alpha1 = f1 ./ (f1 + f2);
    alpha2 = f2 ./ (f1 + f2);
    % maximization M步 根据E步中计算的后验概率再计算新的参数
    alpha = mean(alpha1);
    v1 = sum(alpha1 .* x.^2) / sum(alpha1);
    mu1 = sum(alpha2 .* x) / sum(alpha2);
    v2 = sum(alpha2 .* (x - mu1).^2) / sum(alpha2);
end

return