
function [alpha, sg1, mu1, sg2] = EMGaussianZM(x, tol)
%
% estimate Gaussian mixture parameters from data x with EM algorithm
% assume x distributed as alpha * N(0,v1) + (1 - alpha) * N(mu2, v2)

% 最大迭代次数
max_iter = 500;
% 初始化像素被选中的概率
alpha = 0.5;
% mean(x) 返回x的均值 
mu1 = mean(x);
% var(x) 定义为概率密度函数f的二阶矩，返回x的方差
sg2 = var(x);
sg1 = sg2/10;
alpha_old = 1;
k = 1;
while abs(alpha - alpha_old) > tol && k < max_iter
    alpha_old = alpha;
    k = k + 1;
    % expectation E步 根据当前的参数计算后验概率
    f1 = alpha * exp(-x.^2/2/sg1)/sqrt(sg1);
    f2 = (1 - alpha) * exp(-(x - mu1).^2/2/sg2)/sqrt(sg2);
    alpha1 = f1 ./ (f1 + f2);
    alpha2 = f2 ./ (f1 + f2);
    % maximization M步 根据E步中计算的后验概率再计算新的参数
    alpha = mean(alpha1);
    sg1 = sum(alpha1 .* x.^2) / sum(alpha1);
    mu1 = sum(alpha2 .* x) / sum(alpha2);
    sg2 = sum(alpha2 .* (x - mu1).^2) / sum(alpha2);
end

return