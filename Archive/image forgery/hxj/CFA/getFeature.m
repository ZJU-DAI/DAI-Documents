



function [statistics] = getFeature(var_map,Bayer)

% logical 把所有的非零变成1 零变成逻辑0
% sigma() 下标索引
% prod 计算数组元素的连乘积
% func 在模版Bayer下 采样信号的连乘积/插值信号的连乘积


func = @(x) (sqrt(prod(x(logical(Bayer)))))/(sqrt(prod(x(not(logical(Bayer))))));

statistics = blkproc(var_map,[2 2],func);



return

