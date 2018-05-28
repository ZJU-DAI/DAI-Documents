



function [statistics] = getFeature(map,Bayer)

% logical 把所有的非零变成1 零变成逻辑0
% sigma() 下标索引
% prod 计算数组元素的连乘积
% 存在CFA插值时，采集信号区块上的方差比插值信号区块上的方差要大
% 不存在时，方差则没有区别

% dimensione of statistics
Nb = 2;
% func 在模版Bayer下 采样信号的连乘积/插值信号的连乘积
func = @(sigma) (prod(sigma(logical(Bayer))))/(prod(sigma(not(logical(Bayer)))));


% blkproc 对图像进行以[nb, nb]为单位分块处理
statistics = blkproc(map,[Nb Nb],func);

return

