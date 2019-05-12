



function [window]=gaussian_window()

% window length
N_window=7;     
sigma=1;        
% 生成范围(-2,2) 以2/3为步长的矩阵
[x, y] = meshgrid(-(ceil(sigma*2)):4*sigma/(N_window-1):ceil(sigma*2));

% 高斯窗函数
window = (1/(2*pi*sigma^2)).*exp(-0.5.*(x.^2+y.^2)./sigma^2);

return


