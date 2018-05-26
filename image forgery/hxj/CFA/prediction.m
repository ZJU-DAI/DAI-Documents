


function [pred_error] = prediction(im)

% predictor with a bilinear kernel 拉普拉斯核
Hpred = [ 0,   1,    0;
          1,  -4,    1;
          0,   1,    0 ];
      
% imfilter 对任意类型的数组或多维图像进行滤波 
% replicate 图像大小通过复制外边界的值来扩展

pred_error = imfilter(double(im),double(Hpred),'replicate');


return