


function [pred_error] = prediction(im)

% predictor with a bilinear kernel
Hpred = [ 0,   -1,    0;
          -1,  4,    -1;
          0,   -1,    0 ];
      
pred_error = imfilter(double(im),double(Hpred));

return
