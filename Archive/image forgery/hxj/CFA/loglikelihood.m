


function [L] = loglikelihood(statistics, mu, sigma)

% Loglikelihood map
min = 1e-320;
max = 1e304;

statistics(isnan(statistics))=1;
statistics(isinf(statistics))=max;
statistics(statistics == 0) = min;

mu1=mu(1);
mu2=mu(2); 

sigma1=sigma(1);
sigma2=sigma(2);

% log-likelihood ÌØÕ÷¾ØÕó
L = 0.5.*((((log(statistics) - mu1).^2)/sigma1^2)...
    -0.5*(((log(statistics) - mu2).^2)/sigma2^2));

return