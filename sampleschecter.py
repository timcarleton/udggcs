import numpy as np

def sampleschecter( alpha, L_star, L_min, N):
    """ 
        Generate N samples from a Schechter distribution, which is like a gamma distribution 
        but with a negative alpha parameter and cut off on the left somewhere above zero so that
        it converges.
        
        If you pass in stupid enough parameters then it will get stuck in a loop forever, and it
        will be all your own fault.
        
        Based on algorithm in http://www.math.leidenuniv.nl/~gill/teaching/astro/stanSchechter.pdf
        """
    n=0
    output = []
    while n<N:
        L = np.random.gamma(scale=L_star, shape=alpha+2, size=N)
        L = L[L>L_min]
        u = np.random.uniform(size=L.size)
        L = L[u<L_min/L]
        output.append(L)
        n+=L.size
    return np.concatenate(output)[:N]
