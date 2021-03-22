import numpy as np
def lookup_gain_in_table(G, a_post, a_priori, a_post_range, a_priori_range, step):
    a_prioridb = np.round(10*np.log10(a_priori)/step)*step
    a_postdb = np.round(10*np.log10(a_post)/step)*step
    Ia_post = np.minimum(np.maximum(np.min(a_post_range), a_postdb), np.max(a_post_range))
    Ia_post = Ia_post-np.min(a_post_range)+1
    Ia_post = Ia_post/step

    Ia_priori = np.minimum(np.maximum(np.min(a_priori_range), a_prioridb), np.max(a_priori_range))
    Ia_priori = Ia_priori-np.min(a_priori_range)+1
    Ia_priori = Ia_priori/step
    
    import warnings
    warnings.filterwarnings('ignore')
    lookup = Ia_priori.astype('int')+ ((Ia_post.astype('int')) * len(G[:,1]) )
   
    lookup[lookup >8280] = 8280 # it's gonna be 1 anyway so its ok
    f = np.unravel_index(lookup, (91, 91))
    f0 = f[0]
    f1 = f[1]
    f = [f1-1,f0]
    gains = G[tuple(f)]

    return gains
