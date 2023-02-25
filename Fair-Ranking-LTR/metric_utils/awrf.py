import numpy as np

def std_err(p_hat, n):
    """Calculates standard error of a binomially-ditributed observation p_hat
    
    Args:
        p_hat: population estimator, for example calculated using estimate_p
        n: number of items. In a single realization of the ranking without 
        ground truth knowledge of the total number oof items this is
            the size of the rank. 
    """

    return np.sqrt((p_hat)*(1-p_hat)/n)

def estimate_p(alignment, n=None):
    """Calculates the population estimate
    
    Args:
        alignment: L_R matrix describing the alignment with a protected
            attribute, in the shape of |R| x |c|
            (one row per rank, one column per class indicator)
        n: number of elements. If None, the size of the list is used, 
            but it could be a different number (for example in case)
            of aggregates it would be the number of unique items
    """
    if n is None:
        n = alignment.shape[0]
    return np.sum(alignment, axis=0)/n

def subtraction(a, b):
    return a-b

def vlambda(alignment, p_hat = None, delta_max = None,\
            W_R = None, distance=subtraction):

    """
    Finds potentially fair distributions for a given aligntment
    Args:
        alignment: L_R matrix describing the alignment with a protected
            attribute, in the shape of |R| x |c|
            (one row per rank, one column per class indicator)
        p_hat: population estimate, default None. If None, p_hat is calculated
            from alignment
        delta_max: maximum allowable distance between the p_hat and exposure, 
            default None. If None, one standard error from p_hat is used
        W_R: attention vector, default None. If None, geometric distributions
            with varying lambda are used
        distance: function used to calculate the distance between E_R and p_hat

    """
    if p_hat is None:
        p_hat = estimate_p(alignment)

    if delta_max is None:
        delta_max = std_err(p_hat, alignment.shape[0])


    if W_R is None:
        lambda_values = np.arange(0.02, 0.5, 0.02)

        # Matrix of precomputed geometric distribution values. Each row represents a different success parameter `lambda`
        # and each column represents the next trial in the series (i.e. the attention of each search result)
        W_R = np.asarray([(1-lambda_)**np.arange(0, alignment.shape[0])*lambda_ for lambda_ in lambda_values]).transpose()
    else:
        lambda_values = None
   
    W_R = W_R/sum(W_R) # should sum to 1

    E_R = np.matmul(W_R.transpose(), alignment)

    # Sum the absolute values for each category
    # With Lambda value 0.14
    results = distance(E_R, p_hat)
    result = results[results.index==6]
    scores = result.values.tolist()[0]
    
    # Mean normalised scores
    normalised_mean = sum([abs(i) for i in scores])/len(scores)

    return normalised_mean