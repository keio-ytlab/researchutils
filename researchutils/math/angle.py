import math
import copy 

def angle_range_corrector(angle, MAX=math.pi, MIN=-math.pi):
    '''
    Check angle range and correct the range

    Parameters
    -------
    angle : numpy.ndarray 
        unit is radians
    MAX : float
        maximum of range [rad] , default math.pi
    MIN : float
        minimum of range [rad], default -math.pi
    Returns
    -------
    correct_angle : numpy.ndarray
        correct range angle
    '''
    
    correct_angle = copy.deepcopy(angle).flatten()

    for i in range(len(correct_angle)):
        if correct_angle[i] > MAX:
            while correct_angle[i] > MAX:
                correct_angle[i] -=  2 * math.pi
        elif correct_angle[i] < MIN:
            while correct_angle[i] < MIN:
                correct_angle[i] +=  2 * math.pi
    
    correct_angle = correct_angle.reshape(angle.shape)
    
    return correct_angle
