## TODO
1. find out why -q_init and -q_end insteadn of -2*q_init, -2*q_end. This is strange when considering terminal constraint as (q0 - q_init)^2.

2. handle collision. If using straightforward collision potential such as sign()*(x - p_obs)**2, it does not work, because the trajectory will go father away from the obstacle to reduce cost and the term of acceralation will be just ignored.  
