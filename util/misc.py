def clipper(genetic_code, lb, ub):
    """ Small utility function that clips chromosomes according to upper/lower bounds.

    Input:
    - genetic_code: list of chromosomes
    - lb: list of lower bounds for each item in genetic_code
    - ub: list of upper bounds for each item in genetic_code.
    Output:
    - genetic_code: list of clipped chromosomes. """
    for i in range(len(genetic_code)):
        if genetic_code[i] < lb[i]:
            genetic_code[i] = lb[i]
        elif genetic_code[i] > ub[i]:
            genetic_code[i] = ub[i]
    return genetic_code


def BSP(tm, tM, tStep):
    """ Small utility function that returns the binary search values.

    Input:
    - tm: lower bound
    - tM: upper bound
    - tStep: resolution step at which the search must be stopped.
    Output:
    - v: vector returning the binary search values (including upper and lower value). """
    # tm_bak, tM_bak = tm, tM
    v = [tm, tM]
    dt = (tM - tm) / 2
    while (dt >= tStep) and (tM - (tm + dt) >= tStep):
        v.append(dt)
        tm = tm
        tM = tm + dt
        dt = (tM - tm) / 2
    #
    # tm, tM = tm_bak, tM_bak
    # dt = (tM - tm) / 2
    # while (dt >= tStep) and (tM-(tm+dt) >= tStep):
    #     v.append(dt)
    #     tm = tm+dt
    #     tM = tM
    #     dt = (tM - tm) / 2
    return v
