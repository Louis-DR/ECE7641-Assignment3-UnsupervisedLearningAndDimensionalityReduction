import time

time1 = time.time()

def tickinit(nbr_runs):
    ticksize=0
    tickmarks=[]
    if nbr_runs>=20:
        ticksize = 1
        tickmarks = [int(k*nbr_runs/20) for k in range(20)]
    elif nbr_runs>=10:
        ticksize = 2
        tickmarks = [int(k*nbr_runs/10) for k in range(10)]
    elif nbr_runs>=5:
        ticksize = 4
        tickmarks = [int(k*nbr_runs/5) for k in range(5)]
    else:
        ticksize = 20
        tickmarks = [nbr_runs-1]
    print("┌"+20*"─"+"┐")
    print("│",end="")
    return ticksize, tickmarks

def tick(ticksize, tickmarks, run_nbr):
    if (run_nbr in tickmarks) : print(ticksize*"▓",end="")

def chrono(n=0, prt=False):
    global time1
    rt = time.time() - time1
    if (prt) : print(" "*n+"Computation time : {0:.2f} seconds".format(rt))
    time1 = time.time()
    return rt