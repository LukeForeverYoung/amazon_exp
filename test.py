from random import randint, randrange

from math import exp, log

from quality_aware.torch_version.Data import envolu_data_sample


def randabs():
    return randint(1,10)

def lg(p):
    return log(1+exp(p))

def cal(p,t):
    loss = 0
    if t == 1:
        loss += p
    loss -= lg(p)
    return loss

def step(p,t):
    tmp=randrange(0,10)
    if tmp >=p:
        f=1
    else:
        f=-1
    p=f*randabs()

    return cal(p,t),p
envolu_data_sample(None)
input()
num=6
w=5
print('TT','\t'*num,'FT','\t'*num,'TF','\t'*num,'FF')
print(cal(w,1),'\t',cal(-w,1),'\t',cal(w,0),'\t',cal(-w,0))
gt=[1 if randrange(0,10)>=2 else  0for i in range(1000)]

for x in range(10):
    loss = 0
    m = 0
    for i,label in enumerate(gt):
        r1,r2=step(x,label)
        loss+=r1
        m+=r2
    print(loss,m/len(gt))


