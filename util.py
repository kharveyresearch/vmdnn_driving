"""
    util.py

    General utility functions
"""

import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def zeroone(num):
    return max(0,min(1,num))

def rotvec(vec, theta):
    st=sin(theta); ct=cos(theta)
    mat = np.array([[ct, -st, 0],
                    [st, ct, 0],
                    [0, 0, 1]])
    return np.dot(mat, [vec[0],vec[1],1])[:2]


def joint2soft(analog, joint_ranges=[(-1,1),(-1,1)], N_SOFTMAX=10, sigma=0.5):
    """expects n x j"""
    analog = np.array(analog)
    if joint_ranges==[]:
        return analog

    #print('Converting analog:',analog.shape,'to softmax..')
    if len(analog.shape)<2:
        analog = analog.reshape((-1,1))
    joint_centers = [np.linspace(start, end, N_SOFTMAX) for (start, end) in joint_ranges]


    vals = [[np.exp(-(center - analog[:, j]) ** 2) / sigma for center in joint_centers[j]] for j in
            range(analog.shape[1])]
    normalizer = np.repeat(np.sum(vals, axis=1), N_SOFTMAX, 0).reshape(np.shape(vals))
    vals = np.divide(vals, normalizer)
    return vals.transpose(2, 0, 1)



def get_scale(jointvals):
    return [(np.min(jointvals[:, j]), np.max(jointvals[:, j]))
                    for j in range(jointvals.shape[1])]

def get_scale_from_soft(softmaxs, joint_ranges, N_SOFTMAX=10):
    jointvals = soft2joint(softmaxs, joint_ranges, N_SOFTMAX)
    return get_scale(jointvals)

def soft2joint(softmaxs, joint_ranges=[(-1,1),(-1,1)], N_SOFTMAX=10, newscale=None):
    # Input: softmax vals
    # Output: regular (synthesized) vals
    softmaxs = np.array(softmaxs)
    if joint_ranges==[]:
        return softmaxs

    if len(softmaxs.shape)==1:
        softmaxs = softmaxs.reshape((1,-1))

    #Shape
    joint_centers = [np.linspace(start, end, N_SOFTMAX) for (start, end) in joint_ranges]

    if len(softmaxs.shape) == 2:  # one entry
        softmaxs = [softmaxs]

    jointvals = []
    for softmax in softmaxs:
        joint = []
        for j in range(len(softmax)):
            joint.append(softmax[j].dot(joint_centers[j]))
        jointvals.append(joint)

    jointvals = np.array(jointvals)
    #Scale
    if newscale is not None:
        if newscale == 'FROM_DATA':
            newscale = get_scale(jointvals)

        for j in range(len(joint_ranges)):
            newmin, newmax = newscale[j]
            _min, _max = joint_ranges[j]
            oldscale = newmax - newmin
            scaleratio = (_max - _min) / oldscale

            jointvals[:, j] = _min + scaleratio * (jointvals[:, j] - newmin)

    if len(jointvals.shape)==2 and jointvals.shape[0]==1:
        jointvals = jointvals[0]
    if jointvals.shape==(1,):
        jointvals = jointvals[0]
    return jointvals


patterns = [[0.,0.],[-0.5,0.],[0.5,0.],[0.,0.5],[-0.5,0.5],[0.5,0.5]]
lookup = {str(pattern):p for p,pattern in enumerate(patterns)}
soft_template=np.array([0 for p in patterns])

def action2idx(actions):
    return [lookup[str(list(action))] for action in actions.astype(float)]

def action2soft(actions):
    if len(np.array(actions).shape)==1: actions=np.array([actions]) #1 action support
    idxs = action2idx(actions)
    #onehot for pattern
    soft_actions=[]
    for idx in idxs:
        s = soft_template.copy()
        s[idx]=1
        soft_actions.append(s)
    soft_actions = np.array(soft_actions)
    return soft_actions if len(soft_actions)>1 else soft_actions[0]

def soft2action(soft):
    return patterns[np.argmax(soft)]

def onehot2idx(onehot):
    return np.array([np.argmax(row) for row in onehot])


def num2onehot(num, length=6):
    """Convert action to onehot vector labels"""
    onehot =  [0]*length
    onehot[num] = 1
    return onehot

def ctrl2str(x):
    if x[0] == 0:
        turn = 'L'
    elif x[0] == 0.5:
        turn = '|'
    elif x[0] > 0.5:
        turn = 'R'
    if x[1] == 0.5:
        go = '_'
    else:
        go = 'FWD'
    return turn,go



def analyze_acts():
    runname='1534451962'
    data = np.load('logs/'+runname+'/online_acts.npz')

    print([key for key in data.keys()])

    N = len(data['V1'])
    pca = PCA(n_components=2)
    print(data['P0'].shape)

    #Plot proprioceptive predictions
    plt.figure()
    plt.plot(data['P0'][:100])
    plt.savefig('figures/P0')

    #Plot sequence of predictions up to 100 steps
    for i in range(0,110,10):
        plt.figure()
        plt.imshow(data['V0'][i].reshape((20,20)))
        plt.savefig('figures/V0_%d'%i)
        #plt.show()


    #Plot activation trajectory for each unit
    for name in ['V1','V2','V3','P1','P2','P3']:
        plt.figure()
        pc = pca.fit_transform(data[name].reshape((N,-1)))
        plt.plot(pc[:100])
        plt.savefig('figures/'+name)






def data_validate(dataname='ctrl_data'):
    data = np.load('dataset/'+dataname+'.npz')
    keys = [key for key in data.keys()]

    PLOT_LEN=300
    print('-'*90)
    print('Data entries')
    N = len(data['vision'])


    print(data['vision'][0,0])
    raise('ok')


    N_SOFTMAX=int(data['prop'].shape[-1]/2)
    ranges = [(-1, 1), (-1, 1)]
    propsoft = data['prop'].reshape((-1,2,N_SOFTMAX))
    prop = soft2joint(propsoft, ranges, N_SOFTMAX=N_SOFTMAX)


    for i in range(N_SOFTMAX):
        for j in range(2):
            print(np.max(propsoft[:,j,i]) - np.min(propsoft[:,j,i]))

    plt.figure()
    plt.plot(propsoft[:3000,0,:])
    plt.title('Data (soft)')
    plt.figure()
    plt.plot(prop[:3000])
    plt.title('Data (joint)')
    plt.show()


    raise('ok')
    pca = PCA(n_components=2)
    vispc = pca.fit_transform(data['vision'].reshape((N,-1)))


    pca = PCA(n_components=2)
    proppc = pca.fit_transform(data['prop'].reshape((N,-1)))

    plt.figure()
    plt.plot(vispc)
    plt.figure()

    prop = data['prop'].reshape((N,-1,2,10))





    for i in range(30):
        plt.plot(prop[i,:,0,:])
        plt.show()


    raise('ok')


    for b, batch in enumerate(data['vision'][:100]):
        ctrl = data['control'][b][0]


        prop = data['prop'][b][0]
        soft_angle, soft_vel = prop[:10], prop[10:]
        angle = soft2joint(soft_angle, [(0, 2*np.pi)])
        vel = soft2joint(soft_vel.reshape((2,-1)), [(-1, 1),(-1, 1)])

        angles.append(angle)
        vels.append(vel)
        #for img in batch:
        #plt.imshow(batch[0])
        #plt.show()

    plt.plot(vels)
    plt.show()
    raise('ok')
    for x in data['control'].reshape((-1,2)):
        turn_go = ctrl2str(x)
        print(turn,go)

    raise('ok')
    for k,key in enumerate(keys):
        ax = plt.subplot(len(keys)+1,1,k+1)
        print(key)
        print ('Shape:', data[key].shape)
        print('Min:', np.min(data[key]))
        print('Max:', np.max(data[key]))
        print('Sample:', data[key][np.random.choice(data[key].shape[0], 5)])

        #if key=='action':
            #idxs = onehot2idx(data[key])[:PLOT_LEN]+1
            #ax.bar(range(len(idxs)), idxs)
        #else:
        #ax.plot(data[key][:PLOT_LEN, :5])
        #ax.set_title(key)
    #plt.show()


if __name__ == "__main__":
    analyze_acts()