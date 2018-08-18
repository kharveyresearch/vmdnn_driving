"""
    main.py

    Run this script to perform training/testing.
    Loads data, creates a new/ restores a saved network, and performs train/test loop with it.

    Macros:
    MODE: 0=Test, 1=Train, 2=Resume training from latest checkpoint
    VIEW: View data step by step as it is generated
    RANDOM_AMOUNT = Amount of randomness put in deviation of car in data
    RANDOM_PROB = Probability of introducing deviation at start of car trajectory (random position/velocity)
    SEQ_LEN = Length of data sequence to learn
    SHOW_MOTOR_DATA = Display plot of generated training data
    SHOW_MOTOR_DATA_3D = Display plot of generated training data in 3D.
    SHOW_MAP_IN_TEST = Show screenshot of each step during test
    SAVE_ACTS = Save activations of network to logs/runname/online_acts.npz (recommended to use with Test mode only)
"""

import numpy as np
from numpy import pi, sin, cos
import math, random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import sys
import time
import shutil
import subprocess
from mpl_toolkits.mplot3d import Axes3D
from RNN import *
from Network import *
from util import *
from car_sim import CarSim

MODE=1
RANDOM_AMOUNT = 5
RANDOM_PROB = 0.9
if MODE==0:
    RANDOM_PROB=0

VIEW=0
VIEW2=0

SHOW_MOTOR_DATA=0
SHOW_MOTOR_DATA_3D=0
SHOW_MAP_IN_TEST=0

SEQ_LEN=6
SAVE_ACTS=0


assert(torch.cuda.is_available())

def load_flags(data='boxmovesmall', resume=False):
    import argparse

    if resume:
        resume_from, runname = resume
    else:
        resume_from = 0
        runname = str(int(time.time()))

    parser = argparse.ArgumentParser(description='Train/Test the model.')
    parser.add_argument('--resume', type=int, default=resume_from, nargs='?',
                        help='epoch from where to resume training (or nearest saved if no checkpoint exists for that epoch))')
    parser.add_argument('--runname', type=str, default=runname, nargs='?',
                        help='identifying name for this run')
    parser.add_argument('--delete_old_checkpoints', type=bool, default=True, nargs='?',
                        help='whether to delete old checkpoints when saving new ones')
    parser.add_argument('--test', default=False, action='store_true',
                        help='whether to test (default is to train)')
    parser.add_argument('--train_p', type=bool, default=True, nargs='?',
                        help='whether to train the proprioceptive path (default is to train)')
    parser.add_argument('--train_v', type=bool, default=True, nargs='?',
                        help='whether to train the visual path (default is to train)')
    parser.add_argument('--data', type=str, default='sim', nargs='?',
                        help='name of training set to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, nargs='?')
    parser.add_argument('--startp', type=float, default=0.5, nargs='?')
    parser.add_argument('--p_inc_epoch', type=int, default=300, nargs='?')

    parser.add_argument('--cuda', type=bool, default=True, nargs='?',
                        help='whether to use cuda')
    parser.add_argument('--print_epoch', type=int, default=50, nargs='?',
                        help='interval with which to log training progress (print interval is double this)')
    parser.add_argument('--save_epoch', type=int, default=500, nargs='?',
                        help='interval with which to save model')
    parser.add_argument('--save_acts_epoch', type=int, default=SAVE_ACTS, nargs='?',
                        help='interval with which to save activations')
    parser.add_argument('--max_epochs', type=int, default=100000, nargs='?',
                        help='max epochs to train model for')
    parser.add_argument('--min_loss', type=int, default=1e-8, nargs='?',
                        help='stop training if loss drops below this')
    parser.add_argument('--start_epoch', type=int, default=0, nargs='?',
                        help='initial index of epoch counter (DEBUG USE)')
    parser.add_argument('--log_acts', type=bool, default=False, nargs='?',
                        help='whether to log activation of network (DEBUG USE)')
    flags = parser.parse_args()

    #Additional flags generated at runtime
    flags.logdir = 'logs/'+flags.runname
    flags.logfile = flags.logdir+'/loss.log'
    flags.plogfile = flags.logdir+'/ploss.log'
    flags.vlogfile = flags.logdir+'/vloss.log'
    flags.qlogfile = flags.logdir + '/qloss.log'
    flags.cpdir = 'checkpoints/'+flags.runname

    if not resume:
        os.mkdir(flags.cpdir)
        os.mkdir(flags.logdir)
        shutil.copy2('net.json', flags.logdir+'/net.json')  # save settings for analysis

    flags.data = data

    return flags

def save_checkpoint(state, flags, filename='checkpoint.tar'):
    print('[SAVING]',filename)
    if flags.delete_old_checkpoints:
        #Remove old checkpoints except one
        files = sorted(
            glob.iglob(flags.cpdir+'/*'), key=os.path.getctime)[:-1]
        for file in files:
            os.remove(file)
    torch.save(state, flags.cpdir+'/'+filename)

def load_checkpoint(model, optimizer, flags):
    #Get nearest checkpoint file

    filename=flags.cpdir+'/'+str(flags.resume)+'_checkpoint.tar'
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        flags.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}, minloss {})"
              .format(flags.resume, checkpoint['epoch'], checkpoint['minloss']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return model, optimizer, flags


def load_data(dataname):
    global SEQ_LEN
    print('Loading data from dataset:',dataname)

    if dataname == 'sim':
        N_SAMPLES = SEQ_LEN*30000 if MODE>0 else SEQ_LEN*300
        speed = 1
        motor = []
        vision = []
        #,90,120,150,180)
        sim = CarSim(corner_thetas=[pi/PI_DIVISOR for PI_DIVISOR in [70, 140]], straight_lengths=[40], tracks=['EX1'])


        max_track_length = max([len(track[0]) for track in sim.tracks])

        print('max track length:',max_track_length)

        cur_idx=max_track_length+1


        dist_to_target=2
        fail_recover=0
        for n in range(N_SAMPLES):
            #Change track
            if cur_idx>max_track_length and n%SEQ_LEN==0:
                sim.reset()
                sim.set_random_track()
                cur_idx=1



            #Interpolate, gradually adjust heading and accelerate towards target as distance is minimized
            # between POSITIONS.
            target_pos, target_head, target_vel = sim.track[cur_idx%len(sim.track)]


            #ONLY UPDATE VISION ONCE EVERY NEW SEQUENCE)
            if n%SEQ_LEN==0:
                if np.random.random()<RANDOM_PROB: #learn how to recover from disturbance
                    if VIEW or VIEW2: print ('random!')

                    #NEW: randomize doesnt put car ahead of next target
                    #sim.pos += np.random.normal(0,RANDOM_AMOUNT,2)
                    sim.pos += rotvec(sim.head, np.random.normal(pi,pi/4))*np.random.normal(RANDOM_AMOUNT,0.5)
                    sim.vel = max(0.5, np.random.normal(1,0.2))
                if VIEW:
                    pass#print("SPEED:",sim.vel, "TARGET:",target_vel)

                fail_recover=0



            if VIEW2:
                sim.get_img(1, display_target=target_pos, title='VIEW MODE')
                #print("SPEED:",sim.vel, "TARGET:",target_vel, 'HEAD:',sim.head, 'TARGETHEAD:',target_head)


            #Interpolate towards target heading/velocity
            head_to_target = (target_pos+target_head*target_vel) - sim.pos
            dist_to_target = np.linalg.norm(head_to_target)
            if dist_to_target>5:
                fail_recover-=1 #takes more time
            head_to_target/=dist_to_target
            dist_to_target = max(dist_to_target,1)


            a,b = zeroone(dist_to_target-0.9), 1-zeroone(dist_to_target-0.9)
            sim.head = a*head_to_target + b*target_head
            sim.vel = 0.3*sim.vel + 0.7*target_vel

            sim.pos += sim.head*sim.vel
            assert sim.vel >=0.5

            #Get vision now that head has been obtained
            cur_img = sim.get_img(VIEW, title='VIEW MODE')
            #normalize
            head = 0.5 + (sim.head/2)
            motor.append([head[0], head[1], sim.vel/2]) #sim.vel is predicted but cannot control car!
            vision.append(cur_img)

            #print(dist_to_target)

            if dist_to_target<1.4:
                cur_idx+=1 #proceed to next target
                fail_recover=0
            else:
                fail_recover+=1
                assert fail_recover<6

        motor = np.array(motor)
        vision = np.array(vision)
        if SHOW_MOTOR_DATA:
            plt.figure()
            ax1 = plt.subplot(211)
            ax1.plot(motor.reshape((-1,motor.shape[-1]))[:,:-1])
            ax2 = plt.subplot(212, sharex=ax1)
            ax2.plot(motor.reshape((-1,motor.shape[-1]))[:,-1])
            plt.plot(motor.reshape((-1,motor.shape[-1]))[:,0], motor.reshape((-1,motor.shape[-1]))[:,1])
            plt.show()
        if SHOW_MOTOR_DATA_3D:
            plt.figure()
            flatmotor = motor.reshape((-1,motor.shape[-1]))
            ax = plt.subplot(111, projection='3d')
            ax.scatter(flatmotor[:,0],flatmotor[:,1],flatmotor[:,2], c=flatmotor[:,2])
            plt.show()

        motor = motor.reshape((-1, SEQ_LEN, motor.shape[-1]))
        vision = vision.reshape((-1,SEQ_LEN,vision.shape[-2],vision.shape[-1]))
        vision /= vision.max()
        #motor = joint2soft(motor.reshape((-1,2))).reshape((-1,SEQ_LEN,20))
        return sim, motor, vision, np.zeros((len(motor),1))


    else:
        data = np.load('dataset/'+dataname+'.npz')


    print([key for key in data.keys()])
    return data['prop'] if 'prop' in data else None, \
           data['vision'] if 'vision' in data else None, data['labels'] if 'labels' in data else None



def load_model(v1_input_shape, p1_input_size, q1_input_size, flags):
    model = Network(v1_input_shape, p1_input_size, q1_input_size, flags.startp, flags.save_acts_epoch>0, cuda=flags.cuda)
    if flags.cuda: model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=flags.learning_rate)
    minloss = 0
    # ====================================================
    # Resume model
    if flags.resume:
        model, optimizer, flags = load_checkpoint(model, optimizer, flags)

    return model, optimizer, flags





def test_sim(sim, model, vision, prop, labels, v1_input_shape, p1_input_size, flags):
    proporig = prop.copy()
    p_trail=[]
    head_trail=[]
    v_trail=[]
    realptrail=[]
    realheadtrail=[]
    sim.set_random_track(verbose=True)
    sim.reset()
    model.p=1.0
    realpos = sim.pos.copy()

    propX = Variable(Tensor(proporig[0]))
    for i in range(10):#len(propX)*20): #Increase up to *50,%60 for EX2,EX3

        try:
            #Get visual input from the simulator
            img = sim.get_img(showmap=SHOW_MAP_IN_TEST, title='TEST MODE').reshape((1,vision.shape[-2],vision.shape[-1]))
            if SHOW_MAP_IN_TEST:
                print('SPEED:',sim.vel)
            visX = Variable(Tensor(img))
        except Exception as err:
            print('Error:',err)

        CLOSED_LOOP=1
        if not CLOSED_LOOP:
            propX = Variable(Tensor(proporig[i]))

        v_out, p_out, _, _ = model(visX, propX, None)
        if v_out is not None: v_out = v_out.view(-1,v1_input_shape[0],v1_input_shape[1])


        result = p_out.cpu().detach().numpy()

        for t in range(SEQ_LEN-1):
            sim.head = (result[t,:2]-0.5)*2
            sim.vel = result[t,2]+1
            sim.pos += sim.head * sim.vel

            #realpos += (proporig[i][t+1,:2]-0.5)*2 * (proporig[i][t+1,2]*2)
            head_trail.append(result[t])
            #realheadtrail.append(proporig[i][t+1])
            p_trail.append(sim.pos.copy())
            #realptrail.append(realpos.copy())
            v_trail.append(sim.vel.copy())
            


        visX = v_out
        propX = np.zeros((result.shape))
        propX[0] = result[-1]
        propX = Tensor(propX)


    if flags.save_acts_epoch:
        model.save_acts(flags.logdir+'/online_acts')

    head_trail = np.array(head_trail)
    p_trail = np.array(p_trail)
    v_trail = np.array(v_trail)
    realheadtrail = np.array(realheadtrail)
    realptrail = np.array(realptrail)

    sim.showtrail(p_trail.astype(int), v_trail)

    PLOT_P=0
    PLOT_HEAD=0
    if PLOT_P:
        plt.plot(p_trail[:,0], p_trail[:,1])
        plt.scatter(p_trail[:3,0], p_trail[:3,1])
        plt.plot(realptrail[:,0], realptrail[:,1], 'r')

    if PLOT_HEAD:
        plt.figure()
        plt.plot(head_trail[:,:2], c='b')
        plt.plot(realheadtrail[:,:2], c='r')

        plt.figure()
        plt.plot(head_trail[:,-1], c='k')
        plt.plot(realheadtrail[:,-1], c='r')


    plt.show()



def train(model, optimizer, vision, prop, q, labels, v1_input_shape, p1_input_size, q1_input_size, flags, minloss=1):
    print('TRAINING MODE')
    lossavg = 0.0 # filter out noise in loss
    plossavg = 0.0
    vlossavg = 0.0
    qlossavg = 0.0
    losses = []
    mseloss = nn.MSELoss()
    xentloss = nn.CrossEntropyLoss()

    if vision is not None:
        vision = Variable(Tensor(vision))
    prop = Variable(Tensor(prop))


    for epoch in range(flags.start_epoch, flags.max_epochs):
        BATCH = epoch%len(prop)#np.random.randint(len(vision))


        if vision is not None:
            visX = vision[BATCH][:-1]
            visy = vision[BATCH][1:]
        else:
            visX,visY = None, None
        propX = prop[BATCH][:-1]
        propy = prop[BATCH][1:]
        if q is not None:
            qX = Variable(Tensor(q[BATCH][:-1]))
            qy = Variable(Tensor(q[BATCH][1:]))
        else:
            qX,qY = None, None

        if control is not None:
            #Use control as labels for classification
            controly = Variable(LongTensor(control[BATCH, :-1])).view(-1,6)
            controly = torch.max(controly, 1)[1] #loss function takes indices instead of onehot!


        if (epoch+1) % flags.p_inc_epoch == 0:
            model.p = min(model.p + 0.2, 1)  # encourage training longer term predictions
            print('Updated p:',model.p)
            pass

        optimizer.zero_grad()
        v_out, p_out, q_out, _ = model(visX, propX, qX)

        if v_out is not None: v_out = v_out.view(-1, v1_input_shape[0], v1_input_shape[1])
        if q_out is not None: q_out = q_out.view(-1,6)



        v_loss=p_loss=q_loss=0
        if v_out is not None and flags.train_v: v_loss = mseloss(v_out, visy)
        if p_out is not None and flags.train_p: p_loss = mseloss(p_out, propy)
        if q_out is not None: q_loss = xentloss(q_out, controly)


        loss = v_loss + p_loss + q_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % flags.print_epoch == 0:
            # print('epoch:',epoch+1,'vloss:', v_loss.data.numpy(), 'ploss:', p_loss.data.numpy())

            # print('classifier loss:',l_loss.cpu().data.numpy())

            loss = loss.cpu().data.numpy()
            if type(p_loss)!=int: p_loss = p_loss.cpu().data.numpy()
            if type(q_loss)!=int: q_loss = q_loss.cpu().data.numpy()
            if type(v_loss)!=int: v_loss = v_loss.cpu().data.numpy()
            plossavg = 0.7*plossavg + 0.3*p_loss
            qlossavg = 0.7 * qlossavg + 0.3 * q_loss
            vlossavg = 0.7*vlossavg + 0.3*v_loss
            lossavg = 0.7*lossavg + 0.3*loss
            with open(flags.logfile, 'a') as outfile:
                outfile.write('%0.8f\n' % lossavg)
            with open(flags.plogfile, 'a') as outfile:
                outfile.write('%0.8f\n' % plossavg)
            with open(flags.vlogfile, 'a') as outfile:
                outfile.write('%0.8f\n' % vlossavg)
            with open(flags.qlogfile, 'a') as outfile:
                outfile.write('%0.8f\n' % qlossavg)

            if (epoch + 1) % flags.print_epoch * 2 == 0:
                print(epoch, 'vloss:',vlossavg, 'ploss:',plossavg)


            # losses.append(_loss)
            # if len(losses)>30:
            #    losses=[min(losses)]

            if loss < flags.min_loss: break
        # if (epoch+1)%1000 == 0:
        # t=0
        # Image.fromarray(v_out.data.numpy()[t]*255).resize((60,100)).convert('L').show()

        if (epoch + 1) % flags.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'minloss': minloss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, flags, str(epoch + 1) + '_checkpoint.tar')
        if flags.save_acts_epoch and (epoch + 1) % flags.save_acts_epoch == 0:
            model.save_acts(flags.logdir+'/online_acts')

def get_latest_checkpoint():
    files = sorted(
        glob.iglob('logs/*'), key=os.path.getctime, reverse=True)
    runname = files[0][5:]

    files = sorted(
        glob.iglob('checkpoints/'+runname+'/*'), key=os.path.getctime, reverse=True)
    resume_from = int(files[0][len('checkpoints/'+runname+'/'):-15])


    return resume_from, runname


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    RESUME = MODE==2
    # ====================================================
    #Load
    # ====================================================
    flags = load_flags('sim', resume=0 if MODE==1 else get_latest_checkpoint())

    sim, prop, vision, labels = load_data(flags.data)
    control=None

    print(''); print(''); print('LOADED DATA. BUILDING MODEL...'); print('='*90)
    if vision is not None: print('vision shape:',vision.shape)
    if prop is not None: print('prop shape:',prop.shape)
    if control is not None: print('control shape:',control.shape)
    if labels is not None: print('labels shape:',labels.shape)


    v1_input_shape = vision[0].shape[1:] if vision is not None else None
    p1_input_size = prop[0].shape[-1] if prop is not None else None
    q1_input_size = None
    model, optimizer, flags = load_model(v1_input_shape, p1_input_size, q1_input_size, flags)
    if flags.start_epoch >= flags.max_epochs: flags.start_epoch=0
    print('');    print('');   print('BUILT MODEL. RUNNING',('RESUME' if RESUME else ('TRAIN' if MODE>0 else 'TEST')),'...');   print('=' * 90)


    #====================================================
    #Run
    # ====================================================
    if MODE>0: #TRAIN
        train(model, optimizer, vision, prop, control, labels,\
                v1_input_shape, p1_input_size, q1_input_size, flags, minloss=0)
    else: #TEST
        test_sim(sim, model, vision, prop, labels, v1_input_shape, p1_input_size, flags)
