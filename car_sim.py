"""
    car_sim.py

    Simple 2D simulator of a car which generates visual and proprioceptive data to learn
    and can also be used to test trained network.

"""

import numpy as np
from numpy import pi, sin, cos, tan
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from util import joint2soft, soft2joint, get_scale_from_soft, rotvec


IMG_SIZE=350 #increase to 600-800 for EX2 and EX3
N_SOFTMAX=10
SIGMA=0.5
LOOKAHEAD=1
SHOWTRACK=0
SHOW_RUN=1
WINDOW=10
CONTINUOUS_IMG=0

SHRINK_SHOWMAP=1

TRACK_WIDTH=3
START_POS=(IMG_SIZE*.7, IMG_SIZE*.7)
MOVE=1
ACCEL=0.1
STRAIGHT_LENGTH=40
MAX_SPEED=1.5
MIN_SPEED=1

class CarSim:
    def __init__(self, pos=np.array(START_POS), head=np.array((0,1)),
                 corner_thetas=[pi/70], straight_lengths=[STRAIGHT_LENGTH], ranges=[(-1, 1), (-1, 1)], tracks=['EX4']):
        self.VERBOSE=0
        self.track_width=TRACK_WIDTH

        print('CarSim at pos:',pos)
        self.pos = pos.copy(); self.init_pos=pos.copy()
        self.head = head.copy(); self.init_head=head.copy()
        self.vel=1 #velocity
        self.acc=0 #acceleration
        self.joint_ranges = ranges
        #Data storage
        self.vision=[]
        self.prop=[]

        self.track_idx=0 #current track idx
        self.tracks=[]

        self.signs={} #put indicators on track
        for self.straight_length in straight_lengths:
            for self.corner_theta in corner_thetas:

                for track_idx in tracks:
                    self.tracks.append(self.create_track(track_idx))
                    self.reset()

                    if SHOWTRACK:
                        print(self.get_img(1))
                    #plt.imshow(self.tracks[-1][1])
                    #plt.show()
        self.track, self.img_arr = self.tracks[0] #set current track to first in list


    def reset(self, pos=None, RANDOM=False):
        if pos is not None:
            self.pos = pos
        else:
            self.pos = self.init_pos.copy()
        self.head = self.init_head.copy()
        self.vel = 1
        self.acc = 0

        if RANDOM==True:
            start = np.random.randint(len(self.track))
            self.pos, self.head = self.track[start]
            self.pos += np.random.normal(0,20,2)
            self.head += np.random.normal(0,2,2)
            self.head /= np.linalg.norm(self.head)
            return start
        return 0

    def set_random_track(self, verbose=False):
        self.track_idx = np.random.randint(len(self.tracks))
        if verbose:
            print('Set to track:',self.track_idx)
        self.track, self.img_arr = self.tracks[self.track_idx]

    def track_add(self):
        self.track.append((self.pos.copy(), self.head.copy(), self.vel))

    def go_until(self, target):
        target = np.array(target).astype(float)
        while np.linalg.norm((target - self.pos)) > 0.3:
            self.move()
            self.track_add()
            if self.VERBOSE:
                print(self.pos)
        #self.pos = target.astype(int).astype(float)

    def turn_until(self, target):
        target = np.array(target).astype(float)
        while np.linalg.norm((target - self.head)) > 0.3:
            #print('dist:',np.linalg.norm((target - self.head)))
            self.head = rotvec(self.head, self.corner_theta)
            #self.head = (1000*self.head).astype(int)/1000.
            self.move()
            self.track_add()
        self.head = target.astype(float)
        #self.pos = self.pos.astype(int).astype(float)

    def move(self):
        self.vel += self.acc
        if self.vel>MAX_SPEED:
            self.vel=MAX_SPEED
            self.acc=0
        elif self.vel<MIN_SPEED:
            self.vel=MIN_SPEED
            self.acc=0
        self.pos += self.head*self.vel

    def turn_for(self, num, move=0, theta_mul=1):
        for i in range(num):
            self.head = rotvec(self.head, self.corner_theta*theta_mul)

            for j in range(move):
                self.move()
                self.track_add()
            if not move:
                self.track_add()

    def move_for(self, num):
        for i in range(num):
            self.move()
            self.track_add()

    def get_img(self, showmap=False, display_target=None, title=''):
        win=WINDOW

        _pos = (self.pos+self.head*LOOKAHEAD).astype(int)

        if CONTINUOUS_IMG:
            vis = Image.fromarray(self.img_arr[_pos[0]-win:_pos[0]+win, _pos[1]-win:_pos[1]+win]*255).resize((int(win),int(win)), Image.BICUBIC)
            vis = np.array(vis)/255.
        else:
            vis = self.img_arr[_pos[0]-win:_pos[0]+win, _pos[1]-win:_pos[1]+win]



        if showmap:
            win=1
            img = self.img_arr.copy()
            for i, _pos in enumerate((self.pos.astype(int), (self.pos+self.head*LOOKAHEAD).astype(int))):
                img[_pos[0]-win:_pos[0]+win, _pos[1]-win:_pos[1]+win] = 2+i
            if display_target is not None:
                _pos = display_target.astype(int)
                #img[_pos[0]-1:_pos[0]+1,_pos[1]-1:_pos[1]+1]=4

            #Only show visible window (NEW)
            win=WINDOW
            if SHRINK_SHOWMAP:
                img = img[_pos[0]-win:_pos[0]+win, _pos[1]-win:_pos[1]+win]

            plt.imshow(img)
            plt.title(title)
            plt.show()

        return vis

    def set_color(self, color):
        self.signs[len(self.track)] = color

    def create_track(self, track_idx=1, display=False):
        global SAVE_ACC
        self.track=[]
        N_UNITS = int(2*pi/self.corner_theta)

        im = Image.fromarray(np.zeros((IMG_SIZE,IMG_SIZE)))
        draw = ImageDraw.Draw(im)

        if track_idx=='EX1': #OBLONG
            self.vel=1
            self.acc=0
            for i in range(int(pi/self.corner_theta)):
                self.turn_for(1, move=1)
            self.acc=ACCEL
            self.move_for(self.straight_length)
            self.acc=-ACCEL
            for i in range(int(pi/self.corner_theta)):
                self.turn_for(1, move=1)
            self.acc=ACCEL
            self.move_for(self.straight_length)
            self.acc=-ACCEL

        elif track_idx=='EX2':
            self.vel=1
            self.acc=0
            for i in range(int(pi/self.corner_theta)):
                self.turn_for(1, move=2)
            self.acc=ACCEL
            self.move_for(self.straight_length)
            self.acc=-ACCEL
            for i in range(int(pi/self.corner_theta)):
                self.turn_for(1, move=2)
            self.acc=ACCEL
            self.move_for(self.straight_length)
            self.acc=-ACCEL


        elif track_idx=='EX3':
            self.vel=1
            self.acc=0
            for i in range(int(pi/self.corner_theta)):
                self.turn_for(1, move=10)
            self.acc=ACCEL
            self.move_for(self.straight_length)
            self.acc=-ACCEL
            for i in range(int(pi/self.corner_theta)):
                self.turn_for(1, move=10)
            self.acc=ACCEL
            self.move_for(self.straight_length)
            self.acc=-ACCEL

        elif track_idx=='EX4':
            self.vel=1
            self.acc=0
            for i in range(int(pi/self.corner_theta)):
                self.turn_for(1, move=2, theta_mul=-1)
            self.acc=ACCEL
            self.move_for(self.straight_length)
            self.acc=-ACCEL
            for i in range(int(pi/self.corner_theta)):
                self.turn_for(1, move=2, theta_mul=-1)
            self.acc=ACCEL
            self.move_for(self.straight_length)
            self.acc=-ACCEL


        self.track = np.array(self.track)
        print('Generated track:',self.track.shape)

        #for self.pos,self.head in self.track:
        #    draw.line((self.pos[0]+self.head[1]*WIDTH1, self.pos[1]-self.head[0]*WIDTH1*5,
        #               self.pos[0]-self.head[1]*WIDTH1*5, self.pos[1]+self.head[0]*WIDTH1), width=4, fill=4)
        fill=1
        self.vel=1; self.acc=0
        for i, (self.pos, self.head, self.vel) in enumerate(self.track):
            if i in self.signs:
                fill=self.signs[i]

            draw.line((self.pos[0], self.pos[1],
                       self.pos[0]+self.head[0]*self.vel*3, self.pos[1]+self.head[1]*self.vel*3), width=self.track_width, fill=fill)
            #if i in self.checkpoints:
                #draw.ellipse((self.pos[0], self.pos[1], self.pos[0]+1, self.pos[1]+1), 4)


        self.img_arr = np.array(im).T
        if display:
            plt.imshow(self.img_arr)
            plt.show()

        return self.track, self.img_arr


    def showtrail(self, p_trail, v_trail):
        img_arr_ = self.img_arr.copy().T
        for i,p in enumerate(p_trail):
            img_arr_[p[1] - 1:p[1] + 1, p[0] - 1:p[0] + 1] = 2 + v_trail[i]
        img_arr_[p[1] - 1:p[1] + 1, p[0] - 1:p[0] + 1] = 3

        plt.imshow(img_arr_)
        plt.axis('off')
        plt.show()

    def run(self, log=False, visualize=False, start=0):
        for self.track, self.img_arr in self.tracks:
            i=0
            p_trail=[]
            print('track length:',len(self.track))
            self.vel=1; self.acc=0

            for loop in range(1): #check looping
                for origpos, head, self.vel in self.track[start+1:]:

                    self.head = head.copy()
                    p = self.pos.astype(int)
                    p_trail.append(p)
                    vis = self.get_img()


                    if visualize and len(p_trail)%1==0:
                        img_arr_ = self.img_arr.copy().T
                        for p in p_trail:
                            img_arr_[p[1] - 1:p[1] + 1, p[0] - 1:p[0] + 1] = 2
                        img_arr_[p[1] - 1:p[1] + 1, p[0] - 1:p[0] + 1] = 3

                        plt.imshow(img_arr_)
                        plt.show()

                        print(vis)
                    i += 1

                    print('SPEED:',self.vel)

                    self.pos += self.head*self.vel
                    if log:
                        self.vision.append(vis)
                        self.prop.append([self.head[0], self.head[1], self.acc])

        prop = np.array(self.prop)[:,-1].reshape((-1,1))
        prop = joint2soft(prop, [(-ACCEL,ACCEL)], N_SOFTMAX=6)
        plt.plot(prop.reshape((-1,6)))
        plt.show()

    def get_vis(self, p, win=20):
        return np.array(Image.fromarray(self.img_arr[p[1]-win:p[1]+win, p[0]-win:p[0]+win]).resize((int(win/2),int(win/2))))

    def get_softhead(self):
        #print(self.head)
        #print('->',joint2soft(self.head, self.joint_ranges))
        return joint2soft(self.head, self.joint_ranges, N_SOFTMAX=N_SOFTMAX, sigma=SIGMA)

    def save_data(self, SEQ_LEN = 10):
        if len(self.prop)==0:
            print('Nothing to save!')
            return

        N_BATCHES = int(len(self.vision) / SEQ_LEN)
        prop = joint2soft(self.prop, self.joint_ranges, N_SOFTMAX=N_SOFTMAX, sigma=SIGMA)
        prop = np.array(prop[:SEQ_LEN * N_BATCHES]).reshape((N_BATCHES, SEQ_LEN, -1))
        vision = np.array(self.vision[:SEQ_LEN * N_BATCHES]).reshape((N_BATCHES, SEQ_LEN, self.vision[0].shape[0], self.vision[0].shape[1]))
        print('Saving data..')
        print(prop.shape)
        np.savez('dataset/ctrl_data', vision=vision/255., prop=prop)
        return

        print('scale:', get_scale_from_soft(np.array(prop).reshape((-1, 2, N_SOFTMAX)), self.joint_ranges, N_SOFTMAX=N_SOFTMAX))
        plt.figure()
        plt.title('Real values converted from soft')
        reconst_prop = soft2joint(prop.reshape((-1, 2, N_SOFTMAX)), self.joint_ranges, N_SOFTMAX=N_SOFTMAX, newscale='FROM_DATA')
        #plt.plot(prop.reshape((-1,N_SOFTMAX*2)))
        print(vision.shape)
        plt.figure()
        plt.plot(reconst_prop)

        print(np.array(self.prop)[0,0].sum())
        print(reconst_prop[0,0].sum())





if __name__ == "__main__":

    p_trail=[]
    #Generate self.track
    print('Creating track...')
    pos = np.array(START_POS)
    head = np.array((0.,1.))
    sim = CarSim(pos, head)
    sim.create_track()
    sim.reset()


    #Auto drive to generate data!
    print('Creating data...')
    start=-1


    sim.run(visualize=SHOW_RUN, start=start, log=True)

