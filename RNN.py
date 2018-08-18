"""
    RNN.py

    Defines the CTRNN, MSTRNN, and PREDICT nodes for constructing a network.
"""


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np


debug=0
def DEBUG(msg, *args):
    if debug:
        print(msg, *args)


def tanh_mod(x):
    return 1.7159 * torch.tanh(0.66666667 * x)


#Tensor = torch.FloatTensor
#LongTensor = torch.LongTensor
Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

class PREDICT(nn.Linear):
    def __init__(self, **kwargs):
        final_fc = kwargs.pop('final_fc')
        super(PREDICT, self).__init__(**kwargs)
        self.final_fc = final_fc
        self.next_inputs = {'td':[]}
        self.activation = tanh_mod

    def forward(self, **kwargs):
        DEBUG('-' * 90)
        DEBUG(self.name + ' forward pass.....')
        DEBUG('td shape:',self.inputs['td'][0].shape)
        input = self.inputs['td'][0].view(-1)

        #Vision path already deconvolves so doesn't need fc at end
        if self.final_fc:
            result = super(PREDICT, self).forward(input, **kwargs)
        else:
            result = input

        self.prev_u = result
        return self.activation(result) #return prediction to loss function

class CTRNN(nn.Module):
    def __init__(self, **kwargs):
        super(CTRNN, self).__init__()
        args = dict(units=10, tau=1.0, name='CTRNN', cuda=False, lat_input_size=0, output_shapes=None)
        args.update(kwargs)

        self.USE_HEBB=0

        self.name = args['name']
        self.usecuda = args['cuda']
        self.input_size = args['input_size']
        self.units = args['units']
        self.eta = 1.0 / args['tau']
        self.hidden_shape = (1, self.units)
        self.output_shapes = args['output_shapes']

        self.td_input_size = args['input_size'] - args['bu_input_size'] - args['units']
        #Updated each iteration with inputs to cell
        self.bu_input_size = args['bu_input_size']
        self.lat_input_size = args['lat_input_size']
        print('**total input size:', args['input_size'], 'td_input_size:', self.td_input_size,
              'bu_input_size:', args['bu_input_size'], 'lat_input_size:', self.lat_input_size, 'units:', args['units'])

        # bu (input) + td (pi+1 units) + rc (pi units) -> units
        #self.W = nn.Linear(self.input_size, self.units, bias=True)
        #W = Variable(.01 * torch.randn(self.input_size, self.units).type(Tensor),
        #                                       requires_grad=True)

        #Fixed
        #bottom-up/top-down
        wIU = Variable(.01 * torch.randn(self.bu_input_size+self.td_input_size, self.units).type(Tensor),
                                               requires_grad=True)
        #recurrent
        wUU = Variable(.01 * torch.randn(self.units, self.units).type(Tensor),
                                               requires_grad=True)
        #lateral
        if self.lat_input_size:
            wLU = Variable(.01 * torch.randn(self.lat_input_size, self.units).type(Tensor),
                                               requires_grad=True)
            self.register_parameter('wLU', torch.nn.parameter.Parameter(wLU))

        #self.register_parameter('W', torch.nn.parameter.Parameter(W))
        self.register_parameter('wIU', torch.nn.parameter.Parameter(wIU))
        self.register_parameter('wUU', torch.nn.parameter.Parameter(wUU))


        #Plastic
        self.alpha = None# Variable(.01 * torch.randn(self.units, self.units).type(Tensor),
                         #                      requires_grad=True)
        self.hebb_eta = 0.1



        print(self.name,' fixed:',self.wIU.shape, self.wUU.shape)
        if self.alpha is not None: print(self.name,' plastic:',self.alpha.shape)
        self.dropout = None
        self.dropout = nn.Dropout(0.1)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.activation = tanh_mod
        self.initHidden()

        if self.usecuda:
            self.wIU = self.wIU.cuda()
            self.wUU = self.wUU.cuda()
            if self.alpha is not None: self.alpha = self.alpha.cuda()
            if self.dropout is not None: self.dropout = self.dropout.cuda()
        print('='*90)



    def initHidden(self):
        """Call at start of batch to prevent "buffers have already been freed" error"""
        self.prev_u = torch.randn(self.hidden_shape)
        self.next_inputs = {}
        self.next_inputs['bu'] = [torch.zeros([self.bu_input_size])]
        self.next_inputs['td'] = [torch.zeros([self.td_input_size])] if self.td_input_size else []
        self.next_inputs['lat'] = [torch.zeros(1,self.lat_input_size)] if self.lat_input_size else []
        self.next_inputs['rc'] = torch.zeros([self.units])

        self.hebb = None#torch.randn(self.wUU.shape)
        if self.usecuda:
            if self.hebb is not None: self.hebb = self.hebb.cuda()
            self.prev_u = self.prev_u.cuda()
            self.next_inputs = {key:[tensor.cuda() for tensor in tensors] if type(tensors)==list else tensors.cuda()
                                        for key,tensors in self.next_inputs.items()}


    def forward(self):
        DEBUG('-'*90)
        DEBUG(self.name,'forward pass.....')

        DEBUG('bu:', [_.shape for _ in self.inputs['bu']])
        DEBUG('td:', [_.shape for _ in self.inputs['td']])
        DEBUG('rc:', self.inputs['rc'].shape)
        DEBUG('lat:', [_.shape for _ in self.inputs['lat']])
        #Concatenate bottom-up, top-down, and recurrent inputs
        a=torch.cat(self.inputs['bu'],0)
        b=torch.cat(self.inputs['td'],0) if self.td_input_size else []
        c=self.inputs['rc'].view(1,-1)
        d=torch.cat(self.inputs['lat'],1) if self.lat_input_size else []

        DEBUG('a:',a)#[_.shape for _ in a])
        DEBUG('b:',b)#[_.shape for _ in b])
        DEBUG('c:',c)#[_.shape for _ in c])
        DEBUG('d:',d)#[_.shape for _ in d])


        input_combined = torch.cat((a,b),0).view(1,-1) if len(b)>0 else a
        DEBUG('combined input:',input_combined.shape)
        #DEBUG('-'*90)
        #u = self.W(input_combined)

        u1 = input_combined.view(1,-1).mm(self.wIU).view(1,-1)

        if self.USE_HEBB:
            wUU = self.wUU + torch.mul(self.alpha, self.hebb)
        else:
            wUU = self.wUU

        u2 = c.mm(wUU).view(1,-1)
        u3 = d.mm(self.wLU).view(1, -1) if len(d)>0 else 0

        u = u1+u2+u3
        u = (1-self.eta)*self.prev_u + self.eta*u

        if self.dropout is not None:
            u = self.dropout(u)
        y = self.activation(u)

        DEBUG('rc:',self.inputs['rc'].shape)
        DEBUG('y:',y.shape)
        if self.hebb is not None:
            DEBUG('hebb:',self.hebb.shape)
            self.hebb = (1 - self.hebb_eta) * self.hebb + self.hebb_eta * torch.ger(self.inputs['rc'], y.view(-1))  # outer product between previous and current output

        self.next_inputs['rc'] = y.view(-1)
        self.prev_u = u

        y_bu = y_td = y_lat = y

        if self.output_shapes is None:
            y_td = [y_td]
        else:
            y_td = [y_td if shape==0 else y_td.reshape(shape) for shape in self.output_shapes]

        return y_bu, y_td, y_lat



class MSTRNN(nn.Module):
    def __init__(self, **kwargs):#input_chans, input_shape, units, filter, stride,
                        #rc_filter=1, rc_stride=1, tau=1.0, pool=0, name='MSTRNN'):
        super(MSTRNN, self).__init__()
        args = dict(stride=1, rc_filter=1, rc_stride=1, tau=1.0, pool=0, name='MSTRNN',\
                                        output_shapes=None, output_lat_shapes=None,cuda=False)
        args.update(kwargs)
        self.name = args['name']
        self.usecuda = args['cuda']
        self.input_chans = args['input_chans']
        self.input_shape = args['input_shape']
        self.lat_input_size = args['lat_input_size']
        self.output_lat_shapes = args['output_lat_shapes']
        self.hidden_shape = args['hidden_shape']
        self.units = args['units']
        self.output_shapes = args['output_shapes']

        self.eta = 1.0 / args['tau']
        self.make2D = None
        self.dropout = None #nn.Dropout(0.01)

        #Plastic synapses
        self.eta_hebb = 0.1
        wUU = Variable(.01 * torch.randn(np.product(self.hidden_shape), np.product(self.hidden_shape)).type(Tensor),
                            requires_grad=True)

        self.alpha=None
        #alpha = Variable(.01 * torch.randn(np.product(self.hidden_shape), np.product(self.hidden_shape)).type(Tensor),
        #                      requires_grad=True)

        self.register_parameter('wUU', torch.nn.parameter.Parameter(wUU))
        #self.register_parameter('alpha',torch.nn.parameter.Parameter(alpha))
        self.initHidden()

        pool = args['pool']
        #Forward connection--------------------------
        #CTRNN->MSTRNN
        print('lat input size:',self.lat_input_size)
        if self.lat_input_size:
            self.make2D = nn.Linear(self.lat_input_size, np.product(self.input_shape)*self.input_chans)


        #MSTRNN->MSTRNN
                        #input CHANNELS, out chans(before max pooling), kernelsize, stride
        self.conv_fwd = nn.Conv2d(self.input_chans, self.units, args['filter'], args['stride'],\
                                  padding=0, bias=True)



        print(self.name,'hidden shape:', self.hidden_shape)

        if pool:
            self.pool = nn.MaxPool2d(pool, pool, return_indices=True)
            print(self.name,'pool:',self.pool)
        else:
            self.pool = None

        #self.conv_rc = nn.Conv2d(self.units, self.units, args['rc_filter'], args['rc_stride'], \
        #                                    padding=0, bias=False)

        #Backward connection--------------------------
        if pool:
            self.inv_pool = nn.MaxUnpool2d(pool, pool)
        self.conv_tr = nn.ConvTranspose2d(self.units, self.input_chans, args['filter'], args['stride'], \
                                     output_padding=0, bias=False)
        
        self.activation = tanh_mod

        if self.usecuda:
            if self.dropout is not None: self.dropout = self.dropout.cuda()
            if self.make2D is not None: self.make2D = self.make2D.cuda()
            self.conv_fwd = self.conv_fwd.cuda()
            if pool:
                self.pool = self.pool.cuda()
                self.inv_pool = self.inv_pool.cuda()
            self.wUU = self.wUU.cuda()
            if self.alpha is not None: self.alpha = self.alpha.cuda()
            self.conv_tr = self.conv_tr.cuda()


        print('=' * 90)





    def initHidden(self):
        """Call at start of batch to prevent "buffers have already been freed" error"""
        """Prevents Torch from chaining dependencies to previous sequences when doing BPTT"""
        self.prev_u = torch.randn(self.hidden_shape)
        self.hebb = Variable(torch.randn((np.product(self.hidden_shape), np.product(self.hidden_shape))))


        # Updated each iteration with inputs to cell
        self.next_inputs = {'bu': [torch.zeros(1,self.input_chans,self.input_shape[0],self.input_shape[1])],
                            'lat':[torch.zeros(1,self.lat_input_size)] if self.lat_input_size else [],
                            'td': [torch.zeros(self.hidden_shape)],
                            'rc': torch.zeros(self.hidden_shape)}
        if self.usecuda:
            self.prev_u = self.prev_u.cuda()
            self.next_inputs = {key: [tensor.cuda() for tensor in tensors] if type(tensors)==list else tensors.cuda() for key, tensors in self.next_inputs.items()}
            self.hebb = self.hebb.cuda()

    def forward(self):
        DEBUG('-' * 90)
        DEBUG(self.name+' forward pass.....')

        #Fwd conv the unfiltered bottom-up input
        conv_input = torch.zeros(self.hidden_shape)
        if self.usecuda: conv_input = conv_input.cuda()

        lat_input = [torch.cat(self.inputs['lat'],1)] if self.lat_input_size else []

        after_conv_shapes = []
        for input in self.inputs['bu'] + lat_input:
            if len(input.shape)==2: #input from CTRNN, preprocess
                DEBUG('1D lateral input:', input.shape)
                input = self.make2D(input).view(1, self.input_chans, self.input_shape[0], self.input_shape[1])
            #Convolve 2D input
            DEBUG('2D bu input:', input.shape)

            i_bu = self.conv_fwd(input)
            DEBUG('after fwd conv:', i_bu.shape)
            DEBUG('conv_input:',conv_input.shape)
            after_conv_shapes.append(i_bu.shape)
            if self.pool is not None:
                i_bu, self.indices = self.pool(i_bu)
                DEBUG('y_bu after pool:', i_bu.shape)

            conv_input += i_bu

        if len(self.inputs['td']) == 0:
            td_input = torch.zeros(self.hidden_shape)
            if self.usecuda: td_input = td_input.cuda()
        elif len(self.inputs['td'])==1:
            td_input = self.inputs['td'][0]
        else:
            td_input = torch.sum(Tensor(self.inputs['td']), 0)


        if debug:
            if type(conv_input)!=int:
                print('buconv:',conv_input.shape)
            else:
                print('buconv:',0)

        DEBUG('rc:',self.inputs['rc'].shape)
        DEBUG('td:',td_input.shape)
        new_u = conv_input + td_input + self.inputs['rc'] + td_input
        DEBUG('new_u:', new_u.shape)

        u = (1 - self.eta) * self.prev_u + self.eta * new_u
        y = self.activation(u)

        if self.dropout is not None:
            y = self.dropout(y)

        #eta_hebb = 0.01# self.eta_hebb*self.eta
        #self.hebb = (1 - eta_hebb) * self.hebb + eta_hebb * torch.ger(self.inputs['rc'].view(-1), y.view(-1))  # outer product between previous and current output
        self.prev_u = u
        y_bu = [y] #multiple bu output not supported yet


        #Rc conv to create output to send to self next timestep
        #y_rc = self.conv_rc(y)


        #if self.USE_HEBB:
        wUU = self.wUU #+ torch.mul(self.alpha, self.hebb)
        #else:
        #wUU = self.wUU
        y_rc = y.view(1,-1).mm(wUU).reshape(self.hidden_shape)


        DEBUG('[rc] rc output shape:', y_rc.shape)
        self.next_inputs['rc'] = y_rc


        #Tr conv of output to send topdown
        #Invert maxpooling
        if self.pool is not None:
            y_inv = self.inv_pool(y, indices=self.indices, output_size=after_conv_shapes[0])
            DEBUG('[td] inverse pool result:', y_inv.shape)
        else:
            y_inv = y

        #Invert conv
        y_td = self.conv_tr(y_inv, output_size=self.input_shape)
        DEBUG('[td] inverse conv result:', y_td.shape)

        if self.output_shapes is None:
            y_td = [y_td]
        else:
            y_td = [y_td if shape==0 else y_td.reshape(shape) for shape in self.output_shapes]


        y_lat = y.reshape((1,-1))

        return y_bu, y_td, y_lat

