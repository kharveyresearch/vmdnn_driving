"""
    Network.py

    Defines the main network which takes in a sequence of input on each epoch
    and passes it to each node in the graph, then returns a sequence of predictions.
"""


import json
import numpy as np
from RNN import *
import sys
import collections



class VisualCellFactory():
    def __init__(self, input_shape, input_units=1):
        self.input_shape = input_shape
        self.prev_units = input_units
        self.path = []

    def newCell(self, **kwargs):
        kwargs.update(dict(input_chans=self.prev_units, input_shape=self.input_shape))

        #Create input shapes (used to provide zeroed inputs at start of each iteration)
        #bu_input = np.zeros((1, self.input_chans, self.input_shape[0], self.input_shape[1]))
        #td_input = np.zeros((1, self.units, self.hidden_shape[0], self.hidden_shape[1]))

        next_input_shape = np.subtract(self.input_shape, kwargs['filter']) + 1
        if 'pool' in kwargs:
           next_input_shape = np.divide(next_input_shape, kwargs['pool']).astype(int)

        print('input shape:',self.input_shape)
        print('nxt input shape:',next_input_shape)
        kwargs['hidden_shape'] = (1, kwargs['units'], next_input_shape[0], next_input_shape[1])

        #Create the cell
        cell = MSTRNN(**kwargs)
        #Update values for next time newCell is called
        self.prev_units = kwargs['units']
        self.input_shape = next_input_shape


        return cell


def sort_visual_first(todo):
    """Put MSTRNN cells first in order to calculate their hidden shapes"""
    newtodo=[]
    for node in todo:
        if node[0]=='V':
            newtodo.append(node)
    newtodo += [node for node in todo if node not in newtodo]
    return newtodo

class Network(nn.Module):

    def loadGraph(self, v1_input_shape, p1_input_size, q1_input_size):
        with open("net.json", "r") as read_file:
            nodes = json.load(read_file,  object_pairs_hook=collections.OrderedDict)

        print('Nodes:',nodes.keys())
        v1_input_size = np.product(v1_input_shape)
        if 'V0' in nodes: nodes['V0']['units'] = v1_input_size
        if 'P0' in nodes: nodes['P0']['units'] = p1_input_size
        #if 'Q0' in nodes: nodes['Q0']['units'] = q1_input_size

        self.input_nodes = nodes.pop('INPUT')
        print('Input nodes:',self.input_nodes)

        visFactory = VisualCellFactory(v1_input_shape)
        graph = {}

        todo = self.input_nodes.copy()
        done = []

        #Calculate arguments for cell creation
        while len(todo)>0:
            todo = sort_visual_first(todo)
            nodename = todo.pop(0)
            done.append(nodename)
            print('Processing:',nodename)
            args = nodes[nodename]
            args['lat_input_size'] = 0
            args['cuda'] = self.usecuda
            print(args)


            #Find upnodes/ downnodes/ latnodes and add them to queue if needed
            upnodes = [];  downnodes = []; latnodes = []
            if 'u' in args:
                upnodes = args['u'] if type(args['u'])==list else [args['u']]
                for upnode in upnodes:
                    if upnode not in done and upnode not in todo:
                        todo.append(upnode)

            if 'd' in args:
                downnodes = args['d'] if type(args['d']) == list else [args['d']]

            if 'l' in args:
                latnodes = args['l'] if type(args['l']) == list else [args['l']]

            #Set input/output shapes based on upnodes/downnodes/latnodes

            args['output_shapes'] = []; args['output_lat_shapes'] = []
            for latnode in latnodes:
                print('[',nodename,']','latnode',latnode,'type:',nodes[latnode]['type'])
                if nodes[latnode]['type'] == "CTRNN":
                    args['output_lat_shapes'].append(-1) #Flatten output
                    args['lat_input_size'] += nodes[latnode]['units']
                else: #latnode is MSTRNN
                    if args['type'] == 'CTRNN': #MSTRNN->CTRNN lateral connection
                        assert (latnode in graph), "Have to initiate MSTRNN nodes first to get hidden shape!"
                        args['lat_input_size'] += np.product(graph[latnode].hidden_shape)
                        print('set lat input to product of hidden shape:', args['lat_input_size'])
                        args['output_lat_shapes'].append(-1)

            for downnode in downnodes:
                if nodes[downnode]['type'] == "CTRNN":
                    args['output_shapes'].append(-1) #Flatten output
                else:
                    args['output_shapes'].append(0) #Dont reshape output

            #Set celltype and type specific arguments
            if args['type'] == 'PREDICT':
                args = dict(in_features=nodes[args['u']]['units'], out_features=args['units'], final_fc=args['final_fc'], bias=True)
                print(args)
                cell = PREDICT

            elif args['type'] == 'MSTRNN':
                cell = visFactory.newCell

            elif args['type'] == 'CTRNN':
                #Calculate CTRNN input size (bu units + td units + self units)
                #search nodes
                bu_input_size = 0
                input_size = args['units']
                for upnode in upnodes:
                    if nodes[upnode]['type'] == "CTRNN":
                        input_size += nodes[upnode]['units']
                    else: #CTRNN->MSTRNN
                        #assert 'input_shape' in nodes[upnode], "Have to initiate MSTRNN cells first!"
                        #input_size += np.product(nodes[upnode]['input_shape']*nodes[upnode]['input_chans'])
                        assert (upnode in graph), "Have to initiate dependent node first!"
                        input_size += np.product(graph[upnode].input_shape)*graph[upnode].input_chans
                for downnode in downnodes:
                    if 'classify' not in nodes[downnode]:
                        input_size += nodes[downnode]['units']
                        bu_input_size += nodes[downnode]['units']

                args['input_size'] = input_size
                args['bu_input_size'] = bu_input_size
                cell = CTRNN

            #Now make the actual cell
            args = args.copy()
            for nonarg in ['u', 'd', 'l', 'type']:
                if nonarg in args: args.pop(nonarg)

            print('final args:',args)
            newcell = cell(**args)

            print(newcell)
            if self.usecuda: newcell=newcell.cuda()
            newcell.name = nodename
            newcell.upnodes = upnodes
            newcell.downnodes = downnodes
            newcell.latnodes = latnodes
            graph[nodename] = newcell

        print('=============DONE==============')

        self.non_input_nodes = [key for key in graph.keys() if key not in self.input_nodes]
        return(graph)


    """Predict t+1 input"""
    def __init__(self, v1_input_shape, p1_input_size, q1_input_size, startp=0.5, log_acts=False, cuda=False):
        super(Network,self).__init__()
        self.epoch=0
        self.usecuda=cuda
        self.v1_input_size = 1
        self.p1_input_size = p1_input_size
        self.q1_input_size = q1_input_size


        self.log_acts = log_acts
        self.activation = tanh_mod
        self.p = startp


        #Graph -up/down/lateral in dict
        self.graph = self.loadGraph(v1_input_shape, p1_input_size, q1_input_size)
        self._params = []
        self._named_params = {}

        #TODO: register parameters properly
        for node in self.graph:
            self._params += self.graph[node].parameters()
            self._named_params.update({(node+'_'+name):param for name,param in self.graph[node].named_parameters()})
        print(self._named_params.keys())

        self.test_states = [] #Store states for each epoch during testing

        #+1 above is for td dummy input
    def parameters(self):
        return self._params
    def named_parameters(self):
        return self._named_params
    def state_dict(self):
        state_dicts = {'net':super(Network,self).state_dict()}
        for nodename, node in self.graph.items():
            state_dicts[nodename] = node.state_dict()
        return state_dicts

    def load_state_dict(self, state_dicts):
        super(Network,self).load_state_dict(state_dicts['net'])
        for nodename, node in self.graph.items():
            node.load_state_dict(state_dicts[nodename])

    def forward(self, v_seq, p_seq, q_seq, prev_states=None):
        #print('forward..vshape:',v_seq.shape,'pshape:',p_seq.shape)
        out_v = []; out_p = []; out_l = []; out_q = []

        #reinit hidden state to prevent 'buffers have already been freed' error!
        #this prevents the time dependence between sequences
        for nodename, node in self.graph.items():
            if type(node)!= PREDICT:
                node.initHidden()
        #Set up inputs
        v_chunks=p_chunks=q_chunks=[]
        if v_seq is not None and v_seq.size(0):
            v_chunks = v_seq.chunk(v_seq.size(0),dim=0)
            v_in = v_chunks[0]
            v_in = v_in.view(1, v_in.shape[0], v_in.shape[1], v_in.shape[2])

        if p_seq is not None and p_seq.size(0):
            p_chunks = p_seq.chunk(p_seq.size(0),dim=0)
            p_in = p_chunks[0].view(-1)

        if q_seq is not None and q_seq.size(0):
            q_chunks = q_seq.chunk(q_seq.size(0),dim=0)
            q_in = q_chunks[0].view(-1)
        #Process data sequence
        n_chunks = max(len(v_chunks), len(p_chunks))

        for chunk in range(n_chunks):
            if np.random.rand()>self.p:
                if v_chunks != []:
                    v_in = v_chunks[chunk]
                    v_in = v_in.view(1, v_in.shape[0], v_in.shape[1], v_in.shape[2])
                if p_chunks != []:
                    p_in = p_chunks[chunk].view(-1)
                if q_chunks != []:
                    q_in = q_chunks[chunk].view(-1)
            #Feed input for this timestep to each node
            for nodename, node in self.graph.items():
                node.inputs = node.next_inputs
                node.next_inputs = {'bu':[], 'lat':[], 'td':[],'rc':None}


            # Set initial bottom up inputs to network
            if 'V1' in self.graph: self.graph['V1'].inputs['bu'] = [v_in]
            if 'P1' in self.graph: self.graph['P1'].inputs['bu'] = [p_in]
            if 'Q1' in self.graph: self.graph['Q1'].inputs['bu'] = [q_in]

            #Traverse graph
            for normalnode in self.non_input_nodes:
                node = self.graph[normalnode]
                bu_out, td_out, lat_out = node()

                #print('bu out:',[b.shape for b in bu_out])
                #print('td out:',[t.shape for t in td_out])
                for i,downnode in enumerate(node.downnodes):
                    #print('sending to:',i,downnode)
                    self.graph[downnode].next_inputs['td'].append(td_out[i])
                for i,upnode in enumerate(node.upnodes):
                    self.graph[upnode].next_inputs['bu'].append(bu_out[i])
                for i,latnode in enumerate(node.latnodes):
                    self.graph[latnode].next_inputs['lat'].append(lat_out)

            #Form predictions for this timestep
            pred_vision = pred_prop = pred_q = None
            for prednode in self.input_nodes:
                node = self.graph[prednode]
                node.inputs = node.next_inputs
                pred = node()
                #print(prednode,'out:',pred.shape)
                #Save and cycle predictions to next input (in closed loop mode)
                if prednode[0]=='V':
                    pred_vision = pred
                    v_in = pred_vision.view(1, v_in.shape[-3], v_in.shape[-2], v_in.shape[-1])
                elif prednode[0]=='P':
                    pred_prop = pred
                    p_in = pred_prop.view(-1)
                elif prednode[0]=='Q':
                    pred_q = pred
                    q_in = pred_q.view(-1)

            if pred_vision is not None: out_v.append(pred_vision)
            if pred_prop is not None: out_p.append(pred_prop)
            if pred_q is not None: out_q.append(pred_q)

            """Save activations to file"""
            if self.log_acts:
                self.test_states.append({nodename:node.prev_u.cpu().detach().numpy() for nodename,node in self.graph.items()})
            self.epoch+=1
        #end for chunks

        return torch.stack(out_v) if len(out_v) else None, torch.stack(out_p) if len(out_p) else None, \
               torch.stack(out_q) if len(out_q) else None, None# torch.stack(out_l)

    def save_acts(self, filename, labels=None):
        print('SAVING',len(self.test_states),'epochs of activation to:',filename)
        test_states = {key: [] for key in self.test_states[0].keys()}
        for epoch_states in self.test_states:
            for key, state in epoch_states.items():
                test_states[key].append(state)

        test_states['labels'] = labels
        data = np.arange(len(test_states.keys()))  # hack to save dict
        np.savez(filename, data=data, **test_states)