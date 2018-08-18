"""Convenience script to generate net.json"""
import json
from numpy import floor
from pprint import pprint


class Objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def generate(s):
    """Generates net.json for training.
        s: settings object"""
    dic={
    "INPUT": ["V0", "P0", "Q0"],


    "V0": {
        "u": "V1",
        "type": "PREDICT",
        "final_fc": False,
    },

    "P0": {
        "u": "P1",
        "type": "PREDICT",
        "final_fc": True,
        "bias": True

    },

    "Q0": {
        "u":"P2",
        "type": "PREDICT",
        "final_fc": True,
        "units":6,
        "classify": True

    },

    "SETTINGS": s.__dict__
    }



    #Visual path
    tau=s.tau_start
    VISUAL_UNITS=s.VISUAL_UNITS
    for i in range(1,s.N_V_LAYERS+1):
        dic["V%d"%i] =dict(
            d="V%d"%(i-1),
            #l="P%d"%i,
            u="V%d"%(i+1),
            type="MSTRNN",
            units=VISUAL_UNITS,
            filter=s.FILTER,
            tau=tau
        )
        tau += s.tau_delta
        VISUAL_UNITS += s.VISUAL_UNITS_DELTA


    #Proprioceptive path
    tau=s.tau_start
    for i in range(1, s.N_P_LAYERS + 1):
        dic["P%d" % i] = dict(
            d="P%d" % (i - 1),
            #l="Q%d" % i,
            u="P%d" % (i + 1),
            type="CTRNN",
            units=s.PROP_UNITS,
            tau=tau
        )
        tau+=s.tau_delta

    #Control path
    tau=s.tau_start
    for i in range(1, s.N_Q_LAYERS + 1):
        dic["Q%d" % i] = dict(
            d="Q%d" % (i - 1),
            #l="P%d" % i,
            u="Q%d" % (i + 1),
            type="CTRNN",
            units=s.Q_UNITS,
            tau=tau
        )
        tau+=s.tau_delta

    if "P%d"%i in dic: dic["P%d"%i].pop("u")
    if "Q%d"%i in dic: dic["Q%d"%i].pop("u")
    if "V%d"%i in dic: dic["V%d"%i].pop("u")

    #Override default tau
    if s.TAU_LIST != []:
        for i,tau in enumerate(s.TAU_LIST):
            dic['V%d'%i]['tau'] = tau

    #Override default filter
    if s.FILTER_LIST and s.FILTER_FOR_LIST!=s.FILTER:
        for l in (s.FILTER_LIST):
            dic['V%d'%l]['filter'] = s.FILTER_FOR_LIST


    #Delete input and links if a channel isnt present
    if s.N_V_LAYERS==0:
        dic['INPUT'] = [input for input in dic['INPUT'] if input != 'V0']
        dic.pop('V0')

    if s.N_P_LAYERS==0:
        dic['INPUT'] = [input for input in dic['INPUT'] if input != 'P0']
        dic.pop('P0')


    if s.N_Q_LAYERS==0:
        dic['INPUT'] = [input for input in dic['INPUT'] if input != 'Q0']
        dic.pop('Q0')


    LATERAL = 1
    if LATERAL:
        for i in range(0, s.N_V_LAYERS+1):
            dic['V%d'%i]['l'] = ['P%d'%i]
            dic['P%d'%i]['l'] = ['V%d'%i]



    print(dic)
    with open('net.json', 'w') as outfile:
        json.dump(dic, outfile)

if __name__ == "__main__":
    tau_delta = 1
    tau_start = 1
    tau = tau_start

    VISUAL_UNITS = 20#20
    VISUAL_UNITS_DELTA=-2
    FILTER = 7
    PROP_UNITS = 100
    C_UNITS = 20
    Q_UNITS=10

    N = 3
    N_V_LAYERS = N
    N_P_LAYERS = N
    N_Q_LAYERS = 0

    #Not used if same filter is used on each layer
    FILTER_LIST = [1,2,3,4]
    FILTER_FOR_LIST = FILTER


    settings = Objectview(dict(tau_start=tau_start, tau_delta=tau_delta, TAU_LIST=[],
                    VISUAL_UNITS=VISUAL_UNITS, VISUAL_UNITS_DELTA=VISUAL_UNITS_DELTA,
                    PROP_UNITS=PROP_UNITS, Q_UNITS=Q_UNITS, C_UNITS=C_UNITS,
                    FILTER=FILTER, N_V_LAYERS=N_V_LAYERS, N_P_LAYERS=N_P_LAYERS,
                               N_Q_LAYERS=N_Q_LAYERS,
                    FILTER_LIST=(FILTER_LIST if FILTER_FOR_LIST != FILTER else []),
                    FILTER_FOR_LIST=(FILTER_FOR_LIST if FILTER_FOR_LIST != FILTER else 0)))
    generate(settings)