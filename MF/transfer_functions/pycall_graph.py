import pycallgraph
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput


from load_config import *
#import stuff_i_dont_want_to_see

pycallgraph.start_trace()
#Initializations

#
#stuff_i_dont_want_to_see()
def filtercalls(call_stack, modul, clas, func, full):
    mod_ignore = ['numba','scipy','scipy.special','scipy.integrate','scipy.optimize','re','os','sys','json','numpy','argparse','matplotlib','__future__']
    func_ignore = ['CustomFunctionName','pdbcall']
    clas_ignore = ['pdb']
    return modul not in mod_ignore and func not in func_ignore and clas not in clas_ignore
pycallgraph.start_trace(filter_func=filtercalls)
pycallgraph.stop_trace()
#mycode.things()
pycallgraph.make_dot_graph('cleaner_graph.png')



