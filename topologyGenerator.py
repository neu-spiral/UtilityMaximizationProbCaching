#! /usr/bin/env python
'''
	A Cache Network
'''
from abc import ABCMeta,abstractmethod
#from Caches import PriorityCache, EWMACache, LMinimalCache 
from networkx import Graph, DiGraph,shortest_path
import networkx
import random 
from cvxopt import spmatrix,matrix 
from cvxopt.solvers import lp
from simpy import *
from scipy.stats import rv_discrete
import numpy as np
from numpy.linalg import matrix_rank
import logging,argparse
import itertools
#from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import topologies
from SparseVector import SparseVector 
from helpers import clearFile
import math

class CONFIG(object):
    QUERY_MESSAGE_LENGTH = 0.0
    RESPONSE_MESSAGE_LENGTH = 0.0
    EXPLORE_MESSAGE_LENGTH = 0.0
    EXPLORE_RESPONSE_MESSAGE_LENGTH = 0.0


def pp(l):
   return ' '.join(map(str,l))


class Demand:
   """ A demand object. Contains the item requested, the path a request follows, as a list, and the
       rate with which requests are generated. Tallies count various metrics.

       Attributes:
	   item: the id of the item requested
	   path: a list of nodes to be visited
	   rate: the rate with which this request is generated
	   query_source: first node on the path
	   item_source: last node on the path
   """
   def __init__(self,item,path,rate):
   	""" Initialize a new request.
   	"""
	self.item = item
	self.path = path
	self.rate = rate

	self.query_source = path[0]
	self.item_source = path[-1]
   
   def __str__(self):
	return Demand.__repr__(self)

   def __repr__(self):
	return 'Demand('+','.join(map(str,[self.item,self.path,self.rate]))+')'

   def succ(self,node):
   	""" The successor of a node in the path. 
   	"""
	path = self.path
	if node not in path:
	   return None
	i = path.index(node)
	if i+1==len(path):
	   return None
	else: 
	   return path[i+1]

   def pred(self,node):
	"""The predecessor of a node in the path.
	"""
	path = self.path
	if node not in path:
	   return None
	i = path.index(node)
	if i-1<0:
	   return None
	else: 
	   return path[i-1]






class Message(object):
   """A Message object. 

      Attributes:
          header: the header of the message (e.g., query_message, response_message
          payload: the payload, can be set by the programmer
          length: length, to be used in transmission delay calculations
          stats: statistics collected as the message traverses nodes
	  u_bit: indicates if the message is going upstream or downstream
   """

   def __init__(self,header, payload,length,stats,u_bit):
	self.header = header
	self.payload = payload
	self.length = length
	self.u_bit = u_bit
	if stats == None:
		self.stats = {}
		self.stats['delay'] = 0.0
		self.stats['hops'] = 0.0
		self.stats['weight'] =0.0
		self.stats['downweight'] =0.0
	else:
		self.stats = stats 

   def __str__(self):
	return Message.__repr__(self)

   def __repr__(self):
	return pp(['Message(',self.header,',',self.payload,',',self.length,',',self.stats,')'])

class QueryMessage(Message):
   """
	A query message.
   """
   def __init__(self,d,query_id,stats = None):
	Message.__init__(self,header=("QUERY",d,query_id), payload = None, length = CONFIG.QUERY_MESSAGE_LENGTH, stats = stats, u_bit = True)

class ResponseMessage(Message):
   """
	A response message.
   """
   def __init__(self,d,query_id,stats = None):
	Message.__init__(self,header = ("RESPONSE",d,query_id),payload = None, length = CONFIG.RESPONSE_MESSAGE_LENGTH, stats = stats, u_bit = False)




class CacheNetwork(DiGraph):
   """A cache network. 

     A cache network comprises a weighted graph and a list of demands. Each node in the graph is associated with a cache of finite capacity.
     NetworkCaches must support a message receive operation, that determines how they handle incoming messages.

     The cache networks handles messaging using simpy stores and processes. In partiqular, each cache, edge and demand is associated with a 
     Store object, that receives, stores, and processes messages from simpy processes.

     In more detail:
     - Each demand is associated with two processes, one that generates new queries, and one that monitors and logs completed queries (existing only for logging purposes)
     - Each cache/node is associated with a process that receives messages, and processes them, and produces new messages to be routed, e.g., towards neigboring edges
     - Each edge is associated with a process that receives messages to be routed over the edge, and delivers them to the appropriate target node.
       During "delivery", messages are (a) delayed, according to configuration parameters, and (b) statistics about them are logged (e.g., number of hops, etc.)
     
     Finally, a global monitoring process computes the social welfare at poisson time intervals.
   
   """

   def __init__(self,G,cacheGenerator,demands,item_sources,capacities,weights,delays,warmup = 0,monitoring_rate=1.0,demand_change_rate=0,demand_min=1.0,demand_max=1.0):
	self.env = Environment()   
	self.warmup =warmup
	self.demandstats = {}
	self.sw ={}
	self.funstats ={}
	self.optstats = {}
	self.monitoring_rate=monitoring_rate
	self.demand_change_rate = demand_change_rate
        self.demand_min = demand_min
        self.demand_max = demand_max

	DiGraph.__init__(self,G)
	for x in self.nodes():
	   self.node[x]['cache'] =  cacheGenerator(capacities[x],x) 
	   self.node[x]['pipe'] = Store(self.env)
	for e in self.edges():
	   x = e[0]
	   y = e[1]
	   self.edge[x][y]['weight'] = weights[e]
	   self.edge[x][y]['delay'] = delays[e]	
	   self.edge[x][y]['pipe'] = Store(self.env)

	self.demands = {} 
	self.item_set = set()

	for d in demands:
	    self.demands[d]={}
	    self.demands[d]['pipe'] = Store(self.env)
	    self.demands[d]['queries_spawned'] = 0L
	    self.demands[d]['queries_satisfied'] = 0L
	    self.demands[d]['queries_logged'] = 0.0
	    self.demands[d]['pending'] = set([])
	    self.demands[d]['stats'] = {}
	    self.item_set.add(d.item)

	for item in item_sources:
	   for source in item_sources[item]:
	   	self.node[source]['cache'].makePermanent(item)  ###THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE

	for d in self.demands:
	   self.env.process(self.spawn_queries_process(d))
	   self.env.process(self.demand_monitor_process(d))
	   if self.demand_change_rate>0.0:
               self.env.process(self.demand_change_process(d))

        if self.demand_change_rate>0.0:
               self.env.process(self.compute_opt_process())

	for e in self.edges():
	   self.env.process(self.message_pusher_process(e))

	for x in self.nodes():
	   self.env.process(self.cache_process(x))

	self.env.process(self.monitor_process())

   def run(self,finish_time):

	logging.info('Simulating..')
	self.env.run(until=finish_time)
	logging.info('..done simulating')
	   

   
   def spawn_queries_process(self,d):
	""" A process that spawns queries.
	    
	    Queries are generated according to a Poisson process with the appropriate rate. Queries generated are pushed to the query source node.
        """
	while True:
          logging.debug(pp([self.env.now,':New query for',d.item,'to follow',d.path]))
	  _id = self.demands[d]['queries_spawned']
          qm = QueryMessage(d,_id)  # create a new query message at the query_source
	  self.demands[d]['pending'].add(_id)
	  self.demands[d]['queries_spawned'] += 1
	  yield self.node[d.query_source]['pipe'].put((qm,(d,d.query_source))) 	
          yield self.env.timeout(random.expovariate(d.rate))


   def demand_monitor_process(self,d):
        """ A process monitoring statistics about completed requests.
        """
	while True:
	  msg = yield self.demands[d]['pipe'].get()


	  lab,dem,query_id = msg.header 
	  stats = msg.stats
	  now = self.env.now

          if lab is not "RESPONSE":
	     logging.warning(pp([now,':',d,'received a non-response message:',msg]))
	     continue
 	
	  if dem is not d:
	     logging.warning(pp([now,':',d,'received a message',msg,'aimed for demand',dem]))
	     continue
	
	  if query_id not in self.demands[d]['pending']:
	    logging.warning(pp(['Query',query_id,'of',d,'satisfied but not pending']))
            continue
	  else:
	    self.demands[d]['pending'].remove(query_id)
	
	  logging.debug(pp(['Query',query_id,'of',d,'satisfied with stats',stats]))
	  self.demands[d]['queries_satisfied'] += 1L


	  if now>=self.warmup:
		self.demands[d]['queries_logged'] += 1.0

		for key in stats:
   		    if key in self.demands[d]['stats']:
			self.demands[d]['stats'][key] += stats[key]
		    else:
			self.demands[d]['stats'][key] = stats[key]

   def demand_change_process(self,d):
        """ A process changing the demand periodically 
        """
        while True:
	  yield self.env.timeout(1./self.demand_change_rate)
          new_rate = random.uniform(self.demand_min,self.demand_max)
          logging.info(pp([self.env.now,':Demand for ',d.item,'following',d.path,'changing rate from',d.rate,'to',new_rate]))
	  d.rate = new_rate

   def compute_opt_process(self):
	  ''' Process recomputing optimal values after demand changes. Used only if demand changes periodically
          '''
	  yield self.env.timeout(0.01*1./self.demand_change_rate) #Tiny offset, to make sure all demands have been updated by measurement time
	  
          while True:
              Y,res= self.minimizeRelaxation()
              logging.info(pp([self.env.now,': Optimal Relaxation is: ',self.relaxation(Y)]))
              logging.info(pp([self.env.now,': Expected caching gain at relaxation point is: ',self.expected_caching_gain(Y)]))
              optimal_stats= {}
              optimal_stats['res']= res
              optimal_stats['Y'] = Y
              optimal_stats['L'] = self.relaxation(Y)
              optimal_stats['F']  = self.expected_caching_gain(Y)
	      self.optstats[self.env.now] = optimal_stats
	      yield self.env.timeout(1./self.demand_change_rate)



        
		
   def message_pusher_process(self,e):
        """ A process handling message transmissions over edges.

	    It delays messages, accoding to delay specs, and increments stat counters in them.
        """
	
	while True:
	   msg = yield self.edge[e[0]][e[1]]['pipe'].get()
	   logging.debug(pp([self.env.now,':','Pipe at',e,'pushing',msg]))
           time_before = self.env.now
	   yield self.env.timeout(msg.length*random.expovariate(1/self.edge[e[0]][e[1]]['delay'])) 
	   delay = self.env.now	- time_before   	
	   msg.stats['delay'] +=delay
	   msg.stats['hops'] += 1.0
	   msg.stats['weight'] +=  self.edge[e[0]][e[1]]['weight']
	   if not msg.u_bit:
	   	msg.stats['downweight'] +=  self.edge[e[0]][e[1]]['weight'] + self.edge[e[1]][e[0]]['weight']
	   logging.debug(pp([self.env.now,':','Pipe at',e,'delivering',msg]))
	   self.node[e[1]]['pipe'].put((msg,e))

   def cache_process(self,x):
	"""A process handling messages sent to caches.

	   It is effectively a wrapper for a receive call, made to a NetworkedCache object. 

	"""
	while True:
	   (msg,e) = yield self.node[x]['pipe'].get()
           generated_messages = self.node[e[1]]['cache'].receive(msg,e,self.env.now) #THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE!!!!
           for (new_msg,new_e) in generated_messages:
	      if new_e[1] in self.demands:
		yield self.demands[new_e[1]]['pipe'].put(new_msg)
	      else: 
	      	yield self.edge[new_e[0]][new_e[1]]['pipe'].put(new_msg)

   def cachesToMatrix(self):
	"""Constructs a matrix containing cache information.
	"""
	zipped=[]
	n = len(self.nodes())
	m = max(len(self.item_set),max(self.item_set)+1)
	for x in self.nodes():
	    for item in self.node[x]['cache']:
		zipped.append( (1, x, item )  )
	
	val,I,J = zip(*zipped)
 
	return spmatrix(val,I,J,size=(n,m))


   def statesToMatrix(self):
	"""Constructs a matrix containing marginal information. This assumes that caches contain a state() function, capturing maginals. Only LMin implements this
	"""
	zipped=[]
	n = len(self.nodes())
	m = max( [ d.item for d in self.demands ])+1
	Y = matrix()
	for x in self.nodes():
	    for item in self.node[x]['cache'].non_zero_state_items():
		zipped.append( (self.node[x]['cache'].state(item), x, item )  )
	
	val,I,J = zip(*zipped)
 
	return spmatrix(val,I,J,size=(n,m))


   def social_welfare(self):
	""" Function computing the social welfare.
	"""
	dsw  = 0.0
	hsw  = 0.0
	wsw  = 0.0
	sumrate = 0.0
	for d in self.demands:
	    item = d.item
	    rate = d.rate
	    sumrate += rate
	    x = d.query_source
	    while not (item in self.node[x]['cache'] or x is d.item_source): #THIS NEEDS TO BE IMPLEMENTED BY THE NETWORKED CACHE!!!
		s = d.succ(x)
		dsw += rate*(CONFIG.QUERY_MESSAGE_LENGTH*self.edge[x][s]['delay'] + CONFIG.RESPONSE_MESSAGE_LENGTH*self.edge[s][x]['delay'])
		wsw += rate*(self.edge[x][s]['weight'] + self.edge[s][x]['weight'])
		hsw += rate*(2)
		x = s

	return (dsw/sumrate,hsw/sumrate,wsw/sumrate)	

   def caching_gain(self):
	""" Function computing the caching gain under the present caching situation
	"""
	X = self.cachesToMatrix()
	return self.expected_caching_gain(X)

   def cost_without_caching(self):
	""" Function computing the  cost of recovering all items demanded from respective sources."""
	cost = 0.0
	sumrate = 0.0
	for d in self.demands:
	    item = d.item
	    rate = d.rate
	    sumrate += rate

	    x = d.query_source
	    s = d.succ(x)
	    while s is not None:
		cost += rate*(self.edge[s][x]['weight']+self.edge[x][s]['weight'])
		x = s
	        s = d.succ(x)

	return cost/sumrate

   def expected_caching_gain(self,Y):
	""" Function computing the expected caching gain under marginals Y, presuming product form. Also computes deterministic caching gain if Y is integral.
	"""
	ecg = 0.0
	sumrate = 0.0
	for d in self.demands:
	    item = d.item
	    rate = d.rate
	    sumrate += rate

	    x = d.query_source
	    s = d.succ(x)
	    prodsofar = 1-Y[x,item]
	    while s is not None:
		ecg += rate*(self.edge[s][x]['weight']+self.edge[x][s]['weight'])*(1-prodsofar)
		
		x = s
	        s = d.succ(x)
	    	prodsofar *= 1-Y[x,item]

	return ecg/sumrate


   def relaxation(self,Y):
	""" Function computing the relaxation of caching gain under marginals Y. Relaxation equals deterministic caching gain if Y is integral.
	"""
	rel = 0.0
	sumrate = 0.0
	for d in self.demands:
	    item = d.item
	    rate = d.rate
	    sumrate += rate

	    x = d.query_source
	    s = d.succ(x)
	    sumsofar = Y[x,item]
	    while s is not None:
		rel += rate*(self.edge[s][x]['weight']+self.edge[x][s]['weight'])*min(1.0,sumsofar)
		
		x = s
	        s = d.succ(x)
	    	sumsofar += Y[x,item]

	return rel/sumrate

   def demand_stats(self):
	""" Computed stats across demands.
	"""
	stats ={}
	queries_logged = 0.0
	rate =0.0
	for d in self.demands:	    
	    queries_logged += self.demands[d]['queries_logged']
  	    for key in self.demands[d]['stats']:
		if key in stats:
		    stats[key] += self.demands[d]['stats'][key]
		else:
		    stats[key] = self.demands[d]['stats'][key]
	for key in stats:
	    stats[key] = stats[key]/queries_logged	

	return stats	

   def monitor_process(self):
    	while True:
	    now = self.env.now
	    if now >=self.warmup:
	        self.sw[now] = self.social_welfare()
	        self.demandstats[now] = self.demand_stats()
		X = self.cachesToMatrix()
		ecg = self.expected_caching_gain(X)
		rel = self.relaxation(X)
		tot = self.cost_without_caching()
		esw = tot-ecg
		self.funstats[now] = (ecg,rel,esw,tot)
		logging.info(pp([now,':','DSW = %f, HSW = %f, WSW = %f' % self.sw[now],', DEMSTATS =',self.demandstats[now], 'FUNSTATS =',self.funstats[now]])) 
		try:
		#if True:
		    Y = self.statesToMatrix()
		    secg = self.expected_caching_gain(Y)
		    srel = self.relaxation(Y)
		    self.funstats[now] += (secg,srel)
		    logging.info(pp([now,': SECG=',secg,'SREL=',srel]))	
		except AttributeError:
		    logging.debug(pp([now,": No states in this class"]))    
		#M=self.cachesToMatrix()
		#formatted = [  (i,j,'*')  if self.node[i]['cache'].isPermanent(j) else (i,j)  for i,j,v in  zip(M.I,M.J,M.V)   ]
		#logging.info(str(sorted(formatted)))
	    yield self.env.timeout(random.expovariate(self.monitoring_rate))

  

   def minimizeRelaxation(self):
	n = len(self.nodes())
	m = max( [ d.item for d in self.demands ])+1
	number_of_placement_variables = n*m

	def position(node,item):
	    return node*m+item	

	A =[]
	b = []
	row = 0

        #Permanent set constraints
	logging.debug('Creating permanent set constaints...')
	row = 0
	for x in self.nodes():
	    perm_set =  self.node[x]['cache'].perm_set()
	    if len(perm_set)>0:
		for item in perm_set:
		    A += [ (1.0,row,position(x,item)) ]
                    b += [1.0]	
		    row += 1  
	logging.debug('...done. Created %d constraints' % row)
	
	total_equality_constraints = row
	
        G = []
	h = []

        #Capacity constraints
	logging.debug('Creating capacity constaints...')
	G +=    [  (1.0,x,position(x,item))  for item in xrange(m) for x in self.nodes()   ] 
	h += [ self.node[x]['cache'].capacity()+ len(self.node[x]['cache'].perm_set())    for x in self.nodes()]
	#h += [ self.node[x]['cache'].capacity()    for x in self.nodes()]
	logging.debug('...done at %d rows' % len(h))

	row = n
	t = number_of_placement_variables

	#t's smaller than sums of y_vi's in path
	logging.debug('Creating t up constraints...')
	for d in self.demands:
	    item = d.item
	    path = d.path
	    sofar = []
	    for v in path:
		if len(sofar)>0:
	   	    for u in sofar:
		    	G += [ (-1.0,row,position(u,item)) ]
                    G += [(1.0,row,t)]
		    h += [0.0]
   		    row += 1
		    t += 1
                sofar += [v] 

 		
	logging.debug('...done at %d rows' % row)

	total_ts = t - number_of_placement_variables
	

	#t's smaller than 1.0
	logging.debug('Creating t ll one constraints...')
	t = number_of_placement_variables
	while t< (number_of_placement_variables+total_ts):
	    G += [ (1.0,row,t)]
	    h += [1.0]
	    row +=1
	    t += 1

	logging.debug('...done at %d rows' %row)
	
	#y's less than 1
	logging.debug('Creating y ll one gg 0 constraints...')
	y = 0
	while y< number_of_placement_variables:
	    G += [ (1.0,row,y)] 
	    h += [1.0]
	    row +=1
	    G += [ (-1.0,row,y)] 
	    h += [0.0]
	    row +=1
	    y += 1
	logging.debug('...done at %d rows' %row)
	    
	total_inequality_constraints = row
	
       	#objective
	logging.debug('Creating objective vector...')
	c = number_of_placement_variables*[0] 
	for d in self.demands:
	    rate = d.rate
	    path = d.path
	    
	    x = d.query_source
	    s = d.succ(x)
	    while s is not None:
		c += [-rate*(self.edge[s][x]['weight']+self.edge[x][s]['weight'])]
	    	x = s
		s = d.succ(x)

	logging.debug('...done at %d terms' % len(c) )

	val, I,J = zip(*A)
	A = spmatrix(val,I,J,size=(total_equality_constraints,number_of_placement_variables+total_ts))
	b = matrix(b)  

	val, I, J = zip(*G)
	G = spmatrix(val,I,J,size=(total_inequality_constraints, number_of_placement_variables + total_ts))	
	h = matrix(h)
	
	c = matrix(c)

	logging.debug('c has length %d ' % len(c))
	logging.debug('G has dims %d x %d and matrix_rank ' % G.size + str(matrix_rank(G)))
	logging.debug('h has length %d ' % len(h))
	logging.debug('A has dims %d x %d and matrix_rank' % A.size +str(matrix_rank(A)))
	logging.debug('b has length %d ' % len(b))
	

	res = lp(c, G, h,A,b) #, primalstart={'x':matrix(np.zeros(c.size)*1.e-99),'s':matrix( total_inequality_constraints*[1.e-20])})
	
	opt = res['x'][:number_of_placement_variables]
	return np.reshape(opt,(n,m),order ='C'),res
def export_edge(cnx_edge):
    edge_info = []
    for u in cnx_edge:
        for v in cnx_edge[u]:
            edge_info.append((u,v))
    return edge_info
def write_edges(edge_info, file):
    f = open(file, 'w')
    f.write(str(edge_info))
    f.close()	

class NetworkedCache(object):
    """An abstract networked cache.	
    """

    __metaclass__ = ABCMeta


    @abstractmethod
    def __init__(self,capacity,_id):
        pass

    @abstractmethod
    def capacity(self):
	pass

    @abstractmethod
    def perm_set(self):
	pass


    @abstractmethod
    def makePermanent(self,item):
        pass


    @abstractmethod
    def receive(self,message,edge,time):
	pass


    @abstractmethod
    def __contains__(self,item):
	pass
 
    
    @abstractmethod
    def __iter__(self):
	pass

    @abstractmethod
    def isPermanent(self,item):
	pass
   
class PriorityNetworkCache(NetworkedCache):
    """ A Priority Networked Cache. Supports LRU,LFU, and RR policies.

	Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the LRU principle.
    """
    def __init__(self,capacity,_id,principle):
	self.cache = PriorityCache(capacity,_id)
	self.permanent_set=set([])
	self._id = _id
	self._capacity =capacity
	self.stats ={}
	self.stats['queries'] = 0.0
	self.stats['hits'] = 0.0
	self.stats['responses'] = 0.0
	self.principle=principle


	
    def __str__(self):
	return str(self.cache)+'+'+str(self.permanent_set)	

    def __contains__(self,item):
	return item in self.cache or item in self.permanent_set
    
    def __iter__(self):
	return itertools.chain(self.cache,self.permanent_set)

    def isPermanent(self,item):
	return item in self.permanent_set

    def capacity(self):
        return self._capacity

    def perm_set(self):
	return self.permanent_set

    def makePermanent(self,item):
	self.permanent_set.add(item)
    	
    def receive(self,msg,e,now):
	label,d,query_id = msg.header

	if label == "QUERY":
	   item = d.item
	   logging.debug(pp([now,': Query message for item',item,'received by cache',self._id]))
	   self.stats['queries'] +=1.0

	   inside_cache =  item in self.cache
	   inside_permanent_set =  item in self.permanent_set
	   if inside_cache or inside_permanent_set:
              logging.debug(pp([now,': Item',item,'is inside', 'permanent set' if inside_permanent_set else 'cache','of',self._id]))
	      if inside_cache: #i.e., not in permanent set
	        princ_map = { 'LRU': now, 'LFU': self.cache.priority(item)+1, 'RR': random.random(), 'FIFO':self.cache.priority(item)  }
		logging.debug(pp([now,': Priority of',item,'updated to',princ_map[self.principle],'at cache',self._id]))
	        self.cache.add(item,princ_map[self.principle])
	      self.stats['hits'] +=1
	      if self._id == d.query_source:
                logging.debug(pp([now,': Response to query',query_id,'of',d,'delivered to query source by cache',self._id]))
		pred = d
	      else:
		pred = d.pred(self._id)
		logging.debug(pp([now,': Response to query',query_id,'of',d,' generated by cache',self._id]))
	      e = (self._id,pred)
	      rmsg = ResponseMessage(d,query_id,stats=msg.stats)
	      return [(rmsg,e)]
	   else:
              logging.debug(pp([now,': Item',item,'is not inside',self._id,'continue searching']))
	      succ = d.succ(self._id)
	      if succ == None:
		logging.error(pp([now,':Query',query_id,'of',d,'reached',self._id,'and has nowhere to go, will be dropped']))
		return []
	      e = (self._id,succ)
	      return [(msg,e)]

	
	if label == "RESPONSE":
	   logging.debug(pp([now,': Response message for',d,'received by cache',self._id]))
	   self.stats['responses'] +=1.0
	   item = d.item
	   princ_map = { 'LRU': now, 'LFU': self.cache.priority(item)+1 if item in self.cache else 1, 'RR': random.random(), 'FIFO':self.cache.priority(item) if item in self.cache else now }
	   logging.debug(pp([now,': Priority of',item,'updated to',princ_map[self.principle],'at',self._id]))
           self.cache.add(item,princ_map[self.principle]) #add the item to the cache/update priority
	  
	   if d.query_source==self._id:
              logging.debug(pp([now,': Response to query',query_id,'of',d,' finally delivered by cache',self._id]))
	      pred = d
	   else:
              logging.debug(pp([now,': Response to query',query_id,'of',d,'passes through cache',self._id,'moving further down path']))
	      pred = d.pred(self._id)
	   e = (self._id,pred)
	   return [(msg,e)]



class ExploreMessage(Message):
   """
	An exploration message.
   """
   def __init__(self,d,query_id, initiator, stats = None):
	Message.__init__(self,header=("EXPLORE",d,query_id),payload=None, length=CONFIG.EXPLORE_MESSAGE_LENGTH, stats = stats, u_bit = True)
	self. explore_source = initiator

class ExploreResponseMessage(Message):
   """
	An exploration response message.
   """

   def __init__(self,d,query_id, initiator, stats = None):
	Message.__init__(self,header=("EXPLORE_RESPONSE",d,query_id),payload= None, length=CONFIG.EXPLORE_MESSAGE_LENGTH, stats = stats, u_bit = False)
	self. explore_source = initiator


class EWMAGradCache(NetworkedCache):
    """ An EWMA Gradient Networked Cache.

	Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the EWMA principle.
    """

       

    def __init__(self,capacity,_id,beta=1.0):
	self.cache = EWMACache(capacity,_id,beta)
	self._id = _id
	self.permanent_set=set([])
	self._capacity=capacity
	self.stats ={}
	self.stats['queries'] = 0.0
	self.stats['hits'] = 0.0
	self.stats['responses'] = 0.0
	self.stats['explores'] = 0.0
	self.stats['explore_responses'] = 0.0


    def __str__(self):
	return 'Cache: '+str(self.cache)+'Permanent: '+str(self.permanent_set)	

    def __contains__(self,item):
	return item in self.cache or item in self.permanent_set

    def __iter__(self):
	return itertools.chain(self.cache,self.permanent_set)


    def capacity(self):
        return self._capacity

    
    def perm_set(self):
	return self.permanent_set
    
    def isPermanent(self,item):
	return item in self.permanent_set

    def makePermanent(self,item):
	self.permanent_set.add(item)
    	
    def receive(self,msg,e,now):
	label,d,query_id = msg.header

	if label == "QUERY":
	   item = d.item
	   logging.debug(pp([now,': Query message for item',item,'received by cache',self._id]))
	   self.stats['queries'] +=1.0

	   inside = item in self.cache or item in self.permanent_set
	   if inside:
              logging.debug(pp([now,': Item',item,'is inside ',self._id]))
	      self.stats['hits'] +=1
	      if self._id == d.query_source:
                logging.debug(pp([now,': Response to query',query_id,'of',d,' finally delivered by cache',self._id]))
		pred = d
	      else:
		pred = d.pred(self._id)
		logging.debug(pp([now,': Response to query',query_id,'of',d,' generated by cache',self._id]))
	      e = (self._id,pred)
	      rmsg = ResponseMessage(d,query_id,stats=msg.stats)

	      msglist = [(rmsg,e)]

	      if item not in self.permanent_set:
		  succ = d.succ(self._id)
	      	  if succ != None:
	          	logging.debug(pp([now,': Item',item,'is inside cache of ',self._id,' will forward exploration message']))
	          	emsg = ExploreMessage(d,query_id,self._id)
		  	e = (self._id,succ)
		  	msglist += [ (emsg,e)]
		  
	      return msglist
	   else:
              logging.debug(pp([now,': Item',item,'is not inside',self._id,'continue searching']))
	      succ = d.succ(self._id)
	      if succ == None:
		logging.error(pp([now,':Query',query_id,'of',d,'reached',self._id,'and has nowhere to go, will be dropped']))
		return []
	      e = (self._id,succ)
	      return [(msg,e)]

	
	if label == "RESPONSE":
	   logging.debug(pp([now,': Response message for',d,'received by cache',self._id]))
	   self.stats['responses'] +=1.0
	   item = d.item
           logging.debug(pp([now,': Node',self._id,'updating derivative of item',item,'with measurement',msg.stats['downweight']]))
           self.cache.add(item,msg.stats['downweight'],now) #does not really add item, but updates its gradient estimate
           logging.debug(pp([now,': Node',self._id,' now stores',self.cache]))
  
	   if d.query_source==self._id:
              logging.debug(pp([now,': Response to query',query_id,'of',d,' finally delivered by cache',self._id]))
	      pred = d
	   else:
              logging.debug(pp([now,': Response to query',query_id,'of',d,'passes through cache',self._id,'moving further down path']))
	      pred = d.pred(self._id)
	   e = (self._id,pred)
	   return [(msg,e)]

	if label == "EXPLORE":
	   item = d.item
	   logging.debug(pp([now,': Explore message for item',item,'received by cache',self._id]))
	   self.stats['explores'] +=1.0

	   inside = item in self.cache or item in self.permanent_set
	   if inside:
              logging.debug(pp([now,': Item',item,'is inside ',self._id]))
	      pred = d.pred(self._id)
	      logging.debug(pp([now,': Explore Response to query',query_id,'of',d,' generated by cache',self._id]))
	      e = (self._id,pred)
	      ermsg = ExploreResponseMessage(d,query_id,msg.explore_source,stats=msg.stats)

	      return [(ermsg,e)]
		  
	   else:
              logging.debug(pp([now,': Item',item,'is not inside',self._id,'continue exploring']))
	      succ = d.succ(self._id)
	      if succ == None:
		logging.debug(pp([now,':Exploration for',query_id,'of',d,'reached',self._id,'and has nowhere to go, will be dropped']))
		return []
	      e = (self._id,succ)
	      return [(msg,e)]

 
	if label =="EXPLORE_RESPONSE":
	   logging.debug(pp([now,': Explore Response message for',d,'received by cache',self._id]))
	   self.stats['explore_responses'] +=1.0
	   item = d.item
  
	   if msg.explore_source==self._id:
           	logging.debug(pp([now,': Node',self._id,'received final exploration response for',query_id,'of',d]))
           	logging.debug(pp([now,': Node',self._id,'updating derivative of item',item,'with measurement',msg.stats['downweight']]))
           	self.cache.add(item,msg.stats['downweight'],now) #does not really add item, but updates its gradient estimate
           	logging.debug(pp([now,': Node',self._id,' now stores',self.cache]))
       		return [] 
	 
           logging.debug(pp([now,': Explore Response to query',query_id,'of',d,'passes through cache',self._id,'moving further down path']))
	   pred = d.pred(self._id)
	   e = (self._id,pred)
	   return [(msg,e)]

class LMinCache(NetworkedCache):
    """ A Networked Cache minimizing the relaxation L.

	Note: the capacity of the cache does not include its permanent set; i.e., the capacity concerns only files handled through the EWMA principle.
    """

       

    def __init__(self,capacity,_id,gamma=0.1,T=5,expon=0.5,interpolate=False):
	self.cache = LMinimalCache(capacity,_id,gamma,expon,interpolate)
	self._id = _id
	self.permanent_set=set([])
	self._capacity=capacity
	self.T=T
	self.stats ={}
	self.stats['queries'] = 0.0
	self.stats['hits'] = 0.0
	self.stats['responses'] = 0.0
	self.stats['explores'] = 0.0
	self.stats['explore_responses'] = 0.0


    def __str__(self):
	return 'Cache: '+str(self.cache)+'Permanent: '+str(self.permanent_set)	

    def __contains__(self,item):
	return item in self.cache or item in self.permanent_set

    def __iter__(self):
	return itertools.chain(self.cache,self.permanent_set)


    def capacity(self):
        return self._capacity

    
    def perm_set(self):
	return self.permanent_set
    
    def isPermanent(self,item):
	return item in self.permanent_set

    def makePermanent(self,item):
	self.permanent_set.add(item)
    	
    def receive(self,msg,e,now):
	label,d,query_id = msg.header


	if label == "QUERY":
	   msglist = []
	   item = d.item

	   logging.debug(pp([now,': Query message for item',item,'received by cache',self._id]))
	   self.stats['queries'] +=1.0

           if self._id == d.query_source:
	        succ = d.succ(self._id)
		if succ != None:
		    logging.debug(pp([now,': Item',item,'is inside cache of ',self._id,' which is the query source, will prepare an exploration message']))
	   	    emsg = ExploreMessage(d,query_id,self._id)
		    e = (self._id,succ)
		    sumsofar = self.cache.state(item) + float(item in self.permanent_set)
		    emsg.payload = sumsofar
		    if sumsofar < 1.0:
		       msglist.append( (emsg,e) )


	   inside_cache = item in self.cache 
           inside_perm  = item in self.permanent_set
	   if inside_cache or inside_perm:
              logging.debug(pp([now,': Item',item,'is inside %s of %d'% ( 'cache' if inside_cache else 'permanent set',self._id)]))
	      self.stats['hits'] +=1
	      if self._id == d.query_source:
                logging.debug(pp([now,': Response to query',query_id,'of',d,' finally delivered by cache',self._id]))
		pred = d
	      else:
		pred = d.pred(self._id)
		logging.debug(pp([now,': Response to query',query_id,'of',d,' generated by cache',self._id]))
	      e = (self._id,pred)
	      rmsg = ResponseMessage(d,query_id,stats=msg.stats)

	      msglist.append( (rmsg,e) )
		  
	   else:
              logging.debug(pp([now,': Item',item,'is not inside',self._id,'continue searching']))
	      succ = d.succ(self._id)
	      if succ == None:
		logging.error(pp([now,':Query',query_id,'of',d,'reached',self._id,'and has nowhere to go, will be dropped']))
		return []
	      e = (self._id,succ)
	      msglist.append( (msg,e) )

	   return msglist

	if label == "RESPONSE":
	   logging.debug(pp([now,': Response message for',d,'received by cache',self._id]))
	   self.stats['responses'] +=1.0
	   item = d.item
           #logging.debug(pp([now,': Node',self._id,'updating derivative of item',item,'with measurement',msg.stats['downweight']]))
           #self.cache.add(item,msg.stats['downweight'],now) #does not really add item, but updates its gradient estimate
           #logging.debug(pp([now,': Node',self._id,' now stores',self]))
  
	   if d.query_source==self._id:
              logging.debug(pp([now,': Response to query',query_id,'of',d,' finally delivered by cache',self._id]))
	      pred = d
	   else:
              logging.debug(pp([now,': Response to query',query_id,'of',d,'passes through cache',self._id,'moving further down path']))
	      pred = d.pred(self._id)
	   e = (self._id,pred)
	   return [(msg,e)]

	if label == "EXPLORE":
	   item = d.item
	   logging.debug(pp([now,': Explore message for item',item,'received by cache',self._id,'sumsofar is %f local state is %f in permanent is %f' % (msg.payload,self.cache.state(item),float(item in self.permanent_set))] ))
	   self.stats['explores'] +=1.0

           new_sum = msg.payload+self.cache.state(item) + float(item in  self.permanent_set)
	   if new_sum > 1.0 or item in self.permanent_set:
              logging.debug(pp([now,': Item',item,'has aggregate state value',new_sum,'on node',self._id,'of demand',d]))
	      pred = d.pred(self._id)
	      logging.debug(pp([now,': Explore Response to query',query_id,'of',d,' generated by cache',self._id]))
	      e = (self._id,pred)
	      ermsg = ExploreResponseMessage(d,query_id,msg.explore_source,stats=msg.stats)
	      
	      return [(ermsg,e)]
		  
	   else:
              logging.debug(pp([now,': Item',item,'is has agreggate',new_sum,'at node',self._id,'continue exploring']))
	      succ = d.succ(self._id)
	      if succ == None:
		logging.error(pp([now,':Exploration for',query_id,'of',d,'reached',self._id,'and has nowhere to go, will be dropped']))
		return []
	      e = (self._id,succ)
	      msg.payload = new_sum
	      return [(msg,e)]

 
	if label =="EXPLORE_RESPONSE":
	   logging.debug(pp([now,': Explore Response message for',d,'received by cache',self._id]))
	   self.stats['explore_responses'] +=1.0
	   item = d.item
           
	   logging.debug(pp([now,': Node',self._id,'received exploration response for',query_id,'of',d]))
           logging.debug(pp([now,': Node',self._id,'updating derivative of item',item,'with measurement',msg.stats['downweight']]))
           self.cache.updateGradient(item,msg.stats['downweight']) 
  
	   if msg.explore_source==self._id:
		logging.debug(pp([now,': Node',self._id,'is terminal node for exploration response for',query_id,'of',d]))
           	#logging.debug(pp([now,': Node',self._id,' now stores',self.cache]))
       		return [] 
	 
           logging.debug(pp([now,': Explore Response to query',query_id,'of',d,'passes through cache',self._id,'moving further down path']))
	   pred = d.pred(self._id)
	   e = (self._id,pred)
	   return [(msg,e)]


    def shuffleProcess(self):
	"""A process handling messages sent to caches.

	   It is effectively a wrapper for a receive call, made to a NetworkedCache object. 

	"""
	while True:
          logging.debug(pp([self.env.now,': New suffling of cache',self._id]))
	  #key,placements,probs,distr=self.cache.shuffle(self.env.now)
	  self.cache.shuffle(self.env.now)
	  logging.debug(pp([self.env.now,': New cache at',self._id, ' is',self.cache]))
	  #logging.debug('Setting cache at %d to %s, probability was:%f' %(self._id,str(self.cache),probs[key]) )

          yield self.env.timeout(self.T)


    def startShuffleProcess(self,env):
	self.env = env
	self.env.process(self.shuffleProcess())

    def state(self,item):
	return max(self.cache.state(item), float(item in self.permanent_set)) 

    def non_zero_state_items(self):
	return self.cache._state.keys()+list(self.permanent_set)
def  write_demands(demands, file):
    f = open(file,'w')
    for Demand in demands:
        dict_demand = Demand.__dict__
        f.write(str(dict_demand)+"\n")
    f.close()    
class Problem:
    def __init__(self,G,capacities,demands,bandwidths):
        self.capacities = capacities
        self.bandwidths = bandwidths
        self.demands = demands
        #Create caching and utility reminder variables, i.e., VAR and REM, respectively, and initizlie them to 0
        self.REM = SparseVector( {} )
        self.VAR = SparseVector( {} )
        self.BOX = {}
        for demand in self.demands:
            item = demand['item']
            maxRate = demand['rate']
            path = tuple(demand['path'])
            self.REM[(item, path)] = 0.5
            self.BOX[(item ,path)] = maxRate
            for node in path[:-1]:
                if (item, node) not in self.VAR:
                    self.VAR[(item, node)]  = 0.0
                    self.BOX[(item, node)]  = 1.0
        self.VAR.update( self.REM  ) 
        #Margin for log utilities
        self.log_margin = 1.e-1

        

    def setVAR2Zero(self):
        for var in self.VAR:
            self.VAR[var]  = 0.0

    def evalGradandUtilities(self, degree=2):
        grads = {}
        utility_func = {}
        Hessian  = {}
        for demand in self.demands:
            item = demand['item']
            maxRate = demand['rate']
            path = tuple(demand['path'])
            #*
            #log_margin  = 1.e-1
            utility_func[(item, path)] = -1.0 * np.log(maxRate - self.VAR[(item,path)] + self.log_margin)
            if degree<1:
                continue
            #Grad
            grads[(item, path)] = {(item, path): 1./ (maxRate - self.VAR[(item,path)] + self.log_margin)}
            if degree<2:
                continue
            #Hessian
            Hessian[(item, path)] = {((item, path), (item, path)): (maxRate - self.VAR[(item,path)] + self.log_margin) ** -2}
        return utility_func, grads, Hessian
    def genDep(self):
        #NOT Used!
        """Generate a dict of dependencaies, s.t., the kyes are variables (item . node) and the values 
         are the list of edges tha depend on the caching variable corresponding to the item and node."""

        Var2edge_dep = {}
        for demand in self.demands:
            item = demand['item']
            rate = demand['rate']
            path = tuple(demand['path'])
            tmp_VarSoFar = []
            for node_i in range(len(path)-1):
                edge = (path[node_i],path[node_i+1])
                tmp_VarSoFar.append( (path[node_i], item)  )
                for iterm_node_pair in tmp_VarSoFar:
                    if iterm_node_pair not in Var2edge_dep:
                        Var2edge_dep[iterm_node_pair] = [edge]
                    else:
                        Var2edge_dep[iterm_node_pair].append( edge )
        return edge2vars_dep
    
                
    def evalConstraint(self, edge, edge2rem_dep):
        #* NOT USED 
        cnst = self.bandwidth[edge]
        sumSoFar = 0.0
        for (item, path, maxRate) in edge2rem_dep[edge]:
            prod_ip = maxRate - self.REM[(item, path)]
            for node_i in path:
                prod_ip *= (1.0 - self.VAR[(item, node_i)])
            sumSoFar += prod_ip
        return cnst - sumSoFar

    def evalDelta(self, given_item, given_node):
        "Evaluate the total decrease in link constraints when item is cached at node"
        totalDelta = 0.0
        for demand in self.demands:
            item = demand['item']
            maxRate = demand['rate']
            path = tuple(demand['path']) 
            if item != given_item or given_node not in path:
                continue

            current_prod_without_given_pair = maxRate - self.VAR[(item, path)]
            given_item_node_VSISTED = False
            for i in range(len(path)-1):
                if item == given_item and path[i] == given_node:
                    given_item_node_VSISTED = True
                else:
                    current_prod_without_given_pair *=  (1.0 - self.VAR[(item, path[i])])

                if current_prod_without_given_pair > 0.0 and given_item_node_VSISTED:
                    totalDelta += current_prod_without_given_pair
        return totalDelta

   
    def evalGradandConstraints(self, degree=2):
        grads = {}
        Hessian = {}
        edge_func = {}
        for demand in self.demands:
            item = demand['item']
            maxRate = demand['rate']
            path = tuple(demand['path']) 
            zero_literals = []
            current_prod = maxRate - self.VAR[(item, path)]
            if  maxRate - self.VAR[(item, path)] <= 0.0:
                zero_literals.append( (item, path) )
                current_prod_no_zero_literals  = 1.0
            else:
                current_prod_no_zero_literals = current_prod
            for i in range(len(path)-1):
                edge = (path[i],path[i+1])
     
                if 1.0 - self.VAR[(item, path[i])] <= 0.0:
                    zero_literals.append( (item, path[i]) )
                else:
                    current_prod_no_zero_literals *= (1.0 - self.VAR[(item, path[i])])
                current_prod *=  (1.0 - self.VAR[(item, path[i])])


                if edge not in grads:
                    grads[edge] = {}
        
                if edge not in Hessian:
                    Hessian[edge] = {}


                if edge not in edge_func:
                    edge_func[edge] = self.bandwidths[edge]                  
                edge_func[edge] -= current_prod


                if degree<1:
                    continue 
                for j in range(i+1):
                    try:
                        current_grad_j = current_prod / (1.0 -   self.VAR[(item, path[j])])
                    except ZeroDivisionError:
                        if len(zero_literals) > 1:
                            current_grad_j = 0.0
                        else:
                            current_grad_j = current_prod_no_zero_literals
                    if (item, path[j]) not in grads[edge]:
                         grads[edge][(item, path[j])] = current_grad_j
                    else:
                         grads[edge][(item, path[j])] += current_grad_j
            
                        
                        
                try:
                    current_grad_j = current_prod / (maxRate - self.VAR[(item, path)])
                except ZeroDivisionError:
                    if len(zero_literals) > 1:
                      current_grad_j = 0.0
                    else:
                      current_grad_j = current_prod_no_zero_literals 
                grads[edge][(item, path)] = current_grad_j
                
                if degree<2:
                    continue
                #*Compute Hessian
                for j in range(i+1):
                    for k in range(j+1, i+1):
                        try:
                            current_Hessian_j_k = current_prod / ( (1.0 -   self.VAR[(item, path[j])]) *  (1.0 -   self.VAR[(item, path[k])]) )
                        except ZeroDivisionError:
                            if len(zero_literals) > 2:
                                current_Hessian_j_k = 0.0
                            else:
                                current_Hessian_j_k = current_prod_no_zero_literals
                                if  (item, path[j]) not in zero_literals:
                                    current_Hessian_j_k /= (1.0 - self.VAR[(item, path[j])])
                                if (item, path[k]) not in zero_literals:
                                    current_Hessian_j_k /= (1.0 - self.VAR[(item, path[k])])
        
                        if ((item, path[j]), (item, path[k])) not in Hessian[edge]:
                            Hessian[edge][  ((item, path[j]), (item, path[k])) ] = -1.0 * current_Hessian_j_k
                            Hessian[edge][  ((item, path[k]), (item, path[j])) ] = -1.0 * current_Hessian_j_k
                        else:
                            Hessian[edge][  ((item, path[j]), (item, path[k])) ] += -1.0 * current_Hessian_j_k
                            Hessian[edge][  ((item, path[k]), (item, path[j])) ] += -1.0 * current_Hessian_j_k
                for j in range(i+1):
                    try:
                        current_Hessian_j_r = current_prod / ( (maxRate - self.VAR[(item, path)]) *  (1.0 -   self.VAR[(item, path[j])]) )

                    except ZeroDivisionError:
                         if len(zero_literals) > 2:
                             current_Hessian_j_r = 0.0
                         else:
                             current_Hessian_j_r =  current_prod_no_zero_literals
                             if  (item, path[j]) not in zero_literals:
                                 current_Hessian_j_r /=  (1.0 - self.VAR[(item, path[j])])
                             if (item, path) not in zero_literals:
                                 current_Hessian_j_r /= (maxRate - self.VAR[(item, path)])
                    Hessian[edge][ ( (item, path), (item, path[j])) ] = -1.0 * current_Hessian_j_r
                    Hessian[edge][ ( (item, path[j]), (item, path)) ] = -1.0 * current_Hessian_j_r 
          
                                
                                    
        return edge_func, grads, Hessian
    def evalGradandCapcityConstraints(self, degree=2):
        grads = {}
        cap_nodes = {}
        Hessian = {}
        for (item, node) in self.VAR:
             if type(node) == tuple:
                 continue 
             if node not in cap_nodes:
                 cap_nodes[node] =  self.capacities[node]

             cap_nodes[node] -= self.VAR[(item, node)]
             if degree<1:
                 continue              
             #Grad
             if node not in grads:
                 grads[node] = {}
             grads[node][(item, node)] = -1.0
             if node not in Hessian:
                 Hessian[node] = {}
        return cap_nodes, grads, Hessian
    def evalFullConstraintsGrad(self, degree=2):
        cap_nodes, cap_grads, cap_Hessian = self.evalGradandCapcityConstraints(degree)
        edge_func, edge_grads, edge_Hessian = self.evalGradandConstraints(degree)

        #Merge
        cap_grads.update( edge_grads )
        cap_nodes.update( edge_func ) 
        cap_Hessian.update( edge_Hessian ) 
        return cap_nodes, cap_grads, cap_Hessian

    def evaluate(self):
        "Compute the toal objective and the ratio of satisfied constraints."
        utility_func, dummy1, dummy2 = self.evalGradandUtilities(0)
        obj = sum(utility_func.values())
        constraint_func, dummy1, dummy2 = self.evalFullConstraintsGrad(0)
        satisfied_constrints = [constraint_func[constraint] >= 0.0 for constraint  in constraint_func]
        tot_constrinats = sum(constraint_func.values())
        return obj, sum(satisfied_constrints)/(1.0* len(constraint_func.items() ))
                
        
    def pickle_cls(self,fname):
        f = open(fname,'w')
        pickle.dump(self, f)
        f.close()
    @staticmethod
    def unpickle_cls(fname):
        with file(fname,'r') as f:
            return pickle.load(f)    
class RelaxedProblem(Problem):
    def evalGradandConstraints(self, degree=2):
        "Evalauate relaxed constraints and their gradients and Hessian."
        grads = {}
        Hessian = {}
        edge_func = {}
        for demand in self.demands:
            item = demand['item']
            maxRate = demand['rate']
            path = tuple(demand['path']) 
            current_sum = self.VAR[(item, path)] / maxRate
            current_VARS = [(item, path)]
            for i in range(len(path) -1):
                edge = (path[i],path[i+1])
                current_VARS.append((item, path[i]))
                if edge not in edge_func:
                    edge_func[ edge]  = self.bandwidths[edge] / (1.0 - 1.0 / math.e)
                edge_func[ edge] -= maxRate / (1.0 - 1.0 / math.e)
     
                current_sum += self.VAR[(item, path[i])] 
                #Evaluate constraints 
                edge_func[ edge] += maxRate * min(1.0, current_sum)

                if degree <= 0:
                    continue
                #Evaluate gradients 
                if edge not in grads:
                    grads[edge] = {}
                #Evaluate Hessian (which is zero!)
                if edge not in Hessian:
                    Hessian[edge] = {}
               
                if current_sum >= 1.0:
                    #In this case the contribution of the current demand to the gradients is zero.
                    continue
                for var in current_VARS:
                    if type(var[1]) == tuple:
                        grad_var = 1.0/maxRate
                    else:
                        grad_var = 1.0
                    if  var not in grads[edge]:
                        grads[edge][var] = grad_var
                    else:
                        grads[edge][var] += grad_var
        return edge_func, grads, Hessian
                                        


def generate_bandwidths_old(edge_dict,demands,V,I,mode='bi'):
    EDGE_mu = {}
    X = matrix(0,(V,I))
    rho_uv = ro_uv(edge_dict,demands ,X)
    max_rho = max(rho_uv.values())
    if mode=='mono':
        for edge in edge_dict:
            (u,v) = edge
            EDGE_mu[(u,v)] = max_rho*1.05
    else:
        max_rho = max(rho_uv.values())
        high_load_links = []
        for (u,v) in rho_uv:
            if rho_uv[(u,v)] >= max_rho:
                high_load_links.append((u,v))
                high_load_links.append((v,u))
        high_load_links = list(set(high_load_links))
        for edge in edge_dict:
            (u,v) = edge
            if (v,u) in EDGE_mu:
                EDGE_mu[(u,v)] = EDGE_mu[(v,u)]
            else:

                Pr = np.random.random()
                if Pr > 0.3:
                    EDGE_mu[(u,v)] = max_rho*1.05
                else:
                    EDGE_mu[(u,v)] = max_rho*200
        for edge in high_load_links:
            EDGE_mu[edge] = max_rho*1.05
    return EDGE_mu
            
def ro_uv(edge_dict,demands ,X):
    ro_uv = {}
        #Initialize the functions...
    for edge in edge_dict:
        ro_uv[edge] = 0.
        # Go through demands
    for demand in demands:
        path = demand['path']
        item = demand['item']
        rate = demand['rate']
        if  X[path[0],item]==0 and len(path)>1:
            for node_i in range(len(path)-1):
                edge = (path[node_i],path[node_i+1])
                ro_uv[edge] = ro_uv[edge]+rate
                if X[path[node_i+1],item]==1:
                     break

    return ro_uv    
def generate_bandwidths(demands, margin=0.95):
    '''Given the demands generate bandwidths per edge, s.t., they are margin times the sum of max rates over an edge.'''
    total_rate = {}
    for demand in demands:
        path = demand['path']
        item = demand['item']
        rate = demand['rate']
        for node_i in range(len(path)-1):
            edge = (path[node_i],path[node_i+1])
            if edge not in total_rate:
                 total_rate[edge] = rate
            else:
                 total_rate[edge] += rate
    bandwidths = dict( [(edge, total_rate[edge] * margin) for edge in total_rate] )
    return bandwidths
        
def main():
   #logging.basicConfig(filename='execution.log', filemode='w', level=logging.INFO) 

   parser = argparse.ArgumentParser(description = 'Simulate a Network of Caches',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('outputfile',help = 'Output file')
   parser.add_argument('problem_type', choices=['Problem', 'RelaxedProblem'], help='Type of problem')
   parser.add_argument('--logfile',default='generator.log',type=str, help='logfile')
   parser.add_argument('--max_capacity',default=2,type=int, help='Maximum capacity per cache')
   parser.add_argument('--min_capacity',default=2,type=int, help='Minimum capacity per cache')
   parser.add_argument('--max_rate',default=1.0,type=float, help='Maximum demand rate')
   parser.add_argument('--catalog_size',default=10,type=int, help='Catalog size')
   parser.add_argument('--demand_size',default=20,type=int, help='Demand size')
   parser.add_argument('--demand_change_rate',default=0.0,type=float, help='Demand change rate')
   parser.add_argument('--demand_distribution',default="powerlaw",type=str, help='Demand distribution',choices=['powerlaw','uniform'])
   parser.add_argument('--powerlaw_exp',default=1.2,type=float, help='Power law exponent, used in demand distribution')
   parser.add_argument('--query_nodes',default=10,type=int, help='Number of nodes generating queries')
   parser.add_argument('--graph_type',default="erdos_renyi",type=str, help='Graph type',choices=['erdos_renyi','balanced_tree','hypercube',"cicular_ladder","cycle","grid_2d",'lollipop','expander','hypercube','star', 'barabasi_albert','watts_strogatz','regular','powerlaw_tree','small_world','geant','abilene','dtelekom','servicenetwork', 'linear'])
   parser.add_argument('--graph_size',default=100, type=int, help='Network size')
   parser.add_argument('--graph_degree',default=4, type=int, help='Degree. Used by balanced_tree, regular, barabasi_albert, watts_strogatz')
   parser.add_argument('--graph_p',default=0.10, type=int, help='Probability, used in erdos_renyi, watts_strogatz')
   parser.add_argument('--random_seed',default=4156910908,type=int, help='Random seed')
   parser.add_argument('--debug_level',default='INFO',type=str,help='Debug Level', choices=['INFO','DEBUG','WARNING','ERROR'])
   parser.add_argument('--cache_type',default='LRU',type=str,help='Networked Cache type', choices=['LRU','FIFO','LFU','RR','EWMAGRAD','LMIN'])
   parser.add_argument('--query_message_length',default=0.0,type=float,help='Query message length')
   parser.add_argument('--response_message_length',default=0.0,type=float,help='Response message length')
   parser.add_argument('--monitoring_rate',default=1.0,type=float,help='Monitoring rate')
   parser.add_argument('--interpolate',default=False,type=bool,help='Interpolate past states, used by LMIN')
   parser.add_argument('--beta',default=1.0,type=float,help='beta used in EWMA')
   parser.add_argument('--gamma',default=0.1,type=float,help='gamma used in LMIN')
   parser.add_argument('--expon',default=0.5,type=float,help='exponent used in LMIN')
   parser.add_argument('--congestion',default=0.95,type=float,help='Number in [0,1] shows th ratio of each edge`s bandwidth to the maximum rates for demands passing over an edge') 
   parser.add_argument('--T',default=5.,type=float,help='Suffling period used in LMIN')
   args = parser.parse_args()
   	
   args.debug_level= eval("logging."+args.debug_level)

   def graphGenerator():
	if args.graph_type == "erdos_renyi":
	    return networkx.erdos_renyi_graph(args.graph_size,args.graph_p)
	if args.graph_type == "balanced_tree":
            #* Changed ceiling to floor
	    ndim = int(np.floor(np.log(args.graph_size)/np.log(args.graph_degree)))
	    return networkx.balanced_tree(args.graph_degree,ndim)
	if args.graph_type == "cicular_ladder":
	    ndim = int(np.ceil(args.graph_size*0.5))
	    return  networkx.circular_ladder_graph(ndim)
	if args.graph_type == "cycle":
	    return  networkx.cycle_graph(args.graph_size)
	if args.graph_type == 'grid_2d':
	    ndim = int(np.ceil(np.sqrt(args.graph_size)))
            return networkx.grid_2d_graph(ndim,ndim)
	if args.graph_type == 'lollipop':
	    ndim = int(np.ceil(args.graph_size*0.5))
	    return networkx.lollipop_graph(ndim,ndim)
	if args.graph_type =='expander':
	    ndim = int(np.ceil(np.sqrt(args.graph_size)))
	    return networkx.margulis_gabber_galil_graph(ndim)
	if args.graph_type =="hypercube":
	    ndim = int(np.ceil(np.log(args.graph_size)/np.log(2.0)))
	    return networkx.hypercube_graph(ndim)
	if args.graph_type =="star":
	    ndim = args.graph_size-1
	    return networkx.star_graph(ndim)
	if args.graph_type =='barabasi_albert':
	    return networkx.barabasi_albert_graph(args.graph_size,args.graph_degree)
	if args.graph_type =='watts_strogatz':
	    return networkx.connected_watts_strogatz_graph(args.graph_size,args.graph_degree,args.graph_p)
	if args.graph_type =='regular':
	    return networkx.random_regular_graph(args.graph_degree,args.graph_size)
	if args.graph_type =='powerlaw_tree':
	    return networkx.random_powerlaw_tree(args.graph_size)
	if args.graph_type =='small_world':
	    ndim = int(np.ceil(np.sqrt(args.graph_size)))
	    return networkx.navigable_small_world_graph(ndim)
	if args.graph_type =='geant':
	    return topologies.GEANT()
	if args.graph_type =='dtelekom':
	    return topologies.Dtelekom()
	if args.graph_type =='abilene':
	    return topologies.Abilene()
	if args.graph_type =='servicenetwork':
	    return topologies.ServiceNetwork()
        if args.graph_type == 'linear':
            return networkx.path_graph(args.graph_size)

   def cacheGenerator(capacity,_id):
   	if args.cache_type == 'LRU':
            return  PriorityNetworkCache(capacity,_id,'LRU')	    		
   	if args.cache_type == 'LFU':
            return  PriorityNetworkCache(capacity,_id,'LFU')	    		
   	if args.cache_type == 'FIFO':
            return  PriorityNetworkCache(capacity,_id,'FIFO')	    		
   	if args.cache_type == 'RR':
            return  PriorityNetworkCache(capacity,_id,'RR')	    		
	if args.cache_type == 'EWMAGRAD': 
	    return EWMAGradCache(capacity,_id,beta=args.beta)
	if args.cache_type == 'LMIN': 
	    return LMinCache(capacity,_id,gamma=args.beta,T = args.T, expon = args.expon,interpolate = args.interpolate)


   logging.basicConfig(level=args.debug_level) 
   random.seed(args.random_seed)
   np.random.seed(args.random_seed+2015 )
  
  
   construct_stats ={}
   
   logging.info('Generating graph and weights...')
   temp_graph = graphGenerator()
   #networkx.draw(temp_graph)
   #plt.draw()	
   logging.debug('nodes: '+str(temp_graph.nodes()))
   logging.debug('edges: '+str(temp_graph.edges()))
   G = DiGraph()
   
   number_map = dict( zip(temp_graph.nodes(),range(len(temp_graph.nodes())) ))
   G.add_nodes_from(number_map.values()) 
   for (x,y) in temp_graph.edges():
        xx = number_map[x]
	yy = number_map[y]
   	G.add_edges_from( ((xx,yy),(yy,xx)) ) 
   
   graph_size = G.number_of_nodes()
   edge_size = G.number_of_edges()
   logging.info('...done. Created graph with %d nodes and %d edges' % (graph_size,edge_size))
   logging.debug('G is:'+str(G.nodes())+str(G.edges()))
   construct_stats['graph_size'] = graph_size
   construct_stats['edge_size'] = edge_size
   logging.info('Generating item sources...')
   item_sources = dict( (item,[list(G.nodes)[source]]) for item,source in zip(range(args.catalog_size),np.random.choice(range(graph_size),args.catalog_size)) )
   if args.graph_type == 'linear':
       item_sources = {}
       for item in range(args.catalog_size):
           item_sources[item] = [args.graph_size -1]
   logging.info('...done. Generated %d sources'%len(item_sources))
   logging.debug('Generated sources:')
   for item in item_sources:
	logging.debug(pp([item,':',item_sources[item]]))	

   construct_stats['sources'] = len(item_sources)

   logging.info('Generating query node list...')
   
   query_node_list = [ list(G.nodes)[i] for i  in random.sample(xrange(graph_size),args.query_nodes)]

    #*HARD CODED
   if args.graph_type == 'linear':
        query_node_list = [0]
   
   logging.info('...done. Generated %d query nodes.'%len(query_node_list) )

   construct_stats['query_nodes'] = len(query_node_list)
	
   logging.info('Generating demands...')
   if args.demand_distribution == 'powerlaw':
	factor = lambda i : (1.0+i)**(-args.powerlaw_exp) 
   else:
	factor = lambda i : 1.0
   pmf =  np.array( [ factor(i) for i in range(args.catalog_size) ] ) 
   pmf /= sum(pmf)
   distr = rv_discrete(values=(range(args.catalog_size),pmf))
   if args.catalog_size <= args.demand_size: 
   	items_requested = list(distr.rvs(size = (args.demand_size-args.catalog_size))) +range(args.catalog_size)
   else:
	items_requested =  list(distr.rvs(size = args.demand_size))
   
   random.shuffle(items_requested)
	
   demands_per_query_node= args.demand_size // args.query_nodes
   remainder = args.demand_size % args.query_nodes
   demands =[]

   for x in query_node_list:
      dem = demands_per_query_node
      if x < remainder:
	dem = dem+1
	
      new_dems = [ Demand(items_requested[pos], shortest_path(G,x,item_sources[   items_requested[pos] ][0],weight='weight'),  args.max_rate)   for pos in range(len(demands),len(demands)+dem)]
      logging.debug(pp(new_dems))
      demands = demands + new_dems
   
   Demand_list =[]
   for demand in demands:
        Demand_list.append(demand.__dict__)

   demands = Demand_list 


   logger = logging.getLogger('Topology Generator')
   clearFile(args.logfile)
   fh = logging.FileHandler(args.logfile)
   fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
   logger.addHandler(fh)

   logger.info('...done. Generated %d demands'%len(demands))
   #plt.hist([ d.item for d in demands], bins=np.arange(args.catalog_size)+0.5)
   #plt.show()
   construct_stats['demands'] = len(demands)

   logger.info('Generating capacities...')
   capacities = dict( (x,random.randint(args.min_capacity,args.max_capacity)) for  x in G.nodes() )
   logger.info('...done. Generated %d caches' % len(capacities))
   logger.debug('Generated capacities:')
   for key in capacities:
   	logger.debug(pp([key,':',capacities[key]]))
   dem_items=[]
   for demand in demands:
         item = demand['item']
         dem_items.append(item)
   V,I = (len(capacities),len(dem_items))
   bandwidths = generate_bandwidths(demands, margin=args.congestion) 


   logger.info('Building CacheNetwork')
   logger.info('...done')
   out = args.outputfile+ "_" + args.problem_type + "_" + args.graph_type +"_"+str(args.demand_size) +"demands_"+ str(args.catalog_size)+"catalog_size_"+"mincap_"+str(args.min_capacity)+"maxcap_"+str(args.max_capacity)+"_"+str(args.graph_size)+"_"+str(args.demand_distribution)+"_"+"rate"+str(args.max_rate) +"_"+ str(args.query_nodes) + "qnodes" + "_" + str(args.congestion)  + "congestion"
   problem_class  = eval(args.problem_type)
   pr = problem_class(G,capacities,demands,bandwidths)
   print "The number of variables is ", len(pr.VAR.keys())
   pr.pickle_cls(out)
   
#   print 'Demands are: ', pr.demands
#   print 'bandwidths are : ', pr.bandwidths
#   print 'Capacities are : ', pr.capacities 
#   print pr.evalGradandConstraints()
#   print 'Capacity Grads are: ', pr.evalGradandCapcityConstraints()
 #  print 'Full Grads are: ', pr.evalFullConstraintsGrad()
   
   
if __name__=="__main__":
   main()
