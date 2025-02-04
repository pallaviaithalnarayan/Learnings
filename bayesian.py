import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb

#import graphviz
#print(graphviz.__version__)

bn = gum.BayesNet('Weather')

#Add nodes

cloudy = bn.add('cloudy', 2) # Binary: cloudy or not
rain = bn.add('rain', 2) # Binary: rainy or not
sprinkler = bn.add('sprinkler', 2) # on or off
grass_wet = bn.add('grass_wet', 2) # wet or not

# Add arcs(dependencies)
bn.addArc(cloudy, rain)
bn.addArc(cloudy, sprinkler)
bn.addArc(rain, grass_wet)
bn.addArc(sprinkler, grass_wet)

# Define conditional probability tables
#P(cloudy)
bn.cpt(cloudy).fillWith([0.5,0.5])

#P(rain|cloudy)
bn.cpt(rain).fillWith([0.8, 0.2, # when cloudy
                       0.2, 0.8]) #when not cloudy

#P(sprinkler|cloudy)
bn.cpt(sprinkler).fillWith([0.1,0.9, # when cloudy
                            0.5, 0.5]) # when not cloudy

#P(grass_wet|rain,sprinkler)
bn.cpt(grass_wet).fillWith([0.99, 0.01, #rain and sprinkler
                            0.90, 0.10, # rain and no sprinkler
                            0.90, 0.10, # no rain and sprinkler
                            0.00, 1.00]) # no rain and no sprinkler

print(bn)
print(f'{bn.cpt(cloudy), bn.cpt(rain), bn.cpt(sprinkler), bn.cpt(grass_wet)}')
# gnb.showBN(bn)