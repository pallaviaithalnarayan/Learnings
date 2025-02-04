import pyAgrum as gum
# import pyAgrum.lib.notebook as gnb

bn = gum.BayesNet('Weather')

# nodes
cloudy = bn.add('cloudy', 2)
rain = bn.add('rain', 2)
sprinkler = bn.add('sprinkler', 2)
wet_grass = bn.add('wet_grass', 2)

print(bn, wet_grass)

#add dependecies
bn.addArc(cloudy, rain)
bn.addArc(cloudy, sprinkler)
bn.addArc(rain, wet_grass)
bn.addArc(sprinkler, wet_grass)

# define cpts
#P(cloudy)
bn.cpt(cloudy).fillWith([0.5,0.5])

#P(rain|cloudy)
bn.cpt(rain).fillWith([0.8, 0.2,
                       0.2, 0.8])
#P(sprinkler|cloudy)
bn.cpt(sprinkler).fillWith([0.90, 0.10,
                            0.10, 0.90])
#P(wet_grass|rain,sprinkler)
bn.cpt(wet_grass).fillWith([0.99, 0.01, #rain, sprinkler
                            0.90, 0.10, #rain, no sprinkler
                            0.10, 0.90, #no rain, sprinkler
                            0.00, 1.00]) # no rain, no sprinkler

print(bn)
# gnb.showBN(bn)
ie = gum.LazyPropagation(bn)
ie.makeInference()

print(ie.posterior('wet_grass'))