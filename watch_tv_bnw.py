import pyAgrum as gum
# import pyAgrum.lib.notebook as gnb

bn = gum.BayesNet('Watch TV')

# add nodes

is_bored = bn.add(gum.LabelizedVariable('is_bored', 'Is Bored?', ['yes', 'no']))
goto_livingroom = bn.add(gum.LabelizedVariable('goto', 'Go to living room?', ['yes', 'no']))
watch_tv = bn.add(gum.LabelizedVariable('watch_tv', 'Watch TV?', ['yes', 'no']))

# add dependecies

bn.addArc(is_bored, watch_tv)
bn.addArc(is_bored, goto_livingroom)
bn.addArc(goto_livingroom, watch_tv)

# cpt

#P(isbored)
bn.cpt(is_bored).fillWith([0.5, 0.5])

# P(goto| is_bored)
bn.cpt(goto_livingroom).fillWith([0.9, 0.1, 
                                  0.1, 0.9])

#P(watch_tv|isbored, goto)
bn.cpt(watch_tv).fillWith([0.9, 0.1, 
                           0.2, 0.8,
                           0.8, 0.2,
                           0.0, 1.0])

ie = gum.LazyPropagation(bn)
ie.setEvidence({'goto': 'yes'})
ie.makeInference()

p_watchtv = ie.posterior('watch_tv')

print(p_watchtv[0])

ie.eraseAllEvidence()

ie.setEvidence({'is_bored':'yes'})
ie.makeInference()

p_watchtv1 = ie.posterior('watch_tv')
print(p_watchtv1)

