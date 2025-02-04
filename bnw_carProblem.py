import pyAgrum as gum
# import pyAgrum.lib.notebook as gnb

bn = gum.BayesNet('Car Problem')

# add nodes

battery = bn.add(gum.LabelizedVariable('battery', 'Battery', ['good', 'bad']))
fuel = bn.add(gum.LabelizedVariable('fuel', 'Fuel', ['has_fuel', 'no_fuel']))
weather = bn.add(gum.LabelizedVariable('weather', 'Weather', ['cold', 'normal']))
starter_motor = bn.add(gum.LabelizedVariable('starter_motor', 'starter motor', ['works', 'doesnt_work']))

engine_start = bn.add(gum.LabelizedVariable('engine_start', 'Engine Start', ['yes', 'no']))

# add dependencies

bn.addArc(battery, engine_start)
bn.addArc(fuel, engine_start)
bn.addArc(weather, engine_start)
bn.addArc(starter_motor, engine_start)

# cpt for each node

#P(battery)
bn.cpt(battery).fillWith([0.5, 0.5])

#P(fuel)
bn.cpt(fuel).fillWith([0.9,0.1])

#P(weather)
bn.cpt(weather).fillWith([0.7, 0.3])

#P(starter_moter)
bn.cpt(starter_motor).fillWith([0.8, 0.2])

#P(engine_start|battery, fuel, weather, starter_motor)


bn.cpt(engine_start).fillWith([
    0.99, 0.01,  # battery=good, fuel=has, weather=cold, starter=works
    0.0, 1.0,    # battery=good, fuel=has, weather=cold, starter=doesn't work
    0.8, 0.2,    # battery=good, fuel=has, weather=normal, starter=works
    0.0, 1.0,    # battery=good, fuel=has, weather=normal, starter=doesn't work
    0.0, 1.0,    # battery=good, fuel=no, weather=cold, starter=works
    0.0, 1.0,    # battery=good, fuel=no, weather=cold, starter=doesn't work
    0.0, 1.0,    # battery=good, fuel=no, weather=normal, starter=works
    0.0, 1.0,    # battery=good, fuel=no, weather=normal, starter=doesn't work
    0.6, 0.4,    # battery=bad, fuel=has, weather=cold, starter=works
    0.0, 1.0,    # battery=bad, fuel=has, weather=cold, starter=doesn't work
    0.7, 0.3,    # battery=bad, fuel=has, weather=normal, starter=works
    0.0, 1.0,    # battery=bad, fuel=has, weather=normal, starter=doesn't work
    0.0, 1.0,    # battery=bad, fuel=no, weather=cold, starter=works
    0.0, 1.0,    # battery=bad, fuel=no, weather=cold, starter=doesn't work
    0.0, 1.0,    # battery=bad, fuel=no, weather=normal, starter=works
    0.0, 1.0,    # battery=bad, fuel=no, weather=normal, starter=doesn't work
])

# print(gnb.showBN(bn))
bn.cpt(weather)

ie = gum.LazyPropagation(bn)
# ie.setEvidence({'battery': 'good'})
ie.makeInference()

ie.posterior('engine_start')