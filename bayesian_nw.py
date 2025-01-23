'''
We want to predict whether a student will pass or fail an exam. The result depends on:

Studied: Did the student study? (Yes/No)
Difficulty: Was the exam difficult? (Easy/Hard)

'''
import pyAgrum as gum

# Create a simple Bayesian Network
bn = gum.BayesNet("Student Performance")

# Add variables (nodes)
# Studied: Root node with two states
studied = bn.add(gum.LabelizedVariable("Studied", "Did the student study?", ["No", "Yes"]))

# Difficulty: Root node with two states
difficulty = bn.add(gum.LabelizedVariable("Difficulty", "How difficult was the exam?", ["Easy", "Hard"]))

# Performance: Target variable (dependent on Studied and Difficulty)
performance = bn.add(gum.LabelizedVariable("Performance", "Exam Performance", ["Fail", "Pass"]))

# Add arcs (dependencies)
bn.addArc(studied, performance)
bn.addArc(difficulty, performance)

# Set probability tables (CPTs)
# P(Studied)
bn.cpt(studied).fillWith([0.6, 0.4])  # 60% did not study, 40% studied

# P(Difficulty)
bn.cpt(difficulty).fillWith([0.7, 0.3])  # 70% of exams are easy, 30% are hard

# P(Performance | Studied, Difficulty)
bn.cpt(performance)[{'Studied': 0, 'Difficulty': 0}] = [0.8, 0.2]  # Didn't study, Easy: 80% Fail, 20% Pass
bn.cpt(performance)[{'Studied': 0, 'Difficulty': 1}] = [0.9, 0.1]  # Didn't study, Hard: 90% Fail, 10% Pass
bn.cpt(performance)[{'Studied': 1, 'Difficulty': 0}] = [0.3, 0.7]  # Studied, Easy: 30% Fail, 70% Pass
bn.cpt(performance)[{'Studied': 1, 'Difficulty': 1}] = [0.5, 0.5]  # Studied, Hard: 50% Fail, 50% Pass

# Print the Bayesian Network structure
print("Bayesian Network structure:")
print(bn)

# Perform inference
ie = gum.LazyPropagation(bn)

# Example 1: What is the probability of passing without setting any evidence?
ie.makeInference()
print("\nPosterior probabilities for 'Performance':")
print(ie.posterior(performance))

# Example 2: Set evidence: The student studied and the exam was hard
ie.setEvidence({"Studied": 1, "Difficulty": 1})
ie.makeInference()
print("\nPosterior probabilities for 'Performance' given evidence (Studied=Yes, Difficulty=Hard):")
print(ie.posterior(performance))
