# Exploring Value Function Factorisation in MADDPG

COMP390 Final Year Project at University Of Liverpool researching Value Function Factorisation techniques found in Q-learner MARL algorthms and testing use in Actor-Critic MARL (see: https://sam.csc.liv.ac.uk/COMP390/2022/project/sgjtrevo)

To correctly use and run experiments on this code, it must be used with the ePyMARL library (https://github.com/uoe-agents/epymarl). The folders config, modules, learners and envs must be replaced with the files found here. The current files are setup in such a way that they should give no errors for predator prey task testing, however, because the Matrix Game environemnt needs certain adjustments (which are clearly commented in code) you will have to replace some lines to test there.

The plotting tool "originalPlot.py" has been included. This is a minor modified file that uses matplotlib to create graphs from results, which are read in through JSON data created through the logging library Sacred (https://sacred.readthedocs.io/en/stable/index.html) and its simple file observer. Full sets of results will be incrementally added to this repo, expect to see 3 folders of results for each test (testing on 3 random seeds).

Any questions about this library should be directed to Joseph (sgjtrevo@liverpool.ac.uk), we can only pray that TurnItIn isnt going to flag this as me plagerising myself! (Hence why this library is private)
