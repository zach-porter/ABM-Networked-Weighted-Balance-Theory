# ABM-Networked-Weighted-Balance-Theory
Extension on the agent based model of Weighted Balance Theory by Schweighofer, Schweitzer, and Garcia. The code is based off of their work, and uses and extends their code via the Mesa package in python. 

For a Simple Overview, read Section 1.1 A-C

1.1a: Purpose:

The purpose of this model is to simulate the opinion dynamics of different types of network using the Weighted Balanced Model. More specifically, this model simulates how individuals change their opinion based upon their neighbors opinions and the equanimity their neighbor’s have. The individuals can only access other neighbors to update their opinion that are a distance of 1 from them in a certain type of network. Depending on the position of the individual in the network, they may update their opinion if and only if they are below a certain threshold in terms of their Burt’s constraint value, that is to say, individuals can only update their opinion if they fall below a threshold, otherwise their opinions remain the same. An aggregate score of the polarization, the alignment, and the extremeness of all individuals in the network is taken at each time step. These scores allow for the simulation to explore the followings impact on opinion dynamics : 
1) The type of the network 
2) The amount of individuals
3) The amount of opinions
4) The individuals place within the network

1.1b: Entities, State Variables, and Scale:

  The following paragraph explains the structure of the model, in terms of the entities, their low level state-variables, and the spatial and temporal scales within the model. Each entity in the model represents an individual. Each individual is placed in a position within a certain type of network, thus each individual has their own position. From this position, each individual is given a Burt’s Score, to identify the structural hole value they have within the network. In terms of temporality, each individual agent is able to update their opinions during each step, as long as their Burt’s Score is less than the threshold value. Thus, certain individual agents experience time at each time step, while others do not. Each agent is also given their own opinions, which are represented as numerical values. Lastly, each agent is given an equanimity score, also a numerical value, which represents how well liked they are by other individuals within the network. For this model, these agent entities that represent individuals do not have many low-level state variables. This distinguishable characteristic can be explored in further development of the agent-based model for future research.

1.1c: Process Overview and Scheduling: 

Furthermore, it is important to consider the processes and scheduling that occurs within the agent-based model. First, a network is created and subsequently the structural hole value is calculated based upon the positions of nodes within the network. For the initialization of the agents, which will be discussed in further detail below in the initialization section, an opinion matrix which contains the varied opinions is created which is based upon the number of agents and the number of opinions. This is also accompanied by the creation of an epsilon matrix that contains the varied equanimity scores. The two matrix creations allow for the following to happen in which each agent is placed in a node in the network and given opinion values, an epsilon value (equanimity score), their Burt score and the threshold value which will determine if they can update their opinions in the interaction with other agents each step. After the initiation of the model, in terms of scheduling for each step, if an agent has a Burt score less than the threshold, they choose at random a neighbor with a distance of one from their position in the network and updates their opinions based upon their neighbors opinions and the epsilon value (equanimity score). If an agent has a Burt score equal or greater than the threshold value given by the modeler to the model,  then the agent keeps their initial opinion. At each step, it is determined the aggregate elements of polarization, alignment, and extremeness of all the agents within the network.


Design Concepts of the Model

1.2a: Basic Principles:

The model is designed to investigate whether an individual's position in a network can affect the polarization and alignment of the group with the opinion dynamics.  The design of the model is heavily based upon the previous work of the Weighted Balanced Model (Scweighofor et al., 2020), in which Balance Theory in opinion dynamics is updated to include not just if an individual is liked or disliked, but the extent to which someone is liked or disliked.  In Heider’s (1946) original work on Balance Theory, an individual’s attitude can have negative or positive valence regarding objects, people, ideas, and events. Individuals attempt to increase the balance within themselves by having similar views on objects with those that they have positive attitudes towards, and reversely strive for balance by having the opposite opinion on attitudes towards objects that individuals they have negative attitudes towards. Cartwright and Harray (1956) extend this to include social networks and make the specification that an individual i, an alter j,  and an object d can only be in balance if each relation between i, j, and d is the product of the signs(positive or negative) of the other two’s relation. The Weighted Balance Model extends this by not making the relations just positive or negative represented in a binary by 1 and -1, instead the relations are weighted with values between 1 and -1. They further incorporate the concept of issue constraint by adding the possibility of equanimity and a certain level of equanimity that individuals may have within the group. However, what is lacking in the Weighted Balance Model is the inclusion of networks as the Weighted Balance Model has individuals interacting with all others at random and are thus not embedded in networks. 
  
Therefore, we extend the Weighted Balance Model to include networks and to also discover how the position individuals may have in the network can play a role. This is based on the work of Burt’s Structural Holes (2018), in which individuals who sit in between two different clusters of networks may have more social capital as they have the ability to control the information between the two groups. Thus, we test this theory in the Weighted Balance Model by not allowing certain individuals with a Burt’s Constraint (2004) value to update their opinions. Importantly, we did not remove these nodes from the network, as that would only lead to testing of a different type of network and would not reveal much about the original network, opinion flow, and the social capital an individual may have from their position in the network. Instead, we chose to keep the individuals above a certain threshold of 1-Burt’s Constraint from updating their opinions. This is important as it stops those above this threshold from spreading their opinions and importantly, they cannot update their opinion based upon one cluster and transfer that updated opinion to another cluster. Thus, the basic principle of the agent-based model is one in which the Weighted Based Model is updated to include networks, and explores the notion that those within certain positions in the network cannot update their opinions nor spread the opinions from one cluster to another. The notions revealed from this exploration demonstrates that when below a certain threshold, that of .75, polarization in the network can remain. However, above this threshold, the agents who have a higher structural hole value can decrease the amount of polarization and alignment in the network. This demonstrates the social capital, in terms of Burt’s structural hole’s, and the ability for the agents at the micro level, in terms of updating or not updating opinions, to lead to macro level outcomes. 

1.2b Emergence:

The emergence of alignment and polarization occurs in three different types of networks dependent on the set threshold of the Burt Constraint, as well as the level of equanimity individuals have in the network. These can be adjusted in the model and it's visualizition. 
 
1.2c Adaption:

Agents adept their opinions based upon the Weighted Balance Theory. More specifically, as discussed above, agents adept their opinions based on their interpersonal attitudes of other agents' opinions they interact with as well as their level of equanimity. 

1.2d Objectives:

Agents' objective is to increase the cognitive balance based upon their interactions with others and their opinions. 

1.2e Learning:

There is no learning in this model.

1.2f Predicting:

There is no predicting in this model.

1.2g Sensing:

Agents adept their opinions based upon those within 1 degree to them in the network. As addressed in Grennoveter’s Strength of Weak Ties (1973), if an individual A has a strong tie with individual B, and individual A also has a strong tie with individual C, individual B and C will either have a connection or there will emerge a connection between the two. Thus, for this reason, agents adept their opinion, not only based upon the weighted balanced model,  but with individuals with whom they sense are a distance of one in the network. This is important as it demonstrates that the network is static and the network is not dynamic. It is only the agents that adapt their opinions that can be sensed. 

1.2h Interaction:

At each time step, only agents with below the set threshold value of the Burt Constraint score interact with other agents to update their opinions. First, agents choose at random a neighbor and then they compute the updated balanced opinion. While the code in Table 1 describes both the interaction in the step function, the following code in Table 2 describes the helper functions that determine the relationship and the updated balance opinion values. Then, each agent simultaneously updates their opinion, either with the newly computed opinion or with their original opinion. The simultaneous update is chosen as if not, who goes first may have an impact on the outcomes of the model (Comer, 2014). The purpose of setting a threshold value which determines which agents can update their opinion is not arbitrary, but is used as a test for the mechanism at play within the model. The particular values chosen do not have a strong theory behind them and further research should be explored in determining these values. In the code below, the alpha value as well as the e value are both stylized facts that remain from the original Weighted Balanced Model and they do not appear to have any theory or empirical evidence in the choice of these values. This code is mainly that of the original model, although some was missing and was updated in attempts to replicate their meaning from the original paper. 

1.2i Stochasticity:

Stochasticity plays a role in the model in the creation of the three types of networks and within the interaction of the agents. In terms of the creation of the networks, the three types of networks the traits of Barabasi’s preferential attachment network, Watt-Strogatz’s small world network, or Connected Watts-Strogatz small world connected network, but are randomly generated in python at the initiation of the model. The interaction between agents also contains a random element as some agents will only pick one agent at random with whom they will use to calculate their updated opinion. 

1.2j Collectives:

There are no collectives in the model as agents are not grouped socially. 

1.2k Observations:

The data collected from the model is the agent's opinions at each time step as well as the aggregate level of polarization, alignment, and extremeness. 

Details of the model

1.3a Initialization:

The model is initialized with one of three types of networks as well as with placing individuals within the network. The agents are each given opinions, the equanimity score, and their Burt Score, which is one minus their Burt constraint. The initial parameters of the model are then the type of network, the threshold value that will determine the individuals who are below a certain Burt Score, the number of agents in the network, as well as the number of opinions. The code above in Table 1 demonstrates this initiation phase in action. 

1.3b Input:

The model does not use any external inputs.

1.3c Submodels:

The model does contain any submodels.
