# Usage as a researcher

We have identified two SIREN usage scenarios for researchers:
1. Use SIREN's interface to adjust the provided parameters and perform controlled experiments. 
2. Use SIREN's code to adjust even more parameters related to recommendations/articles/user behavior.

The interface gives access to certain recommendation/article/user variables, such as the recommendation algorithms to be investigated, the amount of simulated users, the distribution of topics among the simulated articles and so on. However, some researchers might find the collection of ajustable parameters limited. As such, SIREN's code allows for a greater control on variables such as the likelihood of reading an article at the top versus the bottom of a recommendation list. It should be mentioned that the default settings correspond to a typical news environment (for the reasoning behind the default settings please refer to the corresponding paper). 

The next sections aim to briefly describe all adjustable parameters provided by SIREN's interface and code.

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/interface.png?raw=true "Interface")


#### Recommendation settings

The left-most form controls the recommendation settings. The most important variable is the set of recommendation [MyMediaLite](www.mymedialite.net/) algorithms that will be run on the simulation. Other variables include the number and the prominence of the recommended articles presented to the simulated users. "Days" corresponds to the number of simulation iterations that will run per recommendation algorithm.

|          | GUI Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Number of recommended articles per user per iteration.           |
|  Recommenders|       +     |  Factor by which distance decreases for recommended articles (salience)       |
|          |            |  Ranking-based decay of recommender salience     |
|          |       +     |  Number of simulation iterations per recommender      |

Inside SIREN's code, these functions and parameters are encapsulated in the "Recommendations" class. 

####  Article settings

The middle form controls the articles settings i.e., the content-provider's intent: what the content-provider wants users to read. The variables include the number of new articles published per day, the distribution of topics among articles and the prominence of each topic (how likely it is for a topic to appear in the headlines). The default settings correspond to a content-provider with focus on politics. 

|          | GUI Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Total number of articles (number of iterations x articles per day)           |
|  Articles|       +     |  Percentage of articles added per day/iteration per topic      |
|          |       +     |  Awareness: initial article prominence per topic     |
|          |            |  Awareness weight placed on prominent versus neighborhood articles      |

Inside SIREN's code, these functions and parameters are encapsulated in the "Items" class. 

#### User settings
The right-most form controls the users (readers) settings, including the number of active daily readers, the amount of articles they read and their focus on sought out vs editorially-promoted articles. The default settings correspond to an average reading behavior of online news. It should be mentioned, that most of the user settings are only accessible through the code itself.


|          | GUI Adjustable| Description |
| ---      |  :---:        | ---         |
|          |      +     |  Total number of active, daily users/readers.           |
|          |            |  Awareness decay with distance       |
|          |            |  Awareness decay with article prominence      |
|          |      +      |  Awareness weight placed on prominent versus neighborhood articles      |
|     Users     |            |  Maximum size of awareness pool      |
|          |            |  Choice model: the user’s sensitivity to distance on the map      |
|          |            |  User-drift: user’s sensitivity to distance on the map      |
|          |            |  User-drift: distance covered between the article and user       |
|          |       +     |  Amount of articles read per iteration per user (session size)       |

Inside SIREN's code, these functions and parameters are encapsulated in the "Users" class. 

#### Simulation

In order to deal with the cold-start problem, the simulation firsts run a "Control" period where the simulated users read articles without recommendations. The state of the simulation (e.g., reading history) after the "Control" is used as the common starting point for the all the recommendation algorithms.

```
                  Reading history    /---->  Recommendation algorithm 1
Control period => User preferences ------->  Recommendation algorithm ...
                  Articles           \---->  Recommendation algorithm n

```

At each iteration of the simulation, SIREN computes three metrics to be plotted as seen in the figure below: long-tail diversity, unexpectedness diversity and the distribution of topics among the read articles so far. 

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/figures.png?raw=true "Figures")

Inside SIREN's code, these functions and parameters are encapsulated in the "SimulationGUI" class.


### Code structure

For the sake of visual inspection, we show a call graph visualization using PythonCallGraph:

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/graph.png?raw=true "Graph")





