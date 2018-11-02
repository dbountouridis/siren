# Usage as a content provider

This documentation is for content providers that want to use SIREN to investigate the recommender effects (in terms of diversity) specific to the provider's own characteristics (users, articles). 

First of all, when running SIREN you will be first prompted with the following interface:

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/interface.png?raw=true "Interface")

The interface gives access to certain recommendation/article/user variables, such as the recommendation algorithms to be investigated, the amount of simulated users, the distribution of topics among the simulated articles and so on. The default settings correspond to a typical news environment (for the reasoning behind the default settings please refer to the corresponding paper). Now, let us see how to set up and run a simulation as a content provider.

#### User settings
We start with the user settings. As a content provider, you should at least know:
1. How many active readers visit your news website per day. Active readers are those that receive personalized recommendations and actively read content almost every day. 
2. How many articles in average your readers read per day.

It should be noted that for a high  number of readers (e.g., more than 500), the simulation will take a long time to conclude. 



#### Recommendation settings

The left-most form controls the recommendation settings. The most important variable is the set of recommendation [MyMediaLite](www.mymedialite.net/) algorithms that will be run on the simulation. Other variables include the number and the prominence of the recommended articles presented to the simulated users. "Days" corresponds to the number of simulation iterations that will run per recommendation algorithm.

|          | GUI Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Number of recommended articles per user per iteration.           |
|  Recommenders|       +     |  Factor by which distance decreases for recommended articles (salience)       |
|          |            |  Ranking-based decay of recommender salience     |
|          |       +     |  Number of simulation iterations per recommender      |


####  Article settings

The middle form controls the articles settings i.e., the content-provider's intent: what the content-provider wants users to read. The variables include the number of new articles published per day, the distribution of topics among articles and the prominence of each topic (how likely it is for a topic to appear in the headlines). The default settings correspond to a content-provider with focus on politics. 

|          | GUI Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Total number of articles (number of iterations x articles per day)           |
|  Articles|       +     |  Percentage of articles added per day/iteration per topic      |
|          |       +     |  Awareness: initial article prominence per topic     |
|          |            |  Awareness weight placed on prominent versus neighborhood articles      |



#### Simulation

By clicking on "Start", and after adjusting the settings for the recommendations/articles/users, the simulation will initiate. In order to deal with the cold-start problem, the simulation firsts run a "Control" period where the simulated users read articles without recommendations. The state of the simulation (e.g., reading history) after the "Control" is used as the common starting point for the all the recommendation algorithms.

At each iteration of the simulation, three figures are plotted as seen in the figure below: long-tail diversity, unexpectedness diversity and the distribution of topics among the read articles so far. Due to the evolving user preferences, the recommendation algorithms might have different effects on the diversity and topic distribution.

![Alt text](images/figures.png?raw=true "Figures")



