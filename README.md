# Siren: A Simulation Framework for Understanding the Effects of Recommender Systems in Online News Environments


A simulation framework for the visualization and analysis of the effects of different recommenders systems in an online news environment. This simulation draws mainly on the e-commerce simulation work of Fleder and Hosanagar (2017). However, SIREN also accounts for the specificities of news consumption, such as evolving users preferences and editorial priming (promoted articles on a news website). 

SIREN's interface currently offers recommendations based the [MyMediaLite](www.mymedialite.net/) toolbox and visualizations for two diversity metrics (long-tail and unexpectedness). SIREN can be used by content providers (news outlets) to investigate which recommendation strategy fits better their diversity needs. At the same time, SIREN's code can be adapted/expanded by researchers to analyse various recommender effects in a news environment.


## Overview

SIREN assumes that there are |U| users (i.e. readers) and |T| items (i.e. articles) placed in an 2-dimensional attribute space. Each iteration of the simulation corresponds to a news cycle (e.g., a day). Readers are aware of: 
1. articles in their proximity, corresponding to preferred/sought out topics (via search or navigation bars)
2. promoted articles by the editors (as they appear on the news website)
3. personalized recommended articles

At each iteration, each user decides to read a number of unique articles from those they are aware of. At the end of each iteration, the users’ preferences are updated. The article pool and the personalized recommendations are also updated at every iteration, while each article has a limited life-span.



## 1. Usage

After all the dependencies have been installed, run SIREN:

```python
python3 simulation.py
```
An interface similar to the one below will be presented:

![Alt text](images/interface.png?raw=true "Interface")

The interface gives access to certain recommendation/article/user variables, such as the recommendation algorithms to be investigated, the amount of simulated users, the distribution of topics among the simulated articles and so on. The default settings correspond to a typical news environment (for the reasoning behind the default settings please refer to the corresponding paper).



#### 1.1 Recommendation settings

The left-most form controls the recommendation settings. The most important variable is the set of recommendation [MyMediaLite](www.mymedialite.net/) algorithms that will be run on the simulation. Other variables include the number and the prominence of the recommended articles presented to the simulated users. "Days" corresponds to the number of simulation iterations that will run per recommendation algorithm.

|          | GUI Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Number of recommended articles per user per iteration.           |
|  Recommenders|       +     |  Factor by which distance decreases for recommended articles (salience)       |
|          |            |  Ranking-based decay of recommender salience     |
|          |       +     |  Number of simulation iterations per recommender      |


#### 1.2 Article settings

The middle form controls the articles settings i.e., the content-provider's intent: what the content-provider wants users to read. The variables include the number of new articles published per day, the distribution of topics among articles and the prominence of each topic (how likely it is for a topic to appear in the headlines). The default settings correspond to a content-provider with focus on politics. 

|          | GUI Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Total number of articles (number of iterations x articles per day)           |
|  Articles|       +     |  Percentage of articles added per day/iteration per topic      |
|          |       +     |  Awareness: initial article prominence per topic     |
|          |            |  Awareness weight placed on prominent versus neighborhood articles      |

#### 1.3 User settings
The right-most form controls the users (readers) settings, including the number of active daily readers, the amount of articles they read and their focus on sought out vs editorially-promoted articles. The default settings correspond to an average reading behavior of online news. Other variables related to the users' behavior can only be accessed via SIREN's code. It should be noted that for a high  number of users (e.g., more than 500), the simulation will take a long time to conclude. 


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

#### 1.4 Simulation

By clicking on "Start", and after adjusting the settings for the recommendations/articles/users, the simulation will initiate. In order to deal with the cold-start problem, the simulation firsts run a "Control" period where the simulated users read articles without recommendations. The state of the simulation (e.g., reading history) after the "Control" is used as the common starting point for the all the recommendation algorithms.

At each iteration of the simulation three figures are plotted as seen in the figure below: long-tail diversity, unexpectedness diversity and the distribution of topics among the read articles so far.

![Alt text](images/figures.png?raw=true "Figures")










## Citation

If you use this code, please cite the following publication:

D. Bountouridis, J. Harambam, M. Makhortykh, M. Marrero, N. Tintarev, C. Hauff (2018). _SIREN: A Simulation Framework for Understanding the Effects of Recommender Systems in Online News Environments._ ACM Conference on Fairness, Accountability, and Transparency
