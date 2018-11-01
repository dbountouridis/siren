# Siren: A Simulation Framework for Understanding the Effects of Recommender Systems in Online News Environments


A simulation framework  for the visualization and analysis of the effects of different recommenders systems. This simulation draws mainly on the work of Fleder and Hosanagar (2017). To account for the specificities of news consumption, it includes both users preferences and editorial priming as they interact in a news-webpage context. The simulation models two main components: users (preferences and behavior) and items (article content, publishing habits). Users interact with items and are recommended items based on their interaction.

## Usage

After all the dependencies have been installed, run SIREN:

```python
python3 simulation.py
```
An interface similar to the one below will be presented:
![Alt text](images/interface.png?raw=true "Interface")

## Overview

Our model assumes that there are |U| users (i.e. readers) and |T| items (i.e. articles) placed in an 2-dimensional attribute space. Each iteration of the simulation corresponds to a news cycle (e.g., a day). Readers are aware of: 
1. articles in their proximity, corresponding to preferred/sought out topics (via search or navigation bars)
2. promoted articles by the editors (as they appear on the news website)
3. personalized recommended articles

At each iteration, each user decides to read a number of unique articles from those they are aware of. At the end of each iteration, the users’ preferences are updated. The article pool and the personalized recommendations are also updated at every iteration, while each article has a limited life-span.

Under this model, we identify three main interacting components that SIREN’s interface gives control over: 
1. the articles (that translate to specific publishing habits)
2. the users (the readers’ preferences and reading behavior) 
3. the recommendations (articles promoted to each user). 


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
|          |            |  Amount of articles read per iteration per user (session size)       |

|          | GUI Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Number of recommended articles per user per iteration.           |
|  Recommenders|       +     |  Factor by which distance decreases for recommended articles (salience)       |
|          |            |  Ranking-based decay of recommender salience     |
|          |       +     |  Number of simulation iterations per recommender      |

|          | GUI Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Total number of articles (number of iterations x articles per day)           |
|  Articles|       +     |  Percentage of articles added per day/iteration per topic      |
|          |       +     |  Awareness: initial article prominence per topic     |
|          |            |  Awareness weight placed on prominent versus neighborhood articles      |




## Citation

If you use this code, please cite the following publication:

D. Bountouridis, J. Harambam, M. Makhortykh, M. Marrero, N. Tintarev, C. Hauff (2018). _SIREN: A Simulation Framework for Understanding the Effects of Recommender Systems in Online News Environments._ ACM Conference on Fairness, Accountability, and Transparency
