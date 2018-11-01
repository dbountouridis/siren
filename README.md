# Siren: A Simulation Framework for Understanding the Effects of Recommender Systems in Online News Environments


A simulation framework  for the visualization and analysis of the effects of different recommenders systems. This simulation draws mainly on the work of Fleder and Hosanagar (2017). To account for the specificities of news consumption, it includes both users preferences and editorial priming as they interact in a news-webpage context. The simulation models two main components: users (preferences and behavior) and items (article content, publishing habits). Users interact with items and are recommended items based on their interaction.


|          | Adjustable | Description |
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

|          | Adjustable | Description |
| ---      |  :---:        | ---         |
|          |      +     |  Total number of articles (number of iterations x articles per day).           |
|  Articles|       +     |  Percentage of articles added per day/iteration per topic.       |
|          |       +     |  Awareness: initial article prominence per topic     |
|          |            |  Awareness weight placed on prominent versus neighborhood articles      |



Amount of articles read per iteration per user (session size)

## Citation

If you use this code, please cite the following publication:

D. Bountouridis, J. Harambam, M. Makhortykh, M. Marrero, N. Tintarev, C. Hauff (2018). _SIREN: A Simulation Framework for Understanding the Effects of Recommender Systems in Online News Environments._ ACM Conference on Fairness, Accountability, and Transparency
