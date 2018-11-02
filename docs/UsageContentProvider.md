# Usage as a content provider

This documentation is for content providers that want to use SIREN to investigate the recommender effects (in terms of diversity) specific to the provider's own characteristics (users, articles). 

First of all, when running SIREN you will be first prompted with the following interface:

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/interface.png?raw=true "Interface")

The interface gives access to certain recommendation/article/user variables, such as the recommendation algorithms to be investigated, the amount of simulated users, the distribution of topics among the simulated articles and so on. The default settings correspond to a typical news environment (for the reasoning behind the default settings please refer to the corresponding paper). Now, let us see how to set up and run a simulation as a content provider.

#### User and article settings
In order to make use of SIREN as a content provider, you should at least know:
1. The amount of active readers that visit your news website per day. Active readers are those that receive personalized recommendations and actively read content almost every day. 
2. The amount of articles in average your readers read per day.
3. The number of new articles that you publish per day.
4. The distribution of topics among articles, e.g, how much more politics-related articles you publish compared to other topics (sports, technology, business, entertainment). 
5. How much you promote certain topics on the headlines. The default SIREN settings correspond to a content provider with a focus on politics (e.g. CNN). However, your news outlet might be more focused on entertainment or sports.

Providing this information to SIREN is easy. The answers to questions 1 and 2 should allow you to set the user variables on the interface:
![Alt text](https://github.com/dbountouridis/siren/blob/master/images/users.png?raw=true "users")

The answers to question 3,4 and 5 should allow you to set the article variables on the interface (via the tabs):
![Alt text](https://github.com/dbountouridis/siren/blob/master/images/articles.png?raw=true "articles")

#### Recommendation settings

The left-most form controls the recommendation settings. The settings can be adjusted to correspond to a hypothetical scenario or your outlet's current recommendation strategy. For the latter you need to at least know:
1. Your current recommendation strategy/algorithm. The most typical strategies are ItemKNN, UserKNN, or ItemAttributeKNN.
2. The amount of recommended articles that your readers receive daily.

 The most important variable is the set of recommendation [MyMediaLite](www.mymedialite.net/) algorithms that will be run on the simulation. As a content provider you might be already using a specific algorithm that you might want to compare to others. Or you might be thinking of providing recommendations and you might be wondering which algorithm to choose. Whatever the case, SIREN lets you compare different algorithms on the same contolled setting.


 Other variables include the number and the prominence of the recommended articles presented to the simulated users. "Days" corresponds to the number of simulation iterations that will run per recommendation algorithm.


![Alt text](https://github.com/dbountouridis/siren/blob/master/images/recommendations.png?raw=true "articles")






#### Simulation

By clicking on "Start", and after adjusting the settings for the recommendations/articles/users, the simulation will initiate. In order to deal with the cold-start problem, the simulation firsts run a "Control" period where the simulated users read articles without recommendations. The state of the simulation (e.g., reading history) after the "Control" is used as the common starting point for the all the recommendation algorithms.

At each iteration of the simulation, three figures are plotted as seen in the figure below: long-tail diversity, unexpectedness diversity and the distribution of topics among the read articles so far. Due to the evolving user preferences, the recommendation algorithms might have different effects on the diversity and topic distribution.

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/figures.png?raw=true "Figures")



