# Usage as a content provider

This documentation is for content providers that want to use SIREN to investigate the recommender effects (in terms of diversity) specific to the provider's own characteristics (users, articles). 

First of all, when running SIREN you will be first prompted with the following interface:

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/interface.png?raw=true "Interface")

The interface gives access to certain recommendation/article/user variables, such as the recommendation algorithms to be investigated, the amount of simulated users, the distribution of topics among the simulated articles and so on. The default settings correspond to a typical news environment (for the reasoning behind the default settings please refer to the corresponding paper). Now, let us see how to set up and run a simulation as a content provider.

#### User and article settings
In order parameterize SIREN as a content provider, you should at least know:
1. The amount of active readers that visit your news website per day. Active readers are those that receive personalized recommendations and actively read content almost every day. 
2. The amount of articles in average your readers read per day.
3. The number of new articles that you publish per day.
4. The distribution of topics among articles, e.g, how much more politics-related articles you publish compared to other topics (sports, technology, business, entertainment). 
5. How much you promote certain topics on the headlines. The default SIREN settings correspond to a content provider with a focus on politics (e.g. CNN). However, your news outlet might be more focused on entertainment or sports.

Providing this information to SIREN is easy. The answers to questions 1 and 2 should allow you to set the "User settings" (right-most form) on the interface:

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/users.png?raw=true "users")

The answers to question 3,4 and 5 should allow you to set the "Article settings" on the interface (via the tabs, middle form):

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/articles.png?raw=true "articles")

#### Recommendation settings

The left-most form controls the recommendation settings. We have identified two possible SIREN scenarios for content providers with regard to the recommendations:
1. You (as content-provider) are interested in employing a recommendation strategy and want to know which one performs better in terms of diversity.
2. You are already using a recommendation strategy and want to evaluate its diversity performance or compare it to other strategies.

For scenario (1) you need to at least know:
1. Your current recommendation strategy/algorithm. The most typical strategies are ItemKNN, UserKNN, or ItemAttributeKNN.
2. The amount of recommended articles that your active readers receive daily. These recommendations can be either in the form of personalized emails to your users or articles in an allocated frame in your website. 

For scenario (2) you just need to have an indication of how many recommended articles you want your readers to receive daily. Regarding the recommendation algorithms, you can play around with all of them. But again, the most typical strategies are ItemKNN, UserKNN, or ItemAttributeKNN.

The remaining variables in the "Recommendation settings" include the prominence of the recommended articles. The current prominence value corresponds to a default scenario where one out of five article-reads come from a recommendation. "Days" corresponds to the number of simulation iterations that will run per recommendation algorithm. The more the "Days", the more time it will take for the simulation to complete. At the same time, for the recommenders to have an effect on diversity, a number of "Days" is required. We suggest a number from 30 to 50.

This information should be enough to set the "Recommendation settings" (left-most form):


![Alt text](https://github.com/dbountouridis/siren/blob/master/images/recommendations.png?raw=true "articles")


#### Simulation

By clicking on "Start" (and after adjusting the settings) for the recommendations/articles/users, the simulation will initiate. The simulation firsts run a "Control" period where the simulated users read articles without recommendations. This allows SIREN to create user-profiles for each reader (what they like to read).

Later, at each iteration of the simulation, three figures are plotted as seen in the figure below: long-tail diversity, unexpectedness diversity and the distribution of topics among the read articles so far:
1. Long-tail diversity informs you on whether unpopular items are read by the users.
2. Unexpectedness diversity informs you on whether users read surprising content (not agreeing with their previous reading behavior).
3. The distribution of topics informs you about the topics that users read.

Due to the readers possibly adjusting their preferences, the recommendation algorithms might have different effects on the diversity and topic distribution over time.

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/figures.png?raw=true "Figures")

After the simulation is complete, a pop-up will appear. You will also be informed on the folder/location of where the raw data is stored for future analysis.


