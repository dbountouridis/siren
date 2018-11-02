# Overview

Most related works interested in the recommender effects consider the interaction of users/readers with the recommendations in isolation. However, such recommendations typically appear in the context of an online news environment (as seen in the screenshot from Washington Post below). Therefore, the reader's interaction with the recommendations is not completely straightforward: editors promote certain articles more than others, the recommendations have a certain visual salience, readers might change preferences over time and so on. SIREN tries to accommodate for a number of those issues such that the observed recommender effects have a better correspondence to reality.

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/washingtonpost.png?raw=true "washingtopost")

### Conceptual model

SIREN assumes a typical news consumption scenario where users/readers distribute their reading time between preferred/sought out articles (e.g., a sports fan would seek out the sports news), editorially-promoted articles (e.g., stories appearing in the headlines) and recommended articles. 

SIREN assumes that there are |U| users (i.e. readers) and |T| items (i.e. articles) placed in an 2-dimensional attribute/topic space as seen  below. Users are represented as crosses "+" while articles as circles. The users' preferences and articles' content are defined by their position in relation to the topical centers (politics, sports, technology, entertainment, business). For example, users close to the "politics" center are more interested in politics than other topics. At the same time, each article is promoted to a certain degree by the news editors (visualized as the circles' size).

![Alt text](https://github.com/dbountouridis/siren/blob/master/images/featurespace.png?raw=true "Feature space")


Each iteration of the simulation corresponds to a news cycle (e.g., a day). At each iteration readers are aware of: 
1. articles in their proximity, corresponding to preferred/sought out topics
2. promoted articles by the editors (as they appear on the news website)
3. personalized recommended articles

At each iteration, each user decides to read a number of unique articles from those they are aware of. At the end of each iteration, the users’ preferences are updated. The article pool and the personalized recommendations are also updated at every iteration, while each article has a limited life-span.
