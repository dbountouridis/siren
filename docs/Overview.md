# Overview

SIREN assumes that there are |U| users (i.e. readers) and |T| items (i.e. articles) placed in an 2-dimensional attribute/topic space as seen  below:

![Alt text](images/featurespace.png?raw=true "Feature space")


Each iteration of the simulation corresponds to a news cycle (e.g., a day). Readers are aware of: 
1. articles in their proximity, corresponding to preferred/sought out topics (via search or navigation bars)
2. promoted articles by the editors (as they appear on the news website)
3. personalized recommended articles

At each iteration, each user decides to read a number of unique articles from those they are aware of. At the end of each iteration, the users’ preferences are updated. The article pool and the personalized recommendations are also updated at every iteration, while each article has a limited life-span.
