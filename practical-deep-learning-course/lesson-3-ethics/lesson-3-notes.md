# Notes from Lesson 3: Ethics
* examples of data ethics
  * recourse and accountability
  * feedback loops: when a model controlls the next round of data received eventually resulting in data flawed by the system itself
    * when an algorithm has a metric to optimize it will do everything it can to optimize that number
  * bias
    * ex: Googling historically Black names results in advertisements suggesting the person had a criminal record, while white names result in neutral ads
      * might lead to a potential employer turning down an applicant due to suspicion of a criminal record
    * historical bias: a fundamental, structural issue with the first step of the data generation process
      * people/processes/society is biased
      * can exist even given perfect sampling and feature selection
      * IBM computer vision classifier initially had 34.7% error rate for darker females vs. 0.3% for lighter males
        * a year later the models produced almost similar results for both
        * developers initially failed to utilize datasets containing enough darker faces or test the product with darker faces

## Questionnare
1. Does ethics provide a list of "right answers"?
  * no, ethics is complicated/context-dependent
2. How can working with people of different backgrounds help when considering ethical questions?
  * by providing different perspectives of input
3. What was the role of IBM in Nazi Germany? Why did the company participate as it did? Why did the workers participate?
  * IBM provided the Nazis with data products to track the extermination of Jews and other groups at a massive scale.
  * IBM President Thomas Watson approved the special IBM machines to help organize the deportation of Polish Jews in 1939
  * Watson was awarded a "Service to the Reich" medal by Adolf Hitler
  * IBM provided regular training/maintenance at the concentration camps
  * IBM set up a punch card system to track/label the people that were killed through the Holocaust system
  * the company/workers participated due to monetary incentives, weighing the means rather than the ends
4. What was the role of the first person jailed in the Volkswagen diesel scandal?
  * an engineer, James Liang, who was following orders
5. What was the problem with a database of suspected gang members maintained by California law enforcement officials?
  * the database was full of errors including 42 babies added to the database less than a year old that had been marked as suspected gang members
6. Why did YouTube's recommendation algorithm recommend videos of partially clothed children to pedophiles, even though no employee at Google had programmed this feature?
  * The algorithm expeienced feedback loops which essentially curated playlists for pedophiles.
7. What are the problems with the centrality of metrics?
  * The centrality of metrics in a financially important system leads to an algorithm optimizing a metric by any means
8. Why did Meetup.com not include gender in its recommendation system for tech meetups?
  * Men expressed more interest than women in tech meetups, so including gender would result in the algorithm recommending less tech meetups to women which would result in less women finding out and attending the tech meetups, resulting in the algorithm suggesting even fewer tech meetups for women 
  * self-reinforcing feedback loop
9. What are the six types of bias in machine learning, according to Suresh and Guttag?
  * historical, representation, measurement, aggregation, evaluation, and deployment bias 
10. Give two examples of historical race bias in the US.
  * an all-white jury was 16 % more likely to convict a Black defendant than a white one
  * doctors less likely to recommend helpful cardiac procedures to Black patients than white patients with identical files
11. Where are most images in ImageNet from?
12. In the paper [Does Machine Learning Automate Moral Hazard and Error](https://scholar.harvard.edu/files/sendhil/files/aer.p20171084.pdf) why is sinusitis found to be predictive of a stroke?
13. What is representation bias?
14. How are machines and people different, in terms of their use for making decisions?
15. Is disinformation the same as "fake news"?
16. Why is disinformation through auto-generated text a particularly significant issue?
17. What are the five ethical lenses described by the Markkula Center?
18. Where is policy an appropriate tool for addressing data ethics issues?
