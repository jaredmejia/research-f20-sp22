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
    * measurement bias
      * occurs when our models make mistakes because we are measuring the wrong thing, or measuring it in the wrong way, or incorporating that measurement into the model inappropriately
    * aggregation bias
      * occurs when models do not aggregate data in a way that incorporates all of the appropriate factors, or when a model does not include the necessary interaction terms, nonlinearities, or so forth
      * ex: diabetes affects people across ethnicities and genders in different ways, so people may be misdiagnosed or treated incorrectly when medical decisions are based on a model that doesn't include variables such as different ethnicities or genders
    * properties of machine learning algorithms
      * it can create feedback loops
        * small amounts of bias can rapidly increase exponentially due to feedback loops
      * machine learning can amplify bias
        * human bias can lead to larger amounts of machine learning bias
      * algorithms and humans are used differently
        * human decision makers and algorithmic decision makers are not interchangeable
      * technology is power and hence is accompanied by responsibility
  * disinformation
    * not necessarily getting someone to believe something false
    * used to create disharmony and uncertainty and to get people to give up on seeking the truth
    * creates distrust
* questions to consider
  * Should we even be doing this?
  * What bias is in the data?
  * Can the code and data be audited?
  * What are the error rates for different sub-groups?
  * What is the accuracy of a simple rule-based alternative?
  * What processes are in place to handle appeals or mistakes?
  * How diverse is the team that built it?
  * How could this be used in a way in which we did not intend it to be used for?
  * Whose interests, desires, skills, experiences, and values have we simply assumed, rather than actually consulted?
  * Who are all the stakeholders who will be directly affected by our product? How have their interests been protected? How do we know what their interests really are—have we asked? 
  * Who/which groups and individuals will be indirectly affected in significant ways?
  * Who might use this product that we didn’t expect to use it, or for purposes we didn’t initially intend?
  * Who will be directly affected by this project? Who will be indirectly affected?
  * Will the effects in aggregate likely create more good than harm, and what types of good and harm?
  * Are we thinking about all relevant types of harm/benefit (psychological, political, environmental, moral, cognitive, emotional, institutional, cultural)?
  * How might future generations be affected by this project?
  * Do the risks of harm from this project fall disproportionately on the least powerful in society? Will the benefits go disproportionately to the well-off?
  * Have we adequately considered "dual-use"?

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
  * the US and other Western countries
12. In the paper [Does Machine Learning Automate Moral Hazard and Error](https://scholar.harvard.edu/files/sendhil/files/aer.p20171084.pdf) why is sinusitis found to be predictive of a stroke?
  * because people likely to go to the doctor due to sinusitis are people who have access to healthcare, can afford their co-pay, don't experience racial or gender-based medical disinormation, and hence are the same type of people who are likely to go to the doctor when they're having a stroke
13. What is representation bias?
  * when a model predicting outcomes reflects an imbalance and ends up amplifying it
14. How are machines and people different, in terms of their use for making decisions?
  * The privileged are processed by people, whereas the poor are processed by algorithms
  * people are more likely to assume algorithms are objective or error-free (even if they're given the option of a human override)
  * algorithms are more likely to be implemented with no appeals process in place
  * algorithms are often used at scale
  * algorithmic systems are cheap
15. Is disinformation the same as "fake news"?
  * no
  * disinformation can contain parts of the truth taken out of context
16. Why is disinformation through auto-generated text a particularly significant issue?
  * because it can spread high-fidelity disinformation so effectively and very inexpensively
17. What are the five ethical lenses described by the Markkula Center?
  * The rights approach
  * The justice approach
  * the utilitarian approach
  * the common good approach
  * the virtue approach
18. Where is policy an appropriate tool for addressing data ethics issues?
