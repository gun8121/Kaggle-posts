The primary goal of **machine learning is prediction**, while, on the contrary, the main objective of **statistics is to provide an explanation**. However, as we all know, they are much more integrated.

I found the following Venn diagram from the SAS Institute helpful in understanding the boundaries of AI, Machine Learning, and Statistics.

![Data Mining Venn Diagram](https://blogs.sas.com/content/subconsciousmusings/files/2014/08/data-mining-Venn-diagram-300x184.png)

### How is Statistics different?
Statistics emphasizes the reliability of models and pursues simplicity over complexity. It focuses on interpreting the impact of each variable, assumptions about samples extracted from the population, and statistical fitness. In other words, traditional statistics aims to verify hypotheses through probability and interpret data through estimated models. 

From this perspective, traditional statistics may seem somewhat naive and not entirely aligned with what we truly want in data science. However, **machine learning and data science are built on the probabilistic theories and principles of minimizing errors in statistics.**

The table below is an effort to distinguish between the two.

|   | Statistics | Machine Learning |
|----|----|---|
| Approach Method | Understanding the process of data generation through a probability variable. | Creating an algorithm model |
| Basis | Math, Theory | Nonlinear data fitting |
| Objective | Hypothesis testing, interpretation of phenomena | Improvement of prediction accuracy |
| Variables(Dimensions) | Fewer than 10 prime variables | Multidimensional variables |
| Usage | Interpretation of phenomena or event through past and present data | Prediction of the future using past and present data |
| Approach Direction | Hypothesis --> Data | Data --> Hypothesis |

### To Summarize the Differences
Statistics approaches statistical testing through probability variables, while machine learning focuses on creating models applying algorithms to enhance predictive accuracy. Machine learning typically utilizes dozens to even hundreds of variables to build models. 

Due to these differences, statistics generally involves setting hypotheses first, then checking the data, while machine learning tends to derive and validate hypotheses based on the data. However, this can vary depending on the situation, so it is not an absolute distinction.



I decided to write this post because I was always confused about why everyone was talking about learning statistics. I soon realized statistics and machine learning were much more intertwined. To truly understand and analyze data, knowledge of statistical theories and testing methods was a must. It seems like this might be the difference between a developer and a data scientist.

I really liked [PANNMIE](https://www.kaggle.com/discussions/general/425291)'s notebook titles on how straight forward and easy it was to understand. So I tried a similar format.

Well, if you made it this far... Thankyou for reading my first post!
If you find any errors or inconsistencies, please let me know.
I am looking forward to growing with the community.