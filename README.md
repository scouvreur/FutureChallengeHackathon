#   Future Challenge Hackathon: Routing balloons in weather conditions
##  By Alibaba Cloud and the Met Office
### Jan 20th 2018

Highest ranking achieved was 100 / 61792 on [(26th January 2018)](https://tianchi.aliyun.com/competition/rankingList.htm?spm=5176.11165261.5678.4.7c1a54e0HN1zFj&raceId=231622)

The dataset was provided by [Alibaba Cloud](bit.ly/AliCloudHackathon) and the Met Office.

The training data set contained 10 forecasts of wind speed from Met Office models at each location on a geographical 548 x 421 grid map of the UK. This data was available for 18 hours during a day - between 03:00 and 21:00.

The goal of the project was to route the balloons such that they would not crash under excessive windspeeds (15m/s was the threshold).

The tutorial code was provided by Ruixiong from Alibaba's Big Data Analytics Division. The remainder of the code was developed by Paul Diemoz, Ilan Pillemer, Ann Cn and myself (participated in the Hackathon leaderboard as team galileo, and in the general leaderboard as team grey-london).

Sample plots of windspeeds at the same time of day with different forecasting models (differences are subtle):

* [Day 2 Hour 3 Model 2](windMaps/Day-2-Hour-3-Model-2.pdf "Day 2 Hour 3 Model 2")
* [Day 2 Hour 3 Model 5](windMaps/Day-2-Hour-3-Model-5.pdf "Day 2 Hour 3 Model 5")
* [Day 2 Hour 3 Model 9](windMaps/Day-2-Hour-3-Model-9.pdf "Day 2 Hour 3 Model 9")
* [Day 2 Hour 3 Linear Regression Model](windMaps/Day-2-Hour-3-Model-LinReg.pdf "Day 2 Hour 3 Linear Regression Model")