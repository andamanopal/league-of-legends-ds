# Data Science & League of Legends
(In progress...)

League of Legends is one of the most renowed MOBA games in the world with a competitive pro scene. In League of Legends or LoL, 5 players are teamed up and fight againts another team, and which team can destroy the opponent's nexus (opponent's homebase) first gets the victory. Each player can level up, upgrade skills, and buy extra items to make their character stronger within a match. The items are bought with 'Gold' which is earned from killing opponent AI-controlled minions or eliminating the enemy champion.

As all of the matches in LoL are different from each other and there are over 150 unique champions that a player can pick and play, data science can be utilized to identify the trend or key features which can lead to more successful gameplay. Moreover, a machine learning model is trained and used to predict whether a team is winning or losing based on their stats at specific point of time during the match.


## Win Prediction
Correlation plot shows that team total gold and total experience point at 10-min mark are the most important numbers, even more important than kills, that largely increase the chance of winning at the end of the match. Notes that team total exp. and average level are highly correlated to each other so we can drop 'team_average_lvl' column since it's less correlated to the match final result.

Another interesting observation is large negative correlations between deaths and team total cs, total exp, and average level. That means dying in League of Legends considerably widen the gap between both teams' levels and total creep scores. However, dying doesn't really affect the total gold at all as the disadvantaged team is more likely to lose because of lower levels. By the ways, dying a lot can still put your team behind the opponents as seen by negative correlations with every other features.

![](/images/win_corr.png)

As a binary classification problem, the following classifier models are investigated, hyperparameter-tuned, and compared. The result shows that SVC yields the highest testing accuracy of around 74.2% while other models can also perform equally good at predicting the win/lose result as well. Also, every model shows an improvement in performance and accuracy after hyperparameter tuning.

![](/images/cv_scores.png)



<!-- ## Custom RL Environment with OpenAI Gym
To properly create a custom environment for single-agent RL environment, below is the list of attributes/properties that are needed to be defined to simulate the characteristics of the environment
- Environment initial condition
- Step function (how environment is affected after an agent takes an action)
- Reward function (what is the criterion to separate good actions from bad actions)
- Observation space (to what extent our agent can see)
- Action space (to what extent our agent can do)

## Case Study : SmartAC
In this mini-project, a case study of Smart airconditioner which can automatically adjust the temperature based on the current room temperature is replicated by OpenAI Gym API as above. 

For simplicity, the possible temperature range is a range of integers from 0 to 99 degrees. The optimum temperature range where the occupants will feel most comfortable is between 23-25 degrees. After each timestep, the temperature can randomly changed due to heat transfer or any other uncontrollable external factors. The goal of our SmartAC is to be able to maintain the room temperature within the optimum range.

![](/images/smart_ac.png)

Using the reward function which gives negative score when the temperature violates the optimum range, our SmartAC is trained by PPO algorithm imported from Stable-Baselines3 and the result between trained and non-trained models from multiple episodes are shown as below histogram. It can be seen that our SmartAC has successfully learn how to better maintain the temperature to reduce the number of timesteps that the temperature isn't within the proper range.

![](/images/trained_vs_non-trained.png)

Note: Huge thanks to Nicolas Renotte for such a wonderful tutorial on Reinforcement Learning with OpenAI Gym ![Reinforcement Learning in 3 Hours | Full Course using Python](https://youtu.be/Mut_u40Sqz4) -->
