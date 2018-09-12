# crowededness at campus gym dataset from kaggle
# from dataset overview, count of people was taken every 10 minutes during a school year

import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir('C:\\Data')

#explore data
gym = pd.read_csv('crowdedness-at-the-campus-gym.csv')
#print(gym.head())
#print(gym.info())
#print(gym.describe())

#clean data
gym['date'] = pd.to_datetime(gym['date'])

def label_days(row):
    if row['day_of_week'] == 0:
        return 'Mon'
    elif row['day_of_week'] == 1:
        return 'Tues'
    elif row['day_of_week'] == 2:
        return 'Weds'
    elif row['day_of_week'] == 3:
        return 'Thurs'
    elif row['day_of_week'] == 4:
        return 'Fri'
    elif row['day_of_week'] == 5:
        return 'Sat'
    elif row['day_of_week'] == 6:
        return 'Sun'

gym['day_of_week_desc'] = gym.apply(label_days,axis=1)

gymDOW = gym.groupby(['day_of_week_desc'], as_index=False).mean()

plt.subplot(2, 1, 1)
sns.barplot(x='day_of_week_desc', y='number_people',
            data=gymDOW, order=['Mon','Tues','Weds','Thurs','Fri','Sat','Sun'])
plt.title('Average Number of People at Gym Per Day')
plt.xlabel('Day of the Week')
plt.ylabel('Avg # People per Day')

#What effect does the temperature have on gym attendance?

plt.subplot(2, 1, 2)
sns.scatterplot(x='date',y='temperature', data=gym)
#sns.lineplot(x='date',y='number_people', data=gym)
plt.ylabel('Temperature')
plt.xlabel('Date')
plt.show()


#Are there specific hours where the gym is less crowded?




#Apply ML to see if features can predict gym attendance
