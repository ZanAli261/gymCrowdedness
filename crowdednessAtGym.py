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
print('Date range of measurements: ' + str(min(gym.date)) + ' to ' + str(min(gym.date)))

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
gym['day_of_month'] = gym.date.dt.day

gymDOW = gym.groupby(['day_of_week_desc'], as_index=False).mean()

#Do people go to the gym everyday of the week?
sns.barplot(x='day_of_week_desc', y='number_people',
            data=gymDOW, order=['Mon','Tues','Weds','Thurs','Fri','Sat','Sun'])
plt.title('Average Number of People at Gym Per Day')
plt.xlabel('Day of the Week')
plt.ylabel('Avg # People per Day')

#What effect does the temperature have on gym attendance?
##focus on 2016 only, plot graph of each month

plotSpot = [1,2,3,4]
qtrBegin = [1,4,7,10]
for i in range(0,4):
    plt.subplot(2, 2, plotSpot[i])
    x = gym.loc[(gym.date.dt.year == 2016) & (gym.date.dt.month == qtrBegin[i]), 'date']
    yTemp = gym.loc[(gym.date.dt.year == 2016) & (gym.date.dt.month == qtrBegin[i]), 'temperature']
    yPeop = gym.loc[(gym.date.dt.year == 2016) & (gym.date.dt.month == qtrBegin[i]), 'number_people']
    sns.lineplot(x=x,y=yTemp, alpha=0.5)
    sns.lineplot(x=x,y=yPeop, alpha=0.5)
    plt.ylabel('Temp and People Count')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.title('Month' + str(qtrBegin[i]))
    
plt.legend(['temp', 'peop'], loc=3)
plt.tight_layout()
plt.show()


#Are there specific hours where the gym is less crowded?




#Apply ML to see if features can predict gym attendance
