
# Commented out IPython magic to ensure Python compatibility.
import numpy as np 
import pandas as pd 
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm 
# %matplotlib inline 
import matplotlib.pyplot as plt 
import time 
import os

rating_map = {}
rating_map['AAA+'] = 1
rating_map['AAA'] = 2
rating_map['Aaa'] = 2
rating_map['#AAA'] = 2
rating_map['AAApre'] = 2
rating_map['Aa1'] = 3
rating_map['AA+'] = 3
rating_map['AA'] = 4
rating_map['Aa2'] = 4
rating_map['AA-'] = 5
rating_map['Aa3'] = 5
rating_map['A+'] = 6
rating_map['A1'] = 6
rating_map['A'] = 7
rating_map['A2'] = 7
rating_map['A-'] = 8
rating_map['A3'] = 8
rating_map['Baa1'] = 9
rating_map['BBB+'] = 9
rating_map['Baa2'] = 10
rating_map['BBB'] = 10
rating_map['Baa3'] = 11
rating_map['BBB-'] = 11
rating_map['BBB3'] = 11
rating_map['Ba1'] = 12
rating_map['BB+'] = 12
rating_map['Ba2'] = 13
rating_map['BB'] = 13
rating_map['Ba3'] = 14
rating_map['BB-'] = 14
rating_map['B+'] = 15
rating_map['B1'] = 15
rating_map['B'] = 16
rating_map['B2'] = 16
rating_map['B-'] = 17
rating_map['B3'] = 17
rating_map[''] = np.nan
rating_map['P-1'] = np.nan

mub_data['rating_num'] = mub_data['bbgRating'].map(rating_map)

## Aggregate Energy Signal data into MUB benchmark
def aggregate_data(data, item): 
    grouped_data = data.groupby(['date', item]) 
    
    stats = pd.DataFrame() 
    stats['count'] = grouped_data.count()['cusip'] 
    stats['size'] = grouped_data.sum()['cur_face'] 
    stats['mkt_value'] = grouped_data.sum()['mkt_value'] 
    stats['bmk_wgt'] = grouped_data.sum()['bmk_wgt']  
    columns = ['model_duration', 'oas', 'ytm', 'ytw','Green_Tag','rating_num'] 
#     columns.extend(datapoints) 
    for col in columns: 
        stats[col] = grouped_data.apply(lambda x: (x['bmk_wgt'] * x[col]).sum() / x['bmk_wgt'].sum()) 
    stats['next_ret_total'] = grouped_data.apply(lambda x: (x['bmk_wgt'] * x['next_ret_total']).sum() / x.loc[x['next_ret_total'].notnull(), 'bmk_wgt'].sum()) 
    stats['next_ret_xs'] = grouped_data.apply(lambda x: (x['bmk_wgt'] * x['next_ret_xs']).sum() / x.loc[x['next_ret_xs'].notnull(), 'bmk_wgt'].sum()) 
    return stats 

mub_data['Green_Tag'] = 0 
mub_data.loc[mub_data['SC_TAG`GREEN_BOND'] == 'Y','Green_Tag'] = 1

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# mub_data = mub_data.set_index(['date', 'cusip']) 
# mub_data['bmk_wgt'] = mub_data.groupby('date', group_keys=False).apply(lambda x: x['mkt_value'] / x['mkt_value'].sum()) 
# # mub_data['Green_Tag_wgt'] = mub_data['mkt_value'] * mub_data['Green_Tag'] 
# 
# ret_table = mub_data.pivot_table(index = 'date', columns = 'cusip', values = 'total_ret') 
# ret_table = ret_table.shift(-1).stack().reset_index() 
# mub_data = mub_data.reset_index() 
# mub_data_upd =  pd.merge(mub_data, ret_table, left_on = ['date', 'cusip'], right_on = ['date','cusip'], how = 'inner') 
# mub_data['next_ret_total']= mub_data_upd[0] 
# del mub_data_upd 
# bmk_ret_total = mub_data.groupby('date').apply(lambda x: (x['bmk_wgt'] * x['next_ret_total']).sum() / x['bmk_wgt'].sum()) 
# # take 5mins as a lot of issuers 
# 
# ret_table = mub_data.pivot_table(index = 'date', columns = 'cusip', values = 'xs_ret') 
# ret_table = ret_table.shift(-1).stack().reset_index()  
# mub_data = mub_data.reset_index()  
# mub_data_upd =  pd.merge(mub_data, ret_table, left_on = ['date', 'cusip'], right_on = ['date','cusip'], how = 'inner') 
# mub_data['next_ret_xs']= mub_data_upd[0] 
# del mub_data_upd 
# 
# bmk_ret_xs = mub_data.groupby('date').apply(lambda x: (x['bmk_wgt'] * x['next_ret_xs']).sum() / x['bmk_wgt'].sum()) 
# 
# mub_data['state_ticker'] = mub_data['state'] 
# # benchmark total return 
# bmk_ret = mub_data.groupby('date').apply(lambda x: (x['bmk_wgt'] * x['next_ret']).sum() / x['bmk_wgt'].sum()) 
# # take 5mins as a lot of issuers 
# 
# ticker_data = aggregate_data(mub_data, 'state_ticker') 
# 
# ticker_data.reset_index(inplace = True) 
# ticker_data['year'] = ticker_data['date'].dt.year 
# ticker_data.head()

"""## Testing State level signal data into MUB benchmark"""

plt.hist(stats_df['raw_quality'], bins = 50)
plt.gca().set(title = 'Raw Quality', ylabel = 'Frequency');

# Merging Benchmark with Energy Data 
Yearly_state_energy_data = pd.merge(ticker_data, Measure_df[cols], left_on = ['year','state_ticker'], right_on = ['Year_avail','StateCode'], how = 'left') 
Yearly_state_energy_data.drop(['StateCode', 'Year_avail'], axis = 1, inplace = True) 

Yearly_state_energy_data = pd.merge(Yearly_state_energy_data, Measure_df1[cols1], left_on = ['year','state_ticker'], right_on = ['Year_avail','StateCode'], how = 'left') 
# check_Sates(Yearly_state_energy_data) 
Yearly_state_energy_data = pd.merge(Yearly_state_energy_data, stats_df, left_on = ['date','state_ticker'], right_on = ['data_date','state'], how = 'left') 
# check_Sates(Yearly_state_energy_data) 
print(Yearly_state_energy_data.columns) 
Yearly_state_energy_data.drop(['StateCode', 'Year_avail','data_date','state'], axis = 1, inplace = True) 
print(Yearly_state_energy_data.columns)

# Getting correlation values for each state 
## checking weights 'bmk_wgt' variable 
group_data_sum = Yearly_state_energy_data.groupby(['date','state_ticker'])[Measures[i1]].mean().sort_values(ascending = False).reset_index() 
group_data_sum.pivot_table(index = 'date', columns = 'state_ticker', values = Measures[i1]).iloc[:,:10].plot(title = Measures[i1],figsize = (9,9))

# Yearly_state_energy_data.rename({'year': 'Year'}, axis = 1, inplace = True) 
# # Correlation Analysis 
cor_cols = ['bmk_wgt', 'size', 'mkt_value', 'ytm', 'model_duration', 'Green_Tag', 'raw_value', 'raw_carry', 'raw_quality','Net_change_in_Assets','Assets_Liabilities','Revenue_Expense'] 
cor_cols.append(Measures[i])
cor_cols.append(Measures[i1])
# cor_cols.extend(datapoints) 

cor_cols.append(signal) 
Correlation_Analysis_pear = Yearly_state_energy_data[cor_cols].corr(method = 'pearson') 

Correlation_Analysis_pear.style.background_gradient(cmap = 'coolwarm').set_precision(2)

def get_xs_scores(in_data, wins_upp = 3, wins_low = -3, percent = False): 
    # winsorize 
    in_data = in_data.clip(in_data.mean(1) + wins_low * in_data.std(1), in_data.mean(1) + wins_upp * in_data.std(1), axis = 0) 
    # XS z-score 
    if percent: 
        return in_data.rank(1, pct = True).subtract(in_data.rank(1, pct = True).mean(1), 0) 
    else: 
        return in_data.subtract(in_data.mean(1), 0).divide(in_data.std(1), 0) 

def construct_Ground_Transportation_signal(data, signal): 
    Pivot_scores_df = data[['date','state_ticker',signal]].copy() 
    data['raw_'+signal] = data[signal] 
    Green_energy_scores = get_xs_scores(Pivot_scores_df.pivot_table(index = 'date', columns = 'state_ticker', values = signal)) 
    data.set_index(['date','state_ticker'], inplace = True) 
    data[signal] = Green_energy_scores.stack() 
    return data

def quintile_analysis(data, signal, bmk_ret,flag = 0): 
    data[signal] = data[signal].fillna(0.0) 
    data['quintile'] = data.groupby(level = 0, group_keys = False).apply(lambda x: pd.qcut(x[signal].rank(method = 'first'), 5, labels = range(1, 6))).astype(int) 
    if(flag==0):
        tot_ret_q = data.reset_index(level=0).groupby(['date', 'quintile']).apply(lambda x: (x['bmk_wgt'] * x['next_ret_total']).sum() / x.loc[x['next_ret_total'].notnull(), 'bmk_wgt'].sum()).unstack() 
    else:
        tot_ret_q = data.reset_index(level=0).groupby(['date', 'quintile']).apply(lambda x: (x['bmk_wgt'] * x['next_ret_xs']).sum() / x.loc[x['next_ret_xs'].notnull(), 'bmk_wgt'].sum()).unstack() 
        
    Bmk_weights_quintile = data.reset_index(level = 0).groupby(['date', 'quintile']).apply(lambda x: (x['bmk_wgt']).sum() ).unstack() 
#     Policy_scores_quintile = data.reset_index(level = 0).groupby(['date', 'quintile']).apply(lambda x: (x['bmk_wgt'] * x['Policy_Scores']).sum()/ x.loc[x['Policy_Scores'].notnull(), 'bmk_wgt'].sum()).unstack()     
    tot_ret_q['bmk'] = bmk_ret 
    signal_stats = pd.DataFrame(index = ['Total Return', 'Total Vol', 'SR', 'Alpha', 'Beta', 'Beta Adjusted IR','model_duration', 'ytw','rating_num','Green_Tag'], columns = [1, 2, 3, 4, 5,'bmk']) 

    signal_stats.loc['Total Return'] = tot_ret_q.mean() * 12 
    signal_stats.loc['Total Vol'] = tot_ret_q.std() * np.sqrt(12) 
    # assume risk free rate of 0... need to add rf rate 
    signal_stats.loc['SR'] = (tot_ret_q.mean() * 12) / (tot_ret_q.std() * np.sqrt(12)) 
    
    resid_ret_q = pd.DataFrame(index = tot_ret_q.index, columns = [1, 2, 3, 4, 5]) 

    for i in range(1,6): 
        tmp_reg = sm.OLS(tot_ret_q[i], sm.add_constant(tot_ret_q['bmk']), missing = 'drop').fit() 
        resid_ret_q[i] = tmp_reg.resid + tmp_reg.params['const'] 
        signal_stats.loc['Alpha', i] = resid_ret_q[i].mean() * 12 
        signal_stats.loc['Beta', i] = tmp_reg.params['bmk'] 
        signal_stats.loc['Beta Adjusted IR', i] = (resid_ret_q[i].mean() * 12) / (resid_ret_q[i].std() * np.sqrt(12)) 
        for j in list(signal_stats.index[6:]): 
            signal_stats.loc[j, i] = data.reset_index(level = 0).groupby(['date', 'quintile']).apply(lambda x: (x['bmk_wgt'] * x[j]).sum()/ x.loc[x[j].notnull(), 'bmk_wgt'].sum()).unstack()[i].mean() 
            signal_stats.loc[j, 'bmk'] = data.reset_index().groupby(['date']).apply(lambda x: (x['bmk_wgt'] * x[j]).sum()/ x.loc[x[j].notnull(), 'bmk_wgt'].sum()).mean() 

    tot_ret_q['bmk-1'] = tot_ret_q['bmk'] - tot_ret_q[1] 
    screened_signal = pd.DataFrame() 
    screened_signal = pd.Series(index = ['Return', 'Tracking Error', 'IR']) 
    screened_signal['Return'] = tot_ret_q['bmk-1'].mean() * 12 
    screened_signal['Tracking Error'] = tot_ret_q['bmk-1'].std()* np.sqrt(12) 
    screened_signal['IR'] = (tot_ret_q['bmk-1'].mean() * 12)/(tot_ret_q['bmk-1'].std()* np.sqrt(12)) 

    return tot_ret_q[[1,2,3,4,5,'bmk']], resid_ret_q, signal_stats, Bmk_weights_quintile, screened_signal

test_data = construct_Ground_Transportation_signal(Yearly_state_energy_data.reset_index(drop = True), signal) 
tot_ret_q, resid_ret_q, signal_stats, bmk_weights_quintile, screened_signal  = quintile_analysis(test_data, signal, bmk_ret_total,0)

bmk_weights_quintile.mean()

tot_ret_q.cumsum().plot(title = 'cumumlative total ret') 
resid_ret_q.cumsum().plot(title = 'cumumlative residual total ret')

signal_stats

screened_signal

"""# Excess Returns"""

tot_ret_q, resid_ret_q, signal_stats, bmk_weights_quintile,screened_signal  = quintile_analysis(test_data, signal, bmk_ret_xs,1) 
tot_ret_q.cumsum().plot(title = 'cumumlative total xs_ret') 
resid_ret_q.cumsum().plot(title = 'cumumlative residual xs_ret')

signal_stats

screened_signal
