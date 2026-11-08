from pathlib import Path
import unittest
import numpy as np,pandas as pd
import portfolio_rebalance as p

FIX=Path(__file__).parent/'fixtures'
class VP1Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.ref=pd.read_csv(FIX/'vp1_frozen_reference.csv.gz',index_col=0,parse_dates=True);r=cls.ref;cls.f=p._ab3_features(r.world_pr.to_frame('Close'),r.nasdaq_pr.to_frame('Close')).loc['1999-01-01':];w=pd.read_csv(FIX/'vp1_ordinary.csv');cls.ordinary=[]
  for row in w.to_dict('records'):cls.ordinary.append(dict(row,weights={a:row[a] for a in ['world','nasdaq','cash']}))
  cls.rows={}
  for variant in ['VP1_ORIGINAL','VP1_2MA']:
   key='secondpass_391_vp1_'+('original' if variant=='VP1_ORIGINAL' else '2ma');cls.rows[variant]=p._run_ab3(cls.f,cls.ordinary,r.weekly_release.reindex(cls.f.index).to_numpy(),key,p.STRATEGY_CONFIGS[key])
 def test_all_frozen_daily_states_and_weights(self):
  for variant,rows in self.rows.items():
   df=pd.DataFrame(rows).set_index('date');df.index=pd.to_datetime(df.index);expected=self.ref.reindex(df.index);np.testing.assert_array_equal(df.ab3_active,expected[variant+'_active'].astype(bool));np.testing.assert_array_equal(df.ab3_decision_active,expected[variant+'_decision'].astype(bool))
   for asset in ['world','nasdaq','cash']:
    actual=df.weights.map(lambda w:w[asset]/100);good=expected[variant+'_target_'+asset].notna();np.testing.assert_allclose(actual[good],expected.loc[good,variant+'_target_'+asset],atol=1e-12)
 def test_covid_distinct_restoration_dates(self):
  for variant,date in [('VP1_ORIGINAL','2020-04-03'),('VP1_2MA','2020-04-09')]:self.assertTrue(next(r for r in self.rows[variant] if r['date']==date)['ab3_reentry'])
 def test_causal_features_states_and_peak(self):
  for date in ['2008-09-15','2020-02-27','2020-04-02','2020-04-08','2022-07-05','2025-04-25']:
   rr=self.ref.loc[:date];f=p._ab3_features(rr.world_pr.to_frame('Close'),rr.nasdaq_pr.to_frame('Close')).loc['1999-01-01':]
   for variant,full in self.rows.items():
    key='secondpass_391_vp1_'+('original' if variant=='VP1_ORIGINAL' else '2ma');short=p._run_ab3(f,self.ordinary,rr.weekly_release.reindex(f.index).to_numpy(),key,p.STRATEGY_CONFIGS[key]);baseline=full[:len(short)]
    for field in ['weights','ab3_active','ab3_pending_action','vp1_peak_sigma10','ab3_execution','ab3_reentry']:self.assertEqual([r[field] for r in short],[r[field] for r in baseline])
 def test_only_previous_ma_condition_differs(self):
  n=16;f=pd.DataFrame(dict(a=np.full(n,2.),ma10=np.ones(n),sigma10=np.r_[np.ones(6),np.full(n-6,.5)]));f.loc[7,'a']=.5;tr=np.zeros(n,bool);tr[5]=True
  original=p._vp1_states(f,tr,'VP1_ORIGINAL');two=p._vp1_states(f,tr,'VP1_2MA');self.assertEqual(original[1][8],0);self.assertEqual(two[1][8],1);self.assertEqual(two[1][9],0)
 def test_one_effective_target_and_no_double_protection(self):
  for rows in self.rows.values():
   for prev,row in zip(rows,rows[1:]):
    self.assertFalse(row['airbag_s3_armed']);self.assertAlmostEqual(sum(row['weights'].values()),100);self.assertEqual(row['reference_weights'],prev['weights'])
    for a in ['world','nasdaq']:self.assertAlmostEqual(row['weights'][a],row['ordinary_weights'][a]*(.5 if row['ab3_active'] else 1))
 def test_original_and_2ma_are_unique_and_experimental_labelled(self):
  c=p.STRATEGY_CONFIGS;self.assertIn('post hoc',c['secondpass_391_vp1_2ma']['research_status']);self.assertEqual(c['secondpass_391_vp1_original']['airbag_ab3'],'VP1_ORIGINAL');self.assertEqual(c['secondpass_391_vp1_2ma']['airbag_ab3'],'VP1_2MA')
if __name__=='__main__':unittest.main()
