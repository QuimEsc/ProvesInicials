"""AB3 production regressions. Fixtures are immutable CODEX_AB3 outputs,
World/Nasdaq PR common sessions, 2017–2025, including every 2020/2025 cycle.
They exercise the frozen daily clock against ordinary Friday targets. The
original three models and their scores are tested separately on identical inputs.
"""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import portfolio_rebalance as p

FIX=Path(__file__).parent/'fixtures'
LABELS=['391 + Airbag S3','391','1207','391 + AB3_RE5','391 + AB3_S3']

def reference():
 f=pd.read_csv(FIX/'ab3_daily_reference.csv',index_col=0,parse_dates=True)
 for n in [10,20]:f[f'two{n}']=f[f'above{n}'].rolling(2).sum().eq(2)
 f['RE5']=(f.two10&(f.volratio<1.5))|f.two20
 f['trigger']=(f.r10<=-.08)&(f.a<f.ma20)&(f.a.shift(1)<f.ma20.shift(1))&f.ready
 w=pd.read_csv(FIX/'ab3_weekly_reference.csv');rows=[]
 for r in w.to_dict('records'):
  rows.append(dict(r,strategy='secondpass_391',strategy_label='391',review_type='weekly_friday',final_regime=r['ordinary_regime'],weights={k:r[k] for k in ['world','nasdaq','cash']},airbag_s3_armed=False))
 return f,rows

class AB3Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.f,cls.ordinary=reference();cls.rows={}
  for rule in ['RE5','S3']:
   key='secondpass_391_ab3_'+rule.lower();cls.rows[rule]=p._run_ab3(cls.f,cls.ordinary,cls.f.release.to_numpy(),key,p.STRATEGY_CONFIGS[key])

 def test_exact_option_names_order_identifiers_and_default(self):
  options=p._strategy_options();self.assertEqual([r['label'] for r in options],LABELS);self.assertEqual(len({r['value'] for r in options}),5)
  self.assertEqual([r['value'] for r in options[:3]],['secondpass_391_airbag_s3','secondpass_391','secondpass_1207']);self.assertEqual(p.DEFAULT_STRATEGY,'secondpass_391')

 def test_four_ordinary_regimes_for_every_strategy(self):
  expected={'defensiu':(45,15,40),'semi-defensiu':(55,20,25),'normal':(60,25,15),'risc-on':(10,85,5)}
  for key in p.STRATEGY_CONFIGS:
   for regime,values in expected.items():
    self.assertEqual(tuple(p.REGIME_WEIGHTS[regime].values()),values,(key,regime));self.assertEqual(sum(values),100)

 def test_no_double_airbag_or_replaced_1207_logic(self):
  self.assertNotIn('airbag_s3',p.STRATEGY_CONFIGS['secondpass_391']);self.assertNotIn('airbag_ab3',p.STRATEGY_CONFIGS['secondpass_391'])
  self.assertNotIn('airbag_ab3',p.STRATEGY_CONFIGS['secondpass_391_airbag_s3']);self.assertEqual(p.STRATEGY_CONFIGS['secondpass_391_airbag_s3']['airbag_s3'],dict(drawdown=.08,days_since_high=70,volatility_min=.25,release_drawdown=.05))
  for rule in ['re5','s3']:self.assertNotIn('airbag_s3',p.STRATEGY_CONFIGS['secondpass_391_ab3_'+rule])
  self.assertEqual(p.STRATEGY_CONFIGS['secondpass_1207']['risk_on_entry'],70.5036804162987)
  self.assertNotEqual(p.SECOND_PASS_1207,p.SECOND_PASS_391)

 def test_frozen_daily_states_every_session_both_models(self):
  for rule,rows in self.rows.items():
   self.assertEqual([r['date'] for r in rows],self.f.index.strftime('%Y-%m-%d').tolist());np.testing.assert_array_equal([r['ab3_active'] for r in rows],self.f[rule+'_level'].astype(bool))

 def test_frozen_signal_execution_and_all_reentries(self):
  for rule,rows in self.rows.items():
   for field,suffix in [('ab3_signal','signal'),('ab3_execution','execution'),('ab3_reentry','reentry')]:np.testing.assert_array_equal([r[field] for r in rows],self.f[rule+'_'+suffix],err_msg=rule+suffix)

 def test_frozen_ordinary_and_effective_weights_every_session(self):
  base=self.f[['base_world','base_nasdaq','base_cash']].to_numpy()
  for rule,rows in self.rows.items():
   np.testing.assert_allclose([[r['ordinary_weights'][k] for k in ['world','nasdaq','cash']] for r in rows],base,atol=1e-10)
   target=base.copy();target[:,:2]*=(1-.5*self.f[rule+'_level'].to_numpy()[:,None]);target[:,2]=100-target[:,:2].sum(axis=1)
   np.testing.assert_allclose([[r['weights'][k] for k in ['world','nasdaq','cash']] for r in rows],target,atol=1e-10)

 def test_weights_protected_and_unprotected_no_partial_reentry(self):
  for rows in self.rows.values():
   for r in rows:
    self.assertAlmostEqual(sum(r['weights'].values()),100);self.assertTrue(all(0<=x<=100 for x in r['weights'].values()))
    factor=.5 if r['ab3_active'] else 1
    for asset in ['world','nasdaq']:self.assertAlmostEqual(r['weights'][asset],r['ordinary_weights'][asset]*factor)
    self.assertFalse(r['airbag_s3_armed'])
   protected=next(r for r in rows if r['ab3_active'] and r['ordinary_regime']=='risc-on');self.assertEqual(protected['weights'],dict(world=5.,nasdaq=42.5,cash=52.5))

 def test_covid_dates_and_distinct_reentries(self):
  for rule in ['RE5','S3']:
   row=next(r for r in self.rows[rule] if r['date']=='2020-02-28');self.assertTrue(row['ab3_execution']);self.assertEqual(row['ab3_signal_date'],'2020-02-27')
  self.assertTrue(next(r for r in self.rows['RE5'] if r['date']=='2020-03-26')['ab3_reentry']);self.assertTrue(next(r for r in self.rows['S3'] if r['date']=='2020-06-01')['ab3_reentry'])

 def test_2025_all_cycles_compared_not_only_first_alarm(self):
  for rule in ['RE5','S3']:
   x=[r for r in self.rows[rule] if r['date'].startswith('2025')];self.assertGreaterEqual(sum(r['ab3_execution'] for r in x),2);self.assertGreaterEqual(sum(r['ab3_reentry'] for r in x),2)
   self.assertTrue(next(r for r in x if r['date']=='2025-03-05')['ab3_execution'])

 def test_causal_prefix_states_weights_and_execution(self):
  for cut in ['2020-02-27','2020-03-23','2020-03-27','2025-03-04','2025-04-08','2025-05-02']:
   f=self.f.loc[:cut]
   for rule in ['RE5','S3']:
    rows=p._run_ab3(f,self.ordinary,f.release.to_numpy(),'test',dict(p.SECOND_PASS_391,label='test',airbag_ab3=rule));full=self.rows[rule][:len(rows)]
    self.assertEqual([r['weights'] for r in rows],[r['weights'] for r in full]);self.assertEqual([r['ab3_active'] for r in rows],[r['ab3_active'] for r in full])
    for row in rows:
     for field in ['ab3_execution_date','ab3_reentry_date']:
      self.assertTrue(row.get(field) is None or row[field]<=row['date'])

 def test_pending_signal_does_not_execute_on_signal_close(self):
  row=next(r for r in self.rows['RE5'] if r['date']=='2020-02-27');self.assertFalse(row['ab3_active']);self.assertTrue(row['ab3_decision_active']);self.assertEqual(row['ab3_pending_action'],'proteccio');self.assertNotEqual(row['weights'],row['ab3_pending_weights'])

 def test_restoration_uses_current_regime_and_daily_actions(self):
  changes=0
  for rule,rows in self.rows.items():
   for prev,r in zip(rows,rows[1:]):
    self.assertEqual(r['reference_weights'],prev['weights'])
    self.assertEqual(r['diffs'],p._action(r['weights'],prev['weights'])['diffs'])
    if r['ab3_reentry']:
     self.assertEqual(r['weights'],r['ordinary_weights'])
    if r['ab3_active'] and prev['ab3_active'] and r['ordinary_regime']!=prev['ordinary_regime']:changes+=1
  self.assertGreater(changes,0)

 def test_intraday_close_guard_winter_and_summer(self):
  df=pd.DataFrame({'Close':[100,101]},index=pd.to_datetime(['2025-03-03','2025-03-04']))
  self.assertEqual(len(p._confirmed_closes(df,'2025-03-04T20:59:00Z')),1);self.assertEqual(len(p._confirmed_closes(df,'2025-03-04T21:00:00Z')),2)
  df.index=pd.to_datetime(['2020-06-01','2020-06-02']);self.assertEqual(len(p._confirmed_closes(df,'2020-06-02T19:59:00Z')),1);self.assertEqual(len(p._confirmed_closes(df,'2020-06-02T20:00:00Z')),2)

 def test_missing_common_close_never_forward_filled(self):
  dates=pd.bdate_range('2020-01-01',periods=300);a=pd.DataFrame({'Close':np.arange(300)+100.},index=dates);d=a.drop(dates[-1]);f=p._ab3_features(d,a);self.assertEqual(len(f),299);self.assertNotIn(dates[-1],f.index)

 def test_indicators_use_common_pr_closes_and_lagged_sigma(self):
  dates=pd.bdate_range('2020-01-01',periods=320);a=pd.DataFrame({'Close':100*np.exp(np.cumsum(.002+np.sin(np.arange(320))*.01))},index=dates);f=p._ab3_features(a,a);r=a.Close.pct_change()
  np.testing.assert_allclose(f.r10,a.Close.pct_change(10),equal_nan=True);np.testing.assert_allclose(f.prior_sigma,r.rolling(63).std().shift(1),equal_nan=True);np.testing.assert_allclose(f.ma20,a.Close.rolling(20).mean(),equal_nan=True)

 def test_weekly_release_is_known_before_next_close_exists(self):
  dates=pd.to_datetime(['2020-05-28','2020-05-29']);weekly=pd.DataFrame({'msci_drawdown':[-.04]},index=pd.to_datetime(['2020-05-29']))
  with patch.object(p,'_score_row',return_value=({},60)):
   np.testing.assert_array_equal(p._ab3_s3_release(dates,weekly),[False,True])

 def test_invalid_reentry_not_silently_mapped_to_re5(self):
  with self.assertRaises(ValueError):p._ab3_states(self.f,np.zeros(len(self.f),bool),'RB',self.f.release.to_numpy())

 def test_fractional_orders_keep_the_exact_protection_amount(self):
  action=p._action(dict(world=5.,nasdaq=42.5,cash=52.5),dict(world=10.,nasdaq=85.,cash=5.))
  self.assertIn('Vendre 42.5 pp Nasdaq',action['text']);self.assertIn('Comprar 47.5 pp Monetari',action['text'])

 def test_complete_payload_uses_confirmed_score_without_daily_regime_confirmation(self):
  dates=pd.bdate_range('2019-01-01',periods=410);world=pd.DataFrame({'Close':100*np.exp(np.arange(410)*.0005)},index=dates);ndx=pd.DataFrame({'Close':100*np.exp(np.arange(410)*.0007)},index=dates);rate=pd.Series(2.,index=[dates[0]])
  generated=dates[-1].strftime('%Y-%m-%d')+'T23:00:00Z';payload,h=p.build_rebalance_data(world,ndx,rate,rate,generated_at=generated)
  self.assertEqual(payload['meta']['default_strategy'],'secondpass_391');self.assertEqual([x['label'] for x in payload['strategy_options']],LABELS)
  features=p._prepare_features(world,ndx,rate,rate);expected=round(p._score_row(features.iloc[-1],p.SECOND_PASS_391)[1],2)
  for key in ['secondpass_391_ab3_re5','secondpass_391_ab3_s3']:
   current=payload['rows_by_strategy'][key]['rows'][0];self.assertEqual(current['score'],expected);self.assertEqual(current['score_asof'],current['date']);self.assertFalse(current['airbag_s3_armed'])
   self.assertEqual(set(h[h.strategy.eq(key)].review_type),{'daily_close'})

if __name__=='__main__':unittest.main()
