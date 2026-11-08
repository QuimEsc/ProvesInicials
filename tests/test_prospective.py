from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
import numpy as np,pandas as pd
import prospective_tracking as t

class TrackingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'freeze').mkdir();(self.root/'dades_historiques').mkdir();(self.root/'rule.py').write_bytes(b'frozen\n');manifest={'sha256':{'rule.py':t.digest((self.root/'rule.py').read_bytes())}};(self.root/'freeze/manifest.json').write_bytes(t.canonical(manifest));identity=dict(rules_commit='0'*40,version='synthetic-historical-test-only',frozen_at='2020-02-26T22:00:00Z',first_admissible_session='2020-02-27',manifest_sha256=t.digest((self.root/'freeze/manifest.json').read_bytes()));(self.root/'freeze/identity.json').write_bytes(t.canonical(identity));pd.DataFrame({'Rate':[2.]},index=pd.to_datetime(['2019-01-01'])).to_csv(self.root/'dades_historiques/DFF.csv',index_label='Date');self.inputs=pd.read_csv(Path(__file__).parent/'fixtures/vp1_frozen_reference.csv.gz',index_col=0,parse_dates=True);self.rate=pd.Series([2.],index=pd.to_datetime(['1999-01-01']))
 def tearDown(self):self.tmp.cleanup()
 def record(self,date,world=None,observed_at=None):
  d=self.inputs.loc[:date];w=d.world_pr.to_frame('Close') if world is None else world
  with patch.object(t.subprocess,'check_output',return_value='1'*40):return t.record_from_inputs(self.root,w,d.nasdaq_pr.to_frame('Close'),self.rate,self.rate,observed_at or date+'T23:00:00Z')
 def test_initial_state_not_prospective_and_two_fee_scenarios(self):
  r=self.record('2020-02-26');self.assertFalse(r['prospective_record_written']);r=self.record('2020-02-27');self.assertTrue(r['prospective_record_written']);z=json.loads((self.root/'tracking/sessions/2020-02-27.json').read_text());self.assertEqual(set(z['ledgers']),{'50','10'});self.assertEqual(len(z['models']),7)
  for key in z['models']:self.assertLess(z['ledgers']['50'][key]['nav'],z['ledgers']['10'][key]['nav'])
 def test_source_revision_never_rewrites_primary_record(self):
  self.record('2020-02-26');self.record('2020-02-27');self.record('2020-02-28');file=self.root/'tracking/sessions/2020-02-28.json';before=file.read_bytes();w=self.inputs.loc[:'2020-02-28','world_pr'].to_frame('Close');w.iloc[-1,0]*=1.01;r=self.record('2020-02-28',w);self.assertFalse(r['prospective_record_written']);self.assertEqual(file.read_bytes(),before);self.assertGreaterEqual(len(list((self.root/'tracking/vintages/2020-02-28').glob('*.json'))),2);self.assertEqual(t.verify_registry(self.root)['sessions'],2)
 def test_duplicate_session_no_new_cost_and_fees_reconcile(self):
  self.record('2020-02-26');self.record('2020-02-27');self.record('2020-02-28');r=self.record('2020-02-28');self.assertFalse(r['prospective_record_written']);z=json.loads((self.root/'tracking/sessions/2020-02-28.json').read_text())
  for fee,models in z['ledgers'].items():
   for key,v in models.items():self.assertAlmostEqual(v['cost'],sum(leg['notional']*int(fee)/1e4 for leg in v['legs']));self.assertLessEqual(len(v['legs']),2)
 def test_changed_frozen_code_is_rejected(self):
  (self.root/'rule.py').write_bytes(b'frozen\r\n');t.integrity(self.root)
  (self.root/'rule.py').write_text('changed')
  with self.assertRaises(ValueError):self.record('2020-02-26')
 def test_missing_observation_not_fabricated_as_prospective(self):
  self.record('2020-02-26');r=self.record('2020-02-28');self.assertFalse(r['prospective_record_written']);self.assertEqual(r['status'],'late_gap_not_strictly_observed')
 def test_late_information_cannot_fill_an_already_passed_execution_close(self):
  self.record('2020-02-26');self.record('2020-02-27',observed_at='2020-02-28T23:00:00Z');r=self.record('2020-02-28',observed_at='2020-03-02T12:00:00Z');self.assertFalse(r['prospective_record_written']);self.assertEqual(r['status'],'late_information_after_execution_close');self.assertFalse((self.root/'tracking/sessions/2020-02-28.json').exists())
 def test_future_year_cannot_be_reported_complete(self):
  self.record('2020-02-26');self.record('2020-02-27')
  with self.assertRaises(ValueError):t.annual(self.root,self.root/'report')
 def test_macro_correction_not_backdated(self):
  old={k:[['2020-02-26',1.]] for k in ['world','nasdaq','ecb','fed','dff']};new={k:[['2020-02-26',9.],['2020-02-27',2.]] for k in old};z=t.append_first_seen(old,new,'2020-02-28');self.assertEqual(z['ecb'],[('2020-02-26',1.),('2020-02-28',2.)]);self.assertEqual(dict(z['world'])['2020-02-26'],1.)
 def test_fixed_episode_classification(self):
  z=t.classify_episodes([100,99,89,90,95,100]);self.assertEqual(z[0]['fall_type'],'rapid');self.assertEqual(z[0]['recovery_type'],'V');self.assertEqual(len(t.classify_episodes([100,101,102])),0)
 def test_partial_historical_fixture_generates_separate_tables_and_episode_metrics(self):
  file=self.root/'freeze/identity.json';ident=json.loads(file.read_text());ident.update(first_admissible_session='2020-03-12',frozen_at='2020-03-11T22:00:00Z');file.write_bytes(t.canonical(ident))
  for date in ['2020-03-11','2020-03-12','2020-03-13','2020-03-16']:self.record(date)
  result=t.annual(self.root,self.root/'report',allow_partial=True);self.assertTrue(result['partial']);self.assertGreaterEqual(result['episodes'],1)
  for fee in [50,10]:
   data=pd.read_csv(self.root/f'report/comparison_{fee}PB.csv');self.assertEqual(len(data),7);self.assertTrue(data.fee_bps.eq(fee).all());self.assertTrue(data.partial.all());episode=pd.read_csv(self.root/f'report/episode_comparison_{fee}PB.csv');self.assertEqual(episode.strategy.nunique(),7);self.assertIn('return_difference_vs_391_pp',episode.columns)
if __name__=='__main__':unittest.main()
