"""Frozen observation registry. Research simulation only; never broker orders.

Content-addressed raw vintages, immutable first-observed session records, and
separate 50/10bp ledgers. Missing sessions are flagged, never backfilled as live.
The production price feed is PR USD, not the audited TR performance universe.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import portfolio_rebalance as p

ASSETS=['world','nasdaq','cash']
def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(data):return hashlib.sha256(data).hexdigest()
def rule_digest(data):return digest(data.replace(b'\r\n',b'\n'))
def immutable(path,value):
 data=canonical(value);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  if path.read_bytes()!=data:raise ValueError(f'Immutable record differs: {path}')
 else:
  with path.open('xb') as f:f.write(data)
 return digest(data)
def blob(reg,value):
 data=canonical(value);key=digest(data);path=reg/'blobs'/f'{key}.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert gzip.decompress(path.read_bytes())==data
 else:
  with path.open('xb') as f:f.write(gzip.compress(data,mtime=0))
 return key
def read_blob(reg,key):
 data=gzip.decompress((reg/'blobs'/f'{key}.json.gz').read_bytes());assert digest(data)==key;return json.loads(data)
def series_rows(s):return [[str(d.date()),float(v)] for d,v in s.items() if pd.notna(v)]
def series(data):return pd.Series([r[1] for r in data],index=pd.to_datetime([r[0] for r in data]),dtype=float)
def integrity(base):
 manifest=json.loads((base/'freeze/manifest.json').read_text(encoding='utf-8'))
 for file,expected in manifest['sha256'].items():
  if rule_digest((base/file).read_bytes())!=expected:raise ValueError(f'Frozen rule hash changed: {file}; create an explicit new version')
 identity=json.loads((base/'freeze/identity.json').read_text(encoding='utf-8'))
 if rule_digest((base/'freeze/manifest.json').read_bytes())!=identity['manifest_sha256']:raise ValueError('Frozen manifest differs from identity')
 return identity
def cash_index(data,end):
 s=series(data);s.index=s.index+pd.offsets.CustomBusinessDay(n=2,calendar=USFederalHolidayCalendar());s=s[~s.index.duplicated(keep='last')];cal=pd.date_range(s.index.min(),end,freq='D');r=s.reindex(cal).ffill().shift(1).fillna(0)/100/360
 return (1+r).cumprod()
def trade(before,target,bps):
 total=float(sum(before));after=total
 for _ in range(12):after=total-bps/1e4*np.abs(after*target[:2]-before[:2]).sum()
 hold=after*target;delta=hold[:2]-before[:2];return dict(holdings=hold.tolist(),nav=after,cost=float(total-after),turnover=float(np.abs(delta).sum()/total/2),legs=[dict(asset=ASSETS[i],side='BUY' if d>0 else 'SELL',notional=float(abs(d)),fee=float(abs(d)*bps/1e4),simulated_only=True) for i,d in enumerate(delta) if abs(d)>1e-12*total])
def model_rows(inputs):
 world=series(inputs['world']).to_frame('Close');ndx=series(inputs['nasdaq']).to_frame('Close');ecb=series(inputs['ecb']);fed=series(inputs['fed']);features=p._prepare_features(world,ndx,ecb,fed);weekly=p._weekly_review_frame(features,'W-FRI');daily=p._ab3_features(world,ndx).loc['1999-01-01':];dates=daily.index;last=dates[-1];ordinary=p._run_model(weekly,'weekly_friday',p.DEFAULT_STRATEGY,p.STRATEGY_CONFIGS[p.DEFAULT_STRATEGY]);rel=p._ab3_s3_release(dates,weekly);out={}
 for key,cfg in p.STRATEGY_CONFIGS.items():
  if cfg.get('airbag_ab3'):
   rows=p._run_ab3(daily,ordinary,rel,key,cfg);row=rows[-1]
  else:
   rows=p._run_model(weekly,'weekly_friday',key,cfg);reviews=pd.DatetimeIndex([r['date'] for r in rows]);j=reviews.searchsorted(last,side='left')-1
   if j<0:raise ValueError('No causal inherited ordinary regime')
   row=dict(rows[j]);known=rows[reviews.searchsorted(last,side='right')-1];row['ordinary_regime']=row['final_regime'];row['ordinary_weights']=dict(row['weights']);row['ab3_active']=False;row['ab3_decision_active']=False;row['ab3_pending_weights']=known['weights'] if known['weights']!=row['weights'] else None;row['ab3_pending_action']='reequilibri setmanal' if row['ab3_pending_weights'] else None
  protected=bool(row.get('ab3_active') or row.get('airbag_s3_armed'))
  _,score=p._score_row(features.loc[last],cfg)
  out[key]=dict(strategy=key,label=cfg['label'],regime=row['final_regime'],ordinary_regime=row.get('ordinary_regime',row['final_regime']),score=float(score),ordinary_execution_score=float(row['score']),weights=row['weights'],ordinary_weights=row.get('ordinary_weights',row['weights']),protection=protected,signal=bool(row.get('ab3_signal')),activation=bool(row.get('ab3_execution')),reentry=bool(row.get('ab3_reentry')),decision_protected=bool(row.get('ab3_decision_active') or row.get('airbag_s3_armed')),pending_target=row.get('ab3_pending_weights'),pending_simulated_order=row.get('ab3_pending_order') or row.get('ab3_pending_action'),pending_execution='next observed common close after information availability' if row.get('ab3_pending_weights') else None,signal_date=row.get('ab3_signal_date'),execution_date=row.get('ab3_execution_date'),reentry_date=row.get('ab3_reentry_date'),vp1_peak_sigma10=row.get('vp1_peak_sigma10'),vp1_sigma10=row.get('vp1_sigma10'))
 return str(last.date()),out
def append_first_seen(old,new,last_session):
 """No replacement of old values and no late insertion of historical PR dates.
 Source revisions remain in raw vintages; DFF new publication dates can lag.
 """
 out={}
 for name in new:
  prior=dict(old[name]);cutoff=max(prior) if name in ['world','nasdaq','ecb','fed'] else ''
  if name in ['ecb','fed']:
   # A newly received rate is effective for this observation, never backdated
   # into an already observed Friday/confirmation counter.
   if last_session>max(prior):prior[last_session]=new[name][-1][1]
   out[name]=sorted(prior.items());continue
  for d,v in new[name]:
   if d not in prior and d>cutoff:prior[d]=v
  out[name]=sorted(prior.items())
 return out
def record_from_inputs(base,world,ndx,ecb,fed,generated_at):
 base=Path(base);reg=base/'tracking'
 if not (base/'freeze/identity.json').exists():return {'status':'not_frozen_no_prospective_record'}
 ident=integrity(base);now=pd.Timestamp(generated_at);cw=p._confirmed_closes(world,generated_at);cn=p._confirmed_closes(ndx,generated_at)
 if cw.empty or cn.empty:raise ValueError('No confirmed common inputs')
 common=cw.index.intersection(cn.index);session=str(common[-1].date());effective=pd.Timestamp(session)
 dff=pd.read_csv(base/'dades_historiques/DFF.csv',index_col=0,parse_dates=True).iloc[:,0]
 # Only observations dated no later than this confirmed session are eligible.
 raw=dict(world=series_rows(cw.Close.loc[:effective]),nasdaq=series_rows(cn.Close.loc[:effective]),ecb=series_rows(ecb.loc[:effective]),fed=series_rows(fed.loc[:effective]),dff=series_rows(dff.loc[:effective]))
 rawhash=blob(reg,raw);rules=ident['rules_commit'];version=ident['version'];first=ident['first_admissible_session'];frozen_at=ident['frozen_at'];seedpath=reg/'initial.json';priorfiles=sorted((reg/'sessions').glob('*.json')) if (reg/'sessions').exists() else []
 if not seedpath.exists():
  _,initial=model_rows(raw);immutable(seedpath,dict(observed_at=generated_at,data_session=session,inputs_blob=rawhash,states=initial,rules_commit=rules,version=version,performance_started=False,classification='post_backtest_cutoff_known_before_prospective_start' if session>'2026-11-05' else 'historical_backtest'))
 seed=json.loads(seedpath.read_text(encoding='utf-8'));previous=json.loads(priorfiles[-1].read_text(encoding='utf-8')) if priorfiles else None;previous_date=previous['session'] if previous else seed['data_session'];previous_blob=previous['canonical_inputs_blob'] if previous else seed['inputs_blob'];old=read_blob(reg,previous_blob)
 merged=append_first_seen(old,raw,session);canonicalhash=blob(reg,merged)
 prospective=session>=first and now>=pd.Timestamp(frozen_at) and session>previous_date
 missing=[str(d.date()) for d in common if previous_date<str(d.date())<session]
 quality='late_gap_not_strictly_observed' if missing and prospective else 'confirmed_common_close' if session>previous_date else 'no_new_common_close_or_source_revision'
 if prospective and missing:prospective=False
 # Recalculated latest-vintage decisions are retained separately from the
 # primary append-only canonical input path. They never rewrite a session.
 _,latest_states=model_rows(raw);_,states=model_rows(merged)
 runtime=dict(python=sys.version.split()[0],pandas=pd.__version__,numpy=np.__version__)
 revision=dict(session=session,observed_at=generated_at,effective_data_date=session,raw_inputs_blob=rawhash,canonical_inputs_blob=canonicalhash,source='Yahoo PR USD World/NDX; cached ECB/FED/DFF, captured at generation',data_quality=quality,missing_observed_sessions=missing,rules_commit=rules,generator_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip(),version=version,runtime=runtime,latest_vintage_decisions=latest_states,primary_first_observed_decisions=states,classification='strictly_prospective' if prospective else 'post_cutoff_not_prospective' if session>'2026-11-05' else 'historical',performance_basis='PR_USD_no_dividend_reinvestment; not audited TR universe')
 # The content key ignores repeated generation timestamps: identical inputs
 # and rules do not create 48 duplicate files per unchanged daily close.
 revisionkey=digest(canonical(dict(raw=rawhash,canonical=canonicalhash,rules=rules,runtime=runtime)))
 receipt=reg/'vintages'/session/(revisionkey+'.json')
 if not receipt.exists():immutable(receipt,revision)
 else:revision=json.loads(receipt.read_text(encoding='utf-8'))
 if prospective:
  ledgers={};rates=cash_index(merged['dff'],session);price=[float(cw.Close.loc[effective]),float(cn.Close.loc[effective])]
  for bps in [50,10]:
   rows={}
   for key,row in states.items():
    target=np.array([row['weights'][a]/100 for a in ASSETS]);prior=previous['ledgers'][str(bps)][key] if previous else None
    if prior:
     ratios=np.array([price[0]/previous['prices'][0],price[1]/previous['prices'][1],float(rates.loc[session]/rates.loc[previous['session']])]);before=np.array(prior['holdings'])*ratios;due=not np.allclose(target,np.array(prior['target']),atol=1e-12,rtol=0)
    else:before=np.array([0.,0.,1.]);due=True
    trade_result=trade(before,target,bps) if due else dict(holdings=before.tolist(),nav=float(before.sum()),cost=0.,turnover=0.,legs=[])
    oldprotect=bool(prior['protection']) if prior else bool(row['protection']);activation=bool(row['protection'] and not oldprotect) if prior else False;reentry=bool(oldprotect and not row['protection']) if prior else False
    ordinary_change=bool(prior and prior['ordinary_regime']!=row['ordinary_regime']);cause='INITIAL' if not prior else 'JOINT' if ordinary_change and (activation or reentry) else 'ACTIVATION' if activation else 'RESTORATION' if reentry else 'ORDINARY' if due else 'NONE'
    rows[key]=trade_result|dict(target=target.tolist(),exposure=(np.array(trade_result['holdings'])/trade_result['nav']).tolist(),protection=row['protection'],activation=activation,reentry=reentry,initial=not bool(prior),ordinary_regime=row['ordinary_regime'],score=row['score'],cause=cause,pending_simulated_order=row['pending_simulated_order'],pending_target=row['pending_target'],pending_execution=row['pending_execution'],executed_at=session if due else None,cost_bps=bps,simulated_only=True)
   ledgers[str(bps)]=rows
  entry=dict(session=session,data_effective_date=session,generated_at=generated_at,source_revision=receipt.relative_to(reg).as_posix(),source_revision_sha256=digest(receipt.read_bytes()),canonical_inputs_blob=canonicalhash,raw_inputs_blob=rawhash,previous_record_sha256=digest(priorfiles[-1].read_bytes()) if previous else None,version=version,rules_commit=rules,generator_commit=revision['generator_commit'],data_quality=quality,prices=price,models=states,ledgers=ledgers,performance_basis=revision['performance_basis'])
  immutable(reg/'sessions'/f'{session}.json',entry)
 return dict(session=session,status=quality,prospective_record_written=prospective,receipt=receipt.relative_to(base).as_posix())
def classify_episodes(prices):
 """Non-overlapping running-peak episodes; fixed boundaries, no model fitting."""
 x=np.asarray(prices,float);peak=0;trough=0;cross=None;out=[]
 for i in range(1,len(x)):
  if x[i]>=x[peak]:
   if cross is not None:
    down=trough-peak;up=i-trough;rebound=x[trough]*(1+.5*(x[peak]/x[trough]-1));false_rallies=0;armed=False
    for j in range(trough+1,i):
     if x[j]>=1.05*np.min(x[trough:j+1]):armed=True
     if armed and x[j]<np.max(x[trough:j+1])*.95:false_rallies+=1;armed=False
    out.append(dict(peak=peak,trough=trough,cross10=cross,recovery=i,fall_sessions=cross-peak,fall_type='rapid' if cross-peak<=20 else 'slow' if cross-peak>60 else 'intermediate',recovery_type='V' if up<=20 else 'slow' if up>60 else 'intermediate',false_rallies=false_rallies,mdd=1-x[trough]/x[peak],censored=False))
   peak=trough=i;cross=None
  else:
   if x[i]<x[trough]:trough=i
   if cross is None and x[i]<=.9*x[peak]:cross=i
 if cross is not None:out.append(dict(peak=peak,trough=trough,cross10=cross,recovery=None,fall_sessions=cross-peak,fall_type='rapid' if cross-peak<=20 else 'slow' if cross-peak>60 else 'intermediate',recovery_type='unrecovered',false_rallies=None,mdd=1-x[trough]/x[peak],censored=True))
 return out
def verify_registry(base):
 reg=base/'tracking';ident=integrity(base);prior=None;count=0
 for file in sorted((reg/'sessions').glob('*.json')):
  row=json.loads(file.read_text(encoding='utf-8'));assert row['previous_record_sha256']==prior;assert row['session']>=ident['first_admissible_session'];assert set(row['models'])==set(p.STRATEGY_CONFIGS);read_blob(reg,row['canonical_inputs_blob']);read_blob(reg,row['raw_inputs_blob']);assert digest((reg/row['source_revision']).read_bytes())==row['source_revision_sha256']
  for fee,models in row['ledgers'].items():
   for model,z in models.items():assert abs(sum(z['holdings'])-z['nav'])<1e-10;assert abs(sum(z['target'])-1)<1e-10;assert abs(sum(v['fee'] for v in z['legs'])-z['cost'])<1e-10
  prior=digest(file.read_bytes());count+=1
 for file in (reg/'blobs').glob('*.json.gz'):read_blob(reg,file.name.split('.')[0])
 return dict(sessions=count,status='integrity_verified',annual_period_complete=False if not count else None)
def annual(base,out,allow_partial=False):
 verify_registry(base);reg=base/'tracking';ident=json.loads((base/'freeze/identity.json').read_text(encoding='utf-8'));files=sorted((reg/'sessions').glob('*.json'));records=[json.loads(f.read_text(encoding='utf-8')) for f in files];records=[r for r in records if ident['first_admissible_session']<=r['session']<='2027-11-05']
 if not records:raise ValueError('No prospective sessions; annual performance cannot yet be calculated')
 complete=records[0]['session']==ident['first_admissible_session'] and records[-1]['session']=='2027-11-05'
 if not complete and not allow_partial:raise ValueError('Annual period incomplete or late start; use --allow-partial only for clearly labelled interim report')
 out.mkdir(parents=True,exist_ok=True);dates=pd.DatetimeIndex([r['session'] for r in records]);years=(dates[-1]-dates[0]).days/365.2425;episodes=classify_episodes([r['prices'][1] for r in records]);pd.DataFrame(episodes).to_csv(out/'episodes.csv',index=False)
 for bps in ['50','10']:
  rows=[];details=[];episode_rows=[]
  for key,cfg in p.STRATEGY_CONFIGS.items():
   zs=[r['ledgers'][bps][key] for r in records];nav=np.array([z['nav'] for z in zs]);ret=np.r_[nav[0]-1,np.diff(nav)/nav[:-1]];dd=1-nav/np.maximum.accumulate(np.r_[1.,nav])[1:];under=longest=0
   for v in dd:under=under+1 if v>1e-12 else 0;longest=max(longest,under)
   cagr=nav[-1]**(1/years)-1 if years>0 else None;expo=np.array([z['exposure'] for z in zs]);protected=np.array([z['protection'] for z in zs]);mdd=float(dd.max());peak=int(np.argmax(nav[:dd.argmax()+1]));recovery=np.flatnonzero(nav[dd.argmax()+1:]>=np.maximum.accumulate(np.r_[1.,nav])[1:][dd.argmax()]);recovery_sessions=int(dd.argmax()+1+recovery[0]-peak) if len(recovery) else None
   rows.append(dict(strategy=key,label=cfg['label'],start=str(dates[0].date()),end=str(dates[-1].date()),sessions=len(dates),fee_bps=int(bps),net_return=nav[-1]-1,cagr_effective=cagr,mdd=mdd,max_drawdown_duration_sessions=longest,volatility_annual=float(np.std(ret[1:],ddof=1)*np.sqrt(252)) if len(ret)>2 else None,calmar=cagr/mdd if cagr is not None and mdd else None,turnover=sum(z['turnover'] for z in zs),orders=sum(len(z['legs']) for z in zs),cumulative_cost=sum(z['cost'] for z in zs),activations=sum(z['activation'] for z in zs),reentries=sum(z['reentry'] for z in zs),protected_sessions=int(protected.sum()),protected_fraction=float(protected.mean()),recovery_sessions=recovery_sessions,mean_world_exposure=expo[:,0].mean(),mean_nasdaq_exposure=expo[:,1].mean(),mean_cash_exposure=expo[:,2].mean(),performance_basis='PR_USD',partial=records[-1]['session']<'2027-11-05'))
   rows[-1]['partial']=not complete
   max_calendar=0;under_start=None
   for i,value in enumerate(dd):
    if value>1e-12 and under_start is None:under_start=max(0,i-1)
    if under_start is not None:max_calendar=max(max_calendar,(dates[i]-dates[under_start]).days)
    if value<=1e-12:under_start=None
   rows[-1]['max_drawdown_duration_calendar_days']=max_calendar
   for record,z in zip(records,zs):details.append(dict(session=record['session'],strategy=key,fee_bps=int(bps),**{f:z[f] for f in ['nav','cost','turnover','protection','activation','reentry','cause']}))
   for number,event in enumerate(episodes,1):
    start=event['peak'];end=event['recovery'] if event['recovery'] is not None else len(nav)-1;trough=event['trough'];part=nav[start:end+1]/nav[start];draw=1-part/np.maximum.accumulate(np.r_[1.,part])[1:];activ=[i for i in range(start,end+1) if zs[i]['activation']];rest=[i for i in range(trough,end+1) if zs[i]['reentry']];buy=[i for i in range(trough,end+1) if any(leg['side']=='BUY' for leg in zs[i]['legs'])];peak_value=nav[start];recovered=np.flatnonzero(nav[trough:end+1]>=peak_value)
    episode_rows.append(dict(episode=number,strategy=key,fee_bps=int(bps),start=str(dates[start].date()),trough=str(dates[trough].date()),end=str(dates[end].date()),fall_type=event['fall_type'],recovery_type=event['recovery_type'],censored=event['censored'],total_return=part[-1]-1,mdd=float(draw.max()),activation_delay_sessions=activ[0]-start if activ else None,loss_before_first_activation=nav[activ[0]]/nav[start]-1 if activ else None,first_activation=str(dates[activ[0]].date()) if activ else None,first_full_reentry_after_trough=str(dates[rest[0]].date()) if rest else None,reentry_delay_after_trough_sessions=rest[0]-trough if rest else None,first_real_equity_buy_after_trough=str(dates[buy[0]].date()) if buy else None,protected_sessions=sum(z['protection'] for z in zs[start:end+1]),cost_initial_units=sum(z['cost'] for z in zs[start+1:end+1]),turnover=sum(z['turnover'] for z in zs[start+1:end+1]),orders=sum(len(z['legs']) for z in zs[start+1:end+1]),rebound_return=nav[end]/nav[trough]-1,model_recovery_sessions=int(recovered[0]) if len(recovered) else None))
  df=pd.DataFrame(rows);reference=df[df.strategy==p.DEFAULT_STRATEGY].iloc[0];df['return_difference_vs_391_pp']=(df.net_return-reference.net_return)*100;df['mdd_saved_vs_391_pp']=(reference.mdd-df.mdd)*100;df.to_csv(out/f'comparison_{bps}PB.csv',index=False);pd.DataFrame(details).to_csv(out/f'daily_{bps}PB.csv',index=False)
  ep=pd.DataFrame(episode_rows)
  if len(ep):
   base=ep[ep.strategy==p.DEFAULT_STRATEGY].set_index('episode');ep['mdd_saved_vs_391_pp']=ep.apply(lambda r:100*(base.loc[r.episode,'mdd']-r.mdd),axis=1);ep['return_difference_vs_391_pp']=ep.apply(lambda r:100*(r.total_return-base.loc[r.episode,'total_return']),axis=1)
  ep.to_csv(out/f'episode_comparison_{bps}PB.csv',index=False)
  title=f'Seguiment prospectiu {bps}pb · '+('PARCIAL' if df.partial.any() else 'període anual observat');(out/f'report_{bps}PB.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title><h1>'+title+'</h1><p>Regles congelades. Un any no estima el CAGR de llarg termini. No es calcula MDD P95 de15anys. PR USD, diferent del TR auditat. Episodis descriptius: '+str(len(episodes))+'; si zero, cap evidència nova de protecció extrema. Les revisions de fonts romanen separades de les decisions originals. No hi ha ordres reals.</p>'+df.to_html(index=False)+'<h2>Episodis aparellats</h2>'+ep.to_html(index=False),encoding='utf-8')
 return dict(sessions=len(records),episodes=len(episodes),output=str(out),partial=not complete)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['verify','annual']);ap.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output-dir',type=Path);ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
 print(json.dumps(verify_registry(args.base_dir) if args.command=='verify' else annual(args.base_dir,args.output_dir or args.base_dir/'tracking/review_2027',args.allow_partial),ensure_ascii=False))
if __name__=='__main__':main()
