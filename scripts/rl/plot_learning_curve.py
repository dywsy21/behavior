"""Reproducible success curve, using PRE-update policies and separate curricula."""
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path


def training_points(data):
    actor=data['first_actor_updates'];points=[]
    for index,batch in enumerate(data['batches']):
        if batch['batch']!=index or len(batch['episodes'])!=2:
            raise ValueError('Missing, duplicate, or reordered complete batch')
        if {r['worker'] for r in batch['episodes']}!={0,1}:
            raise ValueError('Missing worker')
        for row in batch['episodes']:
            if type(row['success']) is not bool: raise ValueError('Success must be official Boolean')
            points.append(dict(row,actor_updates=actor,batch=index,run=batch['run']))
        # This update consumes the preceding trajectories; it did NOT generate them.
        actor+=batch['new_actor_updates']
    return points,actor


def curriculum_segments(points,worker):
    segments=[]
    for point in [p for p in points if p['worker']==worker]:
        if not segments or segments[-1][0]['prefix']!=point['prefix']:segments.append([])
        recent=segments[-1][-4:]+[point]
        segments[-1].append(dict(point,rolling_rate=100*sum(p['success'] for p in recent)/len(recent),
                                rolling_n=len(recent)))
    return segments


def fixed_results(data):
    from g05.rl.protocol import EVAL_INSTANCES,EVAL_SEEDS,EVAL_LIMIT
    expected={(i,s) for i in EVAL_INSTANCES for s in EVAL_SEEDS};result=[]
    for stage in data['evals']:
        rows=[r for r in stage['rows'] if r['variant']!='parent_bf16']
        if len(rows)!=6 or {(r['instance'],r['policy_seed']) for r in rows}!=expected:
            raise ValueError('Only a complete, unique fixed six-episode matrix can be plotted')
        for row in rows:
            if (row['expert_prefix_controls']!=0 or row['environment_seed']!=0
                    or row['control_limit']!=EVAL_LIMIT or row.get('invalid')
                    or row['actor_updates']!=stage['actor_updates'] or row['ae_precision']!='float32'
                    or (row['success'] and not row['terminated'])
                    or (not row['terminated'] and not row['truncated'] and row['controls']!=EVAL_LIMIT)):
                raise ValueError('Mismatched or incomplete fixed evaluation')
        result.append(dict(stage=stage['stage'],actor_updates=stage['actor_updates'],
                           successes=sum(r['success'] for r in rows),n=len(rows)))
    return result


def draw(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts=[Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),
           Path('/home/wsy/.local/share/fonts/NotoSerifCJKsc-Regular.otf')]
    font=next((p for p in fonts if p.exists()),None)
    if font is None: raise RuntimeError('Install/provide a CJK font before rendering Chinese labels')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':['DejaVu Sans',font_manager.FontProperties(fname=str(font)).get_name()],
                         'font.size':12,'axes.unicode_minus':False,'savefig.facecolor':'white'})
    points,latest=training_points(data);fixed=fixed_results(data)
    if not points: raise ValueError('No completed TRAIN batch')
    stamp=datetime.fromisoformat(data['snapshot_utc']).astimezone(timezone(timedelta(hours=8)))
    fig,axes=plt.subplots(3,1,figsize=(13,10.2),gridspec_kw={'height_ratios':[2.2,2.2,1.45]})
    fig.subplots_adjust(left=.095,right=.975,top=.845,bottom=.125,hspace=.6)
    fig.suptitle('RL训练效果与更新步数',x=.095,y=.97,ha='left',fontsize=22)
    fig.text(.095,.925,'任务：打开收音机  |  上两图：TRAIN中途接管；下图：完整重置评测',fontsize=13)
    fig.text(.095,.895,'实线：同前缀内最近≤5回合均值；×：单回合结果。步数指RL优化步，与SFT步数／仿真动作步分开。',fontsize=11,color='#4c5666')
    lo=min(p['actor_updates'] for p in points)-3;hi=max(p['actor_updates'] for p in points)+7
    colors=['#176eab','#b6601c']
    for worker,ax in enumerate(axes[:2]):
        segments=curriculum_segments(points,worker)
        for k,segment in enumerate(segments):
            xs=[p['actor_updates'] for p in segment]
            left=lo if k==0 else (segments[k-1][-1]['actor_updates']+xs[0])/2
            right=hi if k==len(segments)-1 else (xs[-1]+segments[k+1][0]['actor_updates'])/2
            ax.axvspan(left,right,color='#e2e8ef' if k%2==0 else '#f7f9fb',alpha=.8,zorder=0)
            if k: ax.axvline(left,color='#a3aab5',linewidth=.8)
            ax.plot(xs,[p['rolling_rate'] for p in segment],color=colors[worker],linewidth=2.3,marker='o',markersize=4)
            ax.scatter(xs,[100*int(p['success']) for p in segment],color='#757e8e',marker='x',s=26,alpha=.7)
            n=sum(p['success'] for p in segment)
            ax.text((left+right)/2,113,f"前缀 {segment[0]['prefix']}\n{n}/{len(segment)}成功",ha='center',va='center',fontsize=11)
        instance=segments[0][0]['instance']
        ax.set_title(f'TRAIN实例 {instance}',loc='left',pad=12,fontsize=14)
        ax.set_xlim(lo,hi);ax.set_ylim(-10,130);ax.set_yticks([0,25,50,75,100])
        ax.set_ylabel('课程成功率（%）');ax.set_xlabel('生成轨迹的actor累计更新数')
        ax.grid(axis='y',color='#cbd0d8',alpha=.6,linewidth=.6)
        ax.spines[['top','right']].set_visible(False)
    ax=axes[2]
    for row in fixed:
        x=row['actor_updates'];y=100*row['successes']/row['n']
        ax.scatter([x],[y],s=60,color='#176eab',zorder=3)
        ax.annotate(f"{row['stage']}：{row['successes']}/{row['n']}",(x,y),xytext=(0,11),textcoords='offset points',
                    ha='left' if x==0 else 'center',fontsize=11)
    if not any(r['stage']=='E3' for r in fixed):
        ax.text(.62,.55,'本轮E3：六回合评测尚未完成\n未画成0%，也未用课程结果代替',transform=ax.transAxes,fontsize=11)
    ax.set_title('固定完整任务评测｜public_test 301/302 × 3 seeds，无专家前缀',loc='left',fontsize=13,pad=12)
    ax.set_xlim(-5,max(latest,max(r['actor_updates'] for r in fixed))+12)
    ax.set_ylim(-10,110);ax.set_yticks([0,50,100]);ax.set_ylabel('完整成功率（%）')
    ax.set_xlabel('被评测checkpoint的actor累计更新数')
    ax.grid(axis='y',color='#cbd0d8',alpha=.6,linewidth=.6);ax.spines[['top','right']].set_visible(False)
    fig.text(.095,.056,f"快照：{stamp:%Y-%m-%d %H:%M} 北京时间  |  完整训练批：{len(data['batches'])}  |  已落盘actor：{latest}",fontsize=10)
    fig.text(.095,.026,'前缀越短，模型越早接管、任务越难；不同前缀不直接比高低。在线训练曲线不是固定checkpoint的独立成功率。',fontsize=10,color='#4c5666')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160);plt.close(fig)
    return dict(saved=str(output),completed_batches=len(data['batches']),latest_saved_actor=latest,
                fixed_results=fixed,curriculum=[dict(worker=w,prefix=s[0]['prefix'],successes=sum(p['success'] for p in s),
                                                    n=len(s)) for w in (0,1) for s in curriculum_segments(points,w)])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(draw(json.loads(args.input.read_text()),args.output),ensure_ascii=False))
