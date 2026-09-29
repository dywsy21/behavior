"""Recover cumulative E3 curriculum/accounting from completed batch receipts."""
from g05.rl.protocol import next_prefix


def resume_state(manifest, batches, checkpoint, physical_controls):
    if not batches or len(batches)>=manifest['max_batches']:
        raise ValueError('Need completed, non-exhausted training batches')
    prefixes={w['worker']:w['prefix_controls'] for w in manifest['workers']}
    recent={0:[],1:[]}; wins=zero_signal=zero_updates=0; actor=94
    for index,batch in enumerate(batches):
        if batch['batch']!=index or {r['worker'] for r in batch['episodes']}!={0,1} or len(batch['episodes'])!=2:
            raise ValueError('Missing, duplicated or reordered E3 batch')
        for row in batch['episodes']:
            w=row['worker']; spec=manifest['workers'][w]
            if row['instance']!=spec['instance'] or row['prefix']!=prefixes[w] or type(row['success']) is not bool:
                raise ValueError('Curriculum identity/history mismatch')
            recent[w].append(row['success']); wins+=row['success']
            after=next_prefix(spec['first_recorded_terminal'],prefixes[w],recent[w])
            if after!=row['next_prefix']:
                raise ValueError('Recorded curriculum progression changed')
            if after!=prefixes[w]: recent[w]=[]
            prefixes[w]=after
        signal=batch['official_reward']!=0 or batch['nonzero_shaping_controls']>0
        zero_signal=0 if signal else zero_signal+1
        zero_updates=0 if batch['new_actor_updates'] else zero_updates+1
        actor+=batch['new_actor_updates']
    if (checkpoint['path']!=batches[-1]['checkpoint'] or checkpoint['actor_updates']!=actor
            or checkpoint['critic_updates']!=16+4*len(batches)
            or checkpoint['controls']!=batches[-1]['controls']
            or not checkpoint['controls']<=physical_controls<manifest['max_training_controls']
            or zero_signal>=3 or zero_updates>=3):
        raise ValueError('Checkpoint/physical accounting mismatch or an existing stop condition')
    return dict(controls=physical_controls,batches=len(batches),actor_updates=actor,
                critic_updates=checkpoint['critic_updates'],successes=wins,
                recent={str(w):recent[w] for w in (0,1)},zero_rewards=zero_signal,zero_updates=zero_updates,
                prefixes={str(w):prefixes[w] for w in (0,1)})


def safe_restart_boundary(status, batch, receipt):
    # During prefix replay there are no new actor/critic updates to lose.
    replaying='replay_progress' in status
    return (status.get('phase')=='training_curriculum'
            and (status.get('batch')==batch['batch']+1 or replaying)
            and status.get('batches')==batch['batch']+1
            and status.get('actor_updates')==receipt['actor_updates']
            and status.get('critic_updates')==receipt['critic_updates']
            and receipt['path']==batch['checkpoint'])
