"""One shared policy version, heterogeneous episodes, then one update barrier."""
from recovery_corpus import digest


class SkillRounds:
    def __init__(self, cases, eval_seeds, *, rounds, run, seed_round_offset=0,evaluation_interval_updates=1):
        if len(cases)<3 or len(set(cases))!=len(cases) or type(rounds) is not int or rounds<1:
            raise ValueError('Need three distinct heterogeneous starts and a finite experiment window')
        if not eval_seeds or len(set(eval_seeds))!=len(eval_seeds) or any(type(x) is not int for x in eval_seeds):
            raise ValueError('Pin distinct paired evaluation seeds')
        if type(seed_round_offset) is not int or seed_round_offset < 0:
            raise ValueError('Training seed offset must be a nonnegative integer')
        if type(evaluation_interval_updates) is not int or evaluation_interval_updates<1:
            raise ValueError('A positive integer paired-evaluation interval is required')
        training_seeds={17000+1009*(seed_round_offset+i) for i in range(1,rounds+1)}
        if training_seeds.intersection(eval_seeds):
            raise ValueError('Training seeds must not overlap fixed evaluation seeds')
        self.seed_round_offset=seed_round_offset
        self.evaluation_interval_updates=evaluation_interval_updates
        self.cases=sorted(cases);self.eval_seeds=eval_seeds;self.limit=rounds;self.run=run
        self.phase='evaluation';self.round=0;self.jobs=[];self.targets=[];self.install()

    def install(self):
        seeds=self.eval_seeds if self.phase=='evaluation' else [17000+1009*(self.seed_round_offset+self.round)]
        self.jobs=[dict(id=digest([self.run,self.phase,self.round,case,seed]),case=case,seed=seed,
            phase=self.phase,round=self.round,status='pending') for case in self.cases for seed in seeds]
        self.targets=[]

    def take(self, case):
        if case not in self.cases:raise ValueError('Unregistered source')
        for job in self.jobs:
            if job['case']==case and job['status']=='pending':
                job['status']='active';return dict(job)
        return None

    def complete(self, job_id, targets):
        job=next(j for j in self.jobs if j['id']==job_id)
        if job['status']!='active':raise ValueError('Duplicate or unstarted episode completion')
        if self.phase=='train':
            if not targets:raise ValueError('On-policy episode has no actual ACKed transitions')
            ids={r['experience_id'] for r in self.targets}
            for r in targets:
                if r['experience_id'] in ids:raise ValueError('Experience reused across episodes')
                ids.add(r['experience_id'])
            self.targets.extend(targets)
        elif targets:raise ValueError('Evaluation may not enter optimizer targets')
        job['status']='done'

    @property
    def ready(self):return all(j['status']=='done' for j in self.jobs)

    def advance(self, *, optimizer_completed=False):
        if not self.ready:raise ValueError('Cannot change policy/phase while episodes are still active')
        if self.phase=='train':
            if optimizer_completed is not True:raise ValueError('Missing actual shared PPO update')
            if self.round==self.limit or self.round%self.evaluation_interval_updates==0:
                self.phase='evaluation'
            else:
                # Only the measurement schedule changes. A fresh complete
                # three-skill on-policy barrier precedes EVERY optimizer step.
                # Train seeds remain a function of scientific round, not of
                # how many evaluation jobs happened between updates.
                self.round+=1
        else:
            if optimizer_completed:raise ValueError('Evaluation cannot update the actor')
            if self.round==self.limit:self.phase='finished';return
            self.round+=1;self.phase='train'
        self.install()


class SkillEvaluationRounds(SkillRounds):
    """One immutable-checkpoint probe set; there is no route to TRAIN."""
    def __init__(self, cases, eval_seeds, *, run, allow_two_logged_cases=False):
        # Joint high/low handover excludes placement, whose source lacks an
        # actually issued command trace. This opt-in never applies to PPO.
        if type(allow_two_logged_cases) is not bool:
            raise ValueError('Explicit read-only logged-handover opt-in required')
        if not allow_two_logged_cases:
            super().__init__(cases, eval_seeds, rounds=1, run=run)
            self.limit=0
            return
        if len(cases)<2 or len(set(cases))!=len(cases):
            raise ValueError('At least two distinct logged read-only handovers required')
        if not eval_seeds or len(set(eval_seeds))!=len(eval_seeds) or any(type(s) is not int for s in eval_seeds):
            raise ValueError('Pin distinct read-only probe seeds')
        self.seed_round_offset=0;self.evaluation_interval_updates=1
        self.cases=sorted(cases);self.eval_seeds=eval_seeds;self.limit=0;self.run=run
        self.phase='evaluation';self.round=0;self.jobs=[];self.targets=[];self.install()

    def advance(self, *, optimizer_completed=False):
        if not self.ready or self.phase != 'evaluation' or optimizer_completed:
            raise ValueError('Read-only probes require complete episodes and zero optimizer updates')
        if self.targets:
            raise ValueError('Read-only probes may not collect optimizer targets')
        self.phase = 'finished'
