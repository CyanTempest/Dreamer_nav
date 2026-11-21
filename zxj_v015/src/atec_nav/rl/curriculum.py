"""Advance navigation distance only after repeated successful training episodes."""
class NavigationCurriculum:
    def __init__(self, lengths, successes_required=3, stages=None):
        self.stages=stages
        if stages:
            lengths=[s['length_m'] for s in stages]
            if any(s.get('start_s',0)<0 for s in stages):raise ValueError('Negative stage start')
        self.lengths = tuple(float(x) for x in lengths)
        if not self.lengths or any(x<=0 for x in self.lengths) or (not stages and any(b <= a for a,b in zip(self.lengths,self.lengths[1:]))):
            raise ValueError('Curriculum lengths must be positive and strictly increasing')
        if successes_required < 1:raise ValueError('Success count must be positive')
        self.required = successes_required
        self.stage = self.streak = 0

    @property
    def length(self):
        return self.lengths[self.stage]

    @property
    def start(self):
        return float(self.stages[self.stage].get('start_s',0)) if self.stages else 0.

    @property
    def noise_enabled(self):
        return bool(self.stages[self.stage].get('noise_enabled',False)) if self.stages else None

    def finish(self, success):
        self.streak = self.streak+1 if success else 0
        if self.streak >= self.required and self.stage < len(self.lengths)-1:
            self.stage += 1
            self.streak = 0
