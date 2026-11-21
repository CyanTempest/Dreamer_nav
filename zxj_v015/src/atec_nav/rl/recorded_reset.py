"""Retain physical episode starts separately from replay's synthetic resets."""


def install_recorded_reset_metadata(agent_class, include_recovery_phase=False, include_teacher_action_mask=False):
    import elements
    original_space=agent_class.ext_space
    original_context=agent_class._apply_replay_context
    metadata = ('teacher_is_first', 'teacher_reapproach') if include_recovery_phase else ('teacher_is_first',)
    if include_teacher_action_mask:
        metadata += ('teacher_action_supervised',)
    def ext_space(self):
        return {**original_space.fget(self), **{key: elements.Space(bool) for key in metadata}}
    def apply_context(self,carry,data):
        carry,obs,prevact,stepid=original_context(self,carry,data)
        count=obs['is_first'].shape[1]
        return carry,{**obs, **{'_'+key: data[key][:,-count:] for key in metadata}},prevact,stepid
    agent_class.ext_space=property(ext_space)
    agent_class._apply_replay_context=apply_context
