"""Teacher-only actor initialization, alongside the unchanged Dreamer losses.

Installed only in the offline pretrainer, never in live navigation evaluation.
No teacher indicator or future action is included in the policy observation.
"""
import numpy as np


def aligned_targets(previous_actions, is_first, is_last, burnin=8, xp=np,physical_is_first=None,
                    teacher_action_supervised=None):
    if not 0 <= burnin < previous_actions.shape[1]-1:
        raise ValueError('Burn-in must leave at least one aligned transition')
    # Replay stores action_t beside observation_t. Dreamer passes the shifted
    # previous-action tensor to loss: target_t = previous_action_(t+1).
    target=previous_actions[:,1:]
    mask=(~is_last[:,:-1]) & (~is_first[:,1:])
    # Unknown mid-episode latent context needs warm-up. A recorded reset,
    # however, exactly initializes the recurrent state: its first actions must
    # be supervised. Unconditionally discarding burn-in would remove the
    # first seconds of EVERY episode from every possible sampled sequence.
    recorded_first=is_first if physical_is_first is None else physical_is_first
    if recorded_first.shape!=is_first.shape:raise ValueError('Unaligned physical reset flags')
    known_context=xp.cumsum(recorded_first[:,:-1].astype(xp.int32),axis=1)>0
    mask=mask & ((xp.arange(mask.shape[1])[None,:] >= burnin) | known_context)
    if teacher_action_supervised is not None:
        if teacher_action_supervised.shape != is_first.shape or teacher_action_supervised.dtype != xp.bool_:
            raise ValueError('Teacher action mask must align with the current observations')
        mask = mask & teacher_action_supervised[:, :-1]
    return target,mask


def recovery_onsets(targets,previous_actions,mask):
    """The first recorded reverse command, not subsequent teacher-forced steps."""
    if targets.shape!=previous_actions.shape or targets.shape[:-1]!=mask.shape:
        raise ValueError('Recovery onset labels require aligned current and previous actions')
    return (targets[...,0]<-.05) & (previous_actions[...,0]>=-.05) & mask


def transition_weights(targets,mask,recovery_weight=1.,xp=np,
                       previous_actions=None,recovery_onset_weight=1.,forward_action_scale=.45):
    if not np.isfinite(forward_action_scale) or forward_action_scale<=0:
        raise ValueError('Positive finite forward action scale required')
    if not np.isfinite(recovery_weight) or recovery_weight<1:
        raise ValueError('Recovery weight must be finite and at least one')
    if not np.isfinite(recovery_onset_weight) or recovery_onset_weight<1:
        raise ValueError('Recovery onset weight must be finite and at least one')
    # Upweight reverse and high forward command labels. In the faster dataset,
    # the latter include ordinary cruise as well as teacher reapproach frames.
    recovery=(targets[...,0]<-.05)|(targets[...,0]>.405/forward_action_scale)
    weights=xp.where(mask,xp.where(recovery,recovery_weight,1.),0.)
    if recovery_onset_weight!=1.:
        if previous_actions is None:
            raise ValueError('Recovery onset weighting needs recorded previous actions')
        onset=recovery_onsets(targets,previous_actions,mask)
        weights=weights*xp.where(onset,recovery_onset_weight,1.)
    return weights,recovery & mask


def waypoint_action_targets(goal,angle_offset,distance_scale,xp=np,cruise_speed_mps=.315,
                            vx_limit=.45,wz_limit=.3,nominal_speed=None):
    """Counterfactual local goals and bounded tracking labels, never physics.

    An alternate waypoint changes the desired command at the same observed
    state. Recorded actions/rewards remain untouched for world-model training.
    """
    if not np.isfinite([vx_limit,wz_limit]).all() or min(vx_limit,wz_limit)<=0:
        raise ValueError('Positive finite command limits required')
    if not np.isfinite(cruise_speed_mps) or not 0<cruise_speed_mps<=vx_limit:
        raise ValueError('Auxiliary waypoint cruise speed must be positive and within the M1 limit')
    angle=xp.arctan2(goal[...,1],goal[...,2])+angle_offset
    distance=xp.maximum(.2,goal[...,0]*distance_scale)
    altered=xp.stack([distance,xp.sin(angle),xp.cos(angle)],axis=-1)
    curvature=2*xp.sin(angle)/xp.maximum(.4,distance)
    speed=cruise_speed_mps if nominal_speed is None else xp.clip(nominal_speed,.01,vx_limit)
    vx=speed*xp.maximum(.2,xp.cos(angle))
    vx=xp.minimum(vx,wz_limit/xp.maximum(xp.abs(curvature),1e-6))
    wz=xp.clip(vx*curvature,-wz_limit,wz_limit)
    return altered,xp.stack([vx/vx_limit,wz/wz_limit],axis=-1)


def stagnation_reconstruction_loss(prediction,proprio,symlog=True,xp=np):
    """One measured sensor's loss in the upstream MSE head's prediction space."""
    if prediction.shape!=proprio.shape or proprio.shape[-1]!=64:
        raise ValueError('Stagnation reconstruction requires aligned physical_v2 vectors')
    target=proprio[...,10]
    if symlog:target=xp.sign(target)*xp.log1p(xp.abs(target))
    return xp.square(prediction[...,10]-target)


def counterfactual_stagnation(proprio, is_first, offset_s, jitter_s, scale_s, clip_s, xp=np):
    """Alter only the measured clock for an auxiliary sensor reconstruction task.

    This is not a physical transition or an action label. The caller must not
    train dynamics, rewards, values or actor commands on these altered inputs.
    """
    if proprio.shape[-1] != 64 or proprio.shape[:-1] != is_first.shape:
        raise ValueError('Aligned physical_v2 clock inputs required')
    if not np.isfinite([scale_s, clip_s]).all() or min(scale_s, clip_s) <= 0:
        raise ValueError('Positive finite clock scaling required')
    age = xp.clip(proprio[..., 10] * scale_s + offset_s + jitter_s, 0., clip_s) / scale_s
    age = xp.where(is_first, 0., age)
    return xp.concatenate((proprio[..., :10], age[..., None], proprio[..., 11:]), axis=-1)


def install_teacher_loss(agent_class, weight=1., burnin=8,recovery_weight=1.,
                         representation_grad=False,goal_reconstruction_weight=1.,waypoint_augmentation=0.,
                         waypoint_temporal_jitter=0.,recovery_onset_weight=1.,stagnation_reconstruction_weight=1.,
                         waypoint_cruise_speed_mps=.315,stagnation_augmentation_weight=0.,
                         stagnation_scale_s=10.,stagnation_clip_s=20.,
                         vx_limit=.45,wz_limit=.3,waypoint_preserve_recorded_speed=False,
                         recovery_phase_weight=0.):
    import jax
    import jax.numpy as jnp
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Positive finite imitation weight required')
    if not np.isfinite(recovery_weight) or recovery_weight<1:
        raise ValueError('Recovery weight must be finite and at least one')
    if not np.isfinite(recovery_onset_weight) or recovery_onset_weight<1:
        raise ValueError('Recovery onset weight must be finite and at least one')
    if not np.isfinite(recovery_phase_weight) or recovery_phase_weight<0:
        raise ValueError('Recovery phase weight must be finite and nonnegative')
    if not np.isfinite(stagnation_reconstruction_weight) or stagnation_reconstruction_weight<1:
        raise ValueError('Stagnation reconstruction weight must be finite and at least one')
    if not np.isfinite(stagnation_augmentation_weight) or stagnation_augmentation_weight<0:
        raise ValueError('Stagnation augmentation weight must be nonnegative and finite')
    if not np.isfinite([stagnation_scale_s,stagnation_clip_s]).all() or min(stagnation_scale_s,stagnation_clip_s)<=0:
        raise ValueError('Positive finite clock scaling required')
    if not np.isfinite(goal_reconstruction_weight) or goal_reconstruction_weight<1:
        raise ValueError('Goal reconstruction weight must be finite and at least one')
    if not np.isfinite(waypoint_augmentation) or not 0<=waypoint_augmentation<=1.2:
        raise ValueError('Waypoint augmentation must be within 0--1.2 radians')
    if not np.isfinite(waypoint_temporal_jitter) or not 0<=waypoint_temporal_jitter<=1.2:
        raise ValueError('Waypoint temporal jitter must be within 0--1.2 radians')
    if waypoint_temporal_jitter and not waypoint_augmentation:
        raise ValueError('Temporal jitter requires waypoint augmentation')
    if not np.isfinite([vx_limit,wz_limit]).all() or min(vx_limit,wz_limit)<=0:
        raise ValueError('Positive finite command limits required')
    if not np.isfinite(waypoint_cruise_speed_mps) or not 0<waypoint_cruise_speed_mps<=vx_limit:
        raise ValueError('Auxiliary waypoint cruise speed must be positive and within the M1 limit')
    original=agent_class.loss
    def loss(self,carry,obs,prevact,training):
        physical_first=obs.get('_teacher_is_first',obs['is_first'])
        phase=obs.get('_teacher_reapproach')
        supervised = obs.get('_teacher_action_supervised')
        obs={key:value for key,value in obs.items()
             if key not in ('_teacher_is_first','_teacher_reapproach','_teacher_action_supervised')}
        total,aux=original(self,carry,obs,prevact,training)
        new_carry,entries,outs,metrics=aux
        targets,mask=aligned_targets(prevact['action'],obs['is_first'],obs['is_last'],burnin,jnp,physical_first,supervised)
        features=jax.tree.map(lambda x:x[:,:-1] if representation_grad else jax.lax.stop_gradient(x[:,:-1]),outs['repfeat'])
        distribution=self.pol(self.feat2tensor(features),2)['action']
        nll=-distribution.logp(jax.lax.stop_gradient(targets))
        assert nll.shape==mask.shape,(nll.shape,mask.shape)
        denominator=jnp.maximum(1.,mask.sum())
        previous_actions=prevact['action'][:,:-1]
        onset=recovery_onsets(targets,previous_actions,mask)
        weights,recovery=transition_weights(targets,mask,recovery_weight,jnp,
            previous_actions,recovery_onset_weight,vx_limit)
        imitation=(jnp.where(mask,nll,0.)*weights).sum()/jnp.maximum(1.,weights.sum())
        phase_loss = jnp.array(0., jnp.float32)
        if recovery_phase_weight:
            if phase is None:
                raise ValueError('Expected training-only teacher reapproach labels')
            # Normalize each phase separately so abundant cruise requests do not
            # dilute either the reverse or the forward recovery action loss.
            reverse = mask & (targets[...,0]<-.05)
            approach = mask & phase[:,:-1] & (targets[...,0]>.405/vx_limit)
            for label, selected in (('reverse',reverse),('reapproach',approach)):
                count = selected.sum()
                error = jnp.where(selected,nll,0.).sum()/jnp.maximum(1.,count)
                phase_loss += error
                metrics[f'offline_bc/phase_{label}_transitions'] = count
                metrics[f'offline_bc/phase_{label}_nll'] = error
        mse=jnp.square(distribution.pred()-targets).mean(-1)
        metrics.update({'offline_bc/nll':imitation,'offline_bc/action_mse':jnp.where(mask,mse,0.).sum()/denominator,
                        'offline_bc/labeled_transitions':mask.sum(),
                        'offline_bc/recovery_transitions':recovery.sum(),
                        'offline_bc/recovery_action_mse':jnp.where(recovery,mse,0.).sum()/jnp.maximum(1.,recovery.sum()),
                        'offline_bc/recovery_onset_transitions':onset.sum(),
                        'offline_bc/recovery_onset_action_mse':jnp.where(onset,mse,0.).sum()/jnp.maximum(1.,onset.sum())})
        goal_extra=(goal_reconstruction_weight-1.)*self.scales['goal']*outs['losses']['goal'].mean()
        metrics['offline_bc/extra_goal_reconstruction']=goal_extra
        age_extra=jnp.array(0.,jnp.float32)
        if stagnation_reconstruction_weight>1.:
            # Upstream aggregates the 64 proprioceptive scalar losses by sum.
            # Add weight only to the measured age slot; preserve other sensors.
            _,_,reconstruction=self.dec(carry[2],outs['repfeat'],obs['is_first'],training)
            age_loss=stagnation_reconstruction_loss(reconstruction['proprio'].pred(),
                jax.lax.stop_gradient(obs['proprio']),self.dec.symlog,jnp).mean()
            age_extra=(stagnation_reconstruction_weight-1.)*self.scales['proprio']*age_loss
            metrics.update({'offline_bc/stagnation_reconstruction_loss':age_loss,
                            'offline_bc/extra_stagnation_reconstruction':age_extra})
        augmentation=jnp.array(0.,jnp.float32)
        if waypoint_augmentation:
            import ninjax as nj
            batch=obs['goal'].shape[0]
            angle=jax.random.uniform(nj.seed(),(batch,1),minval=-waypoint_augmentation,maxval=waypoint_augmentation)
            if waypoint_temporal_jitter:
                # A sequence-constant offset alone lets the recurrent state
                # lag behind current goals. Change only auxiliary observations
                # and request labels at each frame, preserving real transitions.
                angle=angle+jax.random.uniform(nj.seed(),obs['goal'].shape[:2],
                    minval=-waypoint_temporal_jitter,maxval=waypoint_temporal_jitter)
            scale=jax.random.uniform(nj.seed(),(batch,1),minval=.7,maxval=1.3)
            nominal_speed=None
            if waypoint_preserve_recorded_speed:
                # Preserve each recording's slow stair / faster cruise regime
                # in auxiliary turns. These remain command labels, not physics.
                current_actions=jnp.concatenate((prevact['action'][:,1:],prevact['action'][:,-1:]),axis=1)
                nominal_speed=jnp.clip(current_actions[...,0]*vx_limit /
                    jnp.maximum(.2,obs['goal'][...,2]),.01,vx_limit)
            augmented_goal,requested_actions=waypoint_action_targets(obs['goal'],angle,scale,jnp,
                cruise_speed_mps=waypoint_cruise_speed_mps,vx_limit=vx_limit,wz_limit=wz_limit,
                nominal_speed=nominal_speed)
            augmented_obs={**obs,'goal':augmented_goal}
            # This separate branch trains observation/command conditioning.
            # It has no dynamics, reward or value target for the unexecuted
            # requested actions. The original branch retains recorded physics.
            enc_carry,_,tokens=self.enc(carry[0],augmented_obs,obs['is_first'],training)
            dyn_carry,_,feat=self.dyn.observe(carry[1],tokens,prevact,obs['is_first'],training)
            _,_,recons=self.dec(carry[2],feat,obs['is_first'],training)
            policy=self.pol(self.feat2tensor(jax.tree.map(lambda x:x[:,:-1],feat)),2)['action']
            ordinary=mask & (targets[...,0]>=.09/vx_limit)
            if not waypoint_preserve_recorded_speed:ordinary=ordinary & (targets[...,0]<.405/vx_limit)
            command_loss=-policy.logp(jax.lax.stop_gradient(requested_actions[:,:-1]))
            command_loss=jnp.where(ordinary,command_loss,0.).sum()/jnp.maximum(1.,ordinary.sum())
            goal_loss=recons['goal'].loss(jax.lax.stop_gradient(augmented_goal)).mean()
            augmentation=weight*command_loss+goal_reconstruction_weight*self.scales['goal']*goal_loss
            if stagnation_reconstruction_weight>1.:
                augmented_age_loss=stagnation_reconstruction_loss(recons['proprio'].pred(),
                    jax.lax.stop_gradient(obs['proprio']),self.dec.symlog,jnp).mean()
                augmentation+=stagnation_reconstruction_weight*self.scales['proprio']*augmented_age_loss
                metrics['offline_bc/counterfactual_stagnation_reconstruction_loss']=augmented_age_loss
            metrics.update({'offline_bc/counterfactual_action_nll':command_loss,
                            'offline_bc/counterfactual_goal_loss':goal_loss,
                            'offline_bc/counterfactual_labeled_transitions':ordinary.sum()})
        clock_augmentation=jnp.array(0.,jnp.float32)
        if stagnation_augmentation_weight:
            import ninjax as nj
            shape=obs['is_first'].shape
            offset=jax.random.uniform(nj.seed(),(shape[0],1),minval=-stagnation_clip_s,maxval=stagnation_clip_s)
            jitter=jax.random.uniform(nj.seed(),shape,minval=-2.,maxval=2.)
            alternate=counterfactual_stagnation(obs['proprio'],obs['is_first'],offset,jitter,
                stagnation_scale_s,stagnation_clip_s,jnp)
            # Break the training correlation between stair appearance and age.
            # Only reconstruct this deliberately varied sensor. Real recorded
            # transitions above remain the sole dynamics/action supervision.
            alternate_obs={**obs,'proprio':alternate}
            _,_,tokens=self.enc(carry[0],alternate_obs,obs['is_first'],training)
            _,_,features=self.dyn.observe(carry[1],tokens,prevact,obs['is_first'],training)
            _,_,recons=self.dec(carry[2],features,obs['is_first'],training)
            age_loss=stagnation_reconstruction_loss(recons['proprio'].pred(),
                jax.lax.stop_gradient(alternate),self.dec.symlog,jnp).mean()
            clock_augmentation=stagnation_augmentation_weight*self.scales['proprio']*age_loss
            metrics['offline_bc/independent_clock_reconstruction_loss']=age_loss
        return total+weight*imitation+recovery_phase_weight*phase_loss+goal_extra+age_extra+augmentation+clock_augmentation,(new_carry,entries,outs,metrics)
    def report(self,carry,data):
        carry,obs,prevact,_=self._apply_replay_context(carry,data)
        _,(new_carry,_,_,metrics)=self.loss(carry,obs,prevact,training=False)
        return (*new_carry,{k:data[k][:,-1] for k in self.act_space}),metrics
    agent_class.loss=loss
    agent_class.report=report
