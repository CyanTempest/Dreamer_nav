"""Fix1: exact, asserted string replacements in nav3/dreamer_adapter.py and r9_official_runner/diagnose_policy.py."""
from pathlib import Path
ROOT=Path('/home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3')

def patch(rel,edits):
    path=ROOT/rel;text=path.read_text()
    for old,new in edits:
        assert text.count(old)==1,(rel,text.count(old),old)
        text=text.replace(old,new)
    path.write_text(text);print('patched',rel,len(edits))

patch('nav3/dreamer_adapter.py',[
("Only agent.py's priv_ encoder exclusion is patched in the worktree.\n",
 "Patched vs upstream e01491f: agent.py (priv_ keys excluded from the encoder), and replay storage in\n"
 "embodied/core chunk.py, replay.py plus new compress.py (LZ4 in-RAM or mmap chunks, replay-v1 save/restore).\n"),
("    def __init__(self,seed=0,logdir='checkpoints/dreamer',head=True,amnesic=False,jax_overrides=None):\n",
 "    def __init__(self,seed=0,logdir='checkpoints/dreamer',head=True,amnesic=False,jax_overrides=None,deterministic=False):\n"),
("        self._policy=jax.jit(nj.pure(self.agent.model.policy),static_argnames=('mode',))\n",
 "        def mode_policy(carry,obs,mode='train'):\n"
 "            # Agent.policy with act=pol(...).pred() (tanh(mean) of bounded_normal) instead of sample().\n"
 "            # That sample() is its last nj.seed() draw, so the RSSM latent draws are unchanged.\n"
 "            m=self.agent.model;enc_carry,dyn_carry,dec_carry,prevact=carry\n"
 "            kw=dict(training=False,single=True);reset=obs['is_first']\n"
 "            enc_carry,enc_entry,tokens=m.enc(enc_carry,obs,reset,**kw)\n"
 "            dyn_carry,dyn_entry,feat=m.dyn.observe(dyn_carry,tokens,prevact,reset,**kw)\n"
 "            dec_entry={}\n"
 "            if dec_carry:dec_carry,dec_entry,_=m.dec(dec_carry,feat,reset,**kw)\n"
 "            act=jax.tree.map(lambda d:d.pred(),m.pol(m.feat2tensor(feat),bdims=1))\n"
 "            outs={'finite':elements.tree.flatdict(jax.tree.map(lambda x:jnp.isfinite(x).all(range(1,x.ndim)),\n"
 "                dict(obs=obs,carry=carry,tokens=tokens,feat=feat,act=act)))}\n"
 "            if m.config.replay_context:outs.update(elements.tree.flatdict(dict(enc=enc_entry,dyn=dyn_entry,dec=dec_entry)))\n"
 "            return (enc_carry,dyn_carry,dec_carry,act),act,outs\n"
 "        self._policy=jax.jit(nj.pure(mode_policy if deterministic else self.agent.model.policy),static_argnames=('mode',))\n"),
("        _,(carry,act,outs)=self._policy(self.agent.params,self.carry,obs,mode='train',seed=key)\n",
 "        # Upstream Agent.policy ignores mode and always samples; deterministic=True uses pred() instead.\n"
 "        _,(carry,act,outs)=self._policy(self.agent.params,self.carry,obs,mode='train',seed=key)\n"),
])

patch('r9_official_runner/diagnose_policy.py',[
("    p.add_argument('--filter-hold-steps',type=int,default=0)\n",
 "    p.add_argument('--filter-hold-steps',type=int,default=0)\n"
 "    p.add_argument('--action',choices=['sample','mode'],default='sample')\n"),
("    a=p.parse_args()\n",
 "    a=p.parse_args()\n"
 "    if a.action=='mode' and a.method!='dreamer':p.error('--action mode is defined for Dreamer only')\n"),
("policy=DreamerPolicy(0,str(out.parent/'policy_runtime'),jax_overrides=dict(autotune=1,deterministic=True))\n",
 "policy=DreamerPolicy(0,str(out.parent/'policy_runtime'),jax_overrides=dict(autotune=1,deterministic=True),deterministic=(a.action=='mode'))\n"),
("        action=policy(obs)\n"
 "        action=action.at[:,2].set(jnp.minimum(action[:,2],a.speed_cap/.4-1))\n"
 "        active=np.array([i not in rows for i in range(env.n)])\n",
 "        action=policy(obs)\n"
 "        active=np.array([i not in rows for i in range(env.n)])\n"),
("method=a.method,checkpoint=a.checkpoint,",
 "method=a.method,action=a.action if a.method=='dreamer' else None,checkpoint=a.checkpoint,"),
])
