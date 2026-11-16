"""Lossless collector snapshots, including bfloat16 recurrent-state bits."""
from pathlib import Path
import uuid


def save_tree(tree,directory):
    import jax
    import numpy as np
    directory=Path(directory)/('env_'+uuid.uuid4().hex);directory.mkdir()
    leaves,structure=jax.tree.flatten(tree);paths=[];bfloat16=[]
    for i,leaf in enumerate(leaves):
        value=np.asarray(leaf)
        if value.dtype.name=='bfloat16':
            bfloat16.append(i);value=value.view(np.uint16)
        path=directory/f'{i}.npy';np.save(path,value,allow_pickle=False);paths.append(str(path))
    return dict(structure=structure,paths=paths,bfloat16_indices=bfloat16,
        serialization='lwy-device-tree-v2-uint16-bfloat16')


def load_tree(snapshot):
    import jax
    import jax.numpy as jnp
    import numpy as np
    # Earlier LWY snapshots used np.save directly. NumPy retained BF16 bits
    # but labeled those two policy-carry leaves as an opaque two-byte dtype.
    path_tree=jax.tree.unflatten(snapshot['structure'],snapshot['paths'])
    keyed,_=jax.tree_util.tree_flatten_with_path(path_tree)
    carry_indices={i for i,(path,_) in enumerate(keyed)
        if path and isinstance(path[0],jax.tree_util.SequenceKey) and path[0].idx==3}
    leaves=[]
    for i,path in enumerate(snapshot['paths']):
        value=np.load(path,allow_pickle=False)
        if i in snapshot.get('bfloat16_indices',[]):
            assert value.dtype==np.dtype('uint16')
            value=value.view(np.dtype(jnp.bfloat16))
        elif value.dtype==np.dtype('V2'):
            assert 'serialization' not in snapshot and i in carry_indices, 'Unexpected opaque dtype outside legacy policy carry'
            value=value.view(np.dtype(jnp.bfloat16))
        leaf=jnp.asarray(value);leaf.block_until_ready();leaves.append(leaf)
    return jax.tree.unflatten(snapshot['structure'],leaves)
