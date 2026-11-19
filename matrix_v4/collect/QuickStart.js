'use strict';
const UE = require('ue'), pu = require('puerts');
const game = pu.argv.getByIndex(0);
let index = 0, last = null, expires = 0;
const ready = setInterval(() => {
    const ref = pu.$ref();
    UE.WidgetBlueprintLibrary.GetAllWidgetsOfClass(game, ref, UE.UserWidget.StaticClass(), true);
    const widgets = pu.$unref(ref);
    for (let i = 0; i < widgets.Num(); i++) {
        const widget = widgets.Get(i);
        if (!widget.GetName().startsWith('WBP_GenericFileDrop')) continue;
        clearInterval(ready);
        widget.StartStopSim();
        if (typeof UE.GameplayStatics.SetEnableWorldRendering === 'function') {
            UE.GameplayStatics.SetEnableWorldRendering(game, false);
            console.log('COL_WORLD_VIEW disabled');
        }
        setInterval(() => {
            const actor = UE.GameplayStatics.GetActorOfClass(game, UE.MatrixMotionControlActor.StaticClass());
            if (!actor) return;
            let cmd;
            // Drain outstanding writes so a slow UE tick never replays old commands.
            while (true) {
                let next;
                try { next = require('col_cmd_' + index); } catch (_) { break; }
                index++;
                if (next.stair) {
                    actor.GamepadInput = UE.EMatrixMotionControlGamepadInput.Disabled;
                    actor.SetExternalInputEnabled(true);
                    console.log('COL_STAIR_ACK ' + actor.RequestAction(UE.EMatrixMotionAction.Stair));
                }
                cmd = next;
            }
            if (cmd) {
                last = new UE.Vector(cmd.v[0], cmd.v[1], cmd.v[2]);
                expires = cmd.expires;
            }
            if (last && Date.now() > expires) {
                actor.SetWalkCommand(new UE.Vector(0,0,0));
                last = null;
            } else if (last) {
                actor.SetWalkCommand(last);
            }
        }, 50);
        break;
    }
}, 500);
module.exports = {};
