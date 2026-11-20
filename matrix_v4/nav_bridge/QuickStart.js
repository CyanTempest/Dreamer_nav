'use strict';
const UE=require('ue'), pu=require('puerts'), game=pu.argv.getByIndex(0);
const timer=setInterval(()=>{
    const ref=pu.$ref();
    UE.WidgetBlueprintLibrary.GetAllWidgetsOfClass(game,ref,UE.UserWidget.StaticClass(),true);
    const widgets=pu.$unref(ref);
    for(let i=0;i<widgets.Num();i++){
        const widget=widgets.Get(i);
        if(!widget.GetName().startsWith('WBP_GenericFileDrop')) continue;
        clearInterval(timer);
        widget.StartStopSim();
        const pc=UE.GameplayStatics.GetPlayerController(game,0);
        let index=0, cmd=null, heading=require('nav_heading').yaw;
        setInterval(()=>{
            const mc=UE.GameplayStatics.GetActorOfClass(game,UE.MatrixMotionControlActor.StaticClass());
            if(!mc) return;
            while(true){
                let next;
                try {next=require('nav_cmd_'+index);} catch (_) {break;}
                index++;
                if(next.stair){
                    mc.GamepadInput=UE.EMatrixMotionControlGamepadInput.Disabled;
                    mc.SetExternalInputEnabled(true);
                    console.log('NAV_STAIR',mc.RequestAction(UE.EMatrixMotionAction.Stair));
                }
                cmd=next;
                heading=cmd.heading;
            }
            if(cmd){
                const live=Date.now()<cmd.until;
                mc.SetWalkCommand(new UE.Vector(live?cmd.f:0,0,live?cmd.w:0));
            }
        },50);
        let last=0;
        setInterval(()=>{
            if(Date.now()-last<120) return;
            const scene=UE.GameplayStatics.GetActorOfClass(game,UE.MujocoSceneActor.StaticClass());
            if(!scene) return;
            const v=scene.GetRobotWorldLocation(), c=Math.cos(heading), s=-Math.sin(heading);
            const camera=new UE.Vector(v.X-340*c,v.Y-340*s,v.Z+210);
            const look=new UE.Vector(v.X+65*c,v.Y+65*s,v.Z+10);
            UE.GameplayStatics.GetPlayerPawn(game,0).K2_SetActorLocation(camera,false,pu.$ref(),true);
            pc.SetControlRotation(UE.KismetMathLibrary.FindLookAtRotation(camera,look));
            // The independent RGB sensor records the third-person view.
            last=Date.now();
        },125);
        break;
    }
},250);
module.exports={};
