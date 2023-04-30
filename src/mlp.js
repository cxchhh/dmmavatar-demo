import ndarray from "ndarray";
import { GPU } from "gpu.js";
import zeros from "zeros";
class MLP{
    constructor(fc0_w,fc0_b,fc1_w,fc1_b,fc2_w,fc2_b,act0,act1){
        this.fc0_w=fc0_w;
        this.fc0_b=fc0_b;
        this.fc1_w=fc1_w;
        this.fc1_b=fc1_b;
        this.fc2_w=fc2_w;
        this.fc2_b=fc2_b;
        this.act0=act0;
        this.act1=act1;
        this.gpu=new GPU();
    }
    forward(x){
        var out0=zeros([64],'float32');
        for(let j=0;j<64;j++){
            for(let i=0;i<53;i++){
                out0.data[j]+=this.fc0_w.data[j*53+i]*x[i];
            }
            out0.data[j]+=this.fc0_b.data[j];
            out0.data[j]=Math.max(out0.data[j],this.act0.data[j]*out0.data[j]);
        }
        var out1=zeros([64],'float32');
        for(let j=0;j<64;j++){
            for(let i=0;i<64;i++){
                out1.data[j]+=this.fc1_w.data[j*64+i]*out0.data[i];
            }
            out1.data[j]+=this.fc1_b.data[j];
            out1.data[j]=Math.max(out1.data[j],this.act1.data[j]*out1.data[j]);
        }
        var out2=zeros([352],'float32');
        for(let j=0;j<352;j++){
            for(let i=0;i<64;i++){
                out2.data[j]+=this.fc2_w.data[j*64+i]*out1.data[i];
            }
            out2.data[j]+=this.fc2_b.data[j];
        }
        return out2;
    }
}
export default MLP;