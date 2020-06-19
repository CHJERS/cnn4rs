package CNN;


import java.util.HashMap;

class bpFinal{

    HashMap d_bpmat;
    HashMap d_bpbias;
    HashMap d_kernel;
    HashMap d_bias;

    float[] l_update;
    float countmse=0;
    float step;
    float e=(float)Math.E;

    HashMap new_d_bpmat=new HashMap();
    HashMap new_d_bpbias=new HashMap();

    void init_new(){
        for(int nu=0;nu<this.d_bpmat.size();nu++){
            float[][][][][] bpmatt= (float[][][][][]) this.d_bpmat.get(nu);
            int len_1=bpmatt.length;
            int len_2=bpmatt[0].length;
            int len_3=bpmatt[0][0].length;
            //int len_4=bpmatt[0][0][0][0].length;
            new_d_bpmat.put(nu,new float[len_1][len_2][len_3][len_3]);
            new_d_bpbias.put(nu,new float[len_1]);
        }
    }


    void cee_update(bpOutputLayer fc,float[] real_score){
        for (int n=0;n<real_score.length;n++) {
            if (real_score[n]==1){
                l_update[n]=(fc.vector[n]-1);
                this.countmse=(float) Math.log(fc.vector[n])/real_score.length;
            }
            else{
                l_update[n]=fc.vector[n];
            }
        }
    }

    void mse_update(bpOutputLayer fc,float[] real_score){
        float mse;
        float mmse;
        for (int n=0;n<real_score.length;n++){
            mse=(fc.vector[n]-real_score[n]);
            l_update[n]=mse;
            mmse= (float) (0.5*mse*mse);
            this.countmse+=mmse;
        }
        this.countmse/=real_score.length;
    }

    bpFinal(bpOutputLayer fc,float[] real_score,float step,HashMap d_bpmat,HashMap d_bpbias,HashMap d_kernel,HashMap d_bias,String type){
        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;
        this.d_kernel=d_kernel;
        this.d_bias=d_bias;
        this.step=step;
        init_new();

        l_update=new float[real_score.length];
        if(type=="mse"){
            mse_update(fc,real_score);
        }
        else if(type=="softmax"){
            cee_update(fc,real_score);
        }

        //this.step= (float) ((0.2*(Math.pow(e,10*this.countmse)))-0.2);                //动态更新step
        for(int nu=0;nu<this.d_bpmat.size();nu++){
            float[][][][][] bpmat=(float[][][][][])this.d_bpmat.get(nu);

            for(int t=0;t<fc.vector.length;t++){
                for(int i=0;i<bpmat.length;i++){
                    for(int band=0;band<bpmat[0].length;band++){
                        for(int y=0;y<bpmat[0][0].length;y++){
                            for(int x=0;x<bpmat[0][0].length;x++){
                                //bpmat[i][band][y][x][t]*=l_mse[t];
                                ((float[][][][])new_d_bpmat.get(nu))[i][band][y][x]+=(bpmat[i][band][y][x][t]*l_update[t]);
                            }

                        }
                    }
                    //((float[][])bpAPI.d_bpbias.get(nu))[i][t] *=l_mse[t];
                    ((float[])new_d_bpbias.get(nu))[i]+=(((float[][])this.d_bpbias.get(nu))[i][t] *l_update[t]);
                }
            }
        }

        this.d_bpmat.clear();
        this.d_bpmat.putAll(new_d_bpmat);

        this.d_bpbias.clear();
        this.d_bpbias.putAll(new_d_bpbias);

        //this.d_bpmat=new_d_bpmat;
        //this.d_bpbias=new_d_bpbias;
    }

    void update(){
        for(int nu=0;nu<this.d_bpmat.size();nu++){
            float[][][][] kernel=(float[][][][])this.d_kernel.get(nu);
            float[][][][] bpmat=(float[][][][])this.d_bpmat.get(nu);
            for(int i=0;i<kernel.length;i++){
                for(int band=0;band<kernel[0].length;band++){
                    for(int y=0;y<kernel[0][0].length;y++){
                        for(int x=0;x<kernel[0][0].length;x++){
                            kernel[i][band][y][x]-=(step*bpmat[i][band][y][x]);
                        }
                    }
                }
                ((float[])this.d_bias.get(nu))[i]-=(step*(((float[])this.d_bpbias.get(nu))[i]));
            }
        }
    }

    HashMap return_d_kernel(){
        return this.d_kernel;
    }

    HashMap return_d_bias(){
        return this.d_bias;
    }

}