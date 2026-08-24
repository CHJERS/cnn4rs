package CNN;


import java.util.HashMap;

class bpFC extends FC{                                  //新的bpmat和bpbias没有考虑

    float[][][][][] bpmat;
    float[][] bpbias;

    HashMap d_bpmat;               //命名不是特别规范，应该表达bp中的kernel.mat
    HashMap d_bpbias;

    HashMap new_d_bpmat=new HashMap();
    HashMap new_d_bpbias=new HashMap();

    void init_new(int sw){
        bpmat=new float[KernelCount.length][KernelCount.nBandCount][KernelCount.size_Kernel][KernelCount.size_Kernel][KernelCount.length];
        bpbias=new float[KernelCount.length][KernelCount.length];
        if (sw==0){
            for(int nu=0;nu<this.d_bpmat.size();nu++){
                float[][][][][][][] bpmatt= (float[][][][][][][]) this.d_bpmat.get(nu);
                int len_1=bpmatt.length;
                int len_2=bpmatt[0].length;
                int len_3=bpmatt[0][0].length;
                new_d_bpmat.put(nu,new float[len_1][len_2][len_3][len_3][KernelCount.length]);
                new_d_bpbias.put(nu,new float[len_1][KernelCount.length]);
            }
        }
        if (sw==1){
            for(int nu=0;nu<this.d_bpmat.size();nu++){
                float[][][][][] bpmatt= (float[][][][][]) this.d_bpmat.get(nu);
                int len_1=bpmatt.length;
                int len_2=bpmatt[0].length;
                int len_3=bpmatt[0][0].length;
                new_d_bpmat.put(nu,new float[len_1][len_2][len_3][len_3][KernelCount.length]);
                new_d_bpbias.put(nu,new float[len_1][KernelCount.length]);
            }
        }
    }

    void progress_PM(bpPoolMat PM){
        init_new(0);
        for (int i=0;i<KernelCount.length;i++) {
            float n=0;
            for (int band = 0; band < PM.nBandCount; band++) {
                for (int y = 0; y < PM.ySize; y++) {
                    for (int x = 0; x < PM.xSize; x++) {
                        n += (KernelCount.AK[i][band][y][x] * PM.mat[band][y][x]);
                        bpmat[i][band][y][x][i]= PM.mat[band][y][x];                                                                                                     //通用？
                        bpbias[i][i]=1;
                        for(int nu=0;nu<this.d_bpmat.size();nu++){
                            float[][][][][] new_bpmatt=(float[][][][][]) new_d_bpmat.get(nu);
                            float[][][][][][][] old_bpmatt=(float[][][][][][][]) this.d_bpmat.get(nu);
                            for (int t=0;t<new_bpmatt.length;t++) {
                                for (int tt = 0; tt < new_bpmatt[0].length; tt++) {
                                    for (int ttt = 0; ttt < new_bpmatt[0][0].length; ttt++) {
                                        for (int tttt = 0; tttt < new_bpmatt[0][0][0].length; tttt++) {
                                            new_bpmatt[t][tt][ttt][tttt][i]+=(old_bpmatt[t][tt][ttt][tttt][band][y][x]*KernelCount.AK[i][band][y][x]);
                                        }
                                    }
                                }
                                ((float[][])new_d_bpbias.get(nu))[t][i]+=(((float[][][][])this.d_bpbias.get(nu))[t][band][y][x]*KernelCount.AK[i][band][y][x]);
                            }
                        }
                    }
                }
            }
            vector[i]= n+ bias[i];
        }
        this.d_bpmat.clear();
        this.d_bpmat.putAll(new_d_bpmat);
        this.d_bpmat.put(this.d_bpmat.size(),bpmat);
        this.d_bpbias.clear();
        this.d_bpbias.putAll(new_d_bpbias);
        this.d_bpbias.put(this.d_bpbias.size(),bpbias);
    }

    bpFC(bpPoolMat PM, int[] AK_size,HashMap d_bpmat,HashMap d_bpbias){

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        int[] ak_size={AK_size[0],AK_size[1],PM.ySize};
        KernelCount= new Array_Kernel(ak_size);
        biasIniting(KernelCount.length);
        vector=new float[KernelCount.length];

        progress_PM(PM);
    }

    bpFC(bpPoolMat PM, float[][][][] AK,float[] bias,HashMap d_bpmat,HashMap d_bpbias){

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        KernelCount=new Array_Kernel(AK);
        this.bias=bias;
        vector=new float[KernelCount.length];

        progress_PM(PM);
    }

    void progress_FC(bpFC fc){
        init_new(1);
        for (int i=0;i<KernelCount.length;i++) {
            float n=0;
            for (int band=0;band<fc.vector.length;band++){
                n +=KernelCount.AK[i][band][0][0]*fc.vector[band];              //bias是否共享，如果共享则直接加共享值

                bpmat[i][band][0][0][i]= fc.vector[band];                   //通用？
                bpbias[i][i]=1;
                //bpbias[i][i]=bias[i];

                for(int nu=0;nu<this.d_bpmat.size();nu++) {
                    float[][][][][] new_bpmatt = (float[][][][][]) new_d_bpmat.get(nu);
                    float[][][][][] old_bpmatt = (float[][][][][]) this.d_bpmat.get(nu);
                    for (int t = 0; t < new_bpmatt.length; t++) {
                        for (int tt = 0; tt < new_bpmatt[0].length; tt++) {
                            for (int ttt = 0; ttt < new_bpmatt[0][0].length; ttt++) {
                                for (int tttt = 0; tttt < new_bpmatt[0][0].length; tttt++) {
                                    new_bpmatt[t][tt][ttt][tttt][i]+=(old_bpmatt[t][tt][ttt][tttt][band]*KernelCount.AK[i][band][0][0]);
                                }
                            }
                        }
                        ((float[][])new_d_bpbias.get(nu))[t][i]+=(((float[][])this.d_bpbias.get(nu))[t][band]*KernelCount.AK[i][band][0][0]);
                    }
                }
            }
            vector[i]=n+bias[i];
        }

        this.d_bpmat.clear();
        this.d_bpmat.putAll(new_d_bpmat);
        this.d_bpmat.put(this.d_bpmat.size(),bpmat);
        this.d_bpbias.clear();
        this.d_bpbias.putAll(new_d_bpbias);
        this.d_bpbias.put(this.d_bpbias.size(),bpbias);

        /*this.d_bpmat=new_d_bpmat;
        this.d_bpmat.put(this.d_bpmat.size(),bpmat);
        this.d_bpbias=new_d_bpbias;
        this.d_bpbias.put(this.d_bpbias.size(),bpbias);*/

        //bpAPI.d_kenel.put(bpAPI.d_kenel.size(),KernelCount.AK);
        //bpAPI.d_bias.put(bpAPI.d_bias.size(),bias);
    }

    bpFC(bpFC fc,int[] AK_size,HashMap d_bpmat,HashMap d_bpbias){

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        int[] ak_size={AK_size[0],AK_size[1],1};
        KernelCount= new Array_Kernel(ak_size);
        biasIniting(KernelCount.length);
        vector=new float[KernelCount.length];

        progress_FC(fc);
    }

    bpFC(bpFC fc,float[][][][] AK,float[] bias,HashMap d_bpmat,HashMap d_bpbias){

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        KernelCount=new Array_Kernel(AK);
        this.bias=bias;
        vector=new float[KernelCount.length];

        progress_FC(fc);
    }

    void norm(){
        float min=8192;
        float max=0;
        float span=0;
        for (int band=0;band<vector.length;band++){
            float k=vector[band];
            if (k<min){min=k;}
            if (k>max){max=k;}

        };
        span=max-min;
        float[] vec=new float[vector.length];
        for (int band=0;band<vector.length;band++){
            vec[band]=((vector[band]-min)/span);

            float multiple=vec[band]/vector[band];
            for(int nu=0;nu<this.d_bpmat.size();nu++){
                float[][][][][] norm_bpmatt=(float[][][][][]) this.d_bpmat.get(nu);
                for (int t=0;t<norm_bpmatt.length;t++){
                    for(int tt=0;tt<norm_bpmatt[0].length;tt++){
                        for(int ttt=0;ttt<norm_bpmatt[0][0].length;ttt++){
                            for(int tttt=0;tttt<norm_bpmatt[0][0].length;tttt++){
                                norm_bpmatt[t][tt][ttt][tttt][band]*=multiple;
                            }
                        }
                    }
                    ((float[][]) this.d_bpbias.get(nu))[t][band]*=multiple;
                }
            }
        }
        vector=vec;
    }

    void relu(){
        for (int band=0;band<vector.length;band++){
            if (vector[band]<=0){
                vector[band]=0;
                for (int nu=0;nu<d_bpmat.size();nu++){
                    float[][][][][] relu_bpmat=(float[][][][][]) this.d_bpmat.get(nu);
                    int bl1=relu_bpmat.length;
                    int bl2=relu_bpmat[0].length;
                    int bl3=relu_bpmat[0][0].length;
                    for (int i=0;i<bl1;i++){
                        for(int ii=0;ii<bl2;ii++){
                            for(int iii=0;iii<bl3;iii++){
                                for(int iiii=0;iiii<bl3;iiii++){
                                    relu_bpmat[i][ii][iii][iiii][band]=0;
                                }
                            }
                        }
                        ((float[][]) this.d_bpbias.get(nu))[i][band]=0;
                    }
                }
            }
            else{
                continue;
            }
        }
    }
}
