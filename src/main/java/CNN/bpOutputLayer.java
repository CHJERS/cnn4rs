package CNN;


import java.util.HashMap;

class bpOutputLayer extends bpFC{

    void softmax(){
        double sum=0;
        double[] newvec=new double[vector.length];
        for (int band=0;band<vector.length;band++){
            newvec[band]=Math.pow(Math.E,vector[band]);
            sum+= newvec[band];
        }
        for (int band=0;band<vector.length;band++){
            //newvec[band]/=sum;
            vector[band]=(float)(newvec[band]/sum);
        }
    }

    void percentify(){
        float sum=0;
        for(int band=0;band<vector.length;band++){
            sum+=vector[band];
        }
        float[] vec=new float[vector.length];
        for (int band=0;band<vector.length;band++){
            vec[band]=vector[band]/sum;
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

    bpOutputLayer(bpPoolMat PM, int[] AK_size, HashMap d_bpmat, HashMap d_bpbias) {
        super(PM, AK_size,d_bpmat,d_bpbias);
        //this.percentify();
        //System.out.println(Arrays.toString(vector));
    }

    bpOutputLayer(bpPoolMat PM, float[][][][] AK, float[] bias,HashMap d_bpmat,HashMap d_bpbias) {
        super(PM, AK, bias,d_bpmat,d_bpbias);
        //this.percentify();
        //System.out.println(Arrays.toString(vector));
    }

    bpOutputLayer(bpFC fc, int[] AK_size,HashMap d_bpmat,HashMap d_bpbias) {
        super(fc, AK_size,d_bpmat,d_bpbias);
        //this.percentify();
        //System.out.println(Arrays.toString(vector));
    }

    bpOutputLayer(bpFC fc, float[][][][] AK, float[] bias,HashMap d_bpmat,HashMap d_bpbias) {
        super(fc, AK, bias,d_bpmat,d_bpbias);
        //this.percentify();
        //System.out.println(Arrays.toString(vector));
    }

}
