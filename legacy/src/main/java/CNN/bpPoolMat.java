package CNN;


import java.util.HashMap;

class bpPoolMat extends PoolMat{

    HashMap d_bpmat;               //命名不是特别规范，应该表达bp中的kernel.mat
    HashMap d_bpbias;

    HashMap new_d_bpmat=new HashMap();
    HashMap new_d_bpbias=new HashMap();

    void init_new(bpHiddenLayer HL){
        for(int t=0;t<this.d_bpmat.size();t++){
            float[][][][][][][] bpmatt= (float[][][][][][][]) this.d_bpmat.get(t);
            int len_1=bpmatt.length;
            int len_2=bpmatt[0].length;
            int len_3=bpmatt[0][0].length;
            new_d_bpmat.put(t,new float[len_1][len_2][len_3][len_3][HL.nBandCount][this.ySize][this.xSize]);
            new_d_bpbias.put(t,new float[len_1][HL.nBandCount][this.ySize][this.xSize]);

        }
    }



    void pm_max(bpHiddenLayer HL, int poolSize){
        init_new(HL);
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    float n=0;
                    int ny=0;
                    int nx=0;
                    for (int i=0;i<poolSize;i++){
                        for (int ii=0;ii<poolSize;ii++) {
                            if (n < HL.mat[band][y*poolSize+i][x*poolSize+ii]){
                                n=HL.mat[band][y*poolSize+i][x*poolSize+ii];
                                ny=i;
                                nx=ii;
                            }
                        }
                    }
                    mat[band][y][x]=n;
                    for(int nu=0;nu<new_d_bpmat.size();nu++){
                        float[][][][][][][] bpmatt=(float[][][][][][][]) new_d_bpmat.get(nu);
                        for (int t=0;t<bpmatt.length;t++){
                            for(int tt=0;tt<bpmatt[0].length;tt++){
                                for(int ttt=0;ttt<bpmatt[0][0].length;ttt++){
                                    for(int tttt=0;tttt<bpmatt[0][0].length;tttt++){
                                        bpmatt[t][tt][ttt][tttt][band][y][x]=((float[][][][][][][])this.d_bpmat.get(nu))[t][tt][ttt][tttt][band][y*poolSize+ny][x*poolSize+nx];
                                    }
                                }
                            }
                            ((float[][][][]) new_d_bpbias.get(nu))[t][band][y][x]=((float[][][][])this.d_bpbias.get(nu))[t][band][y*poolSize+ny][x*poolSize+nx];
                        }
                    }

                }
            }
        }
        this.d_bpmat.clear();
        this.d_bpmat.putAll(new_d_bpmat);
        this.d_bpbias.clear();
        this.d_bpbias.putAll(new_d_bpbias);
    }

    void pm_mean(bpHiddenLayer HL, int poolSize){
        init_new(HL);
        int PS_squre=poolSize*poolSize;
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    float n=0;
                    for (int i=0;i<poolSize;i++){
                        for (int ii=0;ii<poolSize;ii++) {
                            n+=HL.mat[band][y*poolSize+i][x*poolSize+ii];
                        }
                    }
                    float mean=n/PS_squre;
                    mat[band][y][x]=mean;
                    float[][] mean_mat=new float[poolSize][poolSize];
                    for (int i=0;i<poolSize;i++){
                        for (int ii=0;ii<poolSize;ii++) {
                            mean_mat[i][ii]=mean/(HL.mat[band][y*poolSize+i][x*poolSize+ii]);
                        }
                    }
                    for(int nu=0;nu<new_d_bpmat.size();nu++){
                        float[][][][][][][] bpmatt=(float[][][][][][][]) new_d_bpmat.get(nu);
                        for (int t=0;t<bpmatt.length;t++){
                            for(int tt=0;tt<bpmatt[0].length;tt++){
                                for(int ttt=0;ttt<bpmatt[0][0].length;ttt++){
                                    for(int tttt=0;tttt<bpmatt[0][0].length;tttt++){
                                        for (int ps1=0;ps1<poolSize;ps1++){
                                            for (int ps2=0;ps2<poolSize;ps2++){
                                                bpmatt[t][tt][ttt][tttt][band][y][x]+=((((float[][][][][][][])this.d_bpmat.get(nu))[t][tt][ttt][tttt][band][y*poolSize+ps1][x*poolSize+ps2])/mean_mat[ps1][ps2]);
                                            }
                                        }
                                        bpmatt[t][tt][ttt][tttt][band][y][x]/=PS_squre;
                                    }
                                }
                            }
                            for (int ps1=0;ps1<poolSize;ps1++){
                                for (int ps2=0;ps2<poolSize;ps2++){
                                    ((float[][][][]) new_d_bpbias.get(nu))[t][band][y][x]+=((((float[][][][])this.d_bpbias.get(nu))[t][band][y*poolSize+ps1][x*poolSize+ps2])/mean_mat[ps1][ps2]);
                                }
                            }
                            ((float[][][][]) new_d_bpbias.get(nu))[t][band][y][x]/=PS_squre;
                        }
                    }
                }
            }
        }
        this.d_bpmat.clear();
        this.d_bpmat.putAll(new_d_bpmat);
        this.d_bpbias.clear();
        this.d_bpbias.putAll(new_d_bpbias);
    }


    bpPoolMat(bpHiddenLayer HL, int poolSize, String poolType, HashMap d_bpmat, HashMap d_bpbias) {

        super(HL,poolSize,poolType);

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        if (poolType=="max"){
            pm_max(HL,poolSize);
        }
        else if(poolType=="mean"){
            pm_mean(HL,poolSize);
        }
        else{
            System.out.println("illegal input,please input 'max' or 'mean'.");
        }
    }

}