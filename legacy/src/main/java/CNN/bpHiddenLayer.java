package CNN;


import java.util.HashMap;

class bpHiddenLayer extends HiddenLayer {

    //static int nhl=0;

    HashMap<Integer,float[][][][][][][]> d_bpmat;               //命名不是特别规范，应该表达bp中的kernel.mat
    HashMap<Integer,float[][][][]> d_bpbias;

    float[][][][][][][] bpmat;
    float[][][][] bpbias;

    void receiveYXBM_IL(int L_ySize,int L_xSize,int L_nBandCount, float[][][] L_mat ){

        float[][][] mat=define_pa(L_ySize,L_xSize,L_nBandCount,L_mat);

        bpmat=new float[KernelCount.length][KernelCount.nBandCount][KernelCount.size_Kernel][KernelCount.size_Kernel][KernelCount.length][this.ySize][this.xSize];
        bpbias=new float[this.KernelCount.length][this.KernelCount.length][this.ySize][this.xSize];

        for (int i=0;i<KernelCount.length;i++){
            for (int yhl=0;yhl<this.ySize;yhl++){
                for (int xhl=0;xhl<this.xSize;xhl++){
                    float n=0;
                    for (int band=0;band<KernelCount.nBandCount;band++){
                        for (int y=0;y<KernelCount.size_Kernel;y++){
                            for (int x=0;x<KernelCount.size_Kernel;x++){

                                n+=((KernelCount.AK[i][band][y][x]*mat[band][yhl*step+y][xhl*step+x]));

                                bpmat[i][band][y][x][i][yhl][xhl]= mat[band][yhl*step+y][xhl*step+x];                   //通用？
                                bpbias[i][i][yhl][xhl]=1;
                                //bpbias[i][i][yhl][xhl]=bias[i];
                            }
                        }
                    }

                    this.mat[i][yhl][xhl]=n+bias[i];                           //有问题？
                }
            }
        }


        this.d_bpmat.put(0,bpmat);
        this.d_bpbias.put(0,bpbias);

    }

    void receiveYXBM_bpHLorbpPM(int L_ySize,int L_xSize,int L_nBandCount, float[][][] L_mat ){
        float[][][] mat=define_pa(L_ySize,L_xSize,L_nBandCount,L_mat);

        bpmat=new float[KernelCount.length][KernelCount.nBandCount][KernelCount.size_Kernel][KernelCount.size_Kernel][KernelCount.length][this.ySize][this.xSize];
        bpbias=new float[this.KernelCount.length][this.KernelCount.length][this.ySize][this.xSize];

        HashMap new_d_bpmat=new HashMap();
        HashMap new_d_bpbias=new HashMap();
        for(int n=0;n<this.d_bpmat.size();n++){
            float[][][][][][][] bpmatt= (float[][][][][][][]) this.d_bpmat.get(n);
            int len_1=bpmatt.length;
            int len_2=bpmatt[0].length;
            int len_3=bpmatt[0][0].length;
            new_d_bpmat.put(n,new float[len_1][len_2][len_3][len_3][this.KernelCount.length][this.ySize][this.xSize]);
            new_d_bpbias.put(n,new float[len_1][this.KernelCount.length][this.ySize][this.xSize]);

        }

        for (int i=0;i<this.KernelCount.length;i++){
            for (int yhl=0;yhl<this.ySize;yhl++){
                for (int xhl=0;xhl<this.xSize;xhl++){
                    float n=0;
                    for (int band=0;band<KernelCount.nBandCount;band++){
                        for (int y=0;y<KernelCount.size_Kernel;y++){
                            for (int x=0;x<KernelCount.size_Kernel;x++){

                                n+=((KernelCount.AK[i][band][y][x]*mat[band][yhl*step+y][xhl*step+x]));

                                bpmat[i][band][y][x][i][yhl][xhl]= mat[band][yhl*step+y][xhl*step+x];                   //通用？
                                bpbias[i][i][yhl][xhl]=1;
                                //bpbias[i][i][yhl][xhl]=bias[i];
                                //检查纠错
                                for(int nu=0;nu<this.d_bpmat.size();nu++){
                                    float[][][][][][][] new_bpmatt=(float[][][][][][][]) new_d_bpmat.get(nu);
                                    float[][][][][][][] old_bpmatt=(float[][][][][][][]) this.d_bpmat.get(nu);
                                    for (int t=0;t<new_bpmatt.length;t++){
                                        for(int tt=0;tt<new_bpmatt[0].length;tt++){
                                            for(int ttt=0;ttt<new_bpmatt[0][0].length;ttt++){
                                                for(int tttt=0;tttt<new_bpmatt[0][0].length;tttt++){
                                                    new_bpmatt[t][tt][ttt][tttt][i][yhl][xhl]+=(KernelCount.AK[i][band][y][x]*old_bpmatt[t][tt][ttt][tttt][band][yhl*step+y][xhl*step+x]);
                                                }
                                            }
                                        }
                                        ((float[][][][]) new_d_bpbias.get(nu))[t][i][yhl][xhl]+=(((float[][][][]) this.d_bpbias.get(nu))[t][band][yhl*step+y][xhl*step+x])*KernelCount.AK[i][band][y][x];
                                    }
                                }

                            }
                        }
                    }

                    this.mat[i][yhl][xhl]=n+bias[i];                           //有问题？
                }
            }
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

    }



    void receive(InputLayer IL){
        int L_ySize=receiveYsize(IL);
        int L_xSize=receiveXsize(IL);
        int L_nBandCount=receiveNband(IL);
        float[][][] L_mat=receiveMat(IL);

        receiveYXBM_IL(L_ySize,L_xSize,L_nBandCount,L_mat);

        //bpAPI.d_kenel.put(bpAPI.d_kenel.size(),KernelCount.AK);
        //bpAPI.d_bias.put(bpAPI.d_bias.size(),bias);
    }

    void receive(float[][][] last_mat){
        int L_ySize=receiveYsize(last_mat);
        int L_xSize=receiveXsize(last_mat);
        int L_nBandCount=receiveNband(last_mat);
        float[][][] L_mat=last_mat;

        receiveYXBM_IL(L_ySize,L_xSize,L_nBandCount,L_mat);
    }

    void receive(bpHiddenLayer HL){
        int L_ySize=receiveYsize(HL);
        int L_xSize=receiveXsize(HL);
        int L_nBandCount=receiveNband(HL);
        float[][][] L_mat=receiveMat(HL);

        receiveYXBM_bpHLorbpPM(L_ySize,L_xSize,L_nBandCount,L_mat);

        //bpAPI.d_kenel.put(bpAPI.d_kenel.size(),KernelCount.AK);
        //bpAPI.d_bias.put(bpAPI.d_bias.size(),bias);
    }

    void receive(bpPoolMat PM){
        int L_ySize=receiveYsize(PM);
        int L_xSize=receiveXsize(PM);
        int L_nBandCount=receiveNband(PM);
        float[][][] L_mat=receiveMat(PM);

        receiveYXBM_bpHLorbpPM(L_ySize,L_xSize,L_nBandCount,L_mat);

        //bpAPI.d_kenel.put(bpAPI.d_kenel.size(),KernelCount.AK);
        //bpAPI.d_bias.put(bpAPI.d_bias.size(),bias);
    }



    bpHiddenLayer(InputLayer IL, int step, int[] AK_size, HashMap d_bpmat, HashMap d_bpbias){
        this.step=step;
        this.KernelCount= new Array_Kernel(AK_size);
        biasIniting(KernelCount.length);

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        receive(IL);

    }

    bpHiddenLayer(float[][][] IL_mat, int step, int[] AK_size, HashMap d_bpmat, HashMap d_bpbias){
        this.step=step;
        this.KernelCount= new Array_Kernel(AK_size);
        biasIniting(KernelCount.length);

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        receive(IL_mat);

    }



    bpHiddenLayer(bpHiddenLayer HL, int step, int[] AK_size, HashMap d_bpmat, HashMap d_bpbias){
        this.step=step;
        this.KernelCount= new Array_Kernel(AK_size);
        biasIniting(KernelCount.length);

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        receive(HL);

    }

    bpHiddenLayer(InputLayer IL, int step, float[][][][] nkn, float[] bias, HashMap d_bpmat, HashMap d_bpbias) {                     //上一层为Inputlayer
        this.step=step;
        this.KernelCount=new Array_Kernel(nkn);
        this.bias=bias;

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        receive(IL);
    }

    bpHiddenLayer(float[][][] IL_mat, int step, float[][][][] nkn, float[] bias, HashMap d_bpmat, HashMap d_bpbias) {                     //上一层为Inputlayer
        this.step=step;
        this.KernelCount=new Array_Kernel(nkn);
        this.bias=bias;

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        receive(IL_mat);
    }


    bpHiddenLayer(bpHiddenLayer HL, int step, float[][][][] nkn, float[] bias, HashMap d_bpmat, HashMap d_bpbias){                     //上一层为Inputlayer
        this.step=step;
        this.KernelCount=new Array_Kernel(nkn);
        this.bias=bias;

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        receive(HL);
    }

    bpHiddenLayer(bpPoolMat PM, int step, int[] AK_size, HashMap d_bpmat, HashMap d_bpbias ){
        this.step=step;
        this.KernelCount= new Array_Kernel(AK_size);
        biasIniting(KernelCount.length);

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        receive(PM);
    }

    bpHiddenLayer(bpPoolMat PM, int step, float[][][][] nkn, float[] bias, HashMap d_bpmat, HashMap d_bpbias){                     //上一层为Inputlayer
        this.step=step;
        this.KernelCount=new Array_Kernel(nkn);
        this.bias=bias;

        this.d_bpmat=d_bpmat;
        this.d_bpbias=d_bpbias;

        receive(PM);
    }

    void norm(){
        float[] min=new float[nBandCount];
        float[] span=new float[nBandCount];
        for (int band=0;band<nBandCount;band++){
            float mini=9999999;
            float max=0;
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    float k=mat[band][y][x];
                    if (k<mini){mini=k;}
                    if (k>max){max=k;}
                }
            }
            min[band]=mini;
            span[band]=max-mini;
        };

        float[][][] matrix=new float[this.nBandCount][this.ySize][this.xSize];
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    matrix[band][y][x]=((mat[band][y][x]-min[band])/span[band]);

                    float multiple=matrix[band][y][x]/mat[band][y][x];

                    for(int nu=0;nu<this.d_bpmat.size();nu++){
                        float[][][][][][][] norm_bpmatt=(float[][][][][][][]) this.d_bpmat.get(nu);
                        for (int t=0;t<norm_bpmatt.length;t++){
                            for(int tt=0;tt<norm_bpmatt[0].length;tt++){
                                for(int ttt=0;ttt<norm_bpmatt[0][0].length;ttt++){
                                    for(int tttt=0;tttt<norm_bpmatt[0][0].length;tttt++){
                                        norm_bpmatt[t][tt][ttt][tttt][band][y][x]*=multiple;

                                    }
                                }
                            }
                            ((float[][][][]) this.d_bpbias.get(nu))[t][band][y][x]*=multiple;
                        }
                    }
                }
            }
        }
        mat=matrix;
    }

    void relu(){
        for (int band=0;band<nBandCount;band++){
            for (int y=0;y<ySize;y++){
                for (int x=0;x<xSize;x++){
                    if (mat[band][y][x]<=0){
                        mat[band][y][x]=0;
                        for (int nu=0;nu<d_bpmat.size();nu++){
                            float[][][][][][][] relu_bpmat=(float[][][][][][][]) this.d_bpmat.get(nu);
                            int bl1=relu_bpmat.length;
                            int bl2=relu_bpmat[0].length;
                            int bl3=relu_bpmat[0][0].length;
                            for (int i=0;i<bl1;i++){
                                for(int ii=0;ii<bl2;ii++){
                                    for(int iii=0;iii<bl3;iii++){
                                        for(int iiii=0;iiii<bl3;iiii++){
                                            relu_bpmat[i][ii][iii][iiii][band][y][x]=0;
                                        }
                                    }
                                }
                                ((float[][][][]) this.d_bpbias.get(nu))[i][band][y][x]=0;
                            }
                        }
                    }
                    else{
                        continue;
                    }
                }
            }
        }
    }

}