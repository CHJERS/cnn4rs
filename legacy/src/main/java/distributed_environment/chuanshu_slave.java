package distributed_environment;


import CNN.bpNet;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.text.ParseException;
import java.util.HashMap;

public class chuanshu_slave implements Runnable{
    private int port;
    //private String leixing;
    private float[][][] IL_mat;
    private HashMap d_kernel;
    private HashMap d_bias;
    private float[] real_score;
    private float step;

    static float[][][] txtToMat(String mat_str, int bandCount, int ySize, int xSize) throws IOException {

        float[][][] mat=new float[bandCount][ySize][xSize];
        String[] s_mat=mat_str.split("\\]");
        int band=0;
        for(int i=0;i<bandCount;i++){
            String[] s_i=(s_mat[i].split("\\[")[1]).split("\\, ");

            for(int y=0;y<50;y++){
                for(int x=0;x<50;x++){
                    mat[band][y][x]=Float.parseFloat(s_i[50*y+x]);
                }
            }
            band+=1;
        }

        return mat;

    }

    private void import_kernel_bias(String kernel_, String bias_) throws IOException {
        String[] l_kernel=kernel_.split("\n");
        String[] l_bias=bias_.split("\n");

        HashMap new_d_kernel=new HashMap<Integer,float[][][][]>();
        HashMap new_d_bias=new HashMap<Integer,float[]>();

        float[][][][] kernel;
        String[] kernel_0;
        String[] kernel_1;
        String[] kernel_2;
        String[] kernel_3;
        float[] bias;
        String[] bias_0;

        for(int nn=0;nn<l_kernel.length;nn++){

            kernel_0=((((l_kernel[nn].split("\\]\\]\\]\\]"))[0]).split("\\[\\[\\[\\["))[1]).split("\\]\\]\\], \\[\\[\\[");

            kernel_1=kernel_0[0].split("\\]\\], \\[\\[");

            kernel_2=kernel_1[0].split("\\], \\[");

            kernel_3=kernel_2[0].split("\\, ");

            kernel=new float[kernel_0.length][kernel_1.length][kernel_2.length][kernel_3.length];

            for(int t=0;t<kernel_0.length;t++){
                kernel_1=kernel_0[t].split("\\]\\], \\[\\[");
                for(int tt=0;tt<kernel_1.length;tt++){
                    kernel_2=kernel_1[tt].split("\\], \\[");
                    for(int ttt=0;ttt<kernel_2.length;ttt++){
                        kernel_3=kernel_2[ttt].split("\\, ");
                        for(int tttt=0;tttt<kernel_3.length;tttt++){
                            kernel[t][tt][ttt][tttt]=Float.parseFloat(kernel_3[tttt]);
                        }
                    }
                }
            }

            bias_0=(l_bias[nn].split("\\]")[0]).split("\\[")[1].split("\\, ");
            bias=new float[bias_0.length];
            for(int t=0;t<bias_0.length;t++){
                bias[t]=Float.parseFloat(bias_0[t]);
            }

            new_d_kernel.put(nn,kernel);
            new_d_bias.put(nn,bias);

        }

        this.d_kernel=new_d_kernel;
        this.d_bias=new_d_bias;
    }

    private void get_realscore(String str_realscore){
        String[] ls_rs=(str_realscore.split("\\]")[0]).split("\\[")[1].split("\\,");
        this.real_score=new float[ls_rs.length];

        for (int i=0;i<ls_rs.length;i++){
            this.real_score[i]=Float.parseFloat(ls_rs[i]);
        }
    }

    chuanshu_slave(int port){
        this.port=port;
    }



    public void run(){

        ServerSocket serverSocket;
        try {
            serverSocket = new ServerSocket(this.port);

            //System.out.print(InetAddress.getLocalHost());
            //System.out.print(" : ");
            System.out.print(this.port);
            System.out.println("   启动服务器....");

            String str;
            while (true) {

                Socket socket = serverSocket.accept();
                ObjectInputStream objectInputStream = new ObjectInputStream(socket.getInputStream());
                HashMap js_hm=(HashMap) objectInputStream.readObject();
                //this.IL_mat=txtToMat((String)js_hm.get("IL_mat"),4,50,50);
                this.IL_mat=(float[][][]) js_hm.get("IL_mat");
                this.d_kernel=(HashMap) js_hm.get("d_kernel");
                this.d_bias=(HashMap) js_hm.get("d_bias");
                this.real_score=(float[]) js_hm.get("realscore");
                this.step=Float.parseFloat((String)js_hm.get("step"));
                //System.out.println(fs_hm);
                bpNet bpn=new bpNet(this.IL_mat,this.d_kernel,this.d_bias,this.real_score,this.step);

                ObjectOutputStream objectOutputStream = new ObjectOutputStream(socket.getOutputStream());
                HashMap fs_hm=new HashMap();
                fs_hm.put("kernel",js_hm.get("d_kernel"));
                fs_hm.put("bias",js_hm.get("d_bias"));
                objectOutputStream.writeObject(fs_hm);
                objectOutputStream.flush();


                /*Socket socket = serverSocket.accept();
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(socket.getInputStream(), "UTF-8"));
                str= bufferedReader.readLine();
                JSONObject master_json = JSONObject.parseObject(str);
                //this.leixing=master_json.getString("leixing");
                this.IL_mat=txtToMat(master_json.getString("IL_mat"),4,50,50);
                import_kernel_bias(master_json.getString("d_kernel"),master_json.getString(("d_bias")));
                this.get_realscore(master_json.getString("realscore"));

                bpNet bpn=new bpNet(this.IL_mat,this.d_kernel,this.d_bias,this.real_score);
                JSONObject chuanshu_json=new JSONObject();
                //chuanshu_json.put("leixing",this.leixing);
                chuanshu_json.put("d_kernel",this.d_kernel);
                chuanshu_json.put("d_bias",this.d_bias);
                String to_master_mess=JSON.toJSONString(chuanshu_json);

                BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
                bufferedWriter.write(to_master_mess);
                bufferedWriter.write("\n");
                bufferedWriter.flush();
                socket.shutdownOutput();*/
                //socket.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        }

    }


    public static void main(String[] args){
        /*chuanshu_slave train_1;
        chuanshu_slave train_2;
        train_1=new chuanshu_slave(8888);
        train_2=new chuanshu_slave(8889);
        Thread thread_train1=new Thread(train_1);
        Thread thread_train2=new Thread(train_2);
        thread_train1.start();
        thread_train2.start();*/

        HashMap t_control=new HashMap<Integer,Thread>();

        for(int c=0;c<2;c++){
            t_control.put(c,new Thread(new chuanshu_slave(8888+c)));
            ((Thread)t_control.get(c)).start();
        }

    }
}