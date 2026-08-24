
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.*;

import static java.nio.file.StandardCopyOption.*;

public class copyFile {

    public  static void main(String[] args) throws IOException{
        CopyOption copyOption=COPY_ATTRIBUTES;


        String[] dir_name={"luodi","shuiti","jianzhu","zhibei"};

        for (int dnn=0;dnn<4;dnn++){
            String dirname = "H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//reaction//redo//txt//"+dir_name[dnn];
            File f1 = new File(dirname);
            String[] s = f1.list();
            File source;
            File target;

            for(int i=0;i<s.length;i++){

                source=new File(dirname + "//" + s[i]);

                //sourcePath = Paths.get(dirname + "//" + s[i]);

                if(i%4==0){
                    //targetPath = Paths.get("H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//train//jianzhu" + s[i]);
                    target = new File("H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//reaction//redo//test//"+dir_name[dnn]+"//" + s[i]);
                }
                else{
                    //targetPath = Paths.get("H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//test//jianzhu" + s[i]);
                    target = new File("H://java_wk//remote_sensing_data//sz_1m//sz1m//a___cnn//reaction//redo//train//"+dir_name[dnn]+"//"  + s[i]);
                }
                //Files.copy(sourcePath, targetPath,copyOption);
                Files.copy(source.toPath(), target.toPath(),copyOption);

            }
        }

    }


}
