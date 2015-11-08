import weka.core.Attribute;
import weka.core.Instances;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by mhorowitzgelb on 11/7/15.
 */
public class Main {
    public static void main(String[] args) throws IOException {
        Instances instances = new Instances(new FileReader("/home/mhorowitzgelb/Downloads/leaderboard_final.arff"));
        FileWriter writer = new FileWriter("/home/mhorowitzgelb/Research/leaderboard.csv");
        writer.write("DEATH,");
        for(int i = 0; i < instances.numAttributes(); i ++){
           if(i <= 13)
               continue;
            Attribute attr = instances.attribute(i);
            if(attr.isNominal()){
                for(int)
            }
        }
    }
}
