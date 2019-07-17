package Representations;
import java.util.*;
import java.io.*;
/**
 * @author nikolas
 */


public class Interface{


    private static final String DELIMITER = " ";

    public static void main(String[] args) throws Exception {

        String directory_path = "/root/IdeaProjects/ProbabilisticModel";
        String stop_words_path = "";
        int topics_num = 10;
        for(int i = 0 ; i < args.length - 1 ; i++){
            switch (args[i]) {
                case "-topics": //number of topics for topic model
                    topics_num = Integer.parseInt(args[i + 1]);
                    break;
                case "-path":  // project path that contains a Fasta_Files/ directory containing all fasta files
                    directory_path = args[i + 1];
                    break;
                case "-stop":  // path to a file containing the list of stop words used by the topic model
                    stop_words_path = args[i+1];
                    break;
            }
        }
        System.out.println("Topics num : "+topics_num);
        String mallet_instances_path = directory_path+"/content.txt";
        Probabilistic_Representation model; // Run the topic model representation
        model = new Probabilistic_Representation(mallet_instances_path,topics_num,stop_words_path);
        HashMap<String,Double> TermTopicContribution = model.getTermTopicContribution();
        File file = new File(mallet_instances_path);
        BufferedReader br = new BufferedReader(new FileReader(file));

        String line;

        while ((line = br.readLine()) != null){
            String[] terms = line.split(DELIMITER);
            ArrayList<Double> values = new ArrayList<>();
            for(String term : terms){
                term = term.toLowerCase();
                if(TermTopicContribution.containsKey(term)) {
                    values.add(TermTopicContribution.get(term));
                }
                else{
                    System.out.println("\t"+term+" not represented");
                    values.add(0.0);
                }
            }
            System.out.println(values);
        }
    }
}