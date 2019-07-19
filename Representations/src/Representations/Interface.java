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
        String mallet_instances_path = directory_path+"/sentences.txt";
        Probabilistic_Representation model; // Run the topic model representation
        model = new Probabilistic_Representation(mallet_instances_path,topics_num,stop_words_path);
        HashMap<String,Vector<Double>> TermTopicContribution = model.getTermTopicContribution();


        String line;

        ArrayList<Vector<Vector<Double>>> sentence_representations = new ArrayList<>();

        File file = new File(directory_path+"/dataset.txt");
        BufferedReader br = new BufferedReader(new FileReader(file));
        while ((line = br.readLine()) != null){
            StringTokenizer defaultTokenizer = new StringTokenizer(line);
            defaultTokenizer.nextToken();
            int document_id = Integer.parseInt(defaultTokenizer.nextToken());
            int sentences_num = Integer.parseInt(defaultTokenizer.nextToken());
            System.out.println("Document "+document_id+ ' '+ sentences_num);
            for(int i = 0 ; i < sentences_num ; i++) {
                line = br.readLine();
                String[] fields = line.split("\\[|\\[");
                String text  = fields[1];
                String[] terms = text.split(DELIMITER);
                Vector<Vector<Double>> sentence = new Vector<>();
                for (String term : terms) {
                    term = term.toLowerCase();
                    if (TermTopicContribution.containsKey(term)) {
                        sentence.add(TermTopicContribution.get(term));
                    }
                }
                System.out.println("\t"+sentence);
                sentence_representations.add(sentence);
            }
        }
    }
}