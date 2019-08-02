import cc.mallet.pipe.SerialPipes;
import cc.mallet.topics.ParallelTopicModel;

import java.util.*;
import java.io.*;
/**
 * @author nikolas
 */


public class Probabilistic_Interface {

    private static Vector<Vector<Double>> normalize_vectors(Vector<Vector<Double>> vectors){

        Vector<Vector<Double>> new_v = new Vector<>();
        for(int i = 0 ; i < vectors.size(); i++){
            double sum = 0.0;
            Vector<Double> vector = vectors.get(i);
            for(double val : vector){
                sum = sum + val;
            }
            for(int j = 0 ; j < vector.size() ; j++){
                if(vector.get(j) == 0 )continue;

                vector.set(j,(vector.get(j)*1.0)/(sum));
            }
            new_v.add(vector);
        }
        return new_v;
    }

    private static Vector<Vector<Double>>  average_vectors(Vector<Vector<Double>> vectors){

        for(int i = 0 ; i < vectors.size(); i++) {
            int N = vectors.size();
            double sum = 0.0;
            for (double val : vectors.elementAt(i)) {
                sum = sum + val;
            }
            Vector<Double> new_v = new Vector<>();
            new_v.add(sum / N);
            vectors.set(i, new_v);
        }
        System.out.println(vectors);

        return vectors;
    }


    private static Vector<Vector<Double>>  maximize_vectors(Vector<Vector<Double>> vectors){

        for(int i = 0 ; i < vectors.size(); i++) {
            double max_val = -1.0;
            for (double val : vectors.elementAt(i)) {
               if(val > max_val){
                   max_val = val;
               }
            }
            Vector<Double> new_v = new Vector<>();
            new_v.add(max_val);
            vectors.set(i, new_v);
        }
        System.out.println(vectors);

        return vectors;
    }

    private static final String DELIMITER = " ";
    public static void main(String[] args) throws Exception {

        String directory_path = "/root/IdeaProjects/ProbabilisticModel";
        String stop_words_path = "";
        int topics_num = 10;
        String mode = "train";
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
                case "-test":
                    mode ="test";
                    break;
            }
        }


        System.out.println("Topics num : "+topics_num);
        String mallet_instances_path = directory_path+"/"+mode+"_sentences.txt";
        Probabilistic_Representation model  = null;
        if(mode.equals("test")) {
            model = new Probabilistic_Representation(mallet_instances_path, topics_num, stop_words_path,true);
        }
        else{
            model =  new Probabilistic_Representation(mallet_instances_path, topics_num, stop_words_path,false);
        }

        HashMap<String, Vector<Double>> TermTopicContribution = model.getTermTopicContribution();
        String line;
        ArrayList<Vector<Vector<Double>>> sentence_representations = new ArrayList<>();
        BufferedWriter writer = new BufferedWriter(new FileWriter(directory_path+"/prob_repr_"+mode+".dat"));
        BufferedWriter writer2 = new BufferedWriter(new FileWriter(directory_path+"/prob_labels_"+mode+".dat"));
        File file = new File(directory_path+"/"+mode+"_dataset.txt");
        BufferedReader br = new BufferedReader(new FileReader(file));
        int max_sentence_words = -1;

        //parse dataset and store sentence representations
        while ((line = br.readLine()) != null){
            StringTokenizer defaultTokenizer = new StringTokenizer(line);
            defaultTokenizer.nextToken();
            int document_id = Integer.parseInt(defaultTokenizer.nextToken());
            int sentences_num = Integer.parseInt(defaultTokenizer.nextToken());
            System.out.println("Document "+document_id+ ' '+ sentences_num);


            for(int i = 0 ; i < sentences_num ; i++) {
                line = br.readLine();
                String[] fields = line.split("\\[|\\]");
                String text  = fields[1];
                String[] terms = text.split(DELIMITER);
                Vector<Vector<Double>> sentence = new Vector<>();

                if(terms.length > max_sentence_words){
                    max_sentence_words = terms.length;
                }
                //calculation of term representations
                for (String term : terms) {

                    term = term.toLowerCase();
                    if (TermTopicContribution.containsKey(term)) {
                        sentence.add(TermTopicContribution.get(term));
                    }
                }
                System.out.println("\t"+sentence);
                writer2.write(fields[4]+"\n");
                //sentence = normalize_vectors(sentence);
                //sentence = average_vectors(sentence);
                sentence = maximize_vectors(sentence);
                topics_num = 1;

                sentence_representations.add(sentence);
            }
        }
        // add padding to sentences
        Vector<Double> pad_vector = new Vector<>();
        for(int i = 0 ; i < topics_num ; i++){
            pad_vector.add(0.0);
        }
        for(Vector<Vector<Double>> sentence : sentence_representations ){
            int sentence_words = sentence.size();
            while(sentence_words < max_sentence_words){
                sentence.add(pad_vector);
                sentence_words++;
            }
            writer.write(sentence+"\n");
        }
        writer.close();
        writer2.close();
    }
}