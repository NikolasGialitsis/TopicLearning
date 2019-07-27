
import java.io.*;
import java.util.*;
import java.util.HashMap;
/**
 * @author nikolas
 */

public class TFIDF_Representation    {

    private static final String DELIMITER = " ";
    private static HashMap<String,Double> IDF_values = new HashMap<>();

    public TFIDF_Representation() {
    }

    public static class TFIDFCalculator {
        /**
         * @param doc  list of strings
         * @param term String represents a term
         * @return term frequency of term in document
         */
        double tf(List<String> doc, String term) {
            double result = 0;
            for (String word : doc) {
                if (term.equalsIgnoreCase(word))
                    result++;
            }
            return result / doc.size();
        }

        /**
         * @param docs list of list of strings represents the dataset
         * @param term String represents a term
         * @return the inverse term frequency of term in documents
         */
        double idf(List<List<String>> docs, String term) {
            double n = 0;
            for (List<String> doc : docs) {
                for (String word : doc) {
                    if (term.equalsIgnoreCase(word)) {
                        n++;
                        break;
                    }
                }
            }
            double value = Math.log(docs.size() / n);
            IDF_values.put(term,value);
            return value;
        }

        /**
         * @param doc  a text document
         * @param docs all documents
         * @param term term
         * @return the TF-IDF of term
         */
        double tfIdf(List<String> doc, List<List<String>> docs, String term,String mode) {
            //System.out.println("\tTF ("+ term+") = " + tf(doc,term));
            //System.out.println("\tIDF ("+ term+") = " + idf(docs,term)+"\n\n");
            if(IDF_values.containsKey(term)){
              return tf(doc, term) * IDF_values.get(term);
            }
            else {
                if(mode.equals("train")) {
                    return tf(doc, term) * idf(docs, term);
                }
                else if(mode.equals("test")){
                    return 0.0;
                }
                else {
                    throw new UnsupportedOperationException();
                }
            }

        }

    }
    public static void main(String[] args) throws Exception {

        boolean load_idfs = false;
        String file_suffix = "train";
        String directory_path = "/root/IdeaProjects/TFIDF_Model";
        for(int i = 0 ; i < args.length - 1 ; i++){
            switch (args[i]) {
                case "-path":  // project path that contains a Fasta_Files/ directory containing all fasta files
                    directory_path = args[i + 1];
                    break;
                case "-test":
                    load_idfs = true;
                    file_suffix = "test";
            }
        }


        File file = new File(directory_path+"/"+file_suffix+"_dataset.txt");
        BufferedReader br = new BufferedReader(new FileReader(file));
        int max_sentence_words = -1;
        List<List<String>> documents = new ArrayList<>();
        List<List<Integer>> labels = new ArrayList<>();
        String line;
        BufferedWriter writer2 = new BufferedWriter(new FileWriter(directory_path
                + "/tfidf_labels_"+file_suffix+".dat"));


        //parse dataset and store in vector
        ArrayList<Vector<Vector<String>>> init_sentences = new ArrayList<>();
        while ((line = br.readLine()) != null){
            StringTokenizer defaultTokenizer = new StringTokenizer(line);
            defaultTokenizer.nextToken();
            int document_id = Integer.parseInt(defaultTokenizer.nextToken());
            int sentences_num = Integer.parseInt(defaultTokenizer.nextToken());
            System.out.println(file_suffix+"/Document "+document_id+ ' '+ sentences_num);
            List<String> document = new ArrayList<>();

            Vector<Vector<String>> sentences = new Vector<>();
            for(int i = 0 ; i < sentences_num ; i++) {
                line = br.readLine();
                String[] fields = line.split("\\[|\\]");
                System.out.println(fields[0]);
                System.out.println(fields[1]);
                String text  = fields[1];
                writer2.write(fields[4]+"\n");
                String[] terms = text.split(DELIMITER);
                Vector<String> sentence = new Vector<>();
                int words_num = 0;
                for(String term : terms) {
                    if(term.equals("") || term.equals(" "))continue;
                    document.add(term + " ");
                    sentence.add(term);
                    words_num = words_num + 1;
                }
                if(words_num > max_sentence_words){
                    max_sentence_words = words_num;
                }

                sentences.add(sentence);
                document.add("\n");

            }
            init_sentences.add(sentences);
            documents.add(document);
        }

        //load already saved idfs from map
        if(load_idfs){
            Properties properties = new Properties();
            properties.load(new FileInputStream("IDF_Map.properties"));
            for (String key : properties.stringPropertyNames()) {
                IDF_values.put(key,  Double.parseDouble(String.valueOf(properties.get(key))));
            }
        }

        BufferedWriter writer = new BufferedWriter(new FileWriter(directory_path+
                "/tfidf_repr_"+file_suffix+".dat"));
        //Calculate for each document its words' tfidf representation
        TFIDFCalculator calculator = new TFIDFCalculator();
        ArrayList<HashMap<String,Double>> TFIDF_representations = new ArrayList<>();
        int document_id = 0;
        for(List<String> doc : documents){
            HashMap<String,Double> docmap = new HashMap<>();
            for(String word : doc){
                word = word.toLowerCase();
                double word_tfidf = calculator.tfIdf(doc, documents, word,file_suffix);
                word = word.replaceAll("\\s+$", "");
                if(word_tfidf == 0.0)System.out.println("\t 0 > "+word);
                System.out.println("Insert D"+document_id+ " /"+word+":"+word_tfidf) ;
                docmap.put(word,word_tfidf);
            }
            document_id = document_id + 1;
            TFIDF_representations.add(docmap);
        }


        //Represent each original sentence with the sequence of its tfidf representations
        ArrayList<Vector<Vector<Double>>> sentence_representations = new ArrayList<>();

        for (int doc_id = 0 ; doc_id < document_id ; doc_id++){
            Vector<Vector<String>> doc = init_sentences.get(doc_id);
            for (Vector<String> sentence : doc) {
                Vector<Vector<Double>> tfidf_sequence = new Vector<>();
                for (String word : sentence) {
                    word = word.toLowerCase();
                    if(word.equals("") || word.equals(" "))continue;
                    System.out.println("looking for "+word+" in D"+doc_id);
                    double word_repr = TFIDF_representations.get(doc_id).get(word);
                    System.out.println("word repr " + word_repr);
                    Vector<Double> dummy = new Vector<>(); //dummy replaces scalar value x with [x]
                    dummy.add(word_repr);
                    tfidf_sequence.add(dummy);
                }
                sentence_representations.add(tfidf_sequence);
            }
        }

        //pad sentences with max length
        Vector<Double> pad_vector = new Vector<>();
        pad_vector.add(0.0);
        for(Vector<Vector<Double>> sentence : sentence_representations){
            int sentence_words = sentence.size();
            while(sentence_words < max_sentence_words){
                sentence.add(pad_vector);
                sentence_words++;
            }
            writer.write(sentence+"\n");
        }
        writer.close();
        writer2.close();

        //save IDF map
        if(!load_idfs) {
            Properties properties = new Properties();
            for (HashMap.Entry<String, Double> entry : IDF_values.entrySet()) {
                properties.put(entry.getKey(), entry.getValue().toString());
            }
            properties.store(new FileOutputStream("IDF_Map.properties"), null);
        }
    }

}