
import java.io.*;
import java.util.*;

/**
 * @author nikolas
 */

public class TFIDF_Representation {
    private static final String DELIMITER = " ";
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
            return Math.log(docs.size() / n);
        }

        /**
         * @param doc  a text document
         * @param docs all documents
         * @param term term
         * @return the TF-IDF of term
         */
        double tfIdf(List<String> doc, List<List<String>> docs, String term) {
            //System.out.println("\tTF ("+ term+") = " + tf(doc,term));
            //System.out.println("\tIDF ("+ term+") = " + idf(docs,term)+"\n\n");
            return tf(doc, term) * idf(docs, term);

        }

    }
    public static void main(String[] args) throws Exception {

        String directory_path = "/root/IdeaProjects/TFIDF_Model";
        for(int i = 0 ; i < args.length - 1 ; i++){
            switch (args[i]) {
                case "-path":  // project path that contains a Fasta_Files/ directory containing all fasta files
                    directory_path = args[i + 1];
                    break;
            }
        }


        File file = new File(directory_path+"/dataset.txt");
        BufferedReader br = new BufferedReader(new FileReader(file));
        int max_sentence_words = -1;
        List<List<String>> documents = new ArrayList<>();
        List<List<Integer>> labels = new ArrayList<>();
        String line;

        BufferedWriter writer2 = new BufferedWriter(new FileWriter(directory_path+"/tfidf_labels.dat"));

        ArrayList<Vector<Vector<String>>> init_sentences = new ArrayList<>();
        while ((line = br.readLine()) != null){
            StringTokenizer defaultTokenizer = new StringTokenizer(line);
            defaultTokenizer.nextToken();
            int document_id = Integer.parseInt(defaultTokenizer.nextToken());
            int sentences_num = Integer.parseInt(defaultTokenizer.nextToken());
            System.out.println("Document "+document_id+ ' '+ sentences_num);
            List<String> document = new ArrayList<>();

            Vector<Vector<String>> sentences = new Vector<>();
            for(int i = 0 ; i < sentences_num ; i++) {
                line = br.readLine();
                String[] fields = line.split("[\\[|\\]]");
                String text  = fields[1];
                writer2.write(fields[4]+"\n");
                String[] terms = text.split(DELIMITER);
                Vector<String> sentence = new Vector<>();
                int words_num = 0;
                for(String term : terms) {
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



        BufferedWriter writer = new BufferedWriter(new FileWriter(directory_path+"/tfidf_repr.dat"));

        TFIDFCalculator calculator = new TFIDFCalculator();
        ArrayList<HashMap<String,Double>> TFIDF_representations = new ArrayList<>();
        int document_id = 0;
        for(List<String> doc : documents){
            HashMap<String,Double> docmap = new HashMap<>();
            for(String word : doc){
                word = word.toLowerCase();
                double word_tfidf = calculator.tfIdf(doc, documents, word);
                word = word.replaceAll("\\s+$", "");
                System.out.println("Insert D"+document_id+ " /"+word+":"+word_tfidf + "len("+word.length()+")");
                docmap.put(word,word_tfidf);
            }
            document_id = document_id + 1;
            TFIDF_representations.add(docmap);
        }


        ArrayList<Vector<Vector<Double>>> sentence_representations = new ArrayList<>();

        for (int doc_id = 0 ; doc_id < document_id ; doc_id++){
            Vector<Vector<String>> doc = init_sentences.get(doc_id);
            for (Vector<String> sentence : doc) {
                Vector<Vector<Double>> tfidf_sequence = new Vector<>();
                for (String word : sentence) {
                    word = word.toLowerCase();
                    System.out.println("looking for "+word+" in D"+doc_id+ "len("+word.length()+")");
                    double word_repr = TFIDF_representations.get(doc_id).get(word);
                    System.out.println("word repr " + word_repr);
                    Vector<Double> dummy = new Vector<>();
                    dummy.add(word_repr);
                    tfidf_sequence.add(dummy);
                }
                sentence_representations.add(tfidf_sequence);
            }
        }

        Vector<Double> pad_vector = new Vector<>();
        pad_vector.add(0.0);
        for(Vector<Vector<Double>> sentence : sentence_representations){
            int sentence_words = sentence.size();
            while(sentence_words < max_sentence_words){
                sentence.add(pad_vector);
                sentence_words++;
            };
            writer.write(sentence+"\n");
        }
        writer.close();
        writer2.close();
    }

}