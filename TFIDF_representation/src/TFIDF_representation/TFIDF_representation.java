
package TFIDF_representation;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public interface TFIDF_representation {

    class TFIDFCalculator {
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
            System.out.println("\tTF ("+ term+") = " + tf(doc,term));
            System.out.println("\tIDF ("+ term+") = " + idf(docs,term)+"\n\n");
            return tf(doc, term) * idf(docs, term);

        }

    }

    static void main(String[] args) {

        List<String> doc1 = Arrays.asList("Lorem", "ipsum", "dolor", "ipsum", "sit", "ipsum");
        List<String> doc2 = Arrays.asList("Vituperata", "incorrupte", "at", "ipsum", "pro", "quo");
        List<String> doc3 = Arrays.asList("Has", "persius", "disputationi", "id", "simul");
        List<List<String>> documents = Arrays.asList(doc1, doc2, doc3);

        TFIDFCalculator calculator = new TFIDFCalculator();
        List<List<Double>> TFIDF_representations = new ArrayList<>();
        for(List<String> doc : documents){
            List<Double> docrepr = new ArrayList<>();
            for(String word : doc){
                double word_tfidf = calculator.tfIdf(doc, documents, word);
                docrepr.add(word_tfidf);
            }
            TFIDF_representations.add(docrepr);
        }
        System.out.println("Printing TF-IDF representations");
        for(List<Double> repr : TFIDF_representations){
            System.out.println(repr);
        }
    }
}