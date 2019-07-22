import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.topics.*;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Vector;
/**
 * @author nikolas
 */





class Probabilistic_Representation {


    private String instances_path;
    private int instances_num;
    private int topics_num;
    private String stop_words_path;
    private ArrayList<HashMap<String,Double>> TermTopicContribution;

    Probabilistic_Representation(String instances_path, int topics_num,String stop_words_path) {
        this.instances_path = instances_path;
        this.topics_num = topics_num;
        this.stop_words_path = stop_words_path;
        this.TermTopicContribution = new ArrayList<>();
        this.instances_num = 1;
    }



    HashMap<String,Vector<Double>> getTermTopicContribution() throws IOException {

        // Begin by importing documents from text to feature sequences
        ArrayList<Pipe> pipeList = new ArrayList<>();

        // Pipes: lowercase, tokenize, remove stopwords, map to features
        pipeList.add( new CharSequenceLowercase() );
        pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")) );
       // if(!stop_words_path.isEmpty())pipeList.add( new TokenSequenceRemoveStopwords(new File(stop_words_path), "UTF-8", false, false, false) );
        pipeList.add( new TokenSequence2FeatureSequence() );

        InstanceList instances = new InstanceList (new SerialPipes(pipeList));

        Reader fileReader = new InputStreamReader(new FileInputStream(new File(this.instances_path)), StandardCharsets.UTF_8);
        instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
                3, 2, 1)); // data, label, name fields

        // Create a model with 10 topics, alpha_t = 0.01, beta_w = 0.01
        //  Note that the first parameter is passed as the sum over topics, while
        //  the second is
        ParallelTopicModel model = new ParallelTopicModel(topics_num, 1.0, 0.01);

        model.addInstances(instances);

        // Use two parallel samplers, which each look at one half the corpus and combine
        //  statistics after every iteration.
        model.setNumThreads(2);

        // Run the model for 50 iterations and stop (this is for testing only,
        //  for real applications, use 1000 to 2000 iterations)
        model.setNumIterations(1500);
        model.estimate();

        for( int i = 0 ; i < this.instances_num ; i++){
            // The data alphabet maps word IDs to strings
            Alphabet dataAlphabet = instances.getDataAlphabet();
            // Get an array of sorted sets of word ID/count pairs
            ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();
            for (int topic = 0; topic < topics_num; topic++) {
                double total_occurrences = 0;
                HashMap<String,Double> TermFrequencies = new HashMap<>();
                for (IDSorter idCountPair : topicSortedWords.get(topic)) {
                    Object obj = dataAlphabet.lookupObject(idCountPair.getID());
                    TermFrequencies.putIfAbsent(obj.toString(), 0.0);
                    double occurrences = TermFrequencies.get(obj.toString()) + 1;
                    TermFrequencies.put(obj.toString(), occurrences);
                    total_occurrences = total_occurrences + occurrences;
                }
                for(String s : TermFrequencies.keySet()){
                    if(total_occurrences == 0.0)continue;
                    double old_value = TermFrequencies.get(s);
                    TermFrequencies.put(s,old_value/total_occurrences);
                }
                this.TermTopicContribution.add(TermFrequencies);
            }
        }

        HashMap<String,Vector<Double>> TermRepresentations = new HashMap<>();
        int topic_id = 0;
        for(HashMap<String,Double> TopicMap : this.TermTopicContribution){
            for(String term : TopicMap.keySet()) {
                double contribution_to_topic_percent = TopicMap.get(term);
                if(!TermRepresentations.containsKey(term)){
                    Vector<Double> v = new Vector<>();
                    for(int i = 0 ; i < topics_num ; i++){
                        v.add(0.0);
                    }
                    TermRepresentations.put(term,v);
                }
                Vector<Double> vect = TermRepresentations.get(term);
                vect.set(topic_id,contribution_to_topic_percent);
                TermRepresentations.put(term,vect);
            }
            topic_id = topic_id + 1;
        }


        return TermRepresentations;
    }

}

