package thesisinterface.VectorRepresentation.ProbabilisticModel;

import thesisinterface.VectorRepresentation.BaseClasses.BaseFeatureVector;
import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.topics.*;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.*;
import java.io.*;

import java.util.ArrayList;
import java.util.List;
import thesisinterface.VectorRepresentation.BaseClasses.Sequence;
/**
 * @author nikolas
 */

class TopicsRepresentation extends BaseFeatureVector {

    private String instances_path;
    private int instances_num;
    private int topics_num;
    private String stop_words_path;
    private HashMap<String,Integer> top_words_list;

    TopicsRepresentation(String instances_path, int instances_num,int topics_num,String stop_words_path) {
        this.instances_path = instances_path;
        this.instances_num  = instances_num;
        this.topics_num = topics_num;
        this.stop_words_path = stop_words_path;
        top_words_list = new HashMap<>();
    }



    List<List<Double>> getTopicsDistribution() throws IOException {

        // Begin by importing documents from text to feature sequences
        ArrayList<Pipe> pipeList = new ArrayList<>();

        // Pipes: lowercase, tokenize, remove stopwords, map to features
        pipeList.add( new CharSequenceLowercase() );
        pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")) );
        if(!stop_words_path.isEmpty())pipeList.add( new TokenSequenceRemoveStopwords(new File(stop_words_path), "UTF-8", false, false, false) );
        pipeList.add( new TokenSequence2FeatureSequence() );

        InstanceList instances = new InstanceList (new SerialPipes(pipeList));

        Reader fileReader = new InputStreamReader(new FileInputStream(new File(this.instances_path)), StandardCharsets.UTF_8);
        instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
                3, 2, 1)); // data, label, name fields

        // Create a model with 10 topics, alpha_t = 0.01, beta_w = 0.01
        //  Note that the first parameter is passed as the sum over topics, while
        //  the second is
        ParallelTopicModel model = new ParallelTopicModel(topics_num, 1.0, 0.01);
        model.setRandomSeed(111);
        model.addInstances(instances);

        // Use two parallel samplers, which each look at one half the corpus and combine
        //  statistics after every iteration.
        model.setNumThreads(2);

        // Run the model for 50 iterations and stop (this is for testing only,
        //  for real applications, use 1000 to 2000 iterations)
        model.setNumIterations(1000);
        model.estimate();

        // Estimate the topic distribution of the first instance,
        //  given the current Gibbs state.
        List<List<Double>> list = new ArrayList<>();

        for( int i = 0 ; i < this.instances_num ; i++){
            //System.out.println("Instance :"+i);
            // Show the words and topics in the first instance

            // The data alphabet maps word IDs to strings
            Alphabet dataAlphabet = instances.getDataAlphabet();

            FeatureSequence tokens = (FeatureSequence) model.getData().get(i).instance.getData();
            LabelSequence topics = model.getData().get(i).topicSequence;


            Formatter out = new Formatter(new StringBuilder(), Locale.US);
            for (int position = 0; position < tokens.getLength(); position++) {
                out.format("%s-%d ", dataAlphabet.lookupObject(tokens.getIndexAtPosition(position)),
                       topics.getIndexAtPosition(position));
            }
            //System.out.println(out);

            // Estimate the topic distribution of the first instance,
            //  given the current Gibbs state.
            double[] topicDistribution = model.getTopicProbabilities(i);

            // Get an array of sorted sets of word ID/count pairs
           ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();

            // Show top 10 words in topics with proportions for the first document
            for (int topic = 0; topic < topics_num; topic++) {
                Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();
                //out = new Formatter(new StringBuilder(), Locale.US);
                //out.format("%d\t%.3f\t", topic, topicDistribution[topic]);
                int rank = 0;
                while (iterator.hasNext() && rank < 10) {
                    IDSorter idCountPair = iterator.next();
                    //out.format("%s (%.0f) ", dataAlphabet.lookupObject(idCountPair.getID()), idCountPair.getWeight());
                    Object obj = dataAlphabet.lookupObject(idCountPair.getID());
                    top_words_list.putIfAbsent(obj.toString(),0);
                    int occurrences = top_words_list.get(obj.toString()) + 1;
                    top_words_list.put(obj.toString(),occurrences);
                    rank++;
                }
                //System.out.println(out);
            }

            List<Double> entry_dist = new ArrayList<>();
            for(double d : topicDistribution){
                //System.out.println("Topic:"+d);
                entry_dist.add(d);
            }
            list.add(entry_dist);
        }
        return list;
    }

    void PrintTopWords(){
        for(String s : top_words_list.keySet()){
            System.out.println("String "+s+" / count : "+top_words_list.get(s));
        }
    }

    void SearchTop(Sequence seq){
        if(top_words_list.containsKey(seq.asString()))
            System.out.println("Found "+seq.asString()+" in top words list "+top_words_list.get(seq.asString())+ " times");
        else
            System.out.println("Found "+seq.asString()+" in top words list 0 times");
    }
}

