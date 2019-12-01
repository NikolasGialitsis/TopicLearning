package thesisinterface.VectorRepresentation.ProbabilisticModel;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.io.*;
import thesisinterface.VectorRepresentation.BaseClasses.*;
import thesisinterface.IOHandler;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @author nikolas
 */


public class InterfaceTopicModel {


    private static File outputSparseFile1;
    private static File outputSparseFile2;

    // convert data to WEKA instances
    private static Instances toWekaInstances(ArrayList<BaseFeatureVector> data) {
        ArrayList<Attribute> attributeList = new ArrayList<>();
        int maxDim = data.get(0).size();
        for (int i = 0; i < maxDim; ++i) {
            Attribute att = new Attribute("attribute" + (i + 1), false);
            attributeList.add(att);
        }
        Instances instances = new Instances("data", attributeList, data.size());
        for (BaseFeatureVector vec : data) {
            Instance instance = new SparseInstance(attributeList.size());
            for (int i = 0; i < attributeList.size(); ++i) {
                List<Double> dlist = vec.get(i);
                double val = dlist.get(0);
                instance.setValue(attributeList.get(i), val);
            }
            instances.add(instance);
        }
        return instances;
    }

   /* public static List<String> splitEqually(String text, int size) {
        // Give the list the right capacity to start with. You could use an array
        // instead if you wanted.
        ArrayList<String> ret = new ArrayList<>((text.length() + size - 1) / size);

        for (int start = 0; start < text.length(); start += size) {
            ret.add(text.substring(start, Math.min(text.length(), start + size)));
        }
        return ret;
    }
    */



   //Given an input sequence, returns it's k_mer (ngram) representation , aka. all substrings of length k
    private static Sequences k_mer(Sequence sequence, int k) {

        int number_of_k_mers = sequence.size()-k+1;
        if(number_of_k_mers < 0)number_of_k_mers = 0;
        Sequences ret = new Sequences(number_of_k_mers);

        for (int start = 0; start < number_of_k_mers; start++) {
            ret.add(new Sequence(sequence.asString().substring(start, start + Math.min(sequence.size(),k))));
        }
        return ret;
    }



    public static void main(String[] args) throws Exception {

        String directory_path = "/root/IdeaProjects/ProbabilisticModel";
        String stop_words_path = "";
        int topics_num = 10;
        int word_length = -1;
        for(int i = 0 ; i < args.length - 1 ; i++){
            switch (args[i]) {
                case "-topics": //number of topics for topic model
                    topics_num = Integer.parseInt(args[i + 1]);
                    break;
                case "-length": //word length(k) for k-mer decomposition // higher k equals higher computational time
                    word_length = Integer.parseInt(args[i + 1]);
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
        System.out.println("Word length : "+word_length);
        if(word_length == -1)
            word_length = CalculateZipfTotal(directory_path);//use Zipf's law to calculate the word length

        try (Stream<Path> paths = Files.walk(Paths.get(directory_path+"/Fasta_Files"))) {
            //create a list of the directories of all subfolders and files contained
            List<String> pathList = paths.map(p -> {
                if (Files.isDirectory(p)) {
                    return "/" + p.toString();
                }
                return p.toString();
            })
                    .peek(System.out::println) // write all results in console for debug
                    .collect(Collectors.toList());
            String output_directory = "/Sparse_Files_T"+topics_num+"_K_"+word_length; //create a folder to store results
            //run the list of directories
            //ignore all items of list that are not fasta files
            for (int i = 0; i < pathList.size() ; i++) { //each directory should contain exactly two fasta files
                if (pathList.get(i).endsWith(".fas")) {
                    File file1 = new File(pathList.get(i));
                    //first file in the comparison
                    outputSparseFile1 = new File(pathList.get(i).replace("Fasta_Files", output_directory));
                    i = i+1;
                    while (!pathList.get(i).endsWith(".fas")) { //search the directory for the second fasta file
                     //   System.out.println(pathList.get(i + 1) + " does not end with .fas");
                        i++;
                    }
                    //second file in the comparison
                    outputSparseFile2 = new File(pathList.get(i).replace("Fasta_Files", output_directory));
                    File file2 = new File(pathList.get(i));
                    //transform the sequences to a format that can be used by machine learning algorithms
                    makeTopicSparse(directory_path,file1.getPath(), file2.getPath(), stop_words_path,topics_num, word_length);
                }
            }
        }

    }


    //Zipf's law states that in a large enough text, about 50% of the words occur exactly once.
    //The set of words occuring once are called "Hapax Legomenon"
    //This function returns the  word length that minimizes the distance |hapax_legomenon_percentage - 0.50|.
    //To reduce calculations a smoothing factor is applied so that a k that produces a hapax percentage of 45-55%
    //  is automatically accepted.


    private static int CalculateZipfTotal(String directory_path) throws IOException {
        IOHandler handler = new IOHandler();
        makeJointFile(directory_path);
        ArrayList<String> str = handler.readFAS(directory_path+"/joint_file.txt");
        Sequences seq = new Sequences();
        for(String s : str)
            seq.add(new Sequence(s));
        int k = CalculateZipfLength(seq);
        System.out.println("Zipf's length = " + k);
        return k;
    }

    static int CalculateZipfLength(Sequences seq){
        int k = 3;
        int best_k = k;
        double closer_percent = 1;
        while(k < 20) {
            double hapax_percent = CalculateHapaxLegomenon(k, seq);
            System.out.println("hapax percent = "+ hapax_percent);
            double diff = Math.abs(hapax_percent - 0.5);
            if (diff <= 0.05) {
                return k;
            }
            if (diff < closer_percent) {
                closer_percent = diff;
                best_k = k;
            }
            k++;
        }
        return best_k;
    }

    //Return the percentage  (words occuring once / all word occurences)
    private static double CalculateHapaxLegomenon(int k ,Sequences seq){
        System.out.println("Calculate Hapax Legomenon");
        HashMap<Sequence,Integer> map = new HashMap<>() ;
        for(Sequence s : seq) {
            Sequences subsequence = new Sequences();
            for (int start = 0; start < (s.size()/k); start+=k) {
                subsequence.add(new Sequence(s.asString().substring(start, start + k)));
            }
            for (Sequence ngram : subsequence){
                map.putIfAbsent(ngram,0);
                int occurrences = map.get(ngram) + 1;
                map.put(ngram,occurrences);
            }
        }
        int total = 0;
        int hapax = 0;
        for ( Sequence entry : map.keySet()){
            if(map.get(entry) == 1)hapax++;
            total++;
        }
        return (double)hapax /(double)total;
    }





    //Create a file in the directory that contains all the genomic sequences in a raw form
    static void makeJointFile(String directory_path ) throws IOException {
        try (Stream<Path> paths = Files.walk(Paths.get(directory_path+"/Fasta_Files"))) {

            //create a list of the directories of all subfolders and files contained
            List<String> pathList = paths.map(p -> {
                if (Files.isDirectory(p)) {
                    return "/" + p.toString();
                }
                return p.toString();
            })
                    .peek(System.out::println) // write all results in console for debug
                    .collect(Collectors.toList());

            FileWriter fw = new FileWriter(directory_path+"/joint_file.txt",false);
            BufferedWriter writer = new BufferedWriter(fw);

            //run the list of directories
            //ignore all items of list that are not fasta files
            for (String aPathList : pathList) {
                if (aPathList.endsWith(".fas")) {
                    IOHandler handler = new IOHandler();
                    Sequences content = new Sequences(handler.readFAS(aPathList));
                    for (Sequence s : content) {
                        writer.write(s.asString());
                        writer.newLine();
                    }
                }
            }
            writer.close();
        }
    }

    //to create the sparse ARFF representation file
    private static void makeTopicSparse(String directory_path,String fasPath1, String fasPath2,String stop_words_path,
                                        int topics_num,int word_length) throws IOException {
        IOHandler handler = new IOHandler();
        Sequences DNAseq1 = new Sequences(handler.readFAS(fasPath1));
        Sequences DNAseq2 = new Sequences(handler.readFAS(fasPath2));
        BufferedWriter writer = null;
        /*int min_length = -1;
        String shortest_line = "";
        for (String line : DNAseq) {
            if((min_length == -1)||(line.length() < min_length)) {
                min_length = line.length();
                shortest_line = line;
            }
        }
        */
        //int  min_length = topics_num;
        //min_length = Math.max(3,min_length);
        //System.out.println("min length = "+min_length);

        int instances_num = 0;
        try {//content.txt is used as a temporary file to save the tokenized version of the initial sequences
            writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(directory_path+"/content.txt"), StandardCharsets.UTF_8));

            System.out.println("Word Length = "+word_length);
            for (Sequence line : DNAseq1) {
                instances_num+=1;
                //List<String> substrings = splitEqually(line,min_length);

                Sequences substrings = k_mer(line,word_length);
                for( Sequence sub : substrings){
                    writer.write(sub.asString()+" ");
                }
                writer.newLine();
            }
        } catch (IOException ex) {
            // Report
        } finally {
            try {
                assert writer != null;
                writer.close();} catch (Exception ex) {/*ignore*/}
        }


        int first_num = instances_num;
        try { // append the content.txt with the tokenized sequence of the second fasta file
            FileWriter fw = new FileWriter(directory_path+"/content.txt",true);
            writer = new BufferedWriter(fw);
            System.out.println("Word Length = "+word_length);
            for (Sequence line : DNAseq2) {
                instances_num+=1;
                //List<Sequence> substrings = splitEqually(line,min_length);
                Sequences substrings = k_mer(line,word_length);
                for( Sequence sub : substrings){
                    writer.write(sub.asString()+" ");
                }
                writer.newLine();
            }
        } catch (IOException ex) {
            // Report
        } finally {
            try {
                writer.close();} catch (Exception ex) {/*ignore*/}
        }

        String mallet_instances_path = directory_path+"/content.txt";
        TopicsRepresentation representation; // Run the topic model representation
        representation = new TopicsRepresentation(mallet_instances_path,instances_num,topics_num,stop_words_path);
        List<List<Double>> list;
        list = representation.getTopicsDistribution(); //Get the topic distribution of the comparison file
                                                        //a node represents the distribution of an instance(line)
        representation.PrintTopWords();

        if(word_length <= 6) {
            String cne_seq = "taatta".substring(0, word_length ); //Polychroniou et.al 'CNEs are enriched in TAATAA'
            representation.SearchTop(new Sequence(cne_seq));
        }
        IOHandler io = new IOHandler();
        ArrayList<BaseFeatureVector> data1 = new ArrayList<>();
        ArrayList<BaseFeatureVector> data2 = new ArrayList<>();
        int count = 0;
        for(List<Double> entry : list ){
            int dimension = 0;
            BaseFeatureVector bfv = new BaseFeatureVector();
            for(double ignored : entry){
                List<Double> dlist = new ArrayList<>();
                dlist.add(entry.get(dimension));
                bfv.put(dimension,dlist);
                dimension+=1;
            }
            count++;
            if(count <= first_num)
                data1.add(bfv);
            else
                data2.add(bfv);
        }
        // convert to weka instances
        Instances instances1 = toWekaInstances(data1);
        io.writeARFF(outputSparseFile1 + ".arff", instances1);

        Instances instances2 = toWekaInstances(data2);
        io.writeARFF(outputSparseFile2 + ".arff", instances2);


        

    }
}