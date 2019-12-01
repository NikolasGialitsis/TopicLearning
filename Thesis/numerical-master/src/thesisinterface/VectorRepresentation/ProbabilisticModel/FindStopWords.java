package thesisinterface.VectorRepresentation.ProbabilisticModel;

import thesisinterface.VectorRepresentation.BaseClasses.Sequence;
import thesisinterface.VectorRepresentation.BaseClasses.Sequences;
import thesisinterface.IOHandler;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author nikolas
 */

//Create a list of stop words aka. a list of tokens that the Topic Model should ignore
//in order to improve performance and reduce calculations
public class FindStopWords {

    public static void main(String[] args) throws Exception {
//Two options are available in order to add words into the list.


        int k = 3;
        double tfidf_filter = 0.01;
        double df_filter = 0.5;
        boolean df = false;
        boolean tfidf = false;
        String directory_path = "/root/IdeaProjects/ProbabilisticModel";
        for(int i = 0 ; i < args.length ; i++){
            switch (args[i]) {
                case "-f":
                    if(i == args.length)break;
                    tfidf_filter = Double.parseDouble(args[i + 1]);
                    i++;
                    break;
                case "-maxdocs":
                    if(i == args.length)break;
                    df_filter = Double.parseDouble(args[i + 1]);
                    i++;
                    break;
                case "-k":
                    if(i == args.length)break;
                    k = Integer.parseInt(args[i + 1]);
                    i++;
                    break;
                case "-df":
                    df = true;
                    break;
                case "-tfidf":
                    tfidf = true;
                    break;
                case "-path":
                    if(i == args.length)break;
                    directory_path = args[i + 1];
                    i++;
                    break;
            }

        }
        Sequences stop_words = new Sequences();
        if(tfidf){
            System.out.println("Running TFIDF");
            stop_words.addAll(TFIDF(directory_path,k,tfidf_filter));
        }
        if(df){
            System.out.println("Running DF");
            stop_words.addAll(DF(directory_path,k,df_filter));
        }

        FileWriter fw = new FileWriter(directory_path+"/stopwords.txt",false);
        BufferedWriter writer = new BufferedWriter(fw);
        for(Sequence s : stop_words){
            System.out.println(s.asString());
            writer.write(s.asString());
            writer.newLine();
        }
        writer.close();
    }

    private static Sequences TFIDF(String directory_path,int k ,double filter ){

        Sequences StopWords = new Sequences();
        try (Stream<Path> paths = Files.walk(Paths.get(directory_path+"/Fasta_Files"))) {

            //create a list of the directories of all subfolders and files contained
            List<String> pathList = paths.map(p -> {
                if (Files.isDirectory(p)) {
                    return "/" + p.toString();
                }
                return p.toString();
            })
                    //.peek(System.out::println) // write all results in console for debug
                    .collect(Collectors.toList());

            int total_docs = 0;
            for (String aPathList : pathList) {
                if (aPathList.endsWith(".fas")) {
                    total_docs++;
                }
            }
            HashMap<String,HashMap<Integer,Integer>> TermOccurrences = new HashMap<>(total_docs);
            ArrayList<Integer> DocumentTerms = new ArrayList<>(total_docs);
            int documents_num = 0;
            for (String aPathList : pathList) {
                if (aPathList.endsWith(".fas")) {
                    documents_num++;
                    DocumentTerms.add(documents_num-1,0);
                    //DocumentTerms.set(documents_num-1,0);
                    IOHandler handler = new IOHandler();
                    Sequences seq = new Sequences(handler.readFAS(aPathList));
                    for(int i = 0 ; i < seq.size() ; i++){
                        seq.set(i,new Sequence(seq.get(i).asString().replace('T','t')));
                        seq.set(i,new Sequence(seq.get(i).asString().replace('A','a')));
                        seq.set(i,new Sequence(seq.get(i).asString().replace('C','c')));
                        seq.set(i,new Sequence(seq.get(i).asString().replace('G','g')));
                    }

                    Sequences ngram_list = new Sequences();
                    for (Sequence s : seq) {
                        Sequences subsequence = new Sequences();
                        for (int start = 0; start < (s.size() / k); start += k) {
                            subsequence.add(new Sequence(s.asString().substring(start, start + k)));
                        }
                        ngram_list.addAll(subsequence);
                    }

                    for(Sequence s : ngram_list){
                        TermOccurrences.computeIfAbsent(s.asString(), k1 -> new HashMap<>());
                        TermOccurrences.get(s.asString()).putIfAbsent(documents_num-1,0);
                        int occurrences = TermOccurrences.get(s.asString()).get(documents_num-1) + 1;
                        //System.out.println("occurrences = "+occurrences);
                        HashMap<Integer,Integer> map = TermOccurrences.get(s.asString());
                        map.put(documents_num-1,occurrences);
                        TermOccurrences.put(s.asString(),map);
                        //occurrences = TermOccurrences.get(s).get(documents_num-1) ;
                        //System.out.println("\t"+occurrences);
                        DocumentTerms.set(documents_num-1,DocumentTerms.get(documents_num-1)+1);
                    }
                }
            }


            HashMap<String,Double> TFIDF_map = new HashMap<>();
            for(String s : TermOccurrences.keySet()){
                ArrayList<Double> TermFrequencies = new ArrayList<>(documents_num);
                int document_appears = 0;
               // System.out.println("String = "+s);
                for(int document = 0 ; document < documents_num ; document++){
                    //System.out.println("\tDoc = "+document);
                    int terms = DocumentTerms.get(document);
                    int occurrences = 0;
                    if(TermOccurrences.containsKey(s)) {
                        if(TermOccurrences.get(s) == null)System.out.println("not initialized hash");
                        else if(TermOccurrences.get(s).containsKey(document))
                            occurrences = TermOccurrences.get(s).get(document);
                    }
                    else{
                        System.out.println("tf does not contain"+s);
                        System.exit(-1);
                    }

                    System.out.println(occurrences+"/"+terms);
                    TermFrequencies.add(document, (double)occurrences /(double)terms);
                    if(TermFrequencies.get(document) > 0 ){
                        document_appears++;
                    }
                }

                double DocumentFrequency = (double)document_appears/(double)documents_num;
                //System.out.println("Document frequency = "+DocumentFrequency);
                double sumTFIDF = 0.0;
                for(int document = 0 ; document < documents_num ; document++){
                    sumTFIDF += TermFrequencies.get(document)*DocumentFrequency;
                    //System.out.println(TermFrequencies.get(document) + "*"+ DocumentFrequency);
                }
                TFIDF_map.put(s,sumTFIDF/(double)documents_num);
            }


            for(String s : TFIDF_map.keySet()){
                double val = TFIDF_map.get(s);
                //System.out.println(s+" TFIDF = "+val);
                if(val <= filter) StopWords.add(new Sequence(s));
            }


        } catch (IOException e) {
            e.printStackTrace();
        }
        return StopWords;

    }

    private static Sequences DF(String directory_path,int k,double filter) throws IOException {
        IOHandler handler;
        Sequences stop_words = new Sequences();
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

            //run the list of directories
            //ignore all items of list that are not fasta files

            int fasta_num = 0;
            HashMap<String, Integer> map = new HashMap<>();
            for (String aPathList : pathList) {
                if (aPathList.endsWith(".fas")) {
                    fasta_num++;
                    handler = new IOHandler();
                    ArrayList<String> content = handler.readFAS(aPathList);

                    ArrayList<String> ngram_list = new ArrayList<>();
                    for (String s : content) {
                        List<String> subsequence = new ArrayList<>();
                        for (int start = 0; start < (s.length() / k); start += k) {
                            subsequence.add(s.substring(start, start + k));
                        }
                        ngram_list.addAll(subsequence);
                    }

                    for (String s : ngram_list) {
                        map.putIfAbsent(s, 0);
                        int occurences = map.get(s) + 1;
                        //System.out.println(s+":"+occurences);
                        map.put(s, occurences);
                    }
                }
            }
            System.out.println("Number of fasta files = "+fasta_num);
            for(String entry : map.keySet()){
                System.out.println("entry "+entry+" : "+map.get(entry));
                if(map.get(entry) >= filter*fasta_num){
                    stop_words.add(new Sequence(entry));
                    System.out.println("\t\tentry "+entry+" : "+map.get(entry));
                }

            }
        }
        return stop_words;

    }
    /*
    protected static List<String> returnNonHapax(int k, ArrayList<String> seq){

        HashMap<String, Integer> map = new HashMap<>();
        for (String s : seq) {
            List<String> subsequence = new ArrayList<>();
            for (int start = 0; start < (s.length() / k); start += k) {
                subsequence.add(s.substring(start, start + k));
            }
            for (String ngram : subsequence) {
                map.putIfAbsent(ngram, 0);
                int occurences = map.get(ngram) + 1;
                map.put(ngram, occurences);
            }
        }

        ArrayList<String> non_hapax_list = new ArrayList<>() ;
        for (String entry : map.keySet()) {
            if (map.get(entry) != 1)non_hapax_list.add(entry);
        }
        return non_hapax_list;
    }
    */


}
