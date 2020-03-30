package thesisinterface;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Fasta2Comp {

    /*
    Given a single directory with N fasta files, create (N choose 2) directories
    for each binary classification experiment
    */


    public static void create_directory(String dirname) {
        File directory = new File(dirname);
        if (!directory.exists()) {
            directory.mkdir();
        }
    }


    public static void copy_file_to_dir(String filepath, String dirname) throws IOException {
        File file = new File(filepath);
        String filename = file.getName();
        Files.copy(Paths.get(filepath), Paths.get(dirname + "/" + filename));
        System.out.println("File  " + filename + " renamed and moved successfully");
    }

    public static void main(String[] args) throws Exception {


        String FastasDir = "/root/IdeaProjects/ProbabilisticModel/Mammals/Fasta_Files";
        try (Stream<Path> paths = Files.walk(Paths.get(FastasDir))) {
            //create a list of the directories of all subfolders and files contained
            List<String> pathList = paths.map(p -> {
                if (Files.isDirectory(p)) {
                    return "/" + p.toString();
                }
                return p.toString();
            })
                    .peek(System.out::println) // write all results in console for debug
                    .collect(Collectors.toList());


            //Interfaces only support .fas strin
            ArrayList<String> FastaFiles = new ArrayList<>();
            for (int i = 0; i < pathList.size(); i++) { //each directory should contain exactly two fasta files
                String filename = pathList.get(i);
                if (filename.endsWith(".fas")) {
                    FastaFiles.add(filename);
                }
            }

            int c = 0;
            for (int i = 0; i < FastaFiles.size(); i++) {
                for (int j = 0; j < FastaFiles.size(); j++) {
                    if (i > j) {
                        System.out.println(FastaFiles.get(i) + " VS " + FastaFiles.get(j));
                        c += 1;
                        String dirname = FastasDir + "/" + "Comparison" + c;
                        create_directory(dirname);
                        copy_file_to_dir(FastaFiles.get(i), dirname);
                        copy_file_to_dir(FastaFiles.get(j), dirname);
                    }

                }
            }
            System.out.println("Created " + c + " Comparison Files");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}


