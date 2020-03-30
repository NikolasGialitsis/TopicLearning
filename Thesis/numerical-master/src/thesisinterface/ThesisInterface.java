/*
 * To change this license fastaHeader, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package thesisinterface;

import thesisinterface.VectorRepresentation.BaseClasses.BaseFeatureVector;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static thesisinterface.VectorRepresentation.SingleValue.AtomicNumberRepresentation.atomicNumberRepresentation;

/**
 * @author marina Process: 1)create instances of each class of representation
 * 2)run through the data file once to create the representation and find the
 * number of attributes that will be declared in the arff file 3)create the arff
 * file by declaring the number of attributes found before, running through the
 * representation file and constructing a file that WEKA can read
 */
public class ThesisInterface {

    /**
     * @param args the command line arguments
     * @throws IOException
     */
    private static final Pattern fastaHeader = Pattern.compile("^>.*");

    public static File outputSparseFile;

    public static void main(String[] args) throws IOException {

        //open folder of fasta datasets 
        try (Stream<Path> paths = Files.walk(Paths.get("D:/Marina/Documents/Code Solve/Fasta Files"))) {

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
            for (int i = 0; i < pathList.size(); i++) {

                //ignore all items of list that are not fasta files
                if (!pathList.get(i).endsWith(".fas")) {
                } else {
                    File toReadFile = new File(pathList.get(i));
                    System.out.println(pathList.get(i));
                    //create file for specific type of representation a
                    outputSparseFile = new File(pathList.get(i).replace("Fasta Files", "Atomic"));
                    System.out.println(outputSparseFile.toString());

                    //create sparse representation
                    makeSparse(toReadFile.getPath());
                }
            }

        }
    }

    // Nikiforos method
    //to create the sparse ARFF representation file
    public static int makeSparse(String fasPath) {
        IOHandler io = new IOHandler();
        int maxDim = -1;
        try {
            // read fas
            ArrayList<String> contents = io.readFAS(fasPath);

            // find max vector dimension
            for (String line : contents) {
                if (line.length() > maxDim) {
                    maxDim = line.length();
                }
            }
            // create the sparse vector
            ArrayList<BaseFeatureVector> data = new ArrayList<>();
//            int iInstanceCnt = 0;
//            int count = 0;

            // for each base sequence
            for (String line : contents) {
//                System.out.println("Instance #" + count++ + " : " + line);

                // get repr
                BaseFeatureVector representation = atomicNumberRepresentation(line);

                // pad with zeros, based on the number of inner dimensions per symbol
                // TODO: In the future we need to be able to support varying inner dimension
                int iInnerDim = representation.get(1).size();
                representation.sparsify(maxDim, iInnerDim);
                // add to list
                data.add(representation);
            }

            System.err.println("Dimensions initialized.");

            // convert to weka instances

            System.err.println("Converting to ARFF instances...");
            Instances instances = toWekaInstances(data);

            System.err.println("Converting to ARFF instances... Done.");

            // write arff file
            io.writeARFF(outputSparseFile + ".arff", instances);

        } catch (FileNotFoundException e) {
        } catch (IOException e) {
        }
        System.out.println("DONE making sparse file.");
        return maxDim;

    }

    // convert data to WEKA instances
    public static Instances toWekaInstances(ArrayList<BaseFeatureVector> data) {

        System.out.println("WEKA instances making");

        ArrayList<Attribute> attributeList = new ArrayList();
        int maxDim = data.get(0).size();
        int maxInnerDim = data.get(0).get(0).size();
        System.out.println(maxDim);
        System.out.println(maxInnerDim);
        int iDimCount = 0; // Start with zero

        System.err.println("Creating attributes...");

        // For every symbol-related dimension
        for (int i = 1; i <= maxDim; ++i) {
            // Count number of measurements/components in that dimension
            try {

                // For every component in the related dimension
                for (int innerDim = 0; innerDim < maxInnerDim; ++innerDim) {
                    Attribute att = new Attribute(("attribute" + (iDimCount + 1)), false);
                    attributeList.add(att);
                    iDimCount++; // Get next number
                }

            } catch (NullPointerException e) {
                throw new IllegalStateException(e);
            }
        }
        System.err.println("Creating attributes... Done.");


        System.err.println("Creating instances...");
        Instances instances = new Instances("data", attributeList, data.size());

        // For every data instance
        for (int v = 0; v < data.size(); ++v) {
            System.out.println("Weka'ing instance # " + v + " / " + data.size());

            // Init empty instance
            Instance instance = new SparseInstance(attributeList.size());
            BaseFeatureVector vec = data.get(v);

            //System.out.println("len of key set is " + vec.keySet().size());

            // Count number of dimensions in that instance
            maxDim = vec.size();
            int iCurDim = 0; // Here we keep the overall dimension counter

            // For every dimension
            for (int dim = 0; dim < maxDim; ++dim) {

                // Get list of components for specific dimension
                List<Double> dlist = vec.get(dim);

                // For every component in the related dimension
                for (int innerDim = 0; innerDim < dlist.size(); ++innerDim) {
                    System.out.println("Attempting to access index " + innerDim + " in " + iCurDim + " of the list with len " + dlist.size());
//                    System.out.println("Attempting to access global index " + iCurDim + " of the global attr list with len " + attributeList.size());
                    System.out.println(iCurDim);
                    instance.setValue(attributeList.get(iCurDim), dlist.get(innerDim));

                }
                // Move to next dimension
                iCurDim++;
            }
//
            if (v % 100 == 0) {
                System.err.print(".");
                if (v % 1000 == 0) {
                    System.err.print("X \n");
                }
            }

            instances.add(instance);
        }
        System.err.println("Creating instances... Done.");

        return instances;
    }

}
