package thesisinterface.VectorRepresentation.ProbabilisticModel;/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//import Interface.IOHandler;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Collectors;
import java.util.*;

/**
 *
 * @author marin
 */
public class AddAttribute {

    public static Instances merge(String file1,String file2)throws Exception {
        Instances instances1 = new Instances(new BufferedReader(new FileReader(file1)));
        Instances instances2 = new Instances(new BufferedReader(new FileReader(file2)));
        instances1.addAll(instances2);
        return instances1;
    }

        public static void main(String[] args) throws Exception {

            String parentFolder = "/root/IdeaProjects/ProbabilisticModel/Sparse_Files";
            String procedure = "filter";


            Path folderPath = Paths.get(parentFolder);
            List<Path> subfolder = Files.walk(folderPath, 1)
                    .filter(Files::isDirectory)
                    .collect(Collectors.toList());
            subfolder.remove(0);

            System.out.println(subfolder.toString());

            Instances data1 = null;
            Instances data2 = null;
            Instances newData1 = null;
            Instances newData2 = null;

            Attribute classAttr = null;
            for (int i = 0; i < subfolder.size(); i++) {
                File f = subfolder.get(i).toFile();
                String[] fileList = f.list();
                String comparisonNum = subfolder.get(i).toString().replace(parentFolder, "");
                System.out.println(Arrays.toString(fileList));

                data1 = new Instances(new BufferedReader(new FileReader(subfolder.get(i).toAbsolutePath() + "/" + fileList[0])));
                data2 = new Instances(new BufferedReader(new FileReader(subfolder.get(i).toAbsolutePath() + "/" + fileList[1])));

                String label1 = fileList[0].replace(subfolder.get(i).toString(), "").replace(".fas.arff", "");
                String label2 = fileList[1].replace(subfolder.get(i).toString(), "").replace(".fas.arff", "");

                // load dataset
                if (data1.numAttributes() != data2.numAttributes()) {
                    paddingDatasets(data1, data2);
                }
                switch (procedure) {
                    case "filter":
                        Add filter;
                        // 1. nominal attribute
                        filter = new Add();
                        filter.setAttributeIndex("last");
                        filter.setNominalLabels(label1 + ", " + label2);
                        filter.setAttributeName("Type");
                        filter.setInputFormat(data1);
                        newData1 = Filter.useFilter(data1, filter);
                        newData2 = Filter.useFilter(data2, filter);

                        for (int k = newData1.numInstances() - 1; k >= 0; k--) {
                            newData1.instance(k).setValue(newData1.numAttributes() - 1, label1);
                        }
                        for (int j = newData2.numInstances() - 1; j >= 0; j--) {
                            newData2.instance(j).setValue(newData2.numAttributes() - 1, label2);
                        }

                        newData1.setClassIndex(newData1.numAttributes() - 1);
                        newData2.setClassIndex(newData2.numAttributes() - 1);

                        //append all instances
                        for (int c = 0; c < newData2.numInstances(); c++) {
                            newData1.add(newData2.get(c));
                        }
                        break;
                    default:
                        System.out.println("\nUsage: AddAttribute <Parent_folder> <filter|java>\n");
//                        System.exit(2);
                }

                try {

                    ArffSaver saver = new ArffSaver();
                    saver.setInstances(newData1);

                    saver.setFile(new File(subfolder.get(i).toString().replace("/Comparison", "/Combined") + ".arff"));
                    saver.writeBatch();

                } catch (IOException e) {
                }

            }

        }
        /*
            Instances merge_instances = merge(args[0],args[1]);
            merge_instances.randomize(new Random());
            IOHandler io = new IOHandler();
            io.writeARFF("/root/IdeaProjects/ProbabilisticModel/merge_sparse.arff", merge_instances);
        */

    public static void paddingDatasets(Instances dataset1, Instances dataset2) {
        if (dataset1.numAttributes() < dataset2.numAttributes()) {
            int diff = dataset2.numAttributes() - dataset1.numAttributes();
            int counter = dataset1.numAttributes()+1;
            for (int i = 0; i < diff; i++) {
                dataset1.insertAttributeAt(new Attribute("attribute" + counter), dataset1.numAttributes());
                for (int instCounter = 0; instCounter < dataset1.numInstances(); instCounter++) {
                    dataset1.instance(instCounter).setValue(counter, 0.0);
                    counter++;
                }
            }
        } else {
            int diff = dataset1.numAttributes() - dataset2.numAttributes();
            int counter = dataset2.numAttributes() + 1;
            for (int i = 0; i < diff; i++) {
                dataset2.insertAttributeAt(new Attribute("attribute" + counter), dataset2.numAttributes());
                for (int instCounter = 0; instCounter < dataset1.numInstances(); instCounter++) {
                    dataset1.instance(instCounter).setValue(counter, 0.0);
                    counter++;
                }
            }
        }
    }
}
