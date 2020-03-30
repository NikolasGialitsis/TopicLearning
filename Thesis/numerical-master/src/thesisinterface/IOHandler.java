/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package thesisinterface;

/**
 * @author marina
 */

import thesisinterface.VectorRepresentation.BaseClasses.BaseFeatureVector;
import weka.core.converters.ArffSaver;
import weka.core.Instances;


import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Pattern;


public class IOHandler {
    private static final String FAS_EXTENSION = ".fas";
    private static final Pattern fastaHeader = Pattern.compile("^>.*");


    public ArrayList<String> readFAS(String path) {
        ArrayList<String> contents = new ArrayList<>();
        ArrayList<String> headers = new ArrayList<>();
        try {
            Scanner readDataFile = new Scanner(new FileReader(new File(path)));

            // read all contents and mark max vector dimension
            while (readDataFile.hasNextLine()) {
                String line = readDataFile.nextLine();
                if (line.matches(fastaHeader.pattern())) headers.add(line);
                else contents.add(line);
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
        return contents;
    }

    public void writeCSV(String path, ArrayList<BaseFeatureVector> data) {

        File out = new File(path);
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(out));
            for (BaseFeatureVector vec : data) {
                System.out.println(vec);
                for (int i = 0; i < vec.getNumberOfDimensions(); ++i) {
                    Integer dimName = vec.getDimensionNames().get(i);
                    List<Double> dimValues = vec.get(dimName);
                    for (double val : dimValues) bw.write(Double.toString(val));
                    if (i < vec.getNumberOfDimensions() - 1) bw.write(",");
                }
                bw.write(System.lineSeparator());

            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
        }

    }


    public void read(String path) {
        //if (path.endsWith(FAS_EXTENSION)) return readFAS(path);
        return;

    }

    public void writeARFF(String path, Instances data) {

        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(path));
            saver.writeBatch();
        } catch (Exception e) {

        }


    }
}
