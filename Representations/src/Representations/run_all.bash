echo '@SequenceEncoding '
echo 'Convert json to readable format for training set...'
python Json2Text.py -train
echo '...Done!'
echo 'Convert json to readable format for testing set...'
python Json2Text.py -test;
echo '...Done!'
echo 'Compiling Probabilistic_Interface...'
javac -cp ".:/home/superuser/Mallet/dist/mallet.jar:/home/superuser/Mallet/dist/mallet-depts.jar:/home/superuser/Mallet/lib/mallet-depts.jar:/home/superuser/Mallet/lib/derby.jar;" Probabilistic_Interface.java
echo '...Done!'
echo 'Get representations for training set...'
java Probabilistic_Interface -train -topics 16 -dataset multiling -path /home/superuser/SequenceEncoding/Representations -stop /home/superuser/SequenceEncoding/Representations/stopwords.txt;
echo '...Done!'
echo 'Training model...'
python Train_Model.py -prob;
echo '...Done!'
echo 'Load values for testing set..'
java Probabilistic_Interface -test -topics 16 -dataset multiling -path /home/superuser/SequenceEncoding/Representations -stop /home/superuser/SequenceEncoding/Representations/stopwords.txt;
echo '...Done!'
echo 'Classify sentences based on probabilistic representation...'
python Predict_Label.py -prob > results_prob.txt
echo '...Done!'